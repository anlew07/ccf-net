# utils/visualizer.py
# Save visualization artifacts for C²F-Net (DINOv3 + CP + CoF/TFGM)
#
# ✅ 修改点（按你当前 aux_cache keys 对齐）：
#   - cue_* -> cons_*（cons_full / cons_sem_1_8 / cons_edge_1_4）
#   - cof_* -> tfgm_*（tfgm_freq_in / tfgm_freq_out / tfgm_g_sp / tfgm_g_ch）
# ✅ 修改点（按你要求）：
#   - 不再保存 raw_tensors.pt（避免文件巨大/写盘报错）
# ✅ 兼容：
#   - image / gt / pred 与 aux 可视化分辨率不一致时，自动 resize 对齐
#
# Usage:
#   from utils.visualizer import save_visualization
#   save_visualization(save_dir, image, gt, pred_prob, aux_cache, prefix="xxx_")

import os
from typing import Dict, Optional, Any

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


# ----------------------------
# basic utils
# ----------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _to_4d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [B,C,H,W] format (best-effort)."""
    if x is None:
        return None
    if x.dim() == 2:          # [H,W]
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        # [C,H,W] or [B,H,W]
        if x.shape[0] in (1, 3):  # treat as [C,H,W]
            x = x.unsqueeze(0)
        else:                      # treat as [B,H,W]
            x = x.unsqueeze(1)
    elif x.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported tensor dim: {x.dim()}")
    return x


def _take_first(x: torch.Tensor) -> torch.Tensor:
    """Take batch first => [1,C,H,W]"""
    x = _to_4d(x)
    if x is None:
        return None
    if x.size(0) > 1:
        x = x[:1]
    return x


def _norm_01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize to [0,1] per tensor (over spatial dims)."""
    x = x.float()
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def _as_1ch(x: torch.Tensor) -> torch.Tensor:
    """Convert to [1,1,H,W] (mean over channel if needed)."""
    x = _take_first(x)
    if x is None:
        return None
    if x.size(1) != 1:
        x = x.mean(dim=1, keepdim=True)
    return x


def _as_3ch(x: torch.Tensor) -> torch.Tensor:
    """Convert to [1,3,H,W] (repeat if 1ch)."""
    x = _take_first(x)
    if x is None:
        return None
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.size(1) != 3:
        if x.size(1) >= 3:
            x = x[:, :3]
        else:
            x = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    return x


def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Resize x to ref spatial size if needed."""
    x = _take_first(x)
    ref = _take_first(ref)
    if x is None or ref is None:
        return x
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    align = False if mode in ("nearest", "area") else False  # keep safe
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=align if mode in ("bilinear","bicubic","trilinear","linear") else None)


def _safe_save_image(x: torch.Tensor, path: str):
    """Save image tensor in [0,1], [1,C,H,W]."""
    x = _take_first(x)
    if x is None:
        return
    x = x.clamp(0, 1).cpu()
    save_image(x, path)


def _make_overlay(rgb_01_3ch: torch.Tensor, mask_01_1ch: torch.Tensor, alpha: float = 0.55):
    """
    Simple overlay: rgb*(1-a*m) + red*(a*m)
    """
    rgb = _as_3ch(rgb_01_3ch).clamp(0, 1)
    m = _as_1ch(mask_01_1ch).clamp(0, 1)
    # ensure same size
    m = _resize_like(m, rgb, mode="nearest")

    red = torch.zeros_like(rgb)
    red[:, 0:1] = 1.0
    out = rgb * (1.0 - alpha * m) + red * (alpha * m)
    return out.clamp(0, 1)


def _downsave_heatmap(x: torch.Tensor, ref_rgb: torch.Tensor, path: str):
    """
    Save as grayscale heatmap-like:
    - to 1ch
    - resize to ref
    - normalize
    - save
    """
    x1 = _as_1ch(x)
    if x1 is None:
        return
    x1 = _resize_like(x1, ref_rgb, mode="bilinear")
    x1 = _norm_01(x1).clamp(0, 1)
    _safe_save_image(x1, path)


def _downsave_gate(x: torch.Tensor, ref_rgb: torch.Tensor, path: str):
    """
    For gates:
      - spatial gate [1,1,H,W] -> heatmap
      - channel gate [1,C,1,1] -> barcode map
    """
    x = _take_first(x)
    if x is None:
        return
    x = x.float()

    # channel gate often [1,C,1,1]
    if x.size(-1) == 1 and x.size(-2) == 1 and x.size(1) > 1:
        bar = x.squeeze(-1).squeeze(-1)           # [1,C]
        bar = bar.unsqueeze(0).unsqueeze(0)       # [1,1,1,C]
        bar = _norm_01(bar).clamp(0, 1)
        bar_up = F.interpolate(bar, size=(256, 32), mode="nearest")
        _safe_save_image(bar_up, path)
    else:
        _downsave_heatmap(x, ref_rgb, path)


# ----------------------------
# main API
# ----------------------------

@torch.no_grad()
def save_visualization(
    save_dir: str,
    image: torch.Tensor,
    gt: Optional[torch.Tensor],
    pred: torch.Tensor,
    aux_cache: Optional[Dict[str, Any]] = None,
    prefix: str = "",
):
    """
    Saves:
      - {prefix}rgb.png
      - {prefix}gt.png (if gt)
      - {prefix}pred_prob.png, {prefix}pred_bin.png
      - {prefix}overlay_pred.png, {prefix}overlay_gt.png (if gt)
      - aux_cache -> png (if key exists):
          enc_score_full
          cons_full, cons_sem_1_8, cons_edge_1_4
          tfgm_freq_in, tfgm_freq_out, tfgm_g_sp, tfgm_g_ch
    """
    _ensure_dir(save_dir)

    # ---- rgb ----
    rgb = _take_first(image).float()
    rgb_01 = _norm_01(_as_3ch(rgb))  # safe for any range
    _safe_save_image(rgb_01, os.path.join(save_dir, f"{prefix}rgb.png"))

    # ---- gt ----
    gt1 = None
    if gt is not None:
        gt1 = _as_1ch(gt).float()
        gt1 = _resize_like(gt1, rgb_01, mode="nearest")
        if gt1.max() > 1.5:
            gt1 = (gt1 > 0).float()
        else:
            gt1 = (gt1 > 0.5).float()
        _safe_save_image(gt1, os.path.join(save_dir, f"{prefix}gt.png"))
        overlay_gt = _make_overlay(rgb_01, gt1)
        _safe_save_image(overlay_gt, os.path.join(save_dir, f"{prefix}overlay_gt.png"))

    # ---- pred ----
    pred1 = _as_1ch(pred).float().clamp(0, 1)
    pred1 = _resize_like(pred1, rgb_01, mode="bilinear")
    _safe_save_image(pred1, os.path.join(save_dir, f"{prefix}pred_prob.png"))

    pred_bin = (pred1 > 0.5).float()
    _safe_save_image(pred_bin, os.path.join(save_dir, f"{prefix}pred_bin.png"))

    overlay_pred = _make_overlay(rgb_01, pred1)
    _safe_save_image(overlay_pred, os.path.join(save_dir, f"{prefix}overlay_pred.png"))

    # ---- aux cache ----
    if aux_cache is None:
        aux_cache = {}

    def save_k(name: str, kind: str = "heat"):
        v = aux_cache.get(name, None)
        if v is None or (not torch.is_tensor(v)):
            return
        out_path = os.path.join(save_dir, f"{prefix}{name}.png")
        if kind == "gate":
            _downsave_gate(v, rgb_01, out_path)
        else:
            _downsave_heatmap(v, rgb_01, out_path)

    # ====== aligned with your current aux_cache keys ======
    save_k("enc_score_full", kind="heat")

    # consistency / cue (your keys are cons_*)
    save_k("cons_full", kind="heat")
    save_k("cons_sem_1_8", kind="heat")
    save_k("cons_edge_1_4", kind="heat")

    # CoF/TFGM debug (your keys are tfgm_*)
    save_k("tfgm_freq_in", kind="heat")
    save_k("tfgm_freq_out", kind="heat")
    save_k("tfgm_g_sp", kind="heat")
    save_k("tfgm_g_ch", kind="gate")

    # ✅ 不保存 raw_tensors.pt（按你要求）
