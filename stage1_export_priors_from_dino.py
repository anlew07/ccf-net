#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _import_project_modules():
    try:
        import dataset as dataset_mod  # type: ignore
    except Exception:
        from data import dataset as dataset_mod  # type: ignore

    import dino1_net as dino1_net_mod  # type: ignore
    Net = dino1_net_mod.Net
    return dataset_mod, Net, dino1_net_mod


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_existing_image(image_root: Path, stem: str) -> Path:
    for ext in IMG_EXTS:
        p = image_root / f"{stem}{ext}"
        if p.exists():
            return p
    cands = list(image_root.glob(f"{stem}.*"))
    for p in cands:
        if p.suffix.lower() in IMG_EXTS:
            return p
    raise FileNotFoundError(f"No image found for stem={stem} under {image_root}")


class TrainImageOnlyDataset(Dataset):
    def __init__(self, data_root: str, cfg_mean: np.ndarray, cfg_std: np.ndarray, resize_hw: Tuple[int, int] = (320, 320)):
        self.data_root = Path(data_root)
        self.image_root = self.data_root / "train" / "Image"
        self.resize_hw = resize_hw
        self.mean = cfg_mean.astype(np.float32)
        self.std = cfg_std.astype(np.float32)

        train_txt = self.data_root / "train.txt"
        if not train_txt.exists():
            raise FileNotFoundError(f"train.txt not found: {train_txt}")

        self.samples: List[Tuple[str, Path]] = []
        with open(train_txt, "r", encoding="utf-8") as f:
            for line in f:
                stem = line.strip()
                if not stem:
                    continue
                img_path = find_existing_image(self.image_root, stem)
                self.samples.append((stem, img_path))

        if not self.samples:
            raise RuntimeError(f"No training images found under {self.image_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem, img_path = self.samples[idx]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = img_bgr[:, :, ::-1].astype(np.float32)
        h, w = img.shape[:2]

        img_rs = cv2.resize(img, self.resize_hw[::-1], interpolation=cv2.INTER_LINEAR)
        img_norm = (img_rs - self.mean) / self.std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).contiguous().float()
        return img_tensor, int(h), int(w), stem


def save_prob_png(prob: np.ndarray, path: Path) -> None:
    prob = np.clip(prob, 0.0, 1.0)
    Image.fromarray((prob * 255.0).astype(np.uint8)).save(path)


def _to_numpy_map(x: torch.Tensor, h: int, w: int, apply_sigmoid: bool = False) -> np.ndarray:
    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        x = x[0]
    if apply_sigmoid:
        x = torch.sigmoid(x)
    x = x.detach().float().cpu().numpy()
    if x.shape != (h, w):
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(x, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="CodDataset")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--dinov3_ckpt", type=str, default=None)
    parser.add_argument("--dinov3_arch", type=str, default=None)
    parser.add_argument("--cue_fallback", type=str, default="main", choices=["main", "zeros"])
    args = parser.parse_args()

    dataset_mod, Net, dino1_net_mod = _import_project_modules()
    print(f"[Info] importing Net from: {getattr(dino1_net_mod, '__file__', 'unknown')}")

    out_dir = Path(args.out_dir)
    main_dir = out_dir / "main_prob"
    cue_dir = out_dir / "cue_full"
    fused_dir = out_dir / "fused_prior"
    ensure_dir(main_dir)
    ensure_dir(cue_dir)
    ensure_dir(fused_dir)

    stat_cfg = dataset_mod.Config(datapath=args.data_root, mode="test")
    ds = TrainImageOnlyDataset(args.data_root, stat_cfg.mean, stat_cfg.std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 用 mode='train' 更贴近你当前 validate 的真实调用路径，兼容性更高。
    net_cfg_kwargs = dict(datapath="000", mode="train")
    if args.dinov3_ckpt is not None:
        net_cfg_kwargs["dinov3_ckpt"] = args.dinov3_ckpt
    if args.dinov3_arch is not None:
        net_cfg_kwargs["dinov3_arch"] = args.dinov3_arch
    net_cfg = dataset_mod.Config(**net_cfg_kwargs)

    net = Net(net_cfg)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(state, strict=True)
    net.to(args.device)
    net.eval()

    meta = {
        "ckpt": os.path.abspath(args.ckpt),
        "data_root": os.path.abspath(args.data_root),
        "alpha": float(args.alpha),
        "num_images": len(ds),
        "net_module": getattr(dino1_net_mod, "__file__", "unknown"),
        "cue_fallback": args.cue_fallback,
    }
    with open(out_dir / "export_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    missing_aux_count = 0
    with torch.no_grad():
        for image, hs, ws, stems in tqdm(loader, desc="Export priors"):
            image = image.to(args.device, non_blocking=True)
            outputs = net(image)
            aux = getattr(net, "aux_cache", {}) or {}

            if isinstance(outputs, (tuple, list)):
                main_logit = outputs[0]
            else:
                main_logit = outputs

            bs = image.shape[0]
            for b in range(bs):
                stem = stems[b]
                h, w = int(hs[b]), int(ws[b])

                main_prob = _to_numpy_map(main_logit[b:b+1], h, w, apply_sigmoid=True)

                cue_map = None
                if "cue_full" in aux and aux["cue_full"] is not None:
                    cue_map = _to_numpy_map(aux["cue_full"][b:b+1], h, w, apply_sigmoid=False)
                elif "cue_edge_1_4" in aux and aux["cue_edge_1_4"] is not None:
                    cue_edge = aux["cue_edge_1_4"][b:b+1]
                    if "cue_sem_1_8" in aux and aux["cue_sem_1_8"] is not None:
                        cue_sem = F.interpolate(aux["cue_sem_1_8"][b:b+1], size=cue_edge.shape[-2:], mode="bilinear", align_corners=True)
                        cue_mix = (0.8 * cue_edge + 0.2 * cue_sem).clamp(0, 1)
                    else:
                        cue_mix = cue_edge.clamp(0, 1)
                    cue_map = _to_numpy_map(cue_mix, h, w, apply_sigmoid=False)
                else:
                    missing_aux_count += 1
                    if args.cue_fallback == "main":
                        cue_map = main_prob.copy()
                    else:
                        cue_map = np.zeros_like(main_prob, dtype=np.float32)

                fused_prior = np.clip(args.alpha * main_prob + (1.0 - args.alpha) * cue_map, 0.0, 1.0)

                save_prob_png(main_prob, main_dir / f"{stem}.png")
                save_prob_png(cue_map, cue_dir / f"{stem}.png")
                save_prob_png(fused_prior, fused_dir / f"{stem}.png")

    print(f"[Done] main_prob   -> {main_dir}")
    print(f"[Done] cue_full    -> {cue_dir}")
    print(f"[Done] fused_prior -> {fused_dir}")
    print(f"[Info] missing cue aux fallback used on {missing_aux_count} images")


if __name__ == "__main__":
    main()
