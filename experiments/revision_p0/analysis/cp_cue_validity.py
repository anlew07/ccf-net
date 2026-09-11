#!/usr/bin/env python3
# coding=utf-8

"""Post-hoc validity analysis for CCF-Net consistency-prior cues.

Inference-only: no optimizer, backward, parameter update, or checkpoint write.
Measures whether the existing semantic, edge, and fused CP cues are associated
with prediction errors of the submitted/best checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3] if "experiments/revision_p0/analysis" in str(Path(__file__).resolve()) else Path.cwd()
DINO_SRC_DEFAULT = Path("/root/shared-nvme/dinov3-main")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_dino_src = Path(os.environ.get("DINOV3_SRC", str(DINO_SRC_DEFAULT)))
if str(_dino_src) not in sys.path:
    sys.path.insert(0, str(_dino_src))

from data import dataset  # noqa: E402
from ccfnet import Net  # noqa: E402

CUE_KEYS = {
    "semantic": "cons_sem_1_8",
    "edge": "cons_edge_1_4",
    "fused": "cons_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CP cue/error association using an existing checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "cp_cof_1_1_dino_sem_edge" / "dino_0415" / "model-best.pth",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "CodDataset")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CAMO", "CHAMELEON"],
        help="Test sets under CodDataset/test/. Start small before COD10K/NC4K.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "out_revision_p0" / "cp_cue_validity",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="0 means all images in each requested dataset."
    )
    parser.add_argument(
        "--spearman-max-pixels", type=int, default=20000,
        help="Pixels sampled per image for Spearman rho; 0 uses all pixels."
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _finite_flat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def spearman_score(cue, error, max_pixels, rng) -> float:
    c = np.asarray(cue, dtype=np.float64).reshape(-1)
    e = np.asarray(error, dtype=np.float64).reshape(-1)
    valid = np.isfinite(c) & np.isfinite(e)
    c, e = c[valid], e[valid]
    if c.size < 2 or np.all(c == c[0]) or np.all(e == e[0]):
        return float("nan")
    if max_pixels > 0 and c.size > max_pixels:
        idx = rng.choice(c.size, size=max_pixels, replace=False)
        c, e = c[idx], e[idx]
    result = spearmanr(c, e)
    return float(getattr(result, "statistic", result[0]))


def binary_auc(scores, labels) -> float:
    """Tie-correct ROC AUC via the Mann-Whitney rank statistic."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).astype(bool).reshape(-1)
    valid = np.isfinite(s)
    s, y = s[valid], y[valid]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    rank_sum_pos = float(ranks[y].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_average_precision(scores, labels) -> float:
    """Threshold-grouped AP, avoiding arbitrary ordering within tied scores."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).astype(np.int64).reshape(-1)
    valid = np.isfinite(s)
    s, y = s[valid], y[valid]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    if s.size == 1:
        return float(y[0])
    group_ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), s.size - 1]
    tp_all = np.cumsum(y)
    tp = tp_all[group_ends].astype(np.float64)
    predicted = (group_ends + 1).astype(np.float64)
    precision = tp / predicted
    recall = tp / float(n_pos)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def top_fraction_metrics(cue, abs_error, misclassified, fraction=0.20):
    c = np.asarray(cue, dtype=np.float64).reshape(-1)
    e = np.asarray(abs_error, dtype=np.float64).reshape(-1)
    m = np.asarray(misclassified, dtype=np.float64).reshape(-1)
    valid = np.isfinite(c) & np.isfinite(e) & np.isfinite(m)
    c, e, m = c[valid], e[valid], m[valid]
    if c.size == 0:
        return {k: float("nan") for k in [
            "top20_abs_error", "global_abs_error", "top20_abs_error_enrichment",
            "top20_misclassification_rate", "global_misclassification_rate",
            "top20_misclassification_enrichment",
        ]}
    k = max(1, int(math.ceil(c.size * fraction)))
    top_idx = np.argsort(c, kind="mergesort")[-k:]
    global_err, top_err = float(e.mean()), float(e[top_idx].mean())
    global_mis, top_mis = float(m.mean()), float(m[top_idx].mean())
    return {
        "top20_abs_error": top_err,
        "global_abs_error": global_err,
        "top20_abs_error_enrichment": top_err / global_err if global_err > 0 else float("nan"),
        "top20_misclassification_rate": top_mis,
        "global_misclassification_rate": global_mis,
        "top20_misclassification_enrichment": top_mis / global_mis if global_mis > 0 else float("nan"),
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    a = _finite_flat(np.asarray(list(values), dtype=np.float64))
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "n": int(a.size), "mean": float(a.mean()),
        "median": float(np.median(a)), "std": float(a.std(ddof=0)),
    }


def resize_cue(cue: torch.Tensor, target_hw):
    if cue.shape[-2:] == target_hw:
        return cue
    return F.interpolate(cue, size=target_hw, mode="bilinear", align_corners=False)


def make_loader(data_root: Path, dataset_name: str, num_workers: int):
    datapath = data_root / "test" / dataset_name
    cfg = dataset.Config(datapath=str(datapath), mode="test")
    data = dataset.Data(cfg)
    return DataLoader(data, batch_size=1, shuffle=False, num_workers=num_workers)


def run_dataset(net, data_root, dataset_name, device, max_images,
                spearman_max_pixels, seed, num_workers):
    loader = make_loader(data_root, dataset_name, num_workers)
    rows: List[Dict[str, object]] = []
    total = len(loader.dataset) if max_images <= 0 else min(max_images, len(loader.dataset))
    print(f"\n===== {dataset_name}: analysing {total} image(s) =====")

    with torch.no_grad():
        for idx, (image, mask, shape, name) in enumerate(loader):
            if max_images > 0 and idx >= max_images:
                break
            h, w = int(shape[0].item()), int(shape[1].item())
            image = image.to(device=device, dtype=torch.float32, non_blocking=True)
            main_logit, _ = net(image, (h, w))
            pred = torch.sigmoid(main_logit[0, 0])
            # Match test.py saved-map normalization.
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

            gt = mask[0, 0].to(device=device, dtype=torch.float32)
            if gt.shape != pred.shape:
                gt = F.interpolate(gt[None, None], size=pred.shape, mode="nearest")[0, 0]
            gt_bin = (gt >= 0.5).float()
            abs_error = (pred - gt_bin).abs()
            misclassified = ((pred >= 0.5) != (gt_bin >= 0.5)).float()

            err_np = abs_error.cpu().numpy().astype(np.float64)
            mis_np = misclassified.cpu().numpy().astype(np.uint8)
            cache = getattr(net, "aux_cache", {})
            missing = [v for v in CUE_KEYS.values() if cache.get(v) is None]
            if missing:
                raise RuntimeError(f"Missing cue(s) in net.aux_cache: {missing}")

            for cue_offset, (cue_name, cache_key) in enumerate(CUE_KEYS.items()):
                cue_t = resize_cue(cache[cache_key], pred.shape)[0, 0].clamp(0, 1)
                cue_np = cue_t.cpu().numpy().astype(np.float64)
                rng = np.random.default_rng(seed + idx * 17 + cue_offset)
                rho = spearman_score(cue_np, err_np, spearman_max_pixels, rng)
                auc = binary_auc(cue_np, mis_np)
                ap = binary_average_precision(cue_np, mis_np)
                top = top_fraction_metrics(cue_np, err_np, mis_np)
                rows.append({
                    "dataset": dataset_name,
                    "image": str(name[0]),
                    "cue": cue_name,
                    "prediction_mae": float(err_np.mean()),
                    "misclassification_rate": float(mis_np.mean()),
                    "cue_mean": float(cue_np.mean()),
                    "cue_max": float(cue_np.max()),
                    "cue_nonzero_ratio": float((cue_np > 1e-8).mean()),
                    "spearman_abs_error": rho,
                    "misclassification_auroc": auc,
                    "misclassification_auprc": ap,
                    **top,
                })
            if (idx + 1) % 25 == 0 or (idx + 1) == total:
                print(f"[{dataset_name}] {idx + 1}/{total}")
    return rows


def build_summary(rows):
    result = {}
    metrics = [
        "prediction_mae", "misclassification_rate", "cue_mean", "cue_max",
        "cue_nonzero_ratio", "spearman_abs_error", "misclassification_auroc",
        "misclassification_auprc", "top20_abs_error_enrichment",
        "top20_misclassification_enrichment",
    ]
    for ds in sorted({str(r["dataset"]) for r in rows}):
        result[ds] = {}
        for cue in CUE_KEYS:
            selected = [r for r in rows if r["dataset"] == ds and r["cue"] == cue]
            result[ds][cue] = {
                metric: summarize(float(r[metric]) for r in selected)
                for metric in metrics
            }
    return result


def print_summary(summary):
    print("\n===== CP CUE VALIDITY SUMMARY =====")
    header = (
        f"{'dataset':<12} {'cue':<10} {'rho':>8} {'AUROC':>8} {'AUPRC':>8} "
        f"{'err-x':>8} {'mis-x':>8} {'nz':>8}"
    )
    print(header)
    print("-" * len(header))
    for ds, cue_dict in summary.items():
        for cue, metrics in cue_dict.items():
            m = lambda k: float(metrics[k]["mean"])
            print(
                f"{ds:<12} {cue:<10} {m('spearman_abs_error'):8.4f} "
                f"{m('misclassification_auroc'):8.4f} {m('misclassification_auprc'):8.4f} "
                f"{m('top20_abs_error_enrichment'):8.4f} "
                f"{m('top20_misclassification_enrichment'):8.4f} "
                f"{m('cue_nonzero_ratio'):8.4f}"
            )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if not _dino_src.exists():
        raise FileNotFoundError(f"DINOv3 source not found: {_dino_src}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    print("===== CCF-Net CP cue validity analysis =====")
    print("repo_root:", REPO_ROOT)
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(device))
    print("checkpoint:", args.checkpoint)
    print("datasets:", args.datasets)
    print("max_images per dataset:", args.max_images or "all")

    net_cfg = dataset.Config(datapath=str(args.data_root), mode="test")
    net = Net(net_cfg).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(state, strict=True)
    net.eval()
    print("STRICT_LOAD=OK")

    all_rows = []
    for ds in args.datasets:
        all_rows.extend(run_dataset(
            net, args.data_root, ds, device, args.max_images,
            args.spearman_max_pixels, args.seed, args.num_workers,
        ))

    summary = build_summary(all_rows)
    print_summary(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = args.output_dir / "per_image.csv", args.output_dir / "summary.json"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    print("\nSaved:", csv_path, json_path, sep="\n  ")
    print("NOTE: inference-only; no optimizer/backward/checkpoint write occurred.")


if __name__ == "__main__":
    main()
