#!/usr/bin/env python3
# coding=utf-8

"""Compare CCF-Net CP cues against simple uncertainty baselines.

Inference-only:
- no optimizer
- no backward
- no parameter update
- no checkpoint write

Cues compared:
1) CP semantic discrepancy
2) CP edge discrepancy
3) CP fused discrepancy
4) prediction entropy
5) horizontal-flip TTA disagreement
6) random uniform cue (sanity baseline)

Error targets follow test.py:
- original prediction is sigmoid(logit)
- then per-image min-max normalization
- absolute error and 0.5-threshold misclassification are measured against GT
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

HERE = Path(__file__).resolve()
if "experiments/revision_p0/analysis" in str(HERE):
    REPO_ROOT = HERE.parents[3]
else:
    REPO_ROOT = Path.cwd()

DINO_SRC_DEFAULT = Path("/root/shared-nvme/dinov3-main")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_dino_src = Path(os.environ.get("DINOV3_SRC", str(DINO_SRC_DEFAULT)))
if str(_dino_src) not in sys.path:
    sys.path.insert(0, str(_dino_src))

from data import dataset  # noqa: E402
from ccfnet import Net  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare CP cues with entropy/TTA/random baselines."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "cp_cof_1_1_dino_sem_edge" / "dino_0415" / "model-best.pth",
    )
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "CodDataset")
    p.add_argument("--datasets", nargs="+", default=["CAMO", "CHAMELEON"])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "out_revision_p0" / "cp_vs_uncertainty",
    )
    p.add_argument("--max-images", type=int, default=0, help="0 means all images.")
    p.add_argument(
        "--spearman-max-pixels", type=int, default=20000,
        help="Per-image pixel subsample for Spearman. 0 means all."
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def finite_flat(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return a[np.isfinite(a)]


def summarize(values: Iterable[float]) -> Dict[str, float]:
    a = finite_flat(values)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=0)),
    }


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
    res = spearmanr(c, e)
    return float(getattr(res, "statistic", res[0]))


def binary_auc(scores, labels) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).astype(bool).reshape(-1)
    valid = np.isfinite(s)
    s, y = s[valid], y[valid]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    rank_sum_pos = float(ranks[y].sum())
    return float(
        (rank_sum_pos - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def binary_average_precision(scores, labels) -> float:
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
        return {
            "top20_abs_error": float("nan"),
            "global_abs_error": float("nan"),
            "top20_abs_error_enrichment": float("nan"),
            "top20_misclassification_rate": float("nan"),
            "global_misclassification_rate": float("nan"),
            "top20_misclassification_enrichment": float("nan"),
        }

    k = max(1, int(math.ceil(c.size * fraction)))
    top_idx = np.argsort(c, kind="mergesort")[-k:]
    global_err = float(e.mean())
    top_err = float(e[top_idx].mean())
    global_mis = float(m.mean())
    top_mis = float(m[top_idx].mean())

    return {
        "top20_abs_error": top_err,
        "global_abs_error": global_err,
        "top20_abs_error_enrichment": top_err / global_err if global_err > 0 else float("nan"),
        "top20_misclassification_rate": top_mis,
        "global_misclassification_rate": global_mis,
        "top20_misclassification_enrichment": top_mis / global_mis if global_mis > 0 else float("nan"),
    }


def resize_1ch(x: torch.Tensor, target_hw):
    if x.shape[-2:] == target_hw:
        return x
    return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)


def normalized_binary_entropy(prob: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    p = prob.clamp(eps, 1.0 - eps)
    h = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return h / math.log(2.0)


def make_loader(data_root: Path, dataset_name: str, num_workers: int):
    datapath = data_root / "test" / dataset_name
    cfg = dataset.Config(datapath=str(datapath), mode="test")
    data = dataset.Data(cfg)
    return DataLoader(data, batch_size=1, shuffle=False, num_workers=num_workers)


def evaluate_one_cue(
    dataset_name,
    image_name,
    cue_name,
    cue_np,
    err_np,
    mis_np,
    spearman_max_pixels,
    rng,
):
    rho = spearman_score(cue_np, err_np, spearman_max_pixels, rng)
    auc = binary_auc(cue_np, mis_np)
    ap = binary_average_precision(cue_np, mis_np)
    top = top_fraction_metrics(cue_np, err_np, mis_np, fraction=0.20)
    return {
        "dataset": dataset_name,
        "image": image_name,
        "cue": cue_name,
        "prediction_mae": float(err_np.mean()),
        "misclassification_rate": float(mis_np.mean()),
        "cue_mean": float(np.mean(cue_np)),
        "cue_max": float(np.max(cue_np)),
        "cue_nonzero_ratio": float((cue_np > 1e-8).mean()),
        "spearman_abs_error": rho,
        "misclassification_auroc": auc,
        "misclassification_auprc": ap,
        **top,
    }


def run_dataset(
    net,
    data_root,
    dataset_name,
    device,
    max_images,
    spearman_max_pixels,
    seed,
    num_workers,
):
    loader = make_loader(data_root, dataset_name, num_workers)
    rows: List[Dict[str, object]] = []
    total = len(loader.dataset)
    if max_images > 0:
        total = min(total, max_images)

    print(f"\n===== {dataset_name}: analysing {total} image(s) =====")

    with torch.no_grad():
        for idx, (image, mask, shape, name) in enumerate(loader):
            if max_images > 0 and idx >= max_images:
                break

            h = int(shape[0].item())
            w = int(shape[1].item())
            image = image.to(device=device, dtype=torch.float32, non_blocking=True)

            # Original forward.
            main_logit, _ = net(image, (h, w))
            raw_prob = torch.sigmoid(main_logit[0, 0])

            # Copy CP cues before the TTA forward overwrites aux_cache.
            cache = getattr(net, "aux_cache", {})
            required = {
                "cp_semantic": "cons_sem_1_8",
                "cp_edge": "cons_edge_1_4",
                "cp_fused": "cons_full",
            }
            copied = {}
            for cue_name, key in required.items():
                value = cache.get(key)
                if value is None:
                    raise RuntimeError(f"Missing {key} in net.aux_cache")
                copied[cue_name] = value.detach().clone()

            # Match test.py normalization for the error target.
            pred = (raw_prob - raw_prob.min()) / (raw_prob.max() - raw_prob.min() + 1e-8)

            gt = mask[0, 0].to(device=device, dtype=torch.float32)
            if gt.shape != pred.shape:
                gt = F.interpolate(gt[None, None], size=pred.shape, mode="nearest")[0, 0]
            gt_bin = (gt >= 0.5).float()

            abs_error = (pred - gt_bin).abs()
            misclassified = ((pred >= 0.5) != (gt_bin >= 0.5)).float()
            err_np = abs_error.cpu().numpy().astype(np.float64)
            mis_np = misclassified.cpu().numpy().astype(np.uint8)

            cue_tensors = {}
            for cue_name, value in copied.items():
                cue_tensors[cue_name] = resize_1ch(value, pred.shape)[0, 0].clamp(0, 1)

            cue_tensors["entropy"] = normalized_binary_entropy(raw_prob).clamp(0, 1)

            # Horizontal-flip TTA disagreement.
            image_flip = torch.flip(image, dims=[-1])
            logit_flip, _ = net(image_flip, (h, w))
            prob_flip = torch.sigmoid(logit_flip[0, 0])
            prob_flip_back = torch.flip(prob_flip, dims=[-1])
            cue_tensors["tta_flip"] = (raw_prob - prob_flip_back).abs().clamp(0, 1)

            # Deterministic random sanity baseline.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed + idx * 1009 + 17)
            random_cue = torch.rand(pred.shape, generator=gen, dtype=torch.float32).to(device)
            cue_tensors["random"] = random_cue

            for cue_offset, (cue_name, cue_t) in enumerate(cue_tensors.items()):
                cue_np = cue_t.detach().cpu().numpy().astype(np.float64)
                rng = np.random.default_rng(seed + idx * 37 + cue_offset)
                rows.append(
                    evaluate_one_cue(
                        dataset_name=dataset_name,
                        image_name=str(name[0]),
                        cue_name=cue_name,
                        cue_np=cue_np,
                        err_np=err_np,
                        mis_np=mis_np,
                        spearman_max_pixels=spearman_max_pixels,
                        rng=rng,
                    )
                )

            if (idx + 1) % 25 == 0 or (idx + 1) == total:
                print(f"[{dataset_name}] {idx + 1}/{total}")

    return rows


def build_summary(rows):
    metrics = [
        "prediction_mae",
        "misclassification_rate",
        "cue_mean",
        "cue_max",
        "cue_nonzero_ratio",
        "spearman_abs_error",
        "misclassification_auroc",
        "misclassification_auprc",
        "top20_abs_error_enrichment",
        "top20_misclassification_enrichment",
    ]

    result = {}
    datasets = sorted({str(r["dataset"]) for r in rows})
    cues = []
    for r in rows:
        if str(r["cue"]) not in cues:
            cues.append(str(r["cue"]))

    for ds in datasets:
        result[ds] = {}
        for cue in cues:
            selected = [r for r in rows if r["dataset"] == ds and r["cue"] == cue]
            if not selected:
                continue
            result[ds][cue] = {
                m: summarize(float(r[m]) for r in selected)
                for m in metrics
            }
    return result


def paired_comparisons(rows):
    metrics = [
        "spearman_abs_error",
        "misclassification_auroc",
        "misclassification_auprc",
        "top20_abs_error_enrichment",
        "top20_misclassification_enrichment",
    ]
    baselines = ["cp_edge", "entropy", "tta_flip", "random"]

    by_key = {}
    for r in rows:
        by_key[(str(r["dataset"]), str(r["image"]), str(r["cue"]))] = r

    out = {}
    datasets = sorted({str(r["dataset"]) for r in rows})

    for ds in datasets:
        out[ds] = {}
        images = sorted({
            str(r["image"])
            for r in rows
            if r["dataset"] == ds
        })

        for base in baselines:
            comp_name = f"cp_fused_minus_{base}"
            out[ds][comp_name] = {}
            for metric in metrics:
                deltas = []
                wins = []
                for img in images:
                    a = by_key.get((ds, img, "cp_fused"))
                    b = by_key.get((ds, img, base))
                    if a is None or b is None:
                        continue
                    av = float(a[metric])
                    bv = float(b[metric])
                    if not (np.isfinite(av) and np.isfinite(bv)):
                        continue
                    d = av - bv
                    deltas.append(d)
                    wins.append(float(d > 0))
                out[ds][comp_name][metric] = {
                    "delta": summarize(deltas),
                    "win_rate": float(np.mean(wins)) if wins else float("nan"),
                }

    return out


def print_summary(summary):
    print("\n===== CP VS UNCERTAINTY SUMMARY =====")
    header = (
        f"{'dataset':<12} {'cue':<13} "
        f"{'rho':>8} {'AUROC':>8} {'AUPRC':>8} "
        f"{'err-x':>8} {'mis-x':>8} {'nz':>8}"
    )
    print(header)
    print("-" * len(header))

    for ds, cue_dict in summary.items():
        for cue, metrics in cue_dict.items():
            m = lambda k: float(metrics[k]["mean"])
            print(
                f"{ds:<12} {cue:<13} "
                f"{m('spearman_abs_error'):8.4f} "
                f"{m('misclassification_auroc'):8.4f} "
                f"{m('misclassification_auprc'):8.4f} "
                f"{m('top20_abs_error_enrichment'):8.4f} "
                f"{m('top20_misclassification_enrichment'):8.4f} "
                f"{m('cue_nonzero_ratio'):8.4f}"
            )


def print_paired(paired):
    print("\n===== PAIRED DELTAS: CP_FUSED - BASELINE =====")
    metrics = [
        "spearman_abs_error",
        "misclassification_auroc",
        "top20_abs_error_enrichment",
        "top20_misclassification_enrichment",
    ]

    for ds, comps in paired.items():
        print(f"\n[{ds}]")
        for name, metric_dict in comps.items():
            print(name)
            for metric in metrics:
                block = metric_dict[metric]
                d = block["delta"]
                print(
                    f"  {metric:<36} "
                    f"mean_delta={d['mean']:+.4f} "
                    f"median_delta={d['median']:+.4f} "
                    f"win_rate={block['win_rate']:.3f}"
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

    print("===== CCF-Net CP vs uncertainty analysis =====")
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
        all_rows.extend(
            run_dataset(
                net=net,
                data_root=args.data_root,
                dataset_name=ds,
                device=device,
                max_images=args.max_images,
                spearman_max_pixels=args.spearman_max_pixels,
                seed=args.seed,
                num_workers=args.num_workers,
            )
        )

    summary = build_summary(all_rows)
    paired = paired_comparisons(all_rows)

    print_summary(summary)
    print_paired(paired)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_image.csv"
    summary_path = args.output_dir / "summary.json"
    paired_path = args.output_dir / "paired.json"

    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    with paired_path.open("w", encoding="utf-8") as f:
        json.dump(paired, f, indent=2, allow_nan=True)

    print("\nSaved:")
    print(" ", csv_path)
    print(" ", summary_path)
    print(" ", paired_path)
    print("NOTE: inference-only; no optimizer/backward/checkpoint write occurred.")


if __name__ == "__main__":
    main()
