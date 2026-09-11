#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 / Rank SAM proposals with stricter hard filtering.

Key fixes vs previous version:
1) no longer falls back to all candidates when area filter rejects everything
2) adds hard rejection on bg_leak and semantic margin
3) preserves original scribbles even when no valid candidate exists
4) reduces SAM score dominance in total ranking
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def load_scribble(path: Path) -> np.ndarray:
    arr = load_gray(path)
    uniq = np.unique(arr)
    if set(uniq.tolist()).issubset({0, 1, 2}):
        return arr.astype(np.uint8)
    vals = sorted(uniq.tolist())
    if len(vals) == 3:
        mp = {vals[0]: 0, vals[1]: 1, vals[2]: 2}
        return np.vectorize(mp.get)(arr).astype(np.uint8)
    raise ValueError(f"Unexpected scribble labels in {path}: {uniq.tolist()}")


def find_file_for_stem(root: Path, stem: str) -> Path:
    for ext in IMG_EXTS:
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    cands = list(root.glob(f"{stem}.*"))
    for p in cands:
        if p.suffix.lower() in IMG_EXTS:
            return p
    raise FileNotFoundError(f"No file found for stem={stem} in {root}")


def resolve_prior_path(prior_root: Path, stem: str, prefer: str = "fused_prior") -> Path:
    if (prior_root / prefer).is_dir():
        return find_file_for_stem(prior_root / prefer, stem)
    return find_file_for_stem(prior_root, stem)


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0


def keep_largest_component(mask: np.ndarray, must_overlap: Optional[np.ndarray] = None) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)

    best_idx = None
    best_score = -1.0
    for idx in range(1, num_labels):
        comp = (labels == idx)
        area = float(comp.sum())
        overlap = float((comp & (must_overlap > 0)).sum()) if must_overlap is not None else 0.0
        score = overlap * 1e7 + area
        if score > best_score:
            best_score = score
            best_idx = idx
    return (labels == best_idx).astype(np.uint8)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1)


def compute_candidate_metrics(mask: np.ndarray, sam_score: float, prior: np.ndarray,
                              fg_scrib: np.ndarray, bg_scrib: np.ndarray) -> Dict[str, float]:
    h, w = mask.shape
    area_ratio = float(mask.sum()) / float(h * w)
    fg_cover = float(mask[fg_scrib > 0].mean()) if fg_scrib.sum() > 0 else 0.0
    bg_leak = float(mask[bg_scrib > 0].mean()) if bg_scrib.sum() > 0 else 0.0
    bg_reject = 1.0 - bg_leak

    inside_mean = float(prior[mask > 0].mean()) if mask.sum() > 0 else 0.0
    outside_mean = float(prior[mask == 0].mean()) if (mask == 0).sum() > 0 else 0.0
    semantic_margin = inside_mean - outside_mean

    prior_bin = (prior >= 0.5).astype(np.uint8)
    prior_iou = iou_binary(mask, prior_bin)

    semantic_score = (
        0.45 * max(0.0, semantic_margin)
        + 0.20 * inside_mean
        + 0.20 * prior_iou
        + 0.15 * fg_cover
    )
    prompt_score = 0.65 * fg_cover + 0.35 * bg_reject
    total_score = 0.15 * float(sam_score) + 0.50 * semantic_score + 0.35 * prompt_score
    return {
        "area_ratio": area_ratio,
        "fg_cover": fg_cover,
        "bg_leak": bg_leak,
        "bg_reject": bg_reject,
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "semantic_margin": semantic_margin,
        "prior_iou": prior_iou,
        "semantic_score": semantic_score,
        "prompt_score": prompt_score,
        "sam_score": float(sam_score),
        "total_score": total_score,
    }


def build_trust_map(best_mask: np.ndarray, prior: np.ndarray, fg_scrib: np.ndarray,
                    bg_scrib: np.ndarray, scribble_dilate_radius: int) -> np.ndarray:
    fg_band = dilate(fg_scrib, scribble_dilate_radius)
    bg_band = dilate(bg_scrib, scribble_dilate_radius)

    trust = np.where(best_mask > 0, prior, 1.0 - prior).astype(np.float32)
    trust[fg_band > 0] = np.maximum(trust[fg_band > 0], 0.98)
    trust[bg_band > 0] = np.maximum(trust[bg_band > 0], 0.98)
    return np.clip(trust, 0.0, 1.0)


def build_stage1_label(best_mask: np.ndarray, trust: np.ndarray, fg_scrib: np.ndarray,
                       bg_scrib: np.ndarray, fg_thr: float, bg_thr: float) -> np.ndarray:
    sup = np.zeros(best_mask.shape, dtype=np.uint8)
    sup[(best_mask > 0) & (trust >= fg_thr)] = 1
    sup[(best_mask == 0) & (trust >= bg_thr)] = 2
    sup[fg_scrib > 0] = 1
    sup[bg_scrib > 0] = 2
    return sup


def choose_ids(candidate_infos: List[Dict[str, float]], args) -> (List[int], str):
    hard_ids = []
    soft_ids = []
    for idx, m in enumerate(candidate_infos):
        area_ok = args.tau_small <= m["area_ratio"] <= args.tau_big
        bg_ok = m["bg_leak"] <= args.hard_bg_leak
        sem_ok = m["semantic_margin"] >= args.hard_sem_margin
        fg_ok = m["fg_cover"] >= args.hard_fg_cover
        if area_ok and bg_ok and sem_ok and fg_ok:
            hard_ids.append(idx)
            continue

        area_ok_soft = args.tau_small <= m["area_ratio"] <= min(0.90, args.tau_big + 0.10)
        bg_ok_soft = m["bg_leak"] <= max(0.20, args.hard_bg_leak + 0.10)
        sem_ok_soft = m["semantic_margin"] >= (args.hard_sem_margin - 0.05)
        fg_ok_soft = m["fg_cover"] >= max(0.05, args.hard_fg_cover * 0.5)
        if area_ok_soft and bg_ok_soft and sem_ok_soft and fg_ok_soft:
            soft_ids.append(idx)

    if hard_ids:
        return hard_ids, "hard"
    if soft_ids:
        return soft_ids, "soft"
    return [], "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_root", type=str, required=True)
    parser.add_argument("--prior_root", type=str, required=True)
    parser.add_argument("--scribble_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--prior_name", type=str, default="fused_prior", choices=["fused_prior", "main_prob", "cue_full"])
    parser.add_argument("--tau_small", type=float, default=0.003)
    parser.add_argument("--tau_big", type=float, default=0.80)
    parser.add_argument("--hard_bg_leak", type=float, default=0.10)
    parser.add_argument("--hard_sem_margin", type=float, default=0.00)
    parser.add_argument("--hard_fg_cover", type=float, default=0.15)
    parser.add_argument("--fg_trust_thr", type=float, default=0.72)
    parser.add_argument("--bg_trust_thr", type=float, default=0.80)
    parser.add_argument("--scribble_dilate_radius", type=int, default=5)
    parser.add_argument("--keep_largest", action="store_true")
    parser.add_argument("--save_vis", action="store_true")
    args = parser.parse_args()

    proposal_root = Path(args.proposal_root)
    prior_root = Path(args.prior_root)
    scribble_root = Path(args.scribble_root)
    out_dir = Path(args.out_dir)

    pseudo_dir = out_dir / "PseudoMask"
    trust_dir = out_dir / "TrustMap"
    sup_dir = out_dir / "Stage1Label"
    meta_dir = out_dir / "Meta"
    vis_dir = out_dir / "Vis"
    for d in [pseudo_dir, trust_dir, sup_dir, meta_dir]:
        ensure_dir(d)
    if args.save_vis:
        ensure_dir(vis_dir)

    proposal_files = sorted(proposal_root.glob("*.npz"))
    if not proposal_files:
        raise RuntimeError(f"No proposal files found in {proposal_root}")

    stats = {"hard": 0, "soft": 0, "none": 0}

    for npz_path in tqdm(proposal_files, desc="Rank+export"):
        stem = npz_path.stem
        data = np.load(npz_path)
        masks = data["masks"]
        scores = data["scores"]

        scrib_path = find_file_for_stem(scribble_root, stem)
        prior_path = resolve_prior_path(prior_root, stem, prefer=args.prior_name)

        scrib = load_scribble(scrib_path)
        prior = norm01(load_gray(prior_path))
        fg_scrib = (scrib == 1).astype(np.uint8)
        bg_scrib = (scrib == 2).astype(np.uint8)

        candidate_infos = []
        for idx in range(len(masks)):
            mask = (masks[idx] > 0).astype(np.uint8)
            if args.keep_largest:
                mask = keep_largest_component(mask, must_overlap=fg_scrib)
            candidate_infos.append(compute_candidate_metrics(mask, float(scores[idx]), prior, fg_scrib, bg_scrib))

        valid_ids, pool_type = choose_ids(candidate_infos, args)
        stats[pool_type] += 1

        if valid_ids:
            best_idx = max(valid_ids, key=lambda i: candidate_infos[i]["total_score"])
            best_mask = (masks[best_idx] > 0).astype(np.uint8)
            if args.keep_largest:
                best_mask = keep_largest_component(best_mask, must_overlap=fg_scrib)
            trust = build_trust_map(best_mask, prior, fg_scrib, bg_scrib, args.scribble_dilate_radius)
            sup = build_stage1_label(best_mask, trust, fg_scrib, bg_scrib, args.fg_trust_thr, args.bg_trust_thr)
            meta = {
                "stem": stem,
                "status": "ok",
                "pool_type": pool_type,
                "prior_path": str(prior_path),
                "best_idx": int(best_idx),
                "best_metrics": candidate_infos[best_idx],
                "candidate_metrics": candidate_infos,
                "num_candidates": int(len(masks)),
                "num_valid_after_filter": int(len(valid_ids)),
            }
        else:
            h, w = scrib.shape
            best_mask = np.zeros((h, w), dtype=np.uint8)
            trust = np.zeros((h, w), dtype=np.float32)
            sup = np.zeros((h, w), dtype=np.uint8)
            sup[fg_scrib > 0] = 1
            sup[bg_scrib > 0] = 2
            meta = {
                "stem": stem,
                "status": "no_valid_candidate",
                "pool_type": pool_type,
                "prior_path": str(prior_path),
                "candidate_metrics": candidate_infos,
                "num_candidates": int(len(masks)),
                "num_valid_after_filter": 0,
            }

        Image.fromarray((best_mask * 255).astype(np.uint8)).save(pseudo_dir / f"{stem}.png")
        Image.fromarray((trust * 255).astype(np.uint8)).save(trust_dir / f"{stem}.png")
        Image.fromarray(sup.astype(np.uint8)).save(sup_dir / f"{stem}.png")
        with open(meta_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if args.save_vis:
            canvas = np.zeros((scrib.shape[0], scrib.shape[1], 3), dtype=np.uint8)
            canvas[..., 1] = (best_mask * 255).astype(np.uint8)
            canvas[..., 2] = (bg_scrib * 255).astype(np.uint8)
            canvas[..., 0] = (fg_scrib * 255).astype(np.uint8)
            Image.fromarray(canvas).save(vis_dir / f"{stem}.png")

    print(f"[Done] pseudo masks : {pseudo_dir}")
    print(f"[Done] trust maps   : {trust_dir}")
    print(f"[Done] stage1 label : {sup_dir}")
    print(f"[Done] meta json    : {meta_dir}")
    print(f"[Stats] pool hard={stats['hard']} soft={stats['soft']} none={stats['none']}")


if __name__ == "__main__":
    main()
