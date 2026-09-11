#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 / Prompt Adapter for scribble-supervised WSCOD.

Input:
    - CodDataset/train/Image/*.jpg|png
    - CodDataset/train/Scribble/*.png  (expected labels: 0 unlabeled, 1 fg scribble, 2 bg scribble)
Output:
    - <out_dir>/prompts/<stem>.json
    - <out_dir>/vis/<stem>.png (optional prompt visualization)

This script intentionally stays standalone so it can be dropped into the user's
current project without changing the existing training code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from skimage.morphology import skeletonize  # type: ignore
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_gray_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    uniq = np.unique(arr)
    if set(uniq.tolist()).issubset({0, 1, 2}):
        return arr.astype(np.uint8)

    # Robust remap for files stored as 0 / 128 / 255 or similar.
    vals = sorted(uniq.tolist())
    if len(vals) == 3:
        mapping = {vals[0]: 0, vals[1]: 1, vals[2]: 2}
        remapped = np.vectorize(mapping.get)(arr)
        return remapped.astype(np.uint8)
    if len(vals) == 2:
        # treat missing class conservatively: smaller value -> 0, larger -> 1
        mapping = {vals[0]: 0, vals[1]: 1}
        remapped = np.vectorize(mapping.get)(arr)
        return remapped.astype(np.uint8)
    raise ValueError(f"Unexpected scribble label set in {path}: {uniq.tolist()}")


def to_skeleton(mask: np.ndarray) -> np.ndarray:
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return bin_mask
    if HAS_SKIMAGE:
        skel = skeletonize(bin_mask.astype(bool)).astype(np.uint8)
        if skel.sum() > 0:
            return skel
    return bin_mask


def _cellwise_sample_points(mask: np.ndarray, stride: int, max_points: int) -> List[List[int]]:
    """Sample one point per stride cell if the cell intersects the mask."""
    h, w = mask.shape
    pts: List[List[int]] = []
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return pts

    for y0 in range(0, h, stride):
        y1 = min(h, y0 + stride)
        for x0 in range(0, w, stride):
            x1 = min(w, x0 + stride)
            cell = mask[y0:y1, x0:x1]
            if cell.sum() == 0:
                continue
            cy = (y0 + y1 - 1) / 2.0
            cx = (x0 + x1 - 1) / 2.0
            cell_ys, cell_xs = np.where(cell > 0)
            cell_ys = cell_ys + y0
            cell_xs = cell_xs + x0
            d2 = (cell_ys - cy) ** 2 + (cell_xs - cx) ** 2
            idx = int(np.argmin(d2))
            pts.append([int(cell_xs[idx]), int(cell_ys[idx])])

    if not pts:
        return pts

    # Deduplicate while preserving order.
    uniq = []
    seen = set()
    for x, y in pts:
        key = (x, y)
        if key not in seen:
            seen.add(key)
            uniq.append([x, y])

    if len(uniq) <= max_points:
        return uniq

    # Evenly subsample.
    step = max(1, math.ceil(len(uniq) / max_points))
    sub = uniq[::step]
    return sub[:max_points]


def sample_prompt_points(
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    stride: int,
    max_fg: int,
    max_bg: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    fg_skel = to_skeleton(fg_mask)
    bg_skel = to_skeleton(bg_mask)
    pos_pts = _cellwise_sample_points(fg_skel, stride=stride, max_points=max_fg)
    neg_pts = _cellwise_sample_points(bg_skel, stride=stride, max_points=max_bg)
    return pos_pts, neg_pts


def compute_box_from_fg(fg_mask: np.ndarray, margin_ratio: float) -> List[int] | None:
    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0:
        return None
    h, w = fg_mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    mx = max(2, int(round(bw * margin_ratio)))
    my = max(2, int(round(bh * margin_ratio)))
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w - 1, x1 + mx)
    y1 = min(h - 1, y1 + my)
    return [x0, y0, x1, y1]


def find_image_for_stem(image_root: Path, stem: str) -> Path:
    for ext in IMG_EXTS:
        p = image_root / f"{stem}{ext}"
        if p.exists():
            return p
    candidates = list(image_root.glob(f"{stem}.*"))
    for p in candidates:
        if p.suffix.lower() in IMG_EXTS:
            return p
    raise FileNotFoundError(f"No image found for stem={stem} under {image_root}")


def draw_points(img: np.ndarray, pts: List[List[int]], color: Tuple[int, int, int], radius: int) -> None:
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--scribble_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--grid_stride", type=int, default=16)
    parser.add_argument("--max_fg_points", type=int, default=32)
    parser.add_argument("--max_bg_points", type=int, default=24)
    parser.add_argument("--bbox_margin_ratio", type=float, default=0.12)
    parser.add_argument("--save_vis", action="store_true")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    scribble_root = Path(args.scribble_root)
    out_dir = Path(args.out_dir)
    prompt_dir = out_dir / "prompts"
    vis_dir = out_dir / "vis"
    ensure_dir(prompt_dir)
    if args.save_vis:
        ensure_dir(vis_dir)

    scribble_files = sorted([p for p in scribble_root.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not scribble_files:
        raise RuntimeError(f"No scribble files found in {scribble_root}")

    num_empty = 0
    for s_path in tqdm(scribble_files, desc="Prompt-adapter"):
        stem = s_path.stem
        img_path = find_image_for_stem(image_root, stem)
        scrib = load_gray_mask(s_path)
        fg = (scrib == 1).astype(np.uint8)
        bg = (scrib == 2).astype(np.uint8)

        pos_pts, neg_pts = sample_prompt_points(
            fg_mask=fg,
            bg_mask=bg,
            stride=args.grid_stride,
            max_fg=args.max_fg_points,
            max_bg=args.max_bg_points,
        )
        box = compute_box_from_fg(fg, margin_ratio=args.bbox_margin_ratio)

        if not pos_pts:
            num_empty += 1

        payload = {
            "image_name": img_path.name,
            "stem": stem,
            "height": int(scrib.shape[0]),
            "width": int(scrib.shape[1]),
            "positive_points": pos_pts,
            "negative_points": neg_pts,
            "box_xyxy": box,
            "num_fg_scribble_pixels": int(fg.sum()),
            "num_bg_scribble_pixels": int(bg.sum()),
            "sam_prompt_mode": "points+box" if box is not None else "points",
        }
        with open(prompt_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if args.save_vis:
            img = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
            if box is not None:
                x0, y0, x1, y1 = box
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 180, 0), 2)
            draw_points(img, pos_pts, (0, 255, 0), radius=3)
            draw_points(img, neg_pts, (0, 0, 255), radius=3)
            cv2.imwrite(str(vis_dir / f"{stem}.png"), img)

    print(f"[Done] prompts saved to: {prompt_dir}")
    print(f"[Info] images without positive prompt points: {num_empty}")
    if not HAS_SKIMAGE:
        print("[Warn] skimage is not installed, using raw scribble pixels instead of skeletonization.")


if __name__ == "__main__":
    main()
