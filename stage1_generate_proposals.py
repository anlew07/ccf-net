#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 / Generate SAM proposals from adapted scribble prompts.

Requirements:
    pip install git+https://github.com/facebookresearch/segment-anything.git
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

try:
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore
except Exception as e:
    raise ImportError(
        "Failed to import segment_anything. Install the official SAM package first."
    ) from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def load_prompt(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--prompt_root", type=str, required=True)
    parser.add_argument("--sam_type", type=str, default="vit_h", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    prompt_root = Path(args.prompt_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    prompt_files = sorted(prompt_root.glob("*.json"))
    if not prompt_files:
        raise RuntimeError(f"No prompt json files found in {prompt_root}")

    for p_path in tqdm(prompt_files, desc="SAM-proposals"):
        prompt = load_prompt(p_path)
        stem = prompt["stem"]
        img_path = find_image_for_stem(image_root, stem)
        image = np.array(Image.open(img_path).convert("RGB"))
        predictor.set_image(image)

        pos_pts = prompt.get("positive_points", [])
        neg_pts = prompt.get("negative_points", [])
        pts = np.array(pos_pts + neg_pts, dtype=np.float32)
        labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int64)
        box = prompt.get("box_xyxy", None)
        box_np: Optional[np.ndarray] = None
        if box is not None:
            box_np = np.array(box, dtype=np.float32)

        if len(pts) == 0 and box_np is None:
            # No usable prompt: save empty placeholder so downstream can diagnose it.
            np.savez_compressed(
                out_dir / f"{stem}.npz",
                masks=np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8),
                scores=np.zeros((0,), dtype=np.float32),
                logits=np.zeros((0, image.shape[0], image.shape[1]), dtype=np.float32),
                point_coords=np.zeros((0, 2), dtype=np.float32),
                point_labels=np.zeros((0,), dtype=np.int64),
                box=np.array([], dtype=np.float32),
            )
            continue

        masks, scores, logits = predictor.predict(
            point_coords=pts if len(pts) > 0 else None,
            point_labels=labels if len(labels) > 0 else None,
            box=box_np,
            multimask_output=True,
            return_logits=True,
        )

        np.savez_compressed(
            out_dir / f"{stem}.npz",
            masks=masks.astype(np.uint8),
            scores=scores.astype(np.float32),
            logits=logits.astype(np.float32),
            point_coords=pts.astype(np.float32),
            point_labels=labels.astype(np.int64),
            box=np.array(box if box is not None else [], dtype=np.float32),
        )

    print(f"[Done] proposal files saved to: {out_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
