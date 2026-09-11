#!/usr/bin/env python3
"""Cheap, read-only server-side runtime/path sanity check for CCF-Net revision experiments."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DINO_SRC = Path("/root/shared-nvme/dinov3-main")
DINO_CKPT = Path("/root/shared-nvme/pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
BEST_CKPT = REPO_ROOT / "cp_cof_1_1_dino_sem_edge" / "dino_0415" / "model-best.pth"


def path_status(label: str, path: Path) -> None:
    print(f"{label}: {path} | exists={path.exists()}")


def count_nonempty_lines(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        with path.open("r", encoding="utf-8") as f:
            return str(sum(1 for line in f if line.strip()))
    except Exception as exc:
        return f"error({exc})"


def main() -> None:
    print("===== CCF-Net revision runtime check =====")
    print("repo_root:", REPO_ROOT)
    print("cwd:", Path.cwd())
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    try:
        import torch

        print("torch:", torch.__version__)
        print("torch CUDA runtime:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("visible cuda devices:", torch.cuda.device_count())
            for idx in range(torch.cuda.device_count()):
                print(f"gpu[{idx}]:", torch.cuda.get_device_name(idx))
    except Exception as exc:
        print("torch import failed:", repr(exc))

    try:
        import torchvision

        print("torchvision:", torchvision.__version__)
    except Exception as exc:
        print("torchvision import failed:", repr(exc))

    path_status("DINOv3 source", DINO_SRC)
    path_status("DINOv3 checkpoint", DINO_CKPT)
    path_status("existing best checkpoint", BEST_CKPT)

    data_root = REPO_ROOT / "CodDataset"
    path_status("dataset root", data_root)
    path_status("train image dir", data_root / "train" / "Image")
    path_status("scribble dir", data_root / "train" / "Scribble")
    print("train.txt samples:", count_nonempty_lines(data_root / "train.txt"))

    for name in ("CAMO", "CHAMELEON", "COD10K", "NC4K"):
        ds = data_root / "test" / name
        path_status(f"test/{name}", ds)
        path_status(f"test/{name}/Image", ds / "Image")
        path_status(f"test/{name}/GT", ds / "GT")
        print(f"test/{name} samples:", count_nonempty_lines(ds / "test.txt"))

    toolkit = REPO_ROOT / "PySODEvalToolkit"
    path_status("PySODEvalToolkit", toolkit)
    path_status("PySODEvalToolkit/eval.py", toolkit / "eval.py")
    path_status("cod_method.json", toolkit / "cod_method.json")
    path_status("cod_dataset.json", toolkit / "cod_dataset.json")

    print("NOTE: this script is read-only; it does not create or modify project files.")
    print("==========================================")


if __name__ == "__main__":
    main()
