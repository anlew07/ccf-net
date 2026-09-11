#!/usr/bin/env python3
"""One-mini-batch CP gradient/computation-graph check.

This script performs forward/backward only. It deliberately does NOT construct
an optimizer, call optimizer.step(), or save a checkpoint. Its sole purpose is
to determine whether the encoder-prior projections used by CP receive gradients
under the current formal training objective.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DINO_SRC = Path("/root/shared-nvme/dinov3-main")
DINO_CKPT = Path("/root/shared-nvme/pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

# Preserve an explicitly selected GPU. Otherwise mirror train.py and choose the
# GPU with the most free memory. This must happen before importing torch/loss.py,
# because loss.py creates CUDA-resident loss modules at import time.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    gpu_id = subprocess.getoutput(
        "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader "
        "| nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs"
    ).strip()
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# Project modules should take precedence over any similarly named module inside
# the external DINOv3 source tree.
sys.path.insert(0, str(DINO_SRC))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ccfnet import Net
from data import dataset
from loss import train_loss

BATCH_SIZE = 2  # BatchNorm after 1x1 pooled PPM feature requires N*H*W > 1.


def grad_summary(net: torch.nn.Module, prefix: str) -> dict:
    matched = []
    for name, param in net.named_parameters():
        if name.startswith(prefix):
            grad = param.grad
            matched.append(
                {
                    "name": name,
                    "requires_grad": bool(param.requires_grad),
                    "grad_present": grad is not None,
                    "mean_abs": None if grad is None else float(grad.detach().abs().mean().item()),
                    "max_abs": None if grad is None else float(grad.detach().abs().max().item()),
                }
            )

    present = [x for x in matched if x["grad_present"]]
    nonzero = [x for x in present if (x["max_abs"] or 0.0) > 0.0]
    return {
        "prefix": prefix,
        "n_params": len(matched),
        "n_grad_present": len(present),
        "n_nonzero": len(nonzero),
        "items": matched,
    }


def print_summary(summary: dict) -> None:
    print(
        f"[{summary['prefix']}] params={summary['n_params']} "
        f"grad_present={summary['n_grad_present']} nonzero={summary['n_nonzero']}"
    )
    for item in summary["items"]:
        if item["grad_present"]:
            print(
                "  "
                + item["name"]
                + f" | requires_grad={item['requires_grad']}"
                + f" | mean_abs={item['mean_abs']:.8e}"
                + f" | max_abs={item['max_abs']:.8e}"
            )
        else:
            print(
                "  "
                + item["name"]
                + f" | requires_grad={item['requires_grad']} | grad=None"
            )


def main() -> None:
    if not DINO_SRC.exists():
        raise FileNotFoundError(f"Missing DINOv3 source: {DINO_SRC}")
    if not DINO_CKPT.exists():
        raise FileNotFoundError(f"Missing DINOv3 checkpoint: {DINO_CKPT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this sanity check on the GPU server.")

    # Mirror canonical train.py seed policy.
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("===== CCF-Net one-mini-batch gradient check =====")
    print("repo_root:", REPO_ROOT)
    print("torch:", torch.__version__)
    print("cuda runtime:", torch.version.cuda)
    print("visible gpu:", torch.cuda.get_device_name(0))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("mini-batch size:", BATCH_SIZE)

    cfg = dataset.Config(
        datapath=str(REPO_ROOT / "CodDataset"),
        savepath=str(REPO_ROOT / "out_vdino" / "sanity_grad_check"),
        mode="train",
        batch=BATCH_SIZE,
        lr=1e-3,
        momen=0.9,
        decay=5e-4,
        epoch=150,
        label_dir="Scribble",
        keep_n_eval_ckpt=2,
    )

    data = dataset.Data(cfg)
    if len(data) < BATCH_SIZE:
        raise RuntimeError(f"Training dataset has only {len(data)} samples; need at least {BATCH_SIZE}.")

    # One deterministic mini-batch is sufficient for graph connectivity. We do
    # not need the prefetcher or shuffled sampling used by the long training run.
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    image, mask, _, names = next(iter(loader))
    image = image.cuda(non_blocking=False).float()
    mask = mask.cuda(non_blocking=False).float()

    net = Net(cfg).cuda()
    net.train(True)
    net.zero_grad(set_to_none=True)

    # Exact loss options from the current canonical train.py.
    train_loss_fn = partial(
        train_loss,
        w_ft=0.15,
        ft_st=60,
        ft_fct=0.5,
        ft_dct=dict(crtl_loss=False, w_ftp=1, norm=False, topk=16, step_ratio=2),
        ft_head=False,
        mtrsf_prob=1,
        ops=[0, 1, 2],
        w_l2g=0.3,
        l_me=0.05,
        me_st=20,
        multi_sc=0,
    )

    ctx = dict(
        epoch=1,
        global_step=1,
        sw=None,
        t_epo=150,
        save_dir=str(REPO_ROOT / "out_vdino" / "sanity_grad_check"),
    )

    loss2, loss3, loss4, loss5, loss6 = train_loss_fn(image, mask, net, ctx)
    total = loss2 + 0.8 * loss3 + 0.6 * loss4 + 0.4 * loss5 + 0.2 * loss6

    if not bool(torch.isfinite(total).item()):
        raise RuntimeError(f"Non-finite total loss: {float(total.detach().item())}")

    total.backward()

    print("samples:", list(names) if isinstance(names, (list, tuple)) else names)
    print(
        "losses:",
        f"total={total.detach().item():.8f}",
        f"main={loss2.detach().item():.8f}",
        f"c1={loss3.detach().item():.8f}",
        f"c2={loss4.detach().item():.8f}",
        f"c3={loss5.detach().item():.8f}",
        f"c4={loss6.detach().item():.8f}",
    )

    print("\n--- CP encoder-prior targets ---")
    sem = grad_summary(net, "enc_q_proj_sem")
    edge = grad_summary(net, "enc_q_proj_edge")
    print_summary(sem)
    print_summary(edge)

    print("\n--- Backward controls ---")
    controls = []
    for prefix in (
        "head.0",
        "head.1",
        "head.2",
        "tfgm.dec_out",
        "bkbone.to_f4",
        "bkbone.to_f8",
    ):
        summary = grad_summary(net, prefix)
        controls.append(summary)
        print_summary(summary)

    cp_nonzero = sem["n_nonzero"] + edge["n_nonzero"]
    cp_present = sem["n_grad_present"] + edge["n_grad_present"]
    control_nonzero = sum(x["n_nonzero"] for x in controls)

    print("\n--- Interpretation flag ---")
    if control_nonzero == 0:
        print("BACKWARD_CONTROL=FAILED")
        print("No selected control module received a non-zero gradient; do not interpret the CP result yet.")
    else:
        print("BACKWARD_CONTROL=OK")
        if cp_present == 0:
            print("CP_ENCODER_PRIOR_GRAD=NONE")
            print("The current backward graph provides no gradient to either encoder-prior projection.")
            print("Do NOT start long reproduction/ablation runs before reviewing this result.")
        elif cp_nonzero == 0:
            print("CP_ENCODER_PRIOR_GRAD=ZERO")
            print("Gradient tensors exist but are numerically zero in this mini-batch; inspect another mini-batch before concluding.")
        else:
            print("CP_ENCODER_PRIOR_GRAD=NONZERO")
            print("At least one encoder-prior projection receives a non-zero gradient in this mini-batch.")

    print("No optimizer was constructed; no optimizer.step() was executed; no checkpoint was written.")
    print("================================================")


if __name__ == "__main__":
    main()
