#!/usr/bin/env python3

from pathlib import Path
import os
import sys
import random
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
DINO_SRC = Path("/root/shared-nvme/dinov3-main")

sys.path.insert(0, str(DINO_SRC))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial

from data import dataset
from ccfnet_cp_prior_bce import Net
from loss_cp_prior_bce import train_loss


SEED = 1
BATCH_SIZE = 16
STEPS = 50
CP_PRIOR_W = 0.1

BASE_LR = 1e-5
MAX_LR = 1e-2
TOTAL_EPOCHS = 150


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1.,
                    annealing_decay=1e-2):
    first = int(total_steps * ratio)
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur / total_steps)
    x = np.abs(cur * 2.0 / total_steps - 2.0 * cycle + 1)

    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = (
            (base_lr - min_lr) * cur
            + min_lr * first
            - base_lr * total_steps
        ) / (first - total_steps)

    return float(lr)


def prior_probe(net, image, mask):
    was_training = net.training
    net.eval()

    with torch.no_grad():
        net(image)

        cache = net.aux_cache

        sem_logit = cache["enc_sem_logit_full_raw"]
        edge_logit = cache["enc_edge_logit_full_raw"]

        gt = mask.squeeze(1).long()

        fg_mask = (gt == 1)
        bg_mask = (gt == 0)

        def balanced_bce(logit):
            z = logit.squeeze(1)
            terms = []

            if bool(fg_mask.any()):
                z_fg = z[fg_mask]
                terms.append(
                    F.binary_cross_entropy_with_logits(
                        z_fg,
                        torch.ones_like(z_fg),
                        reduction="mean",
                    )
                )

            if bool(bg_mask.any()):
                z_bg = z[bg_mask]
                terms.append(
                    F.binary_cross_entropy_with_logits(
                        z_bg,
                        torch.zeros_like(z_bg),
                        reduction="mean",
                    )
                )

            if len(terms) == 0:
                return z.sum() * 0.0

            return torch.stack(terms).mean()

        sem_bce = balanced_bce(sem_logit)
        edge_bce = balanced_bce(edge_logit)

        p_sem = torch.sigmoid(sem_logit)
        p_edge = torch.sigmoid(edge_logit)

        def prior_metrics(p):
            p = p.squeeze(1)

            fg_mean = (
                p[fg_mask].mean().item()
                if bool(fg_mask.any())
                else float("nan")
            )
            bg_mean = (
                p[bg_mask].mean().item()
                if bool(bg_mask.any())
                else float("nan")
            )

            return {
                "fg": fg_mean,
                "bg": bg_mean,
                "margin": fg_mean - bg_mean,
            }

        def cue_metrics(x):
            x = x.detach()

            return {
                "mean": x.mean().item(),
                "std": x.std().item(),
                "max": x.max().item(),
                "nz": (
                    (x > 1e-8)
                    .float()
                    .mean()
                    .item()
                ),
            }

        result = {
            "sem_bce": sem_bce.item(),
            "edge_bce": edge_bce.item(),
            "weighted": (
                CP_PRIOR_W
                * 0.5
                * (sem_bce + edge_bce)
            ).item(),

            "sem_prior": prior_metrics(p_sem),
            "edge_prior": prior_metrics(p_edge),

            "sem_cue": cue_metrics(
                cache["cons_sem_1_8"]
            ),
            "edge_cue": cue_metrics(
                cache["cons_edge_1_4"]
            ),
            "fused_cue": cue_metrics(
                cache["cons_full"]
            ),
        }

    net.train(was_training)
    return result


def print_probe(title, d):
    print(f"\n===== {title} =====")
    print(
        "prior_loss:",
        f"sem_bce={d['sem_bce']:.8f}",
        f"edge_bce={d['edge_bce']:.8f}",
        f"weighted={d['weighted']:.8f}",
    )

    for key in ("sem_prior", "edge_prior"):
        x = d[key]
        print(
            f"{key}:",
            f"fg={x['fg']:.6f}",
            f"bg={x['bg']:.6f}",
            f"margin={x['margin']:+.6f}",
        )

    for key in ("sem_cue", "edge_cue", "fused_cue"):
        x = d[key]
        print(
            f"{key}:",
            f"mean={x['mean']:.6f}",
            f"std={x['std']:.6f}",
            f"max={x['max']:.6f}",
            f"nz={x['nz']:.6f}",
        )


def param_delta(before, after):
    d = (after - before).abs()
    return d.mean().item(), d.max().item()


def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print("===== CP PRIOR OPTIMIZER SANITY =====")
    print("GPU:", torch.cuda.get_device_name(0))
    print("steps:", STEPS)
    print("batch:", BATCH_SIZE)
    print("formal LR horizon:", TOTAL_EPOCHS)

    cfg = dataset.Config(
        datapath=str(ROOT / "CodDataset"),
        savepath=str(
            ROOT / "out_revision_p0"
            / "cp_prior_bce_optimizer_sanity"
        ),
        mode="train",
        batch=BATCH_SIZE,
        lr=1e-3,
        momen=0.9,
        decay=5e-4,
        epoch=TOTAL_EPOCHS,
        label_dir="Scribble",
        keep_n_eval_ckpt=2,
    )

    data = dataset.Data(cfg)

    train_loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    probe_loader = DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    probe_image, probe_mask, _, _ = next(iter(probe_loader))
    probe_image = probe_image.cuda().float()
    probe_mask = probe_mask.cuda().float()

    net = Net(cfg).cuda()
    net.train(True)

    base, head = [], []

    for name, param in net.named_parameters():
        if "bkbone" in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": base},
            {"params": head},
        ],
        lr=cfg.lr,
        momentum=cfg.momen,
        weight_decay=cfg.decay,
        nesterov=True,
    )

    train_loss_fn = partial(
        train_loss,
        w_ft=0.15,
        ft_st=60,
        ft_fct=0.5,
        ft_dct=dict(
            crtl_loss=False,
            w_ftp=1,
            norm=False,
            topk=16,
            step_ratio=2,
        ),
        ft_head=False,
        mtrsf_prob=1,
        ops=[0, 1, 2],
        w_l2g=0.3,
        l_me=0.05,
        me_st=20,
        multi_sc=0,
        cp_prior_w=CP_PRIOR_W,
    )

    sem_before = (
        net.enc_q_proj_sem.conv[0].weight
        .detach()
        .clone()
    )
    edge_before = (
        net.enc_q_proj_edge.conv[0].weight
        .detach()
        .clone()
    )

    before_probe = prior_probe(
        net, probe_image, probe_mask
    )
    print_probe("PROBE BEFORE", before_probe)

    total_steps = TOTAL_EPOCHS * len(train_loader)

    total_hist = []
    prior_hist = []

    iterator = iter(train_loader)

    net.train(True)

    for step in range(STEPS):
        try:
            image, mask, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            image, mask, _, _ = next(iterator)

        image = image.cuda().float()
        mask = mask.cuda().float()

        lr = get_triangle_lr(
            BASE_LR,
            MAX_LR,
            total_steps,
            step,
            ratio=1.0,
        )

        optimizer.param_groups[0]["lr"] = 0.1 * lr
        optimizer.param_groups[1]["lr"] = lr

        ctx = dict(
            epoch=1,
            global_step=step + 1,
            sw=None,
            t_epo=TOTAL_EPOCHS,
            save_dir=str(cfg.savepath),
        )

        optimizer.zero_grad(set_to_none=True)

        loss2, loss3, loss4, loss5, loss6 = (
            train_loss_fn(
                image,
                mask,
                net,
                ctx,
            )
        )

        total = (
            loss2
            + 0.8 * loss3
            + 0.6 * loss4
            + 0.4 * loss5
            + 0.2 * loss6
        )

        if not torch.isfinite(total):
            raise RuntimeError(
                f"non-finite loss at step {step + 1}"
            )

        total.backward()
        optimizer.step()

        debug = net.cp_prior_debug

        total_hist.append(total.detach().item())
        prior_hist.append(
            debug["total"].detach().item()
        )

        if (
            step == 0
            or (step + 1) % 10 == 0
            or step + 1 == STEPS
        ):
            print(
                f"step={step + 1:03d}",
                f"lr={lr:.8f}",
                f"total={total.item():.6f}",
                f"cp_prior={debug['total'].item():.6f}",
                f"sem_bce={debug['sem_bce'].item():.6f}",
                f"edge_bce={debug['edge_bce'].item():.6f}",
            )

    sem_after = (
        net.enc_q_proj_sem.conv[0].weight
        .detach()
        .clone()
    )
    edge_after = (
        net.enc_q_proj_edge.conv[0].weight
        .detach()
        .clone()
    )

    after_probe = prior_probe(
        net, probe_image, probe_mask
    )
    print_probe("PROBE AFTER", after_probe)

    sem_mean, sem_max = param_delta(
        sem_before, sem_after
    )
    edge_mean, edge_max = param_delta(
        edge_before, edge_after
    )

    print("\n===== PARAMETER DELTA =====")
    print(
        "semantic:",
        f"mean_abs={sem_mean:.8e}",
        f"max_abs={sem_max:.8e}",
    )
    print(
        "edge:",
        f"mean_abs={edge_mean:.8e}",
        f"max_abs={edge_max:.8e}",
    )

    n = min(10, len(total_hist))

    print("\n===== TRAINING WINDOW =====")
    print(
        "total first10/last10:",
        f"{np.mean(total_hist[:n]):.6f}",
        "->",
        f"{np.mean(total_hist[-n:]):.6f}",
    )
    print(
        "cp_prior first10/last10:",
        f"{np.mean(prior_hist[:n]):.6f}",
        "->",
        f"{np.mean(prior_hist[-n:]):.6f}",
    )

    print("\n===== RESULT =====")

    if sem_max <= 0 or edge_max <= 0:
        print("OPTIMIZER_SANITY=FAILED")
        raise RuntimeError(
            "One or both CP prior heads did not move."
        )

    print("OPTIMIZER_SANITY=OK")
    print(
        "No checkpoint was saved. "
        "This was a 50-step diagnostic only."
    )


if __name__ == "__main__":
    main()
