#!/usr/bin/python3
# coding=utf-8

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _safe_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    denom = w.sum().clamp_min(1.0)
    return (x * w).sum() / denom


def _binary_dice_with_mask(prob: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    prob = prob * valid
    target = target * valid
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
    return dice.mean()


def _pseudo_loss_from_logit(logit: torch.Tensor,
                            stage1_label: torch.Tensor,
                            trust_map: torch.Tensor,
                            pseudo_bce_scale: float = 1.0,
                            pseudo_dice_scale: float = 0.5) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    target = (stage1_label == 1).float()
    valid = (stage1_label != 0).float()
    weight = (0.25 + 0.75 * trust_map).float() * valid

    bce_map = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    pseudo_bce = _safe_mean(bce_map, weight)

    prob = torch.sigmoid(logit)
    pseudo_dice = _binary_dice_with_mask(prob, target, valid)

    loss = pseudo_bce_scale * pseudo_bce + pseudo_dice_scale * pseudo_dice
    return loss, {
        'pseudo_bce': pseudo_bce.detach(),
        'pseudo_dice': pseudo_dice.detach(),
    }


def _scribble_hard_loss(logit: torch.Tensor,
                        scribble: torch.Tensor,
                        scribble_weight: float = 2.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    fg = (scribble == 1).float()
    bg = (scribble == 2).float()
    valid = (fg + bg).clamp(max=1.0)
    target = fg

    if float(valid.sum().item()) < 1.0:
        zero = logit.sum() * 0.0
        return zero, {
            'scribble_bce': zero.detach(),
            'scribble_acc': zero.detach(),
        }

    weight = valid * scribble_weight
    bce_map = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    scribble_bce = _safe_mean(bce_map, weight)

    prob = torch.sigmoid(logit)
    pred = (prob > 0.5).float()
    acc = ((pred == target).float() * valid).sum() / valid.sum().clamp_min(1.0)

    return scribble_bce, {
        'scribble_bce': scribble_bce.detach(),
        'scribble_acc': acc.detach(),
    }


def _simple_smooth_loss(logit: torch.Tensor, weight: float = 0.05) -> torch.Tensor:
    prob = torch.sigmoid(logit)
    dx = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs().mean()
    dy = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs().mean()
    return weight * (dx + dy)


def _single_output_loss(logit: torch.Tensor,
                        stage1_label: torch.Tensor,
                        trust_map: torch.Tensor,
                        scribble: torch.Tensor,
                        lambda_scribble: float,
                        lambda_smooth: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pseudo_loss, pseudo_stats = _pseudo_loss_from_logit(logit, stage1_label, trust_map)
    scribble_loss, scribble_stats = _scribble_hard_loss(logit, scribble)
    smooth_loss = _simple_smooth_loss(logit, weight=lambda_smooth)

    total = pseudo_loss + lambda_scribble * scribble_loss + smooth_loss
    stats = {}
    stats.update(pseudo_stats)
    stats.update(scribble_stats)
    stats['smooth'] = smooth_loss.detach()
    stats['total'] = total.detach()
    return total, stats


def train_loss(image: torch.Tensor,
               stage1_label: torch.Tensor,
               trust_map: torch.Tensor,
               scribble: torch.Tensor,
               net,
               ctx=None,
               lambda_scribble: float = 1.5,
               lambda_smooth: float = 0.05,
               aux_scale: float = 0.7):
    """
    Stage 2 minimal runnable loss.

    Supervision design:
    - Stage1Label (dense pseudo) is the primary supervision.
    - TrustMap weights the pseudo loss.
    - Original scribble remains a hard anchor.
    - A very light smoothness term stabilizes early training.
    """
    outputs = net(image)
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 6:
        raise RuntimeError('Net(image) is expected to return 6 outputs in train mode: main, _, c1, c2, c3, c4')

    main_logit, _, out_c1, out_c2, out_c3, out_c4 = outputs

    loss2, stats2 = _single_output_loss(main_logit, stage1_label, trust_map, scribble,
                                        lambda_scribble=lambda_scribble,
                                        lambda_smooth=lambda_smooth)
    loss3, _ = _single_output_loss(out_c1, stage1_label, trust_map, scribble,
                                   lambda_scribble=lambda_scribble,
                                   lambda_smooth=lambda_smooth)
    loss4, _ = _single_output_loss(out_c2, stage1_label, trust_map, scribble,
                                   lambda_scribble=lambda_scribble,
                                   lambda_smooth=lambda_smooth)
    loss5, _ = _single_output_loss(out_c3, stage1_label, trust_map, scribble,
                                   lambda_scribble=lambda_scribble,
                                   lambda_smooth=lambda_smooth)
    loss6, _ = _single_output_loss(out_c4, stage1_label, trust_map, scribble,
                                   lambda_scribble=lambda_scribble,
                                   lambda_smooth=lambda_smooth)

    loss3 = loss3 * aux_scale
    loss4 = loss4 * aux_scale
    loss5 = loss5 * aux_scale
    loss6 = loss6 * aux_scale

    if ctx is not None and ctx.get('sw', None) is not None:
        sw = ctx['sw']
        global_step = int(ctx.get('global_step', 0))
        sw.add_scalar('stage2/main_total', float(stats2['total'].item()), global_step)
        sw.add_scalar('stage2/main_pseudo_bce', float(stats2['pseudo_bce'].item()), global_step)
        sw.add_scalar('stage2/main_pseudo_dice', float(stats2['pseudo_dice'].item()), global_step)
        sw.add_scalar('stage2/main_scribble_bce', float(stats2['scribble_bce'].item()), global_step)
        sw.add_scalar('stage2/main_scribble_acc', float(stats2['scribble_acc'].item()), global_step)
        sw.add_scalar('stage2/main_smooth', float(stats2['smooth'].item()), global_step)

        if global_step % 200 == 0:
            with torch.no_grad():
                prob = torch.sigmoid(main_logit[0]).detach().cpu()
                tgt = (stage1_label[0] == 1).float().detach().cpu()
                tr = trust_map[0].clamp(0, 1).detach().cpu()
                sc_fg = (scribble[0] == 1).float().detach().cpu()
                sc_bg = (scribble[0] == 2).float().detach().cpu()
                sw.add_image('stage2/img_pred_main', prob, global_step)
                sw.add_image('stage2/img_stage1_fg', tgt, global_step)
                sw.add_image('stage2/img_trust', tr, global_step)
                sw.add_image('stage2/img_scrib_fg', sc_fg, global_step)
                sw.add_image('stage2/img_scrib_bg', sc_bg, global_step)

    return loss2, loss3, loss4, loss5, loss6
