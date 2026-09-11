#!/usr/bin/python3
# coding=utf-8

import sys
sys.path.insert(0, "/root/shared-nvme/dinov3-main")

from functools import partial
import datetime
import glob
import os
import random
import re
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import logging as logger

import stage2_dataset as dataset
from data import dataset as test_dataset
from dino1_net import Net
from stage2_loss import train_loss


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1)

TAG = "wscod_stage2"
logger.basicConfig(
    level=logger.INFO,
    format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=f"train_{TAG}.log",
    filemode="w"
)

GPU_ID = subprocess.getoutput(
    'nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs'
)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps * ratio)
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur / total_steps)
    x = np.abs(cur * 2.0 / total_steps - 2.0 * cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr) * cur + min_lr * first - base_lr * total_steps) / (first - total_steps)

    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1. - x)
        else:
            momentum = momentums[0]

    return lr, momentum


def should_validate(epoch_idx: int) -> bool:
    e = epoch_idx + 1
    if e <= 80:
        return e % 10 == 0
    elif e <= 100:
        return e % 5 == 0
    else:
        return True


def cleanup_eval_checkpoints(save_dir: str, keep: int = 5):
    pattern = os.path.join(save_dir, "model-e*.pth")
    paths = glob.glob(pattern)

    def parse_epoch(p):
        m = re.search(r"model-e(\d+)\.pth$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    items = [(parse_epoch(p), p) for p in paths if parse_epoch(p) >= 0]
    if len(items) <= keep:
        return

    items.sort(key=lambda x: x[0])
    for _, p in items[:-keep]:
        try:
            os.remove(p)
            logger.info(f"[CKPT] removed old eval ckpt: {os.path.basename(p)}")
        except Exception as e:
            logger.warning(f"[CKPT] failed to remove {p}: {e}")


def _forward_main(model, image):
    out = model(image)
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def validate(model, val_loader):
    model.train(False)
    avg_mae = 0.0
    cnt = 0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image = image.cuda().float()
            mask = mask.cuda().float()
            out = _forward_main(model, image)
            out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
            pred = torch.sigmoid(out[0, 0])
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            avg_mae += torch.abs(pred - mask[0]).mean().item()
            cnt += len(image)
    model.train(True)
    return avg_mae / max(cnt, 1)


def validate_multiloader(model, val_loader_list, log_file=None):
    maes = []
    for v in val_loader_list:
        st = time.time()
        mae = validate(model, v)
        cost = time.time() - st
        line = 'Spent %.3fs, %s MAE: %s' % (cost, v.dataset.data_name, mae)
        print(line)
        if log_file is not None:
            log_file.write(line + '\n')
            log_file.flush()
        maes.append(mae)
    return sum(maes) / max(len(maes), 1)


BASE_LR = 1e-5
MAX_LR = 1e-2
TOTAL_EPOCH = 150
ROOT = 'CodDataset'


def train(Dataset, Network, cfg, train_loss_fn, start_from=0):
    data = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)

    val_cfg = [test_dataset.Config(datapath=f'{ROOT}/test/{i}', mode='test') for i in ['CHAMELEON', 'CAMO', 'COD10K']]
    val_data = [test_dataset.Data(v) for v in val_cfg]
    val_loaders = [DataLoader(v, batch_size=1, shuffle=False, num_workers=4) for v in val_data]

    min_mae = 1.0
    best_epoch = 0

    net = Network(cfg)
    net.train(True)
    net.cuda()

    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.SGD(
        [{'params': base}, {'params': head}],
        lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True
    )

    os.makedirs(cfg.savepath, exist_ok=True)
    sw = SummaryWriter(cfg.savepath)
    db_size = len(loader)
    global_step = start_from * db_size
    et = 0

    keep_n_eval_ckpt = int(getattr(cfg, 'keep_n_eval_ckpt', 5))
    val_log_path = os.path.join(cfg.savepath, 'val_log.txt')
    val_log_file = open(val_log_path, 'a', encoding='utf-8')

    for epoch in range(start_from, cfg.epoch):
        batch_idx = -1
        for image, stage1_label, trust_map, scribble, shape, name in loader:
            st = time.time()
            batch_idx += 1
            niter = epoch * db_size + batch_idx

            image = image.cuda(non_blocking=True).float()
            stage1_label = stage1_label.cuda(non_blocking=True).long()
            trust_map = trust_map.cuda(non_blocking=True).float()
            scribble = scribble.cuda(non_blocking=True).long()

            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch * db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum
            global_step += 1

            ctx = dict(epoch=epoch + 1, global_step=global_step, sw=sw, t_epo=cfg.epoch, save_dir=cfg.savepath)
            loss2, loss3, loss4, loss5, loss6 = train_loss_fn(image, stage1_label, trust_map, scribble, net, ctx)
            loss = loss2 * 1.0 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4 + loss6 * 0.2

            if torch.isnan(loss) or torch.isinf(loss):
                print('>>> Found NaN/Inf in TOTAL loss !!!')
                print(f'  epoch={epoch + 1}, step={global_step}, batch_idx={batch_idx}')
                raise RuntimeError('NaN/Inf detected in loss')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalar('loss', loss.item(), global_step=global_step)

            ta = time.time() - st
            et = 0.9 * et + 0.1 * ta if et > 0 else ta

            if batch_idx % 10 == 0:
                msg = '%s| %s | eta:%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f | loss6=%.6f' % (
                    TAG, datetime.datetime.now(),
                    datetime.timedelta(seconds=int((cfg.epoch * db_size - niter) * et)),
                    global_step, epoch + 1, cfg.epoch,
                    optimizer.param_groups[0]['lr'], loss.item(),
                    loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()
                )
                print(msg)
                logger.info(msg)

        if should_validate(epoch):
            mae = validate_multiloader(net, val_loaders, log_file=val_log_file)
            val_line = 'VAL MAE:%s' % (mae)
            print(val_line)
            sw.add_scalar('val', mae, global_step=global_step)
            val_log_file.write(val_line + '\n')
            val_log_file.flush()

            torch.save(net.state_dict(), os.path.join(cfg.savepath, 'model-last.pth'))
            cur_snap = os.path.join(cfg.savepath, f'model-e{epoch + 1}.pth')
            torch.save(net.state_dict(), cur_snap)

            if mae < min_mae:
                min_mae = mae
                best_epoch = epoch + 1
                torch.save(net.state_dict(), os.path.join(cfg.savepath, 'model-best.pth'))
                best_line = 'best epoch is:%d, MAE:%s' % (best_epoch, min_mae)
                print(best_line)
                val_log_file.write(best_line + '\n')
                val_log_file.flush()

            cleanup_eval_checkpoints(cfg.savepath, keep=keep_n_eval_ckpt)

    print('min val mae is {}'.format(min_mae))
    val_log_file.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=ROOT)
    parser.add_argument('--stage1_root', type=str, default='runs/stage1/export_mainprob')
    parser.add_argument('--snapshot', type=str, default=None, help='Warm-start checkpoint. Recommend: current single-stage best model.')
    parser.add_argument('--save_root', type=str, default='./out_stage2')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=TOTAL_EPOCH)
    parser.add_argument('--trainsize', type=int, default=320)
    parser.add_argument('--lambda_scribble', type=float, default=1.5)
    parser.add_argument('--lambda_smooth', type=float, default=0.05)
    parser.add_argument('--aux_scale', type=float, default=0.7)
    args = parser.parse_args()

    run_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.exp_name or f'stage2_v1_{run_time}'

    cfg = dataset.Config(
        datapath=args.data_root,
        savepath=f'{args.save_root}/{exp_name}/',
        mode='train',
        batch=args.batch_size,
        lr=1e-3,
        momen=0.9,
        decay=5e-4,
        epoch=args.epochs,
        label_dir='Scribble',
        keep_n_eval_ckpt=2,
        stage1_root=args.stage1_root,
        trainsize=args.trainsize,
        snapshot=args.snapshot,
    )
    os.makedirs(cfg.savepath, exist_ok=True)

    tm = partial(
        train_loss,
        lambda_scribble=args.lambda_scribble,
        lambda_smooth=args.lambda_smooth,
        aux_scale=args.aux_scale,
    )

    train(dataset, Net, cfg, tm, start_from=0)
