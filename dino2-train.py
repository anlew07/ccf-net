#!/usr/bin/python3
#coding=utf-8
import sys
sys.path.insert(0, "/root/shared-nvme/dinov3-main")

from functools import partial
import sys
import datetime
import os
import time
import re
import glob
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data import dataset
import logging as logger
from lib.data_prefetcher import DataPrefetcher


from dino1_loss import train_loss

from dino1_net import Net

from tools import *


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1)

TAG = "scribblecod"
logger.basicConfig(
    level=logger.INFO,
    format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=f"train_{TAG}.log",
    filemode="w"
)

# 选择空闲显卡（可按需修改/删除）
GPU_ID = subprocess.getoutput(
    'nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs'
)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


""" ---- 学习率策略（作者原版） ---- """
import numpy as np

def get_triangle_lr(
    base_lr,
    max_lr,
    total_steps,
    cur,
    ratio=0.8,                 # ✅ 默认只做 80% 的三角，后 20% 直接低LR收敛
    annealing_decay=1e-2,
    momentums=(0.95, 0.90),    # ✅ 动量跨度收窄，减少震荡（原来 0.95->0.85 太激进）
    max_lr_cap=None            # ✅ 可选：强行上限 max_lr（比如 3e-3）
):
    """
    Triangular LR with safer tail:
      - [0, ratio*total] 做三角
      - [ratio*total, total] 线性衰减到 min_lr
    """
    total_steps = max(1, int(total_steps))
    cur = int(np.clip(cur, 0, total_steps - 1))

    if max_lr_cap is not None:
        max_lr = min(float(max_lr), float(max_lr_cap))

    first = int(total_steps * float(ratio))
    first = max(1, min(first, total_steps - 1))
    last = total_steps - first
    min_lr = float(base_lr) * float(annealing_decay)

    # --- triangle part ---
    if cur < first:
        # 标准 triangle：0->peak->0
        cycle = np.floor(1 + cur / first)
        x = np.abs(cur * 2.0 / first - 2.0 * cycle + 1)
        lr = float(base_lr) + (float(max_lr) - float(base_lr)) * np.maximum(0., 1.0 - x)

        if isinstance(momentums, (int, float)):
            momentum = float(momentums)
        else:
            m0, m1 = float(momentums[0]), float(momentums[1])
            momentum = m0 + (m1 - m0) * np.maximum(0., 1.0 - x)

    # --- tail decay part ---
    else:
        # 线性从 base_lr 衰减到 min_lr（比你原来更“稳”）
        t = (cur - first) / max(1, last)
        lr = (1 - t) * float(base_lr) + t * float(min_lr)

        if isinstance(momentums, (int, float)):
            momentum = float(momentums)
        else:
            momentum = float(momentums[0])  # tail 固定较稳的动量

    return float(lr), float(momentum)



def get_polylr(base_lr, last_epoch, num_steps, power):
    return base_lr * (1.0 - min(last_epoch, num_steps - 1) / num_steps) ** power


""" ---- 评估调度：≤150 每10轮；≤200 每5轮；>200 每轮 ---- """
def should_validate(epoch_idx: int) -> bool:
    """epoch_idx 从 0 开始；返回该轮是否需要评估"""
    e = epoch_idx + 1
    if e <= 80:
        return e % 10 == 0
    elif e <= 100:
        return e % 5 == 0
    else:
        return True


""" ---- 仅保留最近 K 个 model-e{epoch}.pth（不动 best/last） ---- """
def cleanup_eval_checkpoints(save_dir: str, keep: int = 5):
    pattern = os.path.join(save_dir, "model-e*.pth")
    paths = glob.glob(pattern)

    def parse_epoch(p):
        m = re.search(r"model-e(\d+)\.pth$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    items = [(parse_epoch(p), p) for p in paths if parse_epoch(p) >= 0]
    if len(items) <= keep:
        return

    items.sort(key=lambda x: x[0])  # 按 epoch 升序
    to_delete = [p for _, p in items[:-keep]]
    for p in to_delete:
        try:
            os.remove(p)
            logger.info(f"[CKPT] removed old eval ckpt: {os.path.basename(p)}")
        except Exception as e:
            logger.warning(f"[CKPT] failed to remove {p}: {e}")


""" ---- 验证（作者原版，兼容新 Net：train 模式下依然返回 6 个输出） ---- """
def validate(model, val_loader):
    model.train(False)
    avg_mae = 0.0
    cnt = 0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out, _, _, _, _, _ = model(image)
            out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
            pred = torch.sigmoid(out[0, 0])
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            avg_mae += torch.abs(pred - mask[0]).mean().item()
            cnt += len(image)

    model.train(True)
    return (avg_mae / cnt)


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
    return sum(maes) / len(maes)


# —— 作者原版超参（外层三角学习率调度也会用到）—— #
BASE_LR = 1e-5
MAX_LR = 1e-2
total_epoch = 150   # 你也可以改为 500 等

# —— 数据集根目录（按作者原版）。如路径不同，改这里 —— #
root = 'CodDataset'


def train(Dataset, Network, cfg, train_loss_fn, start_from=0):
    """train_loss_fn 即我们从 train_processes_loss 导入的 train_loss（带 ctx）"""
    # dataset
    data = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
    val_cfg = [Dataset.Config(datapath=f'{root}/test/{i}', mode='test')
               for i in ['CHAMELEON', 'CAMO', 'COD10K']]
    val_data = [Dataset.Data(v) for v in val_cfg]
    val_loaders = [DataLoader(v, batch_size=1, shuffle=False, num_workers=4) for v in val_data]

    min_mae = 1.0
    best_epoch = 0

    # network
    net = Network(cfg)
    net.train(True)
    net.cuda()

    # parameter grouping（作者原版：'bkbone' 归为 base）
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

    # log
    os.makedirs(cfg.savepath, exist_ok=True)
    sw = SummaryWriter(cfg.savepath)
    db_size = len(loader)
    global_step = start_from * db_size
    et = 0

    # 允许通过 cfg 控制保留的评估快照个数（默认 5）
    keep_n_eval_ckpt = int(getattr(cfg, "keep_n_eval_ckpt", 5))

    # 验证日志 txt 路径 & 文件句柄
    val_log_path = os.path.join(cfg.savepath, 'val_log.txt')
    val_log_file = open(val_log_path, 'a', encoding='utf-8')

    # -------------------------- training ------------------------------------
    for epoch in range(start_from, cfg.epoch):
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask = prefetcher.next()

        while image is not None:
            st = time.time()
            niter = epoch * db_size + batch_idx
    lr, momentum = get_triangle_lr(
        BASE_LR, MAX_LR,
        cfg.epoch * db_size,
        niter,
        ratio=0.8,          # ✅ 后 20% 低LR磨
        momentums=(0.95, 0.90),
        max_lr_cap=3e-3     # ✅ 关键：把峰值上限砍到 3e-3（你说的“80-100刚好”通常就是峰值太大）
    )

            optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum

            batch_idx += 1
            global_step += 1

            # === 这里把 save_dir 传给 train_loss，供中间可视化使用 === #
            ctx = dict(
                epoch=epoch + 1,
                global_step=global_step,
                sw=sw,
                t_epo=cfg.epoch,
                save_dir=cfg.savepath  # train_processes_loss 里保存中间图和 npy 就靠这个路径
            )
            loss2, loss3, loss4, loss5, loss6 = train_loss_fn(
                image, mask, net, ctx
            )

            # objective function（保持原权重系数）
            loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4 + loss6 * 0.2

            # --------- NaN / Inf 检查 --------- #
            if (torch.isnan(loss) or torch.isinf(loss)):
                print(">>> Found NaN/Inf in TOTAL loss !!!")
                try:
                    print("  loss2, loss3, loss4, loss5, loss6 =",
                          float(loss2), float(loss3),
                          float(loss4), float(loss5), float(loss6))
                except Exception:
                    print("  [warn] cannot safely print sub-losses")

                # 你还可以在这里打印 epoch / step
                print(f"  epoch = {epoch + 1}, global_step = {global_step}, batch_idx = {batch_idx}")

                # 防止继续训练，直接抛错
                raise RuntimeError("NaN/Inf detected in loss")

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalar('loss', loss.item(), global_step=global_step)

            image, mask = prefetcher.next()
            ta = time.time() - st
            et = 0.9 * et + 0.1 * ta if et > 0 else ta
            if batch_idx % 10 == 0:
                msg = '%s| %s | eta:%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f' % (
                    TAG, datetime.datetime.now(),
                    datetime.timedelta(seconds=int((cfg.epoch * db_size - niter) * et)),
                    global_step, epoch + 1, cfg.epoch,
                    optimizer.param_groups[0]['lr'], loss.item(),
                    loss2.item(), loss3.item(), loss4.item(), loss5.item()
                )
                print(msg)
                logger.info(msg)

        # ---- 评估调度 ----
        if should_validate(epoch):
            mae = validate_multiloader(net, val_loaders, log_file=val_log_file)
            val_line = 'VAL MAE:%s' % (mae)
            print(val_line)
            sw.add_scalar('val', mae, global_step=global_step)

            # 写 VAL MAE 到 txt
            val_log_file.write(val_line + '\n')
            val_log_file.flush()

            # 总是保存最近一次评估后的权重（覆盖式）
            torch.save(net.state_dict(), os.path.join(cfg.savepath, 'model-last.pth'))
            # 保存该评估轮的快照（按轮次）
            cur_snap = os.path.join(cfg.savepath, f'model-e{epoch + 1}.pth')
            torch.save(net.state_dict(), cur_snap)

            # 刷新最优
            if mae < min_mae:
                min_mae = mae
                best_epoch = epoch + 1
                torch.save(net.state_dict(), os.path.join(cfg.savepath, 'model-best.pth'))
                best_line = 'best epoch is:%d, MAE:%s' % (best_epoch, min_mae)
                print(best_line)
                val_log_file.write(best_line + '\n')
                val_log_file.flush()

            # 评估后执行清理：仅保留最近 K 个 model-e*.pth（best/last 不受影响）
            cleanup_eval_checkpoints(cfg.savepath, keep=keep_n_eval_ckpt)

    print('min val mae is {}'.format(min_mae))
    val_log_file.close()


if __name__ == '__main__':
    # —— 时间戳命名 —— #
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    EXP_NAME = f"trained_vA_{run_time}"

    # 作者原版超参（传给 train_loss）
    cfg_list = [.15, 60, 16, 1]
    w_ft, ft_st, topk, w_ftp = cfg_list

    cfg = dataset.Config(
        datapath=f'{root}',
        savepath=f'./out_vdino/{EXP_NAME}/',
        mode='train',
        batch=16,
        lr=1e-3,
        momen=0.9,
        decay=5e-4,
        epoch=total_epoch,          # 可改 500
        label_dir='Scribble',
        keep_n_eval_ckpt=2          # 仅保留最近 2 个评估快照
    )
    os.makedirs(cfg.savepath, exist_ok=True)

    # 把 train_processes_loss.train_loss 部分参数固定成 partial，保持原调用风格
    tm = partial(
        train_loss,
        w_ft=w_ft,
        ft_st=ft_st,
        ft_fct=.5,
        ft_dct=dict(crtl_loss=False, w_ftp=w_ftp, norm=False, topk=topk, step_ratio=2),
        ft_head=False,
        mtrsf_prob=1,
        ops=[0, 1, 2],
        w_l2g=0.3,
        l_me=0.05,
        me_st=20,
        multi_sc=0
    )

    train(dataset, Net, cfg, tm, start_from=0)
