#!/usr/bin/python3
# coding=utf-8

import os
import sys
import time
import subprocess
import logging as logger

import torch
from torch.utils.data import DataLoader

sys.dont_write_bytecode = True

GPU_ID = subprocess.getoutput(
    'nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs'
)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

logger.basicConfig(
    level=logger.INFO,
    format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename="vis_%s.log" % ('tmp'),
    filemode="w"
)

# ✅ 只跑 CAMO
DATASET_PATH = 'CodDataset/test/CAMO'

from data import dataset

# 你的网络文件名按实际改（例如：from Bnet_dinov3 import Net）
# from dino1_net import Net
# from cp_cof_1_1_dino_edge import Net
# from cp_cof_1_1_dino_sem import Net
# from cp_cof_0_0_dino import Net

from backbone_dinov2_dino_net import Net
# from backbone_dinov1_dino_net import Net
# from B2_5_8_11_dino_net import Net
# from 5_8_11_dino_net import Net


# ✅ 你的 visualizer（已经按你要求：不保存 raw_tensors.pt）
from utils.visualizer import save_visualization


class VisualizeOnly(object):
    def __init__(self, Dataset, datapath, Network):
        self.datapath = datapath.split("/")[-1]  # CAMO
        print("Visualizing on %s" % self.datapath)

        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)

        self.net = Network
        self.net.cuda()
        self.net.eval()

        # ✅ 只输出到 vis 目录
        self.vis_root = os.path.join('./vis', EXP_NAME, self.datapath)
        os.makedirs(self.vis_root, exist_ok=True)

    def run(self):
        with torch.no_grad():
            cost_time = 0.0
            n = 0

            for i, (image, mask, (H, W), name) in enumerate(self.loader):
                file_stem = name[0].split('.')[0]
                prefix = file_stem + "_"  # ✅ 防止文件名黏连

                start_time = time.perf_counter()
                out2, _ = self.net(image.cuda().float(), (H, W))
                torch.cuda.synchronize()
                cost_time += time.perf_counter() - start_time
                n += 1

                prob = torch.sigmoid(out2)  # [1,1,H,W]

                # ✅ 只打印第一张（一定能触发）
                if i == 0:
                    aux = getattr(self.net, "aux_cache", {})
                    print("aux_cache keys:", sorted(list(aux.keys())), flush=True)

                save_visualization(
                    save_dir=self.vis_root,
                    image=image.cpu(),
                    gt=mask,
                    pred=prob.cpu(),
                    aux_cache=getattr(self.net, "aux_cache", {}),
                    prefix=prefix
                )

            fps = n / max(1e-6, cost_time)
            msg = f'{self.datapath} saved_vis={n}, fps={fps:.4f}, out_dir={self.vis_root}'
            print(msg)
            logger.info(msg)


# EXP_NAME = 'cp_cof_1_1_dino_sem_edge'
# EXP_NAME = 'cp_cof_1_1_dino_edge'
# EXP_NAME = 'cp_cof_1_1_dino_sem'
# EXP_NAME = 'cp_cof_0_0_dino'
# EXP_NAME = 'cp_cof_1_0_dino'

EXP_NAME = 'backbone_dinov2_dino_net'
# EXP_NAME = 'backbone_dinov1_dino_net'
# EXP_NAME = 'B2_5_8_11_dino_net'
# EXP_NAME = '5_8_11_dino_net'



if __name__ == '__main__':
    cfg = dataset.Config(datapath='000', mode='test')
    net = Net(cfg)

    # ✅ 你的权重路径
    # weight_path = 'cp_cof_1_1_dino_sem_edge/dino_0415/model-best.pth'
    # weight_path = 'cp_cof_1_1_dino_edge/cp_cof_1_1_dino_edge20260116_210557/model-best.pth'
    # weight_path = 'cp_cof_1_1_dino_sem/cp_cof_1_1_dino_sem20260116_181056/model-best.pth'
    # weight_path = 'cp_cof_0_0_dino/cp_cof_0_0_dino20260117_143554/model-best.pth'
    # weight_path = 'cp_cof_1_0_dino/cp_cof_1_0_dino20260302_174212/model-best.pth'

    
    weight_path = 'out_backbone_dinov2_dino_net_/backbone_dinov2_dino_net/model-best.pth'
    # weight_path = 'out_backbone_dinov1_dino_net_/backbone_dinov1_dino_net/model-best.pth'
    # weight_path = 'out_B2_5_8_11_dino_net_/B2_5_8_11_dino_net_run/model-best.pth'
    # weight_path = 'out_5_8_11_dino_net_/5_8_11_dino_net_20260318_052734/model-best.pth'
# 
    
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)

    msg = net.load_state_dict(state_dict, strict=False)
    print(f'complete loading: {weight_path}')
    print(f'missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}')
    print('-----------------')
    print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))

    v = VisualizeOnly(dataset, DATASET_PATH, net)
    v.run()
