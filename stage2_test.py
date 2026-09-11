#!/usr/bin/python3
# coding=utf-8

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.dont_write_bytecode = True

GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

from data import dataset
from dino1_net import Net

ROOT = 'CodDataset'
DATASETS = [f'{ROOT}/test/CAMO', f'{ROOT}/test/CHAMELEON', f'{ROOT}/test/COD10K', f'{ROOT}/test/NC4K']
JSON_METHOD = './PySODEvalToolkit/cod_method.json'
JSON_DATA = './PySODEvalToolkit/cod_dataset.json'


class Test(object):
    def __init__(self, Dataset, datapath, Network):
        self.datapath = datapath.split('/')[-1]
        print('Testing on %s' % self.datapath)
        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        self.net = Network
        self.net.train(False)
        self.net.cuda()
        self.net.eval()

    def save(self, exp_name):
        with torch.no_grad():
            cost_time = 0.0
            cnt = 0
            mae = 0.0
            print(f'will save to ./map/{exp_name}')
            head = f'./map/{exp_name}/' + self.cfg.datapath.split('/')[-1]
            os.makedirs(head, exist_ok=True)
            for image, mask, (H, W), name in self.loader:
                start_time = time.perf_counter()
                out = self.net(image.cuda().float(), (H, W))
                if isinstance(out, (list, tuple)):
                    out2 = out[0]
                else:
                    out2 = out
                torch.cuda.synchronize()
                cost_time += time.perf_counter() - start_time

                pred = (torch.sigmoid(out2[0, 0])).cpu()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                mae += (pred - mask).abs().mean()
                cnt += len(image)
                cv2.imwrite(os.path.join(head, name[0]), img_as_ubyte(pred.numpy()))

            fps = len(self.loader.dataset) / max(cost_time, 1e-6)
            print('mae {}'.format(mae / max(cnt, 1)))
            print('%s len(imgs)=%s, fps=%.4f' % (self.datapath, len(self.loader.dataset), fps))

    def change_json(self, exp_name, json_path=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
        k1 = next(iter(js.values()))
        k1.setdefault(self.datapath, {})
        pred_dir = os.path.abspath(os.path.join('./map', exp_name, self.datapath))
        k1[self.datapath]['path'] = pred_dir
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(js, f, indent=4)


def cal_cod_metrics(js_m, js_d):
    js_m = os.path.abspath(js_m)
    js_d = os.path.abspath(js_d)
    os.chdir('./PySODEvalToolkit')
    os.system('python ./eval.py --method {} --dataset {} --record-txt ./results.txt'.format(js_m, js_d))
    os.chdir('../')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default='stage2_eval')
    args = parser.parse_args()

    cfg = dataset.Config(datapath='000', mode='test')
    net = Net(cfg)
    state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    net.load_state_dict(state_dict, strict=True)
    print('complete loading: {}'.format(args.ckpt))
    print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))

    for e in DATASETS:
        t = Test(dataset, e, net)
        t.save(args.exp_name)
        t.change_json(args.exp_name, JSON_METHOD)

    cal_cod_metrics(JSON_METHOD, JSON_DATA)
    print(args.exp_name)
