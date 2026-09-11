#!/usr/bin/python3
# coding=utf-8

from __future__ import annotations

import os
import os.path as osp
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Config(object):
    def __init__(self, **kwargs):
        if kwargs.get('label_dir') is None:
            kwargs['label_dir'] = 'Scribble'
        if kwargs.get('stage1_root') is None:
            kwargs['stage1_root'] = 'runs/stage1/export_mainprob'
        if kwargs.get('trainsize') is None:
            kwargs['trainsize'] = 320
        self.kwargs = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-12s: %s' % (k, v))

        # keep the same default statistics as the user's original dataset.py
        self.mean = np.array([[[0.485 * 256, 0.456 * 256, 0.406 * 256]]], dtype=np.float32)
        self.std = np.array([[[0.229 * 256, 0.224 * 256, 0.225 * 256]]], dtype=np.float32)

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        return None


class Data(Dataset):
    """
    Stage 2 training dataset.

    Returns:
        image:        float tensor [3,H,W]
        stage1_label: long tensor  [1,H,W], disk convention 0=ignore, 1=fg, 2=bg
        trust_map:    float tensor [1,H,W] in [0,1]
        scribble:     long tensor  [1,H,W], disk convention 0=unlabeled, 1=fg, 2=bg
        shape:        original (H, W)
        name:         file name with suffix
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_name = cfg.datapath.split('/')[-1]
        self.trainsize = int(cfg.trainsize)
        self.stage1_root = cfg.stage1_root

        list_path = osp.join(cfg.datapath, 'train.txt')
        if not osp.isfile(list_path):
            raise FileNotFoundError(f'train.txt not found: {list_path}')

        with open(list_path, 'r') as lines:
            self.samples: List[Tuple[str, str, str, str, str]] = []
            for line in lines:
                stem = line.strip()
                if not stem:
                    continue
                imagepath = osp.join(cfg.datapath, 'train', 'Image', stem + '.jpg')
                scribblepath = osp.join(cfg.datapath, 'train', cfg.label_dir, stem + '.png')
                stage1_label_path = osp.join(self.stage1_root, 'Stage1Label', stem + '.png')
                trust_path = osp.join(self.stage1_root, 'TrustMap', stem + '.png')
                self.samples.append((imagepath, stage1_label_path, trust_path, scribblepath, stem + '.png'))

        if len(self.samples) == 0:
            raise RuntimeError(f'No training samples found from {list_path}')

    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        arr = cv2.imread(path, cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f'Failed to read image: {path}')
        return arr[:, :, ::-1].astype(np.float32)

    @staticmethod
    def _load_gray(path: str) -> np.ndarray:
        arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f'Failed to read gray map: {path}')
        return arr

    @staticmethod
    def _remap_scribble(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.uint8)
        uniq = np.unique(arr)
        if set(uniq.tolist()).issubset({0, 1, 2}):
            return arr
        vals = sorted(uniq.tolist())
        if len(vals) == 3:
            mp = {vals[0]: 0, vals[1]: 1, vals[2]: 2}
            out = np.zeros_like(arr, dtype=np.uint8)
            for k, v in mp.items():
                out[arr == k] = v
            return out
        raise ValueError(f'Unexpected scribble labels: {uniq.tolist()}')

    @staticmethod
    def _remap_stage1_label(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.uint8)
        uniq = np.unique(arr)
        if set(uniq.tolist()).issubset({0, 1, 2}):
            return arr
        vals = sorted(uniq.tolist())
        if len(vals) == 3:
            mp = {vals[0]: 0, vals[1]: 1, vals[2]: 2}
            out = np.zeros_like(arr, dtype=np.uint8)
            for k, v in mp.items():
                out[arr == k] = v
            return out
        raise ValueError(f'Unexpected Stage1Label values: {uniq.tolist()}')

    def _resize_all(self, image, stage1_label, trust, scribble):
        size = (self.trainsize, self.trainsize)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        stage1_label = cv2.resize(stage1_label, size, interpolation=cv2.INTER_NEAREST)
        trust = cv2.resize(trust, size, interpolation=cv2.INTER_LINEAR)
        scribble = cv2.resize(scribble, size, interpolation=cv2.INTER_NEAREST)
        return image, stage1_label, trust, scribble

    @staticmethod
    def _hflip_all(image, stage1_label, trust, scribble):
        image = image[:, ::-1, :].copy()
        stage1_label = stage1_label[:, ::-1].copy()
        trust = trust[:, ::-1].copy()
        scribble = scribble[:, ::-1].copy()
        return image, stage1_label, trust, scribble

    def _augment_train(self, image, stage1_label, trust, scribble):
        image, stage1_label, trust, scribble = self._resize_all(image, stage1_label, trust, scribble)
        if random.random() < 0.5:
            image, stage1_label, trust, scribble = self._hflip_all(image, stage1_label, trust, scribble)
        return image, stage1_label, trust, scribble

    def __getitem__(self, idx):
        imagepath, stage1_path, trust_path, scribble_path, name = self.samples[idx]
        image = self._load_rgb(imagepath)
        stage1_label = self._remap_stage1_label(self._load_gray(stage1_path))
        trust = self._load_gray(trust_path).astype(np.float32) / 255.0
        trust = np.clip(trust, 0.0, 1.0)
        scribble = self._remap_scribble(self._load_gray(scribble_path))

        H, W = image.shape[:2]
        image, stage1_label, trust, scribble = self._augment_train(image, stage1_label, trust, scribble)

        image = (image - self.cfg.mean) / self.cfg.std
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
        stage1_label = torch.from_numpy(stage1_label.copy()).unsqueeze(0).long()
        trust = torch.from_numpy(trust.copy()).unsqueeze(0).float()
        scribble = torch.from_numpy(scribble.copy()).unsqueeze(0).long()

        return image, stage1_label, trust, scribble, (H, W), name

    def __len__(self):
        return len(self.samples)
