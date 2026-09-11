# Bnet_dinov3.py
# 单阶段 Decoder + 单尺度 Consistency + dec_c2/dec_c1 频域 TFGM 稳定版
# ✅ Backbone: DINOv3 (ViT-S/16) + 轻量 Pyramid Adapter 输出 (1/4,1/8,1/16,1/32)
# ✅ 只输入 RGB + scribble（不做 4 通道注入）
# ✅ cons1 稀疏化 + 区分 dec_c1 / dec_out 两套 q+dilate
# ✅ CAM edge gate 改成 1 通道输入（mean 后再 gate）
# ✅ 修正 backbone 的上下文（freeze 时 inference_mode，不冻结时保留梯度）
# ✅ TFGM 对 cons 做鲁棒规整（支持 [B,H,W]/[B,1,H,W]/[B,C,H,W]）

import os
import math
from collections import OrderedDict
from contextlib import nullcontext
import sys
sys.path.insert(0, "/root/shared-nvme/dinov3-main")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ============================ 通用初始化 & 卷积封装 ============================ #

def weight_init(module):
    """
    递归地对网络中的卷积/归一化层进行初始化；
    对 nn.Sequential / nn.ModuleList / nn.ModuleDict 递归处理；
    若子模块自带 initialize()，则调用它（安全判定）。
    """
    for _, m in module.named_children():

        # ---- 可训练层：按类型初始化 ----
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d)):
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d,
                            nn.LayerNorm, nn.GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(m.bias, -bound, bound)

        # ---- 容器类：递归 ----
        elif isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            weight_init(m)

        # ---- 无参数/无需初始化的层：跳过 ----
        elif isinstance(m, (
            nn.ReLU, nn.GELU, nn.LeakyReLU, nn.ReLU6, nn.SiLU, nn.ELU, nn.SELU,
            nn.Tanh, nn.Sigmoid, nn.Softmax, nn.Softmax2d, nn.Identity,
            nn.Dropout, nn.Dropout2d, nn.Dropout3d,
            nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
            nn.Upsample, nn.Flatten
        )):
            pass

        # ---- 其它自定义模块：仅在存在 initialize() 时调用 ----
        else:
            if hasattr(m, "initialize") and callable(getattr(m, "initialize")):
                m.initialize()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=bias
    )


# ============================ 基础卷积与金字塔池化 ============================ #

class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel,
                 k=3, s=1, p=1,
                 g=1, d=1,
                 bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(
            in_channel, out_channel,
            k, s, p,
            dilation=d,
            groups=g,
            bias=bias
        )]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)


class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel * 2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size, mode="bilinear", align_corners=False)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size, mode="bilinear", align_corners=False)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size, mode="bilinear", align_corners=False)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size, mode="bilinear", align_corners=False)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

    def initialize(self):
        weight_init(self)


# ============================ Feature Fusion Module ============================ #

class GatedFFM(nn.Module):
    """
    GatedFFM：concat → 通道门 + 空间门 → (x1*x2)*gate + 0.5(x1+x2) → Conv3x3×2
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channel * 2, channel // reduction, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)

        self.spatial = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 3, padding=1, groups=1, bias=False),
            nn.Sigmoid()
        )

        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        cat = torch.cat([x1, x2], dim=1)

        ch = self.avg(cat)
        ch = self.fc2(self.act(self.fc1(ch)))
        ch = self.sigmoid(ch)

        sp = self.spatial(cat)

        x = (x1 * x2) * ch * sp + 0.5 * (x1 + x2)
        x = F.relu(self.bn_1(self.conv_1(x)), inplace=True)
        x = F.relu(self.bn_2(self.conv_2(x)), inplace=True)
        return x


# ============================ CAM：边缘注入式跨层聚合（✅ 1ch edge gate） ============================ #

class CAM(nn.Module):
    """
    CAMEdgeInject:
        out = x_low + α * g(edge(x_low)) * (up(x_high) - x_low)

    ✅ 修改点：
      - edge_from_low 仍输出 [B,C,H,W]（逐通道 laplacian）
      - gate 输入改为 1ch：先 mean -> [B,1,H,W] 再 gate
    """
    def __init__(self, channel, alpha_init=0.5):
        super().__init__()
        self.channel = channel

        lap_kernel = torch.tensor(
            [[0.,  1., 0.],
             [1., -4., 1.],
             [0.,  1., 0.]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap_kernel, persistent=False)

        # ✅ 1通道 gate
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            conv3x3(channel, channel),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            conv3x3(channel, channel),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.alpha = nn.Parameter(
            torch.tensor(alpha_init, dtype=torch.float32),
            requires_grad=True
        )

    def edge_from_low(self, x_low):
        B, C, H, W = x_low.shape
        k = self.lap_kernel.to(x_low.dtype).repeat(C, 1, 1, 1)
        y = F.conv2d(x_low, k, padding=1, groups=C)
        e = y.abs()
        return e

    def forward(self, x_high, x_low):
        B, C, Hl, Wl = x_low.shape

        xh = F.interpolate(
            x_high,
            size=(Hl, Wl),
            mode='bilinear',
            align_corners=True
        )

        # edge: [B,C,H,W] -> mean -> [B,1,H,W]
        e = self.edge_from_low(x_low).mean(dim=1, keepdim=True)
        g = self.edge_gate(e)                         # [B,1,Hl,Wl]

        fused = x_low + self.alpha * g * (xh - x_low)
        out = self.refine(fused) + x_low
        return out


# ============================ Boundary Refinement Module（保留但当前未用） ============================ #

class BRM(nn.Module):
    def __init__(self, channel):
        super(BRM, self).__init__()
        self.conv_atten = conv1x1(channel, channel)
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_edge):
        x = x_1 + x_edge
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten) + x
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# ============================ RFB modified (LSR) ============================ #

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch3 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.conv_cat = basicConv(4 * out_channel, out_channel, 3, p=1, relu=False)
        self.conv_res = basicConv(in_channel, out_channel, 1, relu=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

    def initialize(self):
        weight_init(self)


# ============================ Coordinate Attention ============================ #

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

    def initialize(self):
        weight_init(self)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        weight_init(self)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)                      # [B,C,H,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B,C,1,W] -> [B,C,W,1]

        y = torch.cat([x_h, x_w], dim=2)          # [B,C,H+W,1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

    def initialize(self):
        weight_init(self)


# ============================ Local-Context Contrasted (TFLCC) ============================ #

class Contrast_Block_Deep(nn.Module):
    def __init__(self, planes, d1=4, d2=8):
        super(Contrast_Block_Deep, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 2)

        self.local_1 = nn.Conv2d(self.inplanes, self.outplanes, 3, 1, 1, 1)
        self.context_1 = nn.Conv2d(self.inplanes, self.outplanes, 3, 1, d1, d1)
        self.local_2 = nn.Conv2d(self.inplanes, self.outplanes, 3, 1, 1, 1)
        self.context_2 = nn.Conv2d(self.inplanes, self.outplanes, 3, 1, d2, d2)

        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.ca = nn.ModuleList([
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes)
        ])

    def forward(self, x):
        local_1 = self.local_1(x)
        local_1 = self.ca[0](local_1)
        context_1 = self.context_1(x)
        context_1 = self.ca[1](context_1)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn1(ccl_1)
        ccl_1 = self.relu1(ccl_1)

        local_2 = self.local_2(x)
        local_2 = self.ca[2](local_2)
        context_2 = self.context_2(x)
        context_2 = self.ca[3](context_2)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn2(ccl_2)
        ccl_2 = self.relu2(ccl_2)

        out = torch.cat((ccl_1, ccl_2), 1)
        return out

    def initialize(self):
        weight_init(self)


# ============================ FrequencyEnhance & TFGM（简化稳版） ============================ #

class FrequencyEnhance(nn.Module):
    def __init__(self, channels, bands=3, k_soft=6.0, high_beta=0.75):
        super().__init__()
        self.channels = channels
        self.bands = bands
        self.k_soft = float(k_soft)
        self.high_beta = float(high_beta)

        self.band_logits = nn.Conv2d(channels, channels * bands, kernel_size=1, bias=True)
        self.global_gain = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.high_gain = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1, bias=True))
        weight_init(self)

    def forward(self, x):
        B, C, H, W = x.shape

        X = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(X) + 1e-6
        phase = torch.angle(X)
        logmag = torch.log(mag)

        logits = self.band_logits(logmag)
        B2, Cb, Hf, Wf = logits.shape
        assert B2 == B and Hf == H, "FrequencyEnhance: shape mismatch"

        logits = logits.view(B, self.bands, C, Hf, Wf)
        weights = torch.softmax(logits / max(1e-6, self.k_soft), dim=1)

        mag_exp = mag.unsqueeze(1)
        mag_enh = (weights * mag_exp).sum(dim=1)

        mag_enh = mag_enh * (1.0 + self.global_gain)

        device, dtype = mag.device, mag.dtype
        yy = torch.linspace(-1.0, 1.0, steps=Hf, device=device, dtype=dtype).view(1, 1, Hf, 1)
        xx = torch.linspace(0.0, 1.0, steps=Wf, device=device, dtype=dtype).view(1, 1, 1, Wf)
        r = torch.sqrt(xx * xx + yy * yy)

        high_mask = torch.sigmoid(10.0 * (r - 0.6))
        high_energy = (mag * high_mask).mean(dim=(-2, -1), keepdim=True)
        high_delta = torch.tanh(self.high_gain(high_energy))
        high_delta_map = high_delta * high_mask

        mag_scale = 1.0 + self.high_beta * high_delta_map
        mag_scale = mag_scale.clamp(0.1, 2.0)
        mag_enh = mag_enh * mag_scale

        real = mag_enh * torch.cos(phase)
        imag = mag_enh * torch.sin(phase)
        X_enh = torch.complex(real, imag)

        x_rec = torch.fft.irfft2(X_enh, s=(H, W), norm='ortho')
        return x_rec


class TFGM(nn.Module):
    def __init__(self, channels, bands=3, hidden=16, k_soft=6.0, res_scale=0.5):
        super().__init__()
        self.channels = channels
        self.res_scale = float(res_scale)

        self.freq = FrequencyEnhance(channels, bands=bands, k_soft=k_soft)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.channel_mlp = nn.Sequential(
            nn.Linear(2 * channels + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.debug = {}
        weight_init(self)

    def _norm_cons(self, cons, x_ref):
        """把 cons 规整到 [B,1,H,W] 且 clamp 到 [0,1]"""
        B, C, H, W = x_ref.shape
        if cons is None:
            return x_ref.new_zeros(B, 1, H, W)

        if cons.dim() == 3:
            cons = cons.unsqueeze(1)  # [B,1,H,W]
        elif cons.dim() == 4:
            pass
        else:
            return x_ref.new_zeros(B, 1, H, W)

        if cons.size(0) != B:
            # 不匹配就直接置零，避免隐式广播坑
            return x_ref.new_zeros(B, 1, H, W)

        if cons.size(1) != 1:
            cons = cons.mean(dim=1, keepdim=True)

        if cons.shape[-2:] != (H, W):
            cons = F.interpolate(cons, size=(H, W), mode='bilinear', align_corners=True)

        return cons.clamp(0, 1)

    def forward(self, x, cons: torch.Tensor = None, record: bool = False, tag: str = None):
        B, C, H, W = x.shape
        x_freq = self.freq(x)

        cons = self._norm_cons(cons, x)

        g_sp_1 = self.spatial_gate(cons)          # [B,1,H,W]
        g_spatial = g_sp_1.expand(-1, C, -1, -1)  # [B,C,H,W]

        g_x = F.adaptive_avg_pool2d(x, 1).view(B, C)
        g_f = F.adaptive_avg_pool2d(x_freq, 1).view(B, C)
        g_cons = F.adaptive_avg_pool2d(cons, 1).view(B, 1)

        v = torch.cat([g_x, g_f, g_cons], dim=1)
        g_ch = torch.sigmoid(self.channel_mlp(v)).view(B, C, 1, 1)

        diff = (x_freq - x) * g_spatial * g_ch
        delta = self.refine(diff) * self.res_scale
        out = x + delta

        if record:
            self.debug = {
                "tag": tag,
                "x": x.detach(),
                "x_freq": x_freq.detach(),
                "cons": cons.detach(),
                "g_sp": g_sp_1.detach(),
                "g_ch": g_ch.detach(),
            }
        return out


# ============================ DINOv3 Backbone + Pyramid Adapter ============================ #

class DINOv3PyramidBackbone(nn.Module):
    """
    输出：(enc_c0, enc_c1, enc_c2, enc_c3, enc_c4)
      enc_c1: 1/4
      enc_c2: 1/8
      enc_c3: 1/16
      enc_c4: 1/32

    - 从 DINOv3 取 4 个中间层特征（reshape=True => [B,C,H/16,W/16]）
    - 在 1/16 融合 => f16
    - 通过 up/down 生成 f8, f4, f32
    """
    def __init__(
        self,
        arch="dinov3_vits16",
        ckpt_path="/root/shared-nvme/pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        out_indices=None,
        fuse="mean",           # "mean" or "cat"
        freeze=True,
        fp16_infer=False,
    ):
        super().__init__()

        # ---- 防 None ----
        if arch is None or (isinstance(arch, str) and arch.strip() == ""):
            arch = "dinov3_vits16"
        if ckpt_path is None or (isinstance(ckpt_path, str) and ckpt_path.strip() == ""):
            ckpt_path = "/root/shared-nvme/pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        if fuse is None or (isinstance(fuse, str) and fuse.strip() == ""):
            fuse = "mean"

        self.arch = str(arch)
        self.ckpt_path = str(ckpt_path)
        self.fuse = str(fuse)
        self.fp16_infer = bool(fp16_infer)

        # import dinov3
        try:
            import dinov3.hub.backbones as bb
        except Exception as e:
            raise ImportError(f"Cannot import dinov3. Ensure dinov3-main is in PYTHONPATH. err={e}")

        available = sorted([n for n in dir(bb) if n.startswith("dinov3_")])
        if not hasattr(bb, self.arch):
            raise ValueError(
                f"[DINOv3] Unknown arch='{self.arch}'.\n"
                f"Available dinov3 backbones: {available}"
            )

        build_fn = getattr(bb, self.arch)
        self.backbone = build_fn(pretrained=False)

        # 基本属性
        self.embed_dim = int(getattr(self.backbone, "embed_dim", getattr(self.backbone, "dim", 384)))
        self.patch_size = int(getattr(self.backbone, "patch_size", 16))
        self.depth = int(getattr(self.backbone, "n_blocks", getattr(self.backbone, "depth", 12)))

        # 默认均匀取 4 层
        if out_indices is None:
            if self.depth >= 24:
                out_indices = [4, 11, 17, 23]
            else:
                step = max(1, self.depth // 4)
                out_indices = [step - 1, 2 * step - 1, 3 * step - 1, 4 * step - 1]
        self.out_indices = list(out_indices)

        # 融合投影
        if self.fuse == "cat":
            self.fuse_proj = nn.Sequential(
                nn.Conv2d(self.embed_dim * 4, self.embed_dim, 1, bias=False),
                nn.BatchNorm2d(self.embed_dim),
                nn.GELU(),
            )
        else:
            self.fuse_proj = nn.Identity()

        self.to_f4 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU(),
        )
        self.to_f8 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU(),
        )
        self.to_f32 = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU(),
        )

        # load ckpt
        self._load_ckpt(self.ckpt_path)

        if bool(freeze):
            self.backbone.requires_grad_(False)

        weight_init(self.fuse_proj)
        weight_init(self.to_f4)
        weight_init(self.to_f8)
        weight_init(self.to_f32)

    def _load_ckpt(self, ckpt_path):
        if ckpt_path is None or not os.path.isfile(ckpt_path):
            print(f"[DINOv3] ckpt not found: {ckpt_path}. Use random init.")
            return

        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(raw, (dict, OrderedDict)):
            if "state_dict" in raw and isinstance(raw["state_dict"], (dict, OrderedDict)):
                sd = raw["state_dict"]
            elif "model" in raw and isinstance(raw["model"], (dict, OrderedDict)):
                sd = raw["model"]
            else:
                sd = raw
        else:
            print(f"[DINOv3] Unrecognized ckpt type: {type(raw)}. Use random init.")
            return

        def strip_prefix(d, prefix):
            out = {}
            for k, v in d.items():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = v
                else:
                    out[k] = v
            return out

        for pfx in ["module.", "backbone.", "teacher.", "student."]:
            sd = strip_prefix(sd, pfx)

        model_sd = self.backbone.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        incompatible = self.backbone.load_state_dict(filtered, strict=False)

        print(
            f"[DINOv3] loaded: {ckpt_path} | "
            f"matched={len(filtered)} | "
            f"missing={len(incompatible.missing_keys)} | "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    def _get_intermediate(self, x):
        fn = getattr(self.backbone, "get_intermediate_layers", None)
        if fn is None:
            raise AttributeError("[DINOv3] backbone has no get_intermediate_layers()")

        try:
            feats = fn(x, n=self.out_indices, reshape=True, return_class_token=False)
            return feats
        except TypeError:
            pass

        try:
            feats = fn(x, n=len(self.out_indices), reshape=True, return_class_token=False)
            return feats
        except TypeError:
            pass

        feats = fn(x, n=self.out_indices, reshape=True)
        return feats

    def forward(self, x):
        # autocast：fp16_infer=True 时启用
        use_cuda_amp = (x.is_cuda and self.fp16_infer)
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda_amp)

        # ✅ 冻结：inference_mode；不冻结：nullcontext 保留梯度
        frozen = (not any(p.requires_grad for p in self.backbone.parameters()))
        cm = torch.inference_mode() if frozen else nullcontext()

        with cm:
            with ctx:
                feats = self._get_intermediate(x)

        if not isinstance(feats, (list, tuple)) or len(feats) == 0:
            raise RuntimeError(f"[DINOv3] get_intermediate_layers returned invalid feats: {type(feats)}")

        feats4 = feats[-4:] if len(feats) >= 4 else (list(feats) + [feats[-1]] * (4 - len(feats)))

        if self.fuse == "cat":
            f16 = self.fuse_proj(torch.cat(feats4, dim=1))
        else:
            f16 = sum(feats4) / float(len(feats4))

        f8 = self.to_f8(F.interpolate(f16, scale_factor=2.0, mode="bilinear", align_corners=False))
        f4 = self.to_f4(F.interpolate(f16, scale_factor=4.0, mode="bilinear", align_corners=False))
        f32 = self.to_f32(F.avg_pool2d(f16, kernel_size=2, stride=2))

        enc_c0 = f4  # 占位
        enc_c1 = f4  # 1/4
        enc_c2 = f8  # 1/8
        enc_c3 = f16 # 1/16
        enc_c4 = f32 # 1/32
        return enc_c0, enc_c1, enc_c2, enc_c3, enc_c4


# ============================ 主网络（Backbone = DINOv3） ============================ #

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg

        # ---- DINOv3 backbone ----
        dinov3_ckpt = getattr(cfg, "dinov3_ckpt", "/root/shared-nvme/pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
        dinov3_arch = getattr(cfg, "dinov3_arch", "dinov3_vits16")
        dinov3_fuse = getattr(cfg, "dinov3_fuse", "mean")  # mean/cat
        dinov3_freeze = bool(getattr(cfg, "dinov3_freeze", True))
        dinov3_fp16 = bool(getattr(cfg, "dinov3_fp16", False))

        self.bkbone = DINOv3PyramidBackbone(
            arch=dinov3_arch,
            ckpt_path=dinov3_ckpt,
            fuse=dinov3_fuse,
            freeze=dinov3_freeze,
            fp16_infer=dinov3_fp16,
        )
        ENC_DIM = int(self.bkbone.embed_dim)

        # ---- encoder to decoder adapters ----
        self.pyramid_pooling = PyramidPooling(ENC_DIM, 64)

        self.conv1 = nn.ModuleList([
            basicConv(ENC_DIM, 64, k=1, s=1, p=0),  # enc_c0_64 (unused)
            basicConv(ENC_DIM, 64, k=1, s=1, p=0),  # enc_c1_64, 1/4
            basicConv(ENC_DIM, 64, k=1, s=1, p=0),  # enc_c2_64, 1/8
            basicConv(ENC_DIM, 64, k=1, s=1, p=0),  # reserved
            basicConv(ENC_DIM, 64, k=1, s=1, p=0),  # reserved
        ])

        self.rfb = nn.ModuleList([
            RFB_modified(ENC_DIM, 64),   # enc_c3 -> 1/16
            RFB_modified(ENC_DIM, 64),   # enc_c4 -> 1/32
        ])

        # ===== 分层 TFGM 强度 =====
        self.tfgm = nn.ModuleDict({
            "dec_c4":  TFGM(64, res_scale=0.30),
            "dec_c3":  TFGM(64, res_scale=0.35),
            "dec_c2":  TFGM(64, res_scale=0.45),
            "dec_c1":  TFGM(64, res_scale=0.60),
            "dec_out": TFGM(64, res_scale=0.70),
        })

        self.fusion = nn.ModuleList([
            GatedFFM(64),  # 0: dec_c1 + up(dec_c2) -> dec_out
            GatedFFM(64),  # 1: enc_c3_lsr + up(dec_c4) -> dec_c3
            GatedFFM(64),  # 2: enc_c4_lsr + pp(C5) -> dec_c4
            GatedFFM(64),  # 3: reserved
        ])

        self.aggregation = nn.ModuleList([
            CAM(64),  # 0: dec_c1: (dec_c3, enc_c1_64) -> 1/4
            CAM(64),  # 1: dec_c2: (dec_c4, enc_c2_64) -> 1/8
        ])

        self.head = nn.ModuleList([
            conv3x3(64, 1, bias=True),  # 0: main (dec_out)
            conv3x3(64, 1, bias=True),  # 1: dec_c1
            conv3x3(64, 1, bias=True),  # 2: dec_c2
            conv3x3(64, 1, bias=True),  # 3: dec_c3
            conv3x3(64, 1, bias=True),  # 4: dec_c4
        ])

        self.feature_head = nn.ModuleList([
            basicConv(64, 32, relu=False)
        ])
        self.register_buffer("_iter", torch.zeros(1, dtype=torch.long), persistent=False)

        # ===== 双 cons 的 encoder prior =====
        self.enc_q_proj_sem  = basicConv(ENC_DIM, 1, k=1, s=1, p=0, bn=False, relu=False)  # from enc_c2(1/8)
        self.enc_q_proj_edge = basicConv(ENC_DIM, 1, k=1, s=1, p=0, bn=False, relu=False)  # from enc_c1(1/4)

        self.aux_cache = {}
        self.initialize()

    # ---------- cfg 读取：彻底防 None ----------
    def _cfg_num(self, name, default):
        v = getattr(self.cfg, name, default)
        if v is None:
            return default
        try:
            return type(default)(v)
        except Exception:
            try:
                return float(v) if isinstance(default, float) else int(float(v))
            except Exception:
                return default

    # ---------- cons 稀疏化（top-k 保留） ----------
    def _sparsify_cons(self, cons_1ch: torch.Tensor, q: float = 0.85, dilate: int = 1):
        """
        Top-quantile 稀疏化：保留 >= quantile(q) 的位置
        - q 越大越稀疏（0.95 很稀，0.80 更“松”）
        - dilate 让区域更连贯
        """
        if cons_1ch is None:
            return None
        c = cons_1ch

        if q is None or q <= 0:
            return c.clamp(0, 1)

        with torch.no_grad():
            if float(c.max()) <= 1e-6:
                return c

            B = c.size(0)
            flat = c.view(B, -1)
            thr = torch.quantile(flat, q, dim=1, keepdim=True).view(B, 1, 1, 1)
            m = (c >= thr).float()

            if dilate is not None and int(dilate) > 0:
                k = 2 * int(dilate) + 1
                m = F.max_pool2d(m, kernel_size=k, stride=1, padding=int(dilate))

        return (c * m).clamp(0, 1)

    def forward(self, x, shape=None, epoch=None):
        shape = x.size()[2:] if shape is None else shape

        # ========== Encoder ==========
        enc_c0, enc_c1, enc_c2, enc_c3, enc_c4 = self.bkbone(x)

        enc_c1_64 = self.conv1[1](enc_c1)  # 1/4
        enc_c2_64 = self.conv1[2](enc_c2)  # 1/8

        # ========== warmup ==========
        if self.training:
            self._iter += 1
            it = float(self._iter.item())
        else:
            it = 1e9

        warm_iters = float(self._cfg_num("cons_warmup_iters", 6000))
        ramp = min(1.0, it / max(1.0, warm_iters))

        EDGE_BETA_T = float(self._cfg_num("cons_edge_beta", 0.3))
        Q_EDGE_T    = float(self._cfg_num("cons_sparsify_q_edge", 0.85))
        DILATE_T    = int(self._cfg_num("cons_edge_dilate", 1))

        EDGE_BETA = EDGE_BETA_T * ramp
        Q_EDGE    = (0.95 * (1 - ramp) + Q_EDGE_T * ramp)
        DILATE    = int(round(DILATE_T * ramp))

        # ========== Encoder priors ==========
        # Revision P0:
        # Explicit prior supervision updates only the two 1x1 prior heads.
        # detach() does not change forward values; it isolates the new loss
        # from the shared Pyramid Adapter.
        enc_sem_logit  = self.enc_q_proj_sem(enc_c2.detach())   # [B,1,1/8]
        enc_edge_logit = self.enc_q_proj_edge(enc_c1.detach())  # [B,1,1/4]
        p_enc_sem  = torch.sigmoid(enc_sem_logit)
        p_enc_edge = torch.sigmoid(enc_edge_logit)

        cons_sem_proxy_1_8 = (4.0 * p_enc_sem * (1.0 - p_enc_sem)).clamp(0, 1).detach()

        # ========== Stage-4 (1/32) ==========
        dec_c4_global = self.pyramid_pooling(enc_c4)   # [B,64,1/32]
        dec_c4_local  = self.rfb[1](enc_c4)            # [B,64,1/32]
        dec_c4_base = self.fusion[2](
            dec_c4_local,
            F.interpolate(dec_c4_global, size=dec_c4_local.shape[-2:], mode="bilinear", align_corners=True)
        )

        cons_sem_proxy_1_32 = F.interpolate(cons_sem_proxy_1_8, size=dec_c4_base.shape[-2:], mode="bilinear", align_corners=True)
        dec_c4 = self.tfgm["dec_c4"](dec_c4_base, cons_sem_proxy_1_32)


        # ========== Stage-3 (1/16) ==========
        dec_c3_local  = self.rfb[0](enc_c3)            # [B,64,1/16]  local
        dec_c3_global = self.pyramid_pooling(enc_c3)   # [B,64,1/16]  global (PPM)
        
        # local + global 用 GatedFFM 融合
        dec_c3_base = self.fusion[1](
            dec_c3_local,
            F.interpolate(dec_c3_global, size=dec_c3_local.shape[-2:], mode="bilinear", align_corners=True)
        )
        
        # 一致性引导 + TFGM 修正得到 D3
        cons_sem_proxy_1_16 = F.interpolate(cons_sem_proxy_1_8, size=dec_c3_base.shape[-2:], mode="bilinear", align_corners=True)
        dec_c3 = self.tfgm["dec_c3"](dec_c3_base, cons_sem_proxy_1_16)


        # ========== Stage-2 (1/8) ==========
        dec_c2_high = F.interpolate(
            dec_c4,
            size=[enc_c2_64.size(2) // 2, enc_c2_64.size(3) // 2],
            mode="bilinear", align_corners=True
        )  # [B,64,1/16]
        dec_c2_base = self.aggregation[1](dec_c2_high, enc_c2_64)  # [B,64,1/8]

        dec_sem_logit = self.head[2](dec_c2_base)     # [B,1,1/8]
        p_dec_sem = torch.sigmoid(dec_sem_logit)
        cons_sem_1_8 = (p_enc_sem - p_dec_sem).abs().clamp(0, 1).detach()

        dec_c2 = self.tfgm["dec_c2"](dec_c2_base, cons_sem_1_8)

        # ========== Stage-1 (1/4) ==========
        dec_c1_high = F.interpolate(
            dec_c3,
            size=[enc_c1_64.size(2) // 2, enc_c1_64.size(3) // 2],
            mode="bilinear", align_corners=True
        )  # [B,64,1/8]
        dec_c1_base = self.aggregation[0](dec_c1_high, enc_c1_64)  # [B,64,1/4]

        dec_edge_logit = self.head[1](dec_c1_base)     # [B,1,1/4]
        p_dec_edge = torch.sigmoid(dec_edge_logit)
        cons_edge_1_4 = (p_enc_edge - p_dec_edge).abs()

        def _grad_l1(p):
            gx = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs()
            gy = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs()
            gx = F.pad(gx, (0, 1, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1))
            return gx + gy

        if EDGE_BETA > 0:
            cons_edge_grad = (_grad_l1(p_enc_edge) - _grad_l1(p_dec_edge)).abs()
            cons_edge_1_4 = (cons_edge_1_4 + EDGE_BETA * cons_edge_grad)

        cons_edge_1_4 = cons_edge_1_4.clamp(0, 1).detach()
        cons_edge_1_4 = self._sparsify_cons(cons_edge_1_4, q=Q_EDGE, dilate=DILATE).detach()

        dec_c1 = self.tfgm["dec_c1"](dec_c1_base, cons_edge_1_4)

        # ========== Out (1/4) ==========
        dec_c2_up_ref = F.interpolate(dec_c2, size=dec_c1.shape[-2:], mode="bilinear", align_corners=True)
        dec_out_base = self.fusion[0](dec_c1, dec_c2_up_ref)  # [B,64,1/4]

        cons_sem_1_4 = F.interpolate(cons_sem_1_8, size=cons_edge_1_4.shape[-2:], mode="bilinear", align_corners=True)
        cons_out = (0.8 * cons_edge_1_4 + 0.2 * cons_sem_1_4).clamp(0, 1).detach()

        Q_OUT_T       = float(self._cfg_num("cons_sparsify_q_out", 0.82))  # dec_out 更松
        DILATE_OUT_T  = int(self._cfg_num("cons_out_dilate", 2))          # 更连贯

        Q_OUT       = (0.90 * (1 - ramp) + Q_OUT_T * ramp)
        DILATE_OUT  = int(round(DILATE_OUT_T * ramp))

        cons_out = self._sparsify_cons(cons_out, q=Q_OUT, dilate=DILATE_OUT).detach()
        dec_out = self.tfgm["dec_out"](dec_out_base, cons_out, record=True, tag="dec_out")

        # ========== 输出 ==========
        main_logit = F.interpolate(self.head[0](dec_out), size=shape, mode="bilinear", align_corners=False)
 
        # ========== aux_cache (for tensorboard/analysis) ==========
        enc_score_full = F.interpolate(enc_sem_logit, size=shape, mode="bilinear", align_corners=False)
        cons_full = F.interpolate(cons_out, size=shape, mode="bilinear", align_corners=False)

        # Attached prior logits are exposed only to the explicit weak
        # supervision loss. Existing CP discrepancy cues stay stop-gradient.
        enc_edge_logit_full = F.interpolate(
            enc_edge_logit,
            size=shape,
            mode="bilinear",
            align_corners=False,
        )

        self.aux_cache["enc_sem_logit_full_raw"] = enc_score_full
        self.aux_cache["enc_edge_logit_full_raw"] = enc_edge_logit_full

        self.aux_cache["enc_score_full"]  = enc_score_full.detach()
        self.aux_cache["main_logit_full"] = main_logit.detach()
        self.aux_cache["cons_full"]       = cons_full.detach()
        self.aux_cache["cons_sem_1_8"]    = cons_sem_1_8.detach()
        self.aux_cache["cons_edge_1_4"]   = cons_edge_1_4.detach()

        tfgm_debug = getattr(self.tfgm["dec_out"], "debug", None)
        if isinstance(tfgm_debug, dict):
            self.aux_cache["tfgm_freq_in"]  = tfgm_debug.get("x", None)
            self.aux_cache["tfgm_freq_out"] = tfgm_debug.get("x_freq", None)
            self.aux_cache["tfgm_g_sp"]     = tfgm_debug.get("g_sp", None)
            self.aux_cache["tfgm_g_ch"]     = tfgm_debug.get("g_ch", None)
        else:
            self.aux_cache["tfgm_freq_in"]  = None
            self.aux_cache["tfgm_freq_out"] = None
            self.aux_cache["tfgm_g_sp"]     = None
            self.aux_cache["tfgm_g_ch"]     = None

        mode = getattr(self.cfg, "mode", "train")
        if mode == "train":
            out_c1 = F.interpolate(self.head[1](dec_c1), size=shape, mode="bilinear", align_corners=False)
            out_c2 = F.interpolate(self.head[2](dec_c2), size=shape, mode="bilinear", align_corners=False)
            out_c3 = F.interpolate(self.head[3](dec_c3), size=shape, mode="bilinear", align_corners=False)
            out_c4 = F.interpolate(self.head[4](dec_c4), size=shape, mode="bilinear", align_corners=False)
            return main_logit, None, out_c1, out_c2, out_c3, out_c4
        else:
            return main_logit, None

    def initialize(self):
        print("initialize net (DINOv3 backbone)")
        if getattr(self.cfg, "snapshot", None):
            self.load_state_dict(torch.load(self.cfg.snapshot, map_location="cpu", weights_only=False), strict=False)
        else:
            weight_init(self)
