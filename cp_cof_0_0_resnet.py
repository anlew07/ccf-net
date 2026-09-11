# Bnet_res.py
# 单阶段 Decoder + 单尺度 cueistency + dec_c2/dec_c1 频域 CoF 稳定版


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



# ============================ FrequencyEnhance & CoF（简化稳版） ============================ #

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


class CoF(nn.Module):
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

    def _norm_cue(self, cue, x_ref):
        """把 cue 规整到 [B,1,H,W] 且 clamp 到 [0,1]"""
        B, C, H, W = x_ref.shape
        if cue is None:
            return x_ref.new_zeros(B, 1, H, W)

        if cue.dim() == 3:
            cue = cue.unsqueeze(1)  # [B,1,H,W]
        elif cue.dim() == 4:
            pass
        else:
            return x_ref.new_zeros(B, 1, H, W)

        if cue.size(0) != B:
            # 不匹配就直接置零，避免隐式广播坑
            return x_ref.new_zeros(B, 1, H, W)

        if cue.size(1) != 1:
            cue = cue.mean(dim=1, keepdim=True)

        if cue.shape[-2:] != (H, W):
            cue = F.interpolate(cue, size=(H, W), mode='bilinear', align_corners=True)

        return cue.clamp(0, 1)

    def forward(self, x, cue: torch.Tensor = None, record: bool = False, tag: str = None):
        B, C, H, W = x.shape
        x_freq = self.freq(x)

        cue = self._norm_cue(cue, x)

        g_sp_1 = self.spatial_gate(cue)          # [B,1,H,W]
        g_spatial = g_sp_1.expand(-1, C, -1, -1)  # [B,C,H,W]

        g_x = F.adaptive_avg_pool2d(x, 1).view(B, C)
        g_f = F.adaptive_avg_pool2d(x_freq, 1).view(B, C)
        g_cue = F.adaptive_avg_pool2d(cue, 1).view(B, 1)

        v = torch.cat([g_x, g_f, g_cue], dim=1)
        g_ch = torch.sigmoid(self.channel_mlp(v)).view(B, C, 1, 1)

        diff = (x_freq - x) * g_spatial * g_ch
        delta = self.refine(diff) * self.res_scale
        out = x + delta

        if record:
            self.debug = {
                "tag": tag,
                "x": x.detach(),
                "x_freq": x_freq.detach(),
                "cue": cue.detach(),
                "g_sp": g_sp_1.detach(),
                "g_ch": g_ch.detach(),
            }
        return out


# ============================  Backbone + Pyramid Adapter ============================ #
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3,
            stride=stride,
            padding=(3 * dilation - 1) // 2,
            bias=False,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)

    def initialize(self):
        weight_init(self)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(
            nn.Conv2d(
                self.inplanes, planes * 4,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(planes * 4)
        )
        layers = [Bottleneck(
            self.inplanes, planes,
            stride,
            downsample,
            dilation=dilation
        )]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(
                self.inplanes, planes,
                dilation=dilation
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)  # 1/4
        out2 = self.layer1(out1)  # 1/4
        out3 = self.layer2(out2)  # 1/8
        out4 = self.layer3(out3)  # 1/16
        out5 = self.layer4(out4)  # 1/32
        return out1, out2, out3, out4, out5

    def initialize(self):
        abs_path = "/root/shared-nvme/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations/assets/resnet50-19c8e357.pth"
        rel_path = "./assets/resnet50-19c8e357.pth"
        ckpt_path = abs_path if os.path.isfile(abs_path) else rel_path
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if (isinstance(raw, (dict, OrderedDict)) and
                    "state_dict" in raw and
                    isinstance(raw["state_dict"], (dict, OrderedDict))):
                sd = raw["state_dict"]
            elif isinstance(raw, (dict, OrderedDict)) and all(
                    isinstance(v, torch.Tensor) for v in raw.values()):
                sd = raw
            else:
                sd = None
                if isinstance(raw, (dict, OrderedDict)):
                    for _, v in raw.items():
                        if (isinstance(v, (dict, OrderedDict)) and
                                all(isinstance(t, torch.Tensor)
                                    for t in v.values())):
                            sd = v
                            break
                if sd is None:
                    print(f"[ResNet] Unrecognized checkpoint container: type={type(raw)}. Using random init.")
                    return
            model_sd = self.state_dict()
            filtered = {
                k: v for k, v in sd.items()
                if k in model_sd and model_sd[k].shape == v.shape
            }
            incompatible = self.load_state_dict(filtered, strict=False)
            print(
                f"[ResNet] pretrained loaded from {ckpt_path}: "
                f"matched={len(filtered)}, "
                f"missing_in_model={len([k for k in model_sd.keys() if k not in filtered])}, "
                f"missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
        except Exception as e:
            print(f"[ResNet] load pretrained failed: {e}. Using random init.")




class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg

        # ---- DINOv3 backbone ----
        self.bkbone = ResNet()

        self.pyramid_pooling = PyramidPooling(2048, 64)

        self.conv1 = nn.ModuleList([
            basicConv(64, 64, k=1, s=1, p=0),  # enc_c0_64 (unused)
            basicConv(256, 64, k=1, s=1, p=0),  # enc_c1_64, 1/4
            basicConv(512, 64, k=1, s=1, p=0),  # enc_c2_64, 1/8
            basicConv(1024, 64, k=1, s=1, p=0),  # reserved
            basicConv(2048, 64, k=1, s=1, p=0),  # reserved
        ])

        self.rfb = nn.ModuleList([
            RFB_modified(1024, 64),  # enc_c3 -> 1/16
            RFB_modified(2048, 64),  # enc_c4 -> 1/32
        ])
        
        # ===== 分层 CoF 强度 =====
        self.cof = nn.ModuleDict({
            "dec_c4":  CoF(64, res_scale=0.30),
            "dec_c3":  CoF(64, res_scale=0.35),
            "dec_c2":  CoF(64, res_scale=0.45),
            "dec_c1":  CoF(64, res_scale=0.60),
            "dec_out": CoF(64, res_scale=0.70),
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

        # ===== 双 cue 的 encoder prior =====
        # 语义/区域一致性：来自 enc_c2(1/8)
        self.enc_q_proj_sem  = basicConv(512, 1, k=1, s=1, p=0, bn=False, relu=False)
        # 边界/细节一致性：来自 enc_c1(1/4)
        self.enc_q_proj_edge = basicConv(256, 1, k=1, s=1, p=0, bn=False, relu=False)

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

    # ---------- cue 稀疏化（top-k 保留） ----------
    def _sparsify_cue(self, cue_1ch: torch.Tensor, q: float = 0.85, dilate: int = 1):
        """
        Top-quantile 稀疏化：保留 >= quantile(q) 的位置
        - q 越大越稀疏（0.95 很稀，0.80 更“松”）
        - dilate 让区域更连贯
        """
        if cue_1ch is None:
            return None
        c = cue_1ch

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

        warm_iters = float(self._cfg_num("cue_warmup_iters", 6000))
        ramp = min(1.0, it / max(1.0, warm_iters))

        EDGE_BETA_T = float(self._cfg_num("cue_edge_beta", 0.3))
        Q_EDGE_T    = float(self._cfg_num("cue_sparsify_q_edge", 0.85))
        DILATE_T    = int(self._cfg_num("cue_edge_dilate", 1))

        EDGE_BETA = EDGE_BETA_T * ramp
        Q_EDGE    = (0.95 * (1 - ramp) + Q_EDGE_T * ramp)
        DILATE    = int(round(DILATE_T * ramp))

        # ========== Encoder priors ==========
        enc_sem_logit  = self.enc_q_proj_sem(enc_c2)   # [B,1,1/8]
        enc_edge_logit = self.enc_q_proj_edge(enc_c1)  # [B,1,1/4]
        p_enc_sem  = torch.sigmoid(enc_sem_logit)
        p_enc_edge = torch.sigmoid(enc_edge_logit)

        cue_sem_proxy_1_8 = (4.0 * p_enc_sem * (1.0 - p_enc_sem)).clamp(0, 1).detach()

        # ========== Stage-4 (1/32) ==========
        dec_c4_global = self.pyramid_pooling(enc_c4)   # [B,64,1/32]
        dec_c4_local  = self.rfb[1](enc_c4)            # [B,64,1/32]
        dec_c4_base = self.fusion[2](
            dec_c4_local,
            F.interpolate(dec_c4_global, size=dec_c4_local.shape[-2:], mode="bilinear", align_corners=True)
        )

        cue_sem_proxy_1_32 = F.interpolate(cue_sem_proxy_1_8, size=dec_c4_base.shape[-2:], mode="bilinear", align_corners=True)
        dec_c4 = self.cof["dec_c4"](dec_c4_base, cue_sem_proxy_1_32)

        # ========== Stage-3 (1/16) ==========
        dec_c3_local = self.rfb[0](enc_c3)  # [B,64,1/16]
        dec_c4_up = F.interpolate(dec_c4, size=dec_c3_local.shape[-2:], mode="bilinear", align_corners=True)
        dec_c3_base = self.fusion[1](dec_c3_local, dec_c4_up)

        cue_sem_proxy_1_16 = F.interpolate(cue_sem_proxy_1_8, size=dec_c3_base.shape[-2:], mode="bilinear", align_corners=True)
        dec_c3 = self.cof["dec_c3"](dec_c3_base, cue_sem_proxy_1_16)

        # ========== Stage-2 (1/8) ==========
        dec_c2_high = F.interpolate(
            dec_c4,
            size=[enc_c2_64.size(2) // 2, enc_c2_64.size(3) // 2],
            mode="bilinear", align_corners=True
        )  # [B,64,1/16]
        dec_c2_base = self.aggregation[1](dec_c2_high, enc_c2_64)  # [B,64,1/8]

        dec_sem_logit = self.head[2](dec_c2_base)     # [B,1,1/8]
        p_dec_sem = torch.sigmoid(dec_sem_logit)

        cue_sem_1_8 = (p_enc_sem - p_dec_sem).abs().clamp(0, 1).detach()
        dec_c2 = self.cof["dec_c2"](dec_c2_base, cue_sem_1_8)

        # ========== Stage-1 (1/4) ==========
        dec_c1_high = F.interpolate(
            dec_c3,
            size=[enc_c1_64.size(2) // 2, enc_c1_64.size(3) // 2],
            mode="bilinear", align_corners=True
        )  # [B,64,1/8]
        dec_c1_base = self.aggregation[0](dec_c1_high, enc_c1_64)  # [B,64,1/4]

        dec_edge_logit = self.head[1](dec_c1_base)     # [B,1,1/4]
        p_dec_edge = torch.sigmoid(dec_edge_logit)
        cue_edge_1_4 = (p_enc_edge - p_dec_edge).abs()

        def _grad_l1(p):
            gx = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs()
            gy = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs()
            gx = F.pad(gx, (0, 1, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1))
            return gx + gy

        if EDGE_BETA > 0:
            cue_edge_grad = (_grad_l1(p_enc_edge) - _grad_l1(p_dec_edge)).abs()
            cue_edge_1_4 = (cue_edge_1_4 + EDGE_BETA * cue_edge_grad)

        cue_edge_1_4 = cue_edge_1_4.clamp(0, 1).detach()
        cue_edge_1_4 = self._sparsify_cue(cue_edge_1_4, q=Q_EDGE, dilate=DILATE).detach()
        dec_c1 = self.cof["dec_c1"](dec_c1_base, cue_edge_1_4)
        # ========== Out (1/4) ==========
        dec_c2_up_ref = F.interpolate(dec_c2, size=dec_c1.shape[-2:], mode="bilinear", align_corners=True)
        dec_out_base = self.fusion[0](dec_c1, dec_c2_up_ref)  # [B,64,1/4]

        cue_sem_1_4 = F.interpolate(cue_sem_1_8, size=cue_edge_1_4.shape[-2:], mode="bilinear", align_corners=True)
        cue_out = (0.8 * cue_edge_1_4 + 0.2 * cue_sem_1_4).clamp(0, 1).detach()

        Q_OUT_T       = float(self._cfg_num("cue_sparsify_q_out", 0.82))  # dec_out 更松
        DILATE_OUT_T  = int(self._cfg_num("cue_out_dilate", 2))          # 更连贯

        Q_OUT       = (0.90 * (1 - ramp) + Q_OUT_T * ramp)
        DILATE_OUT  = int(round(DILATE_OUT_T * ramp))

        cue_out = self._sparsify_cue(cue_out, q=Q_OUT, dilate=DILATE_OUT).detach()
        dec_out = self.cof["dec_out"](dec_out_base, cue_out, record=True, tag="dec_out")
        # ========== 输出 ==========
        main_logit = F.interpolate(self.head[0](dec_out), size=shape, mode="bilinear", align_corners=False)
 
        # ========== aux_cache (for tensorboard/analysis) ==========
        enc_score_full = F.interpolate(enc_sem_logit, size=shape, mode="bilinear", align_corners=False)
        cue_full = F.interpolate(cue_out, size=shape, mode="bilinear", align_corners=False)

        self.aux_cache["enc_score_full"]  = enc_score_full.detach()
        self.aux_cache["main_logit_full"] = main_logit.detach()
        self.aux_cache["cue_full"]       = cue_full.detach()
        self.aux_cache["cue_sem_1_8"]    = cue_sem_1_8.detach()
        self.aux_cache["cue_edge_1_4"]   = cue_edge_1_4.detach()

        cof_debug = getattr(self.cof["dec_out"], "debug", None)
        if isinstance(cof_debug, dict):
            self.aux_cache["cof_freq_in"]  = cof_debug.get("x", None)
            self.aux_cache["cof_freq_out"] = cof_debug.get("x_freq", None)
            self.aux_cache["cof_g_sp"]     = cof_debug.get("g_sp", None)
            self.aux_cache["cof_g_ch"]     = cof_debug.get("g_ch", None)
        else:
            self.aux_cache["cof_freq_in"]  = None
            self.aux_cache["cof_freq_out"] = None
            self.aux_cache["cof_g_sp"]     = None
            self.aux_cache["cof_g_ch"]     = None

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
        print("initialize net (res backbone)")

        # 1) 如果有 snapshot：直接 load（通常包含 backbone + decoder 全部权重）
        if getattr(self.cfg, "snapshot", None):
            self.load_state_dict(
                torch.load(self.cfg.snapshot, map_location="cpu", weights_only=False),
                strict=False
            )
            return

        # 2) 没 snapshot：先加载 ResNet50 预训练（只作用在 backbone）
        if hasattr(self, "bkbone") and hasattr(self.bkbone, "initialize"):
            self.bkbone.initialize()
        else:
            print("[Warn] bkbone has no initialize(), skip pretrained loading.")

        # 3) 初始化 decoder/heads 等非 backbone 部分（避免把 backbone 重新 kaiming 掉）
        def _init_except_backbone(m):
            for name, child in m.named_children():
                if name == "bkbone":
                    continue
                weight_init(child)

        _init_except_backbone(self)
