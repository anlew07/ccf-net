# train_processes_loss.py

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from feature_loss import FeatureLoss
from tools import *
from utils import ramps

# ================== 基本模块 & 超参数 ================== #
criterion = torch.nn.CrossEntropyLoss(weight=None,
                                      ignore_index=255,
                                      reduction='mean').cuda()

loss_lsc = FeatureLoss().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5

# LSC 权重（与原版保持一致）
l = 0.3

# 原 q 在训练中实际数值大约为 0.002 左右，这里直接当作常量使用
Q_CONST = 0.002
# 原来深监督 aux_scale = 0.3 + 0.7 * q_mean，这里固定下来
AUX_LOSS_SCALE = 0.3 + 0.7 * Q_CONST  # ≈ 0.3014

# ================== 原作者工具：一致性权重 & 变换 ================== #
def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)


def get_transform(ops=[0, 1, 2]):
    """One of flip, translate, crop"""
    op = np.random.choice(ops)
    if op == 0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op == 1:
        pp = Translate(0.15)
    elif op == 2:
        pp = Crop(0.7, 0.7)
    return pp


def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0, 1), padding=(1, 1), groups=c)
    return c1, b


def unsymmetric_grad(x, y, calc, w1, w2):
    return calc(x, y.detach()) * w1 + calc(x.detach(), y) * w2


# ================== feature_loss（已修复 kernel 掩码 bug） ================== #
def feature_loss(feature_map, pred, kr=4, norm=False,
                 crtl_loss=True, w_ftp=0, topk=16, step_ratio=2):
    """
    feature loss 的完整实现，仅修复了 kernel 的掩码索引错误。
    feature_map: [N,C,H,W]
    pred: [N,1,h,w]
    """
    if norm:
        feature_map = feature_map / feature_map.std(dim=(-1, -2),
                                                    keepdim=True).mean(dim=1, keepdim=True)

    fmap = feature_map
    n, c, h, w = fmap.shape
    ks = 2 * kr
    assert h % ks == 0 and w % ks == 0

    # unfold 成 patch
    uf = lambda x: F.unfold(x, ks, padding=0, stride=ks // step_ratio) \
                     .permute(0, 2, 1).reshape(-1, x.shape[1], ks * ks)
    fcmap = uf(fmap)   # [Npatch, C, ks*ks]
    fcpred = uf(pred)  # [Npatch, 1, ks*ks]

    cfd_thres = .8
    exst = lambda x: (x > cfd_thres).sum(2, keepdim=True) > 0.3 * ks * ks
    coexists = (exst(fcpred) & exst(1 - fcpred))  # [Npatch,1,1]
    coexists = coexists[:, 0, 0]
    fcmap = fcmap[coexists]
    fcpred = fcpred[coexists]

    if not len(fcmap):
        return 0, 0

    mfcmap = fcmap - fcmap.mean(2, keepdim=True)
    mfcpred = fcpred - fcpred.mean(2, keepdim=True)
    cov = mfcmap.matmul(mfcpred.permute(0, 2, 1))
    sgnf_id = cov.abs().topk(topk, dim=1)[1].expand(-1, -1, ks * ks)
    sg_fcmap = fcmap.gather(dim=1, index=sgnf_id)

    device = pred.device
    xy = torch.stack(torch.meshgrid(
        torch.arange(ks, device=device),
        torch.arange(ks, device=device),
        indexing='ij'
    )) / 6.0
    xy = xy.reshape(1, 2, ks * ks).expand(len(sg_fcmap), -1, -1)

    def crf_k(x):
        # x: [N, d, ks*ks] -> [N,1,ks*ks,ks*ks]
        return (-(x[:, :, None] - x[:, :, :, None]) ** 2 * 0.5).sum(1, keepdim=True).exp()

    pred_grvt = lambda x, y: (1 - x) * y + x * (1 - y)
    ft_grvt = lambda x: 1 - crf_k(x)

    ffxy = crf_k(xy)  # [N,1,ks*ks,ks*ks]

    if crtl_loss:
        pmap = fcpred.detach()
        pmap = 0.5 - pred_grvt(pmap.unsqueeze(2), pmap.unsqueeze(-1))
        fpmap = ft_grvt(sg_fcmap) * ffxy
        ice = (pmap * fpmap).mean()
        fffm = crf_k(sg_fcmap.detach())
        kernel = fffm * ffxy   # [N,1,ks*ks,ks*ks]
    else:
        ice = 0
        fffm = crf_k(sg_fcmap)
        kernel = fffm * ffxy
        # ★ 修复点：使用 4D 掩码清零对角线，而不是 2D mask
        eye = torch.eye(ks * ks, device=device,
                        dtype=bool).view(1, 1, ks * ks, ks * ks)
        kernel = kernel.masked_fill(eye, 0.0)

    pp = pred_grvt(fcpred[:, :, None], fcpred.unsqueeze(-1))  # [N,1,ks*ks,ks*ks]

    if w_ftp == 0:
        crf = (kernel * pp).mean()
    elif w_ftp == 1:
        crf = (kernel.detach() * pp).mean() * (1 + w_ftp)
    else:
        crf = unsymmetric_grad(kernel, pp, lambda x, y: (x * y).mean(),
                               1 - w_ftp, 1 + w_ftp)

    return crf, ice


# ================== 可视化工具：tensor → png ================== #
def _save_single_tensor(tensor, path, is_feature=False):
    """
    把 [B,C,H,W] tensor 中第一个样本保存成 png。
    - is_feature=True 时会先在通道维上取 L2 范数，展示整体能量。
    """
    if tensor is None:
        return
    try:
        with torch.no_grad():
            t = tensor.detach().cpu()
            if t.dim() != 4 or t.size(0) == 0:
                return
            t = t[0:1]  # 只取第一个样本
            if is_feature and t.size(1) > 1:
                t = t.norm(p=2, dim=1, keepdim=True)

            minv = float(t.min())
            maxv = float(t.max())
            if maxv - minv > 1e-5:
                t = (t - minv) / (maxv - minv)
            else:
                t = t * 0.0
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(t, path)
    except Exception as e:
        print(f"[vis] save_single_tensor failed at {path}: {e}")


def _save_fft_magnitude(tensor, path):
    """
    在特征图上做 rFFT2，取幅度的 log，再保存成灰度图。
    用于展示 TFGM 前后频谱的变化。
    """
    if tensor is None:
        return
    try:
        with torch.no_grad():
            t = tensor.detach().cpu()
            if t.dim() != 4 or t.size(0) == 0:
                return
            t = t[0:1]  # 取第一个样本
            # rFFT2：只取一半频率，但足够展示结构
            X = torch.fft.rfft2(t, norm='ortho')
            mag = torch.abs(X).mean(dim=1, keepdim=True)  # [1,1,H,Wf]
            mag = torch.log(mag + 1e-6)
            minv = float(mag.min())
            maxv = float(mag.max())
            if maxv - minv > 1e-5:
                mag = (mag - minv) / (maxv - minv)
            else:
                mag = mag * 0.0
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(mag, path)
    except Exception as e:
        print(f"[vis] save_fft_magnitude failed at {path}: {e}")


def dump_intermediate_visuals(image, mask,
                              out2, out3, out4, out5, out6,
                              net, ctx):
    if ctx is None:
        return

    save_dir = ctx.get('save_dir', None)
    epoch     = ctx.get('epoch', None)
    global_step = ctx.get('global_step', None)
    if save_dir is None or epoch is None or global_step is None:
        return

    # ===== 只在指定 epoch（10,100,150）触发 =====
    target_epochs = {10, 100, 150}
    if epoch not in target_epochs:
        return

    # ===== 保证每个目标 epoch 只保存一次 =====
    if not hasattr(net, "_vis_logged_epochs"):
        net._vis_logged_epochs = set()
    if epoch in net._vis_logged_epochs:
        return
    net._vis_logged_epochs.add(epoch)

    vis_root = os.path.join(save_dir,
                            f"vis_epoch_{int(epoch):03d}_step_{int(global_step):06d}")

    # 输入图像 & scribble mask
    _save_single_tensor(image, os.path.join(vis_root, "input_rgb.png"),
                        is_feature=False)
    _save_single_tensor(mask.float(), os.path.join(vis_root, "scribble_mask.png"),
                        is_feature=False)

    # 主预测 & 各尺度预测（使用前景通道 prob）
    try:
        _save_single_tensor(out2[:, 1:2], os.path.join(vis_root, "pred_main_dec_out.png"),
                            is_feature=False)
        _save_single_tensor(out3[:, 1:2], os.path.join(vis_root, "pred_dec_c1.png"),
                            is_feature=False)
        _save_single_tensor(out4[:, 1:2], os.path.join(vis_root, "pred_dec_c2.png"),
                            is_feature=False)
        _save_single_tensor(out5[:, 1:2], os.path.join(vis_root, "pred_dec_c3.png"),
                            is_feature=False)
        _save_single_tensor(out6[:, 1:2], os.path.join(vis_root, "pred_dec_c4.png"),
                            is_feature=False)
    except Exception as e:
        print(f"[vis] save prediction maps failed: {e}")

    # encoder 结构响应 & encoder-decoder 一致性 cons_full
    aux_cache = getattr(net, "aux_cache", {})
    enc_score_full = aux_cache.get("enc_score_full", None)
    cons_full = aux_cache.get("cons_full", None)

    if enc_score_full is not None:
        _save_single_tensor(torch.sigmoid(enc_score_full),
                            os.path.join(vis_root, "enc_struct_score.png"),
                            is_feature=False)
    if cons_full is not None:
        _save_single_tensor(cons_full,
                            os.path.join(vis_root, "cons_full_encoder_decoder.png"),
                            is_feature=False)

    # TFGM 在 dec_c1 尺度的频域前后（空间特征 + 频谱）
    freq_in = aux_cache.get("tfgm_freq_in", None)
    freq_out = aux_cache.get("tfgm_freq_out", None)
    _save_single_tensor(freq_in, os.path.join(vis_root, "tfgm_dec_c1_in_feat.png"),
                        is_feature=True)
    _save_single_tensor(freq_out, os.path.join(vis_root, "tfgm_dec_c1_out_feat.png"),
                        is_feature=True)
    _save_fft_magnitude(freq_in, os.path.join(vis_root, "tfgm_dec_c1_in_fft.png"))
    _save_fft_magnitude(freq_out, os.path.join(vis_root, "tfgm_dec_c1_out_fft.png"))


def _tb_norm01(x: torch.Tensor):
    x = x.detach()
    if x.dim() == 4:
        x = x[0]  # [C,H,W]
    elif x.dim() == 3:
        pass
    elif x.dim() == 2:
        x = x.unsqueeze(0)
    else:
        return None

    # feature -> 取通道 L2 能量显示
    if x.dim() == 3 and x.size(0) > 1:
        x = x.norm(p=2, dim=0, keepdim=True)

    mn = x.min()
    mx = x.max()
    if (mx - mn) > 1e-6:
        x = (x - mn) / (mx - mn)
    else:
        x = x * 0.0
    return x.clamp(0, 1)

def _tb_add_scalars(sw, step: int, prefix: str, d: dict):
    if sw is None:
        return
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                sw.add_scalar(f"{prefix}/{k}", float(v.item()), step)
        else:
            try:
                sw.add_scalar(f"{prefix}/{k}", float(v), step)
            except:
                pass

def _tb_add_image(sw, step: int, tag: str, x: torch.Tensor):
    if sw is None or x is None:
        return
    xx = _tb_norm01(x)
    if xx is None:
        return
    sw.add_image(tag, xx, step)

def _scribble_metrics(prob_fg: torch.Tensor, mask: torch.Tensor):
    """
    prob_fg: [B,1,H,W] in [0,1]
    mask   : [B,1,H,W] 0/1/255
    """
    gt = mask.squeeze(1)
    p = prob_fg.squeeze(1).detach()

    valid = (gt != 255)
    fg = (gt == 1) & valid
    bg = (gt == 0) & valid

    pred = (p > 0.5)

    def _acc(m):
        if m.sum() == 0:
            return None
        return (pred[m] == (gt[m] == 1)).float().mean()

    acc_all = _acc(valid)
    acc_fg = _acc(fg)
    acc_bg = _acc(bg)

    p_fg_on_fg = p[fg].mean() if fg.sum() > 0 else None
    p_fg_on_bg = p[bg].mean() if bg.sum() > 0 else None
    margin = (p_fg_on_fg - p_fg_on_bg) if (p_fg_on_fg is not None and p_fg_on_bg is not None) else None

    return {
        "acc_all": acc_all,
        "acc_fg": acc_fg,
        "acc_bg": acc_bg,
        "p_fg_on_fg": p_fg_on_fg,
        "p_fg_on_bg": p_fg_on_bg,
        "margin": margin,
    }

def _cons_metrics(cons: torch.Tensor):
    if cons is None:
        return {}
    c = cons.detach()
    flat = c.flatten()
    return {
        "mean": c.mean(),
        "max": c.max(),
        "p95": torch.quantile(flat, 0.95),
        "ratio_gt_0.2": (c > 0.2).float().mean(),
        "ratio_gt_0.4": (c > 0.4).float().mean(),
    }

def _tfgm_metrics(aux_cache: dict):
    x = aux_cache.get("tfgm_freq_in", None)
    xf = aux_cache.get("tfgm_freq_out", None)
    gsp = aux_cache.get("tfgm_g_sp", None)
    gch = aux_cache.get("tfgm_g_ch", None)

    out = {}
    if gsp is not None:
        out["sp_gate_mean"] = gsp.mean()
        out["sp_gate_max"] = gsp.max()
    if gch is not None:
        out["ch_gate_mean"] = gch.mean()
        out["ch_gate_max"] = gch.max()

    if x is not None and xf is not None:
        delta = (xf.detach() - x.detach()).abs().mean()
        base = x.detach().abs().mean().clamp_min(1e-6)
        out["delta_mean_abs"] = delta
        out["delta_ratio"] = delta / base
    return out


# ================== 主 train_loss（已移除 q / 频域 loss） ================== #
def train_loss(image, mask, net, ctx,
               w_ft=.1, ft_st=60, ft_fct=.5, ft_dct=None, ft_head=True,
               mtrsf_prob=1, ops=[0, 1, 2], w_l2g=0,
               l_me=0.1, me_st=50, me_all=False, multi_sc=0,
               l=0.3, sl=1):

    if ft_dct is None:
        ft_dct = dict(crtl_loss=False, w_ftp=0, norm=False, topk=16, step_ratio=2)

    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo']
    else:
        epoch = 1
        global_step = 0
        sw = None
        t_epo = 1

    # ---------- feature loss hook（原版） ----------
    fm = []

    def hook(m, i, o):
        if not ft_head:
            fm.extend(get_featuremap(m, i[0]))
        else:
            fm.append(net.feature_head[0](i[0]))

    hh = net.head[0].register_forward_hook(hook)

    # ---------- Saliency Structure Consistency ---------- #
    do_moretrsf = np.random.uniform() < mtrsf_prob
    if do_moretrsf:
        pre_transform = get_transform(ops)
        image_tr = pre_transform(image)
        large_scale = True
    else:
        large_scale = np.random.uniform() < multi_sc
        image_tr = image
    sc_fct = 0.6 if large_scale else 0.3
    image_scale = F.interpolate(image_tr, scale_factor=sc_fct,
                                mode='bilinear', align_corners=True)

    # 注意：这里保留 logit 版本（单通道）
    logit_out2, _, logit_out3, logit_out4, logit_out5, logit_out6 = net(image)
    aux_cache_main = dict(getattr(net, "aux_cache", {}))  # ✅ 关键：先拷贝主尺度缓存
    hh.remove()

    logit_out2_s, _, logit_out3_s, logit_out4_s, logit_out5_s, logit_out6_s = net(image_scale)

    # ---------- intra-consistency (entropy) 原版 ---------- #
    loss_intra = []
    if epoch >= me_st:
        def entrp(t):
            etp = -(F.softmax(t, dim=1) * F.log_softmax(t, dim=1)).sum(dim=1)
            msk = (etp < 0.5)
            return (etp * msk).sum() / (msk.sum() or 1)

        me_fun = lambda x: entrp(torch.cat((x * 0, x), 1))
        if not me_all:
            e = me_fun(logit_out2)
            loss_intra.append(
                e * get_current_consistency_weight(epoch - me_st,
                                                   consistency=l_me,
                                                   consistency_rampup=t_epo - me_st)
            )
            loss_intra = loss_intra + [0, 0, 0, 0]
            if sw is not None:
                sw.add_scalar('intra entropy', e.item(), global_step)
        else:
            ga = get_current_consistency_weight(epoch - me_st,
                                                consistency=l_me,
                                                consistency_rampup=t_epo - me_st)
            for i in [logit_out2, logit_out3, logit_out4, logit_out5, logit_out6]:
                loss_intra.append(me_fun(i) * ga)
            if sw is not None:
                sw.add_scalar('intra entropy', loss_intra[0].item(), global_step)
    else:
        loss_intra.extend([0 for _ in range(5)])

    # ---------- 单通道 logit -> 2 通道 prob ---------- #
    def out_proc(o2, o3, o4, o5, o6):
        a = [o2, o3, o4, o5, o6]
        a = [i.sigmoid() for i in a]
        a = [torch.cat((1 - i, i), 1) for i in a]
        return a

    out2, out3, out4, out5, out6 = out_proc(logit_out2, logit_out3,
                                            logit_out4, logit_out5, logit_out6)
    out2_s, out3_s, out4_s, out5_s, out6_s = out_proc(logit_out2_s, logit_out3_s,
                                                      logit_out4_s, logit_out5_s, logit_out6_s)

    # ---------- A1: 结构一致性损失（SSC） ---------- #
    if not do_moretrsf:
        out2_scale = F.interpolate(out2[:, 1:2], scale_factor=sc_fct,
                                   mode='bilinear', align_corners=True)
        out2_s_aligned = out2_s[:, 1:2]
    else:
        out2_ss = pre_transform(out2)
        out2_scale = F.interpolate(out2_ss[:, 1:2], scale_factor=0.3,
                                   mode='bilinear', align_corners=True)
        out2_s_aligned = F.interpolate(out2_s[:, 1:2],
                                       scale_factor=0.3 / sc_fct,
                                       mode='bilinear', align_corners=True)

    loss_ssc_main = (SaliencyStructureConsistency(out2_s_aligned,
                                                  out2_scale.detach(), 0.85) * (w_l2g + 1)
                     + SaliencyStructureConsistency(out2_s_aligned.detach(),
                                                    out2_scale, 0.85) * (1 - w_l2g)) if sl else 0.0

    # ---------- C1: 涂鸦标签 partial CE（像素监督） ---------- #
    gt = mask.squeeze(1).long()
    bg_label = gt.clone()
    fg_label = gt.clone()
    bg_label[gt != 0] = 255
    fg_label[gt == 0] = 255

    # ---------- B1: feature loss ---------- #
    if epoch >= ft_st:
        wl = get_current_consistency_weight(epoch - ft_st, w_ft, t_epo - ft_st)
        ft_map = fm[0]
        pred_s = out2[:, 1:2].clone()
        pred_s[:, 0][gt != 255] = gt[gt != 255].float()
        pred_s = F.interpolate(pred_s, scale_factor=ft_fct,
                               mode='bilinear', align_corners=False)

        ft_map = F.interpolate(ft_map, out2.shape[-2:], mode='bilinear', align_corners=False)
        ft_map = F.interpolate(ft_map, pred_s.shape[-2:], mode='bilinear', align_corners=False)

        fl, crtl = feature_loss(ft_map, pred_s, **ft_dct)
        if sw is not None:
            sw.add_scalar('ft_loss',
                          fl.item() if isinstance(fl, torch.Tensor) else fl,
                          global_step=global_step)
            sw.add_scalar('fthead_loss',
                          crtl.item() if isinstance(crtl, torch.Tensor) else crtl,
                          global_step=global_step)
        loss_feature_main = fl * wl + crtl
    else:
        loss_feature_main = 0.0

    # ---------- A2: LSC 多尺度结构正则 ---------- #
    image_ = F.interpolate(image, scale_factor=0.25,
                           mode='bilinear', align_corners=True)
    sample = {'rgb': image_}

    def lsc_head(out):
        o_ = F.interpolate(out[:, 1:2], scale_factor=0.25,
                           mode='bilinear', align_corners=True)
        return loss_lsc(o_, loss_lsc_kernels_desc_defaults, loss_lsc_radius,
                        sample, image_.shape[2], image_.shape[3])['loss']

    loss2_lsc = lsc_head(out2)
    loss3_lsc = lsc_head(out3)
    loss4_lsc = lsc_head(out4)
    loss5_lsc = lsc_head(out5)
    loss6_lsc = lsc_head(out6)

    # ---------- 汇总到具体尺度 ----------
    loss_struct2 = loss_ssc_main + loss_feature_main + l * loss2_lsc
    loss_struct3 = l * loss3_lsc
    loss_struct4 = l * loss4_lsc
    loss_struct5 = l * loss5_lsc
    loss_struct6 = l * loss6_lsc

    reg2, reg3, reg4, reg5, reg6 = loss_intra[0], loss_intra[1], loss_intra[2], loss_intra[3], loss_intra[4]

    ce2 = criterion(out2, fg_label) + criterion(out2, bg_label)
    ce3 = criterion(out3, fg_label) + criterion(out3, bg_label)
    ce4 = criterion(out4, fg_label) + criterion(out4, bg_label)
    ce5 = criterion(out5, fg_label) + criterion(out5, bg_label)
    ce6 = criterion(out6, fg_label) + criterion(out6, bg_label)

    loss2 = loss_struct2 + ce2 + reg2
    loss3 = (loss_struct3 + ce3 + reg3) * AUX_LOSS_SCALE
    loss4 = (loss_struct4 + ce4 + reg4) * AUX_LOSS_SCALE
    loss5 = (loss_struct5 + ce5 + reg5) * AUX_LOSS_SCALE
    loss6 = (loss_struct6 + ce6 + reg6) * AUX_LOSS_SCALE

    # ---------- 中间产物可视化（你原来的 png 保存逻辑保留） ----------
    try:
        dump_intermediate_visuals(image, mask,
                                  out2, out3, out4, out5, out6,
                                  net, ctx)
    except Exception as e:
        print(f"[vis] dump_intermediate_visuals failed: {e}")

    # ================== TensorBoard 指标（新增） ==================
    if sw is not None:
        # 1) 总 loss
        _tb_add_scalars(sw, global_step, "loss", {
            "main_total": loss2,
            "aux_c1_total": loss3,
            "aux_c2_total": loss4,
            "aux_c3_total": loss5,
            "aux_c4_total": loss6,
        })

        # 2) main 分解（最关键）
        _tb_add_scalars(sw, global_step, "loss", {
            "main_ssc": loss_ssc_main,
            "main_feature": loss_feature_main,
            "main_lsc": l * loss2_lsc,
            "main_ce": ce2,
            "main_entropy_reg": reg2 if isinstance(reg2, torch.Tensor) else torch.tensor(reg2).to(image.device),
        })

        # 3) 各尺度 CE / LSC（定位哪一层拖后腿）
        _tb_add_scalars(sw, global_step, "ce", {"main": ce2, "c1": ce3, "c2": ce4, "c3": ce5, "c4": ce6})
        _tb_add_scalars(sw, global_step, "lsc", {"main": loss2_lsc, "c1": loss3_lsc, "c2": loss4_lsc, "c3": loss5_lsc, "c4": loss6_lsc})

        # 4) cons 指标（你的 enc-dec 一致性核心）
        cons_map = aux_cache_main.get("cons_full", None)
        _tb_add_scalars(sw, global_step, "cons", _cons_metrics(cons_map))

        # 5) scribble 监督质量（弱监督最实用）
        scrib = _scribble_metrics(out2[:, 1:2], mask)
        _tb_add_scalars(sw, global_step, "scribble", scrib)

        # 6) TFGM 行为（是否真的在“修”）
        _tb_add_scalars(sw, global_step, "tfgm", _tfgm_metrics(aux_cache_main))

        # 7) 图片（别太频繁，避免拖慢）
        img_interval = 200
        if (global_step % img_interval) == 0:
            _tb_add_image(sw, global_step, "img/input", image)
            _tb_add_image(sw, global_step, "img/pred_main", out2[:, 1:2])
            if aux_cache_main.get("enc_score_full", None) is not None:
                _tb_add_image(sw, global_step, "img/enc_score", torch.sigmoid(aux_cache_main["enc_score_full"]))
            if cons_map is not None:
                _tb_add_image(sw, global_step, "img/cons", cons_map)
            if aux_cache_main.get("tfgm_g_sp", None) is not None:
                _tb_add_image(sw, global_step, "img/tfgm_sp_gate", aux_cache_main["tfgm_g_sp"])

    return loss2, loss3, loss4, loss5, loss6
