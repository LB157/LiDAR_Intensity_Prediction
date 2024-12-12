import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_optim_from_kwargs(params, name='', **kwargs):
    optim = getattr(torch.optim, name, None)
    if optim is None:
        raise RuntimeError(f'Optimizer {name} was not found!')
    return optim(params, **kwargs)  # pylint: disable=not-callable


def create_loss_from_kwargs(reflect=False, gamma=2, l2_weight=0.5, ignore_index=4, only_l2=False):
    if reflect:
        if only_l2:

            def fn(output, mask=None, labels=None, mean=True, **kwargs):
                intensity = kwargs['intensity']
                rgb_mask = kwargs.get('rgb_mask', None)
                if rgb_mask is None:
                    rgb_mask = torch.ones_like(intensity)
                if mask is None:
                    mask = torch.ones_like(intensity)
                rgb_mask = rgb_mask >= 0
                mask = mask >= 0
                if labels is not None:
                    label_mask = ~(labels == ignore_index)[:, None, ...]
                else:
                    label_mask = torch.ones_like(mask)
                full_mask = rgb_mask & mask & label_mask
                *_, pred = output
                loss = F.mse_loss(pred, intensity, reduction='none')[full_mask]
                if mean:
                    return torch.mean(loss)
                return loss

        else:

            def fn(output, mask=None, labels=None, mean=True, **kwargs):
                """
                计算损失函数的函数。

                参数:
                - output: 模型的输出，包含预测的二进制分类结果和距离分布。
                - mask: 掩码，用于指示哪些位置需要计算损失。
                - labels: 标签，用于生成标签掩码。
                - mean: 是否对损失取平均值，默认为 True。
                - **kwargs: 其他关键字参数，包含 intensity_bin, intensity_dist, rgb_mask 等。

                返回:
                - 如果 mean=True，返回损失的平均值；否则返回损失张量。
                """
                # 从 kwargs 中获取 intensity_bin 和 intensity_dist
                intensity_bin = kwargs['intensity_bin']
                intensity_dist = kwargs['intensity_dist']
                 # 从 kwargs 中获取 rgb_mask，如果没有提供则创建一个与 intensity_dist 形状相同的全 1 张量
                rgb_mask = kwargs.get('rgb_mask', None)
                 # 如果没有提供 mask，则创建一个与 intensity_dist 形状相同的全 1 张量
                if rgb_mask is None:
                    rgb_mask = torch.ones_like(intensity_dist)
                if mask is None:
                    mask = torch.ones_like(intensity_dist)
                # 将 rgb_mask 和 mask 转换为布尔张量，表示是否需要计算损失
                rgb_mask = rgb_mask >= 0
                mask = mask >= 0
                # 如果提供了 labels，则生成标签掩码，忽略 ignore_index 对应的标签
                if labels is not None:
                    label_mask = ~(labels == ignore_index)[:, None, ...]
                else:
                     # 如果没有提供 labels，则创建一个与 mask 形状相同的全 1 张量
                    label_mask = torch.ones_like(mask)
                 # pred_bin(N,1)。  pred_dist:(N,1) 从 output 中提取预测的二进制分类结果和距离分布
                pred_bin, pred_dist, *_ = output
                # pred_prob (N,C) 对预测的二进制分类结果进行 softmax 操作，得到预测的概率分布
                pred_prob = F.softmax(pred_bin, 1)
                # ce_loss (N,1) 计算交叉熵损失，使用 reduction='none' 表示不进行降维
                ce_loss = F.cross_entropy(pred_bin, intensity_bin, reduction='none')[:, None, ...]
                  # weight (N,1) ；计算权重，权重为 (1 - 预测概率) 的 gamma 次方
                weight = (1 - pred_prob.gather(1, intensity_bin[:, None, ...])) ** gamma
                # l2_loss (N,1) 计算均方误差损失，使用 reduction='none' 表示不进行降维
                l2_loss = F.mse_loss(pred_dist, intensity_dist, reduction='none')
                # 计算总的损失，权重乘以交叉熵损失加上均方误差损失，并应用 rgb_mask, mask 和 label_mask (N)
                loss = (weight * ce_loss + l2_loss * l2_weight)[rgb_mask & mask & label_mask]
                 # 如果 mean=True，返回损失的平均值；否则返回损失张量
                if mean:
                    return torch.mean(loss)
                return loss

    else:
        weight = np.array([0.132528185, 808.046146, 3.53494246, 188.286384], dtype='f4')

        def fn(output, mask=None, labels=None, mean=True, **kwargs):  # pylint: disable=unused-argument
            if mask is None:
                mask = torch.ones_like(output)
            mask = mask >= 0
            tweight = torch.from_numpy(weight).to(output.device)
            b, _, h, w = output.shape
            output_prob = F.softmax(output, 1)
            output_prob = torch.cat((output_prob, torch.ones(b, 1, h, w, device=output_prob.device, dtype=output_prob.dtype)), 1)
            result = F.cross_entropy(output, labels, reduction='none', weight=tweight, ignore_index=ignore_index)[:, None, ...]
            loss_weight = (1 - output_prob.gather(1, labels[:, None, ...])) ** gamma
            loss = (loss_weight * result)[mask]
            if mean:
                return torch.mean(loss)
            return loss

    return fn


def create_image_fn(reflect, ignore_index=4):
    def renumpy(tensor, i, trans=True):
        vec = tensor[i].cpu().numpy()
        if trans:
            vec = np.transpose(vec, (1, 2, 0))
        return vec

    COLORS = np.array([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)], dtype='u1')
    NONE = np.array((128, 128, 128), dtype='u1')
    OK_COLORS = np.array([(0, 255, 0), (255, 0, 0)], dtype='u1')
    BORDER = 5
    if reflect:

        def fn(batch, output, i):
            *_, pred_value = output
            mask = np.squeeze(renumpy(batch['mask'], i) >= 0)
            try:
                rgb_mask = np.squeeze(renumpy(batch['rgb_mask'], i) >= 0)
            except KeyError:
                rgb_mask = np.ones_like(mask)
            try:
                label_mask = ~(renumpy(batch['labels'], i, False) == ignore_index)
            except KeyError:
                label_mask = np.ones_like(mask)
            mask = mask & rgb_mask & label_mask
            pred_value = np.squeeze(renumpy(pred_value, i))
            intensity = np.squeeze(renumpy(batch['intensity'], i))
            pred_value[~mask] = 0
            intensity[~mask] = 0
            ok_map = np.abs(pred_value - intensity)
            score = (ok_map[mask] ** 2).mean()
            h, w, *_ = intensity.shape
            result = np.ones((h * 3 + 2 * BORDER, w))
            result[:h] = intensity
            result[h + BORDER : 2 * h + BORDER] = pred_value
            result[-h:] = ok_map
            return (result * 255).astype('u1'), score

    else:

        def fn(batch, output, i):
            output = torch.argmax(output, 1, keepdim=False)
            output = renumpy(output, i, False)
            labels = renumpy(batch['labels'], i, False)
            mask = np.squeeze(renumpy(batch['mask'], i) >= 0) & ~(labels == ignore_index)
            pred = COLORS[output]
            pred[~mask] = NONE
            ok_map = ~(output == labels)
            score = ok_map[mask].mean()
            ok_map = OK_COLORS[ok_map.astype('u1')]
            ok_map[~mask] = NONE
            correct = COLORS[labels]
            correct[~mask] = NONE
            h, w, c = ok_map.shape
            result = np.ones((h * 3 + 2 * BORDER, w, c), dtype='u1') * 255
            result[:h] = correct
            result[h + BORDER : 2 * h + BORDER] = pred
            result[-h:] = ok_map
            return result, score

    return fn


def info_fn(reflect, ignore_index=4, num_classes=4):
    if reflect:

        def fn(batch, output):
            pred_value = output[-1].detach()
            mask = batch['mask'].detach() >= 0
            if 'rgb_mask' in batch:
                rgb_mask = batch['rgb_mask'].detach() >= 0
            else:
                rgb_mask = torch.ones_like(mask)
            if 'labels' in batch:
                label_mask = ~(batch['labels'].detach() == ignore_index)[:, None, ...]
            else:
                label_mask = torch.ones_like(mask)
            mask = mask & rgb_mask & label_mask
            intensity = batch['intensity'].detach()

            diff = (intensity - pred_value) * (intensity - pred_value)
            diff[~mask] = 0
            return torch.tensor([diff.sum(), mask.sum()])

    else:

        def fn(batch, output):
            output = torch.argmax(output.detach(), 1, keepdim=False)
            labels = batch['labels'].detach()
            mask = torch.squeeze(batch['mask'].detach() >= 0, 1)
            label_mask = ~(labels == ignore_index)
            full_mask = mask & label_mask
            acc = (~(output == labels))[full_mask]
            stats = []
            for i in range(num_classes):
                tp = (output[full_mask & (labels == i)] == i).sum()
                fp = (labels[full_mask & (output == i)] != i).sum()
                fn = (output[full_mask & (labels == i)] != i).sum()
                stats.extend((tp, fp, fn))
            return torch.tensor([acc.sum(), *stats, full_mask.sum()])

    return fn


def scheduler(config, optimizer):
    name = config['name']
    sched = getattr(torch.optim.lr_scheduler, name, None)
    if sched is None:
        return None
    del config['name']
    return sched(optimizer, **config)


class Embed(nn.Embedding):
    def __init__(self, config):
        super().__init__(**config)

    def forward(self, data):
        result = super().forward(data)
        return result.permute(0, 3, 1, 2)
