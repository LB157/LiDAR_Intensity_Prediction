import torch.nn as nn
import torch.nn.functional as F
import yaml

from . import heads
from . import modules as md


class SqueezeWithHead(nn.Module):
    def __init__(self, head_cls, squeeze_kwargs, head_kwargs):
        super().__init__()
        self.squeeze = SqueezeSegBone(**squeeze_kwargs)
        self.head = head_cls(**head_kwargs)

    def forward(self, x):
        features = self.squeeze(x)
        return self.head(x, features)

    @classmethod
    def load_from_kwargs(cls, data):
        if isinstance(data['head_cls'], str):
            head_cls = getattr(heads, data['head_cls'], None)
            if head_cls is None:
                raise RuntimeError(f'Could not find your class {data["head_cls"]}!')
            data['head_cls'] = head_cls
        return cls(**data)

    @classmethod
    def load_from_yaml(cls, yaml_file):
        with open(yaml_file, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return cls.load_from_kwargs(data)


class SqueezeSegBone(nn.Module):
    def __init__(self, input_channels, squeeze_depth=2, cam_depth=1, conv_starts=64, squeeze_start=16, ef_start=64):
        super().__init__()
        self.reduce = 1
        self.start = nn.Sequential(
            md.Conv(input_channels, conv_starts, 3, 1, 2, top_parent=self),
            md.ContextAggregation(conv_starts, top_parent=self),
            md.Conv(conv_starts, conv_starts, 1, top_parent=self),
        )
        self.rest = nn.Sequential(
            md.Pool(3, 2, 1, top_parent=self),
            md.SqueezePart(conv_starts, squeeze_start, ef_start, squeeze_depth, cam_depth, top_parent=self),
            md.DeFire(2 * ef_start, squeeze_start, int(conv_starts / 2), top_parent=self),
            # p 参数的默认值为 0.5，这表示在每次前向传播时，会随机将 50% 的特征（通道）设为 0。
            # 这样可以有效地防止模型在训练过程中发生过拟合。
            nn.Dropout2d(),
        )

    def forward(self, x):
        shape = x.shape
        over = shape[-1] % self.reduce
        # 检查 over 是否为真（即不等于零）。如果是，表示最后一个维度不是 self.reduce 的整数倍，
        # 后续代码将会对输入张量进行填充
        if over:
            # 计算需要填充的数量，以确保最后一个维度是 self.reduce 的整数倍
        # 对输入张量 x 进行填充，使用 'replicate' 模式，填充的数值来自原始张量的边缘
            over = self.reduce - over
            x = F.pad(x, (int(over / 2), int(over / 2), 0, 0), 'replicate')
        pre_add = self.start(x)
        insides = self.rest(pre_add)
        result = pre_add + insides
        # 将初步输出和进一步输出进行相加，得到最终结果
        return result
