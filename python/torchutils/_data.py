import argparse
import builtins
import collections
import contextlib
import enum
import functools
import os.path as osp
import random
import weakref

import numpy as np
import torch
import torch.optim
import torch.utils.data as data

try:
    import tqdm

    _TQDM_FOUND = True
except ImportError:
    _TQDM_FOUND = False


builtins.print = functools.partial(print, flush=True)

# 函数 _compose(*functions) 的作用是实现函数组合（function composition）。
# 函数组合是指将多个函数按顺序组合成一个新函数，使得新函数的输出成为下一个函数的输入。
def _compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


class SimpleDataset(data.Dataset):
    """
    自定义数据集类，继承自 PyTorch 的 data.Dataset
    """
    class weakdict(dict):
        __slots__ = ('__weakref__',) # 为弱字典定义槽位

    class weaklist(list):
        __slots__ = ('__weakref__',)# 为弱列表定义槽位

    def __init__(self, folder: str, name=None, ext='.npy', shuffle=True, keep_ram=False):
        """
        初始化数据集
        :param folder: 数据文件夹路径
        :param name: 数据集名称
        :param ext: 数据文件扩展名
        :param shuffle: 是否打乱数据
        :param keep_ram: 是否保留数据在内存中
        """
        self.folder = folder
        self.ext = ext.lower()
        if name is None:
            # 从路径中提取文件夹名作为数据集名称
            name = osp.split(folder)[1]
        self.name = name
        # 扫描文件夹中的文件
        self.files = self.scan_files()
        if shuffle:
            # 打乱文件顺序
            random.shuffle(self.files)
        self._len = len(self.files)
        self.keep_ram = keep_ram
        if keep_ram:
            self.loaded = dict()
        else:
            # 使用弱引用字典减少内存占用
            self.loaded = weakref.WeakValueDictionary()

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        """
        获取数据项
        :param key: 数据索引
        :return: 返回对应的文件数据
        """
        key = int(key)
        fname = self.files[key]
        data_item = self.loaded.get(fname, None)
        if data_item is None:
            with self.__weakcm():
                data_item = self.load_and_transform(fname, key)
            self.loaded[fname] = data_item
        return data_item

    def load_and_transform(self, fname, key):
        """
        读取和转换数据 (未实现)
        :param fname: 文件名
        :param key: 索引
        """
        raise NotImplementedError

    def scan_files(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def __weakcm(self):
        if not self.keep_ram:
            _stash = dict, list
            builtins.dict = self.weakdict
            builtins.list = self.weaklist
            yield
            builtins.dict, builtins.list = _stash
        else:
            yield


class TorchMode(enum.Enum):
    #名为 TRAIN。enum.auto() 函数会自动分配一个唯一的值给这个成员，通常从 1 开始，并会为后续成员递增
    TRAIN = enum.auto()
    EVAL = enum.auto()


def dict_to_cuda(d, *args, **kwargs):
    """
    将字典中的张量移动到 CUDA 设备
    :param d: 包含张量的字典
    :param args: 其他参数
    :param kwargs: 其他参数
    :return: 移动后的字典
    """
    for key in d:
        if hasattr(d[key], 'cuda'):
            d[key] = d[key].cuda(*args, **kwargs)
    return d

#otils，inten，torchutils
class Runner:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        pass_keys,
        gt_keys,
        embedder=None,
        embed_channel=None,
        verbose=False,
        args=None,
        use_tqdm=False,
        accum_losses=False,
        pass_as_kwargs=False,
        cat_channels=False,
    ):
        """
        初始化 Runner
        :param model: 训练的模型
        :param loss_fn: 损失函数
        :param optimizer: 优化器
        :param pass_keys: 需要传递的键
        :param gt_keys: 真实值的键
        :param embedder: 嵌入层
        :param embed_channel: 嵌入通道
        :param verbose: 是否详细输出
        :param args: 其他参数
        :param use_tqdm: 是否使用 tqdm 进度条
        :param accum_losses: 是否累积损失
        :param pass_as_kwargs: 是否将参数作为关键字传递
        :param cat_channels: 是否连接通道
        """
        # batch[self.embed_channel].shape 为[2,56,512]
        # self.embedder 为Embed(5, 2, max_norm=1)
        # 你的嵌入层 self.embedder 是通过 Embed(5, 2, max_norm=1) 初始化的，
        # 这表示嵌入层有 5 个词（索引从 0 到 4），并且每个词的嵌入大小是 2。这意味着你传入的索引值必须在 0 到 4 的范围内。
        # 标签 应该还要做映射
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.pass_keys = pass_keys
        self.gt_keys = gt_keys
        self.verbose = verbose
         # 判断是否能够使用 tqdm
        self.use_tqdm = use_tqdm & _TQDM_FOUND
        self.embedder = embedder
        self.embed_channel = embed_channel
        if args is None:
            args = argparse.Namespace(cuda=False, keep_ram=False)
        self.args = args
        try:
            self.cuda = args.cuda
        except AttributeError:
            self.cuda = False
        try:
            self.keep_ram = args.keep_ram
        except AttributeError:
            self.keep_ram = False
        if self.use_tqdm:
            self.iter_wrap = tqdm.tqdm
        else:
            self.iter_wrap = functools.partial(map, lambda x, *args, **kwargs: x)
            # self.iter_wrap = lambda iterable: iterable
        self.pass_as_kwargs = pass_as_kwargs
        self.accum_losses = accum_losses
        self.run_times = collections.Counter()
        self.run_losses = dict()
        self.cat_channels = cat_channels

    @contextlib.contextmanager
    def _setup(self):
        _stash = builtins.print
        if self.use_tqdm:
            builtins.print = _compose(lambda x: tqdm.tqdm.write(x) if x.strip() else None, lambda x: '' if not self.verbose else x, str)
        else:
            builtins.print = _compose(lambda x: _stash(x) if x.strip() else None, lambda x: '' if not self.verbose else x, str)
        yield
        builtins.print = _stash

    def __call__(self, dataloader, mode):
        """
        运行训练或评估过程
        :param dataloader: 数据加载器
        :param mode: 模式（训练或评估）
        """
        datalen = len(dataloader.dataset)# 数据集长度
        did = 0# 记录处理过的数据量
        self.model = self.model.train() if mode is TorchMode.TRAIN else self.model.eval() # 设置模型模式
        self.run_pre_epoch(dataloader.dataset, mode)  # 运行前期定义
        if self.accum_losses:
            self.run_losses[dataloader.dataset] = list() # 初始化损失列表
        with self._setup(): # 设置打印上下文
            with torch.set_grad_enabled(mode == TorchMode.TRAIN):# 根据模式设置梯度计算
                for batch_id, batch in self.iter_wrap(enumerate(dataloader)): # 遍历数据
                    batch_len = len(next(iter(batch.values()))) # 获取当前批次的长度
                    did += batch_len # 更新已处理数据量
                    if self.cuda:
                        dict_to_cuda(batch, **({'non_blocking': True} if self.keep_ram else {})) # 移动数据到 CUDA
                    if mode == TorchMode.TRAIN:
                        self.optimizer.zero_grad()  # 清空梯度
                    if self.embedder is not None:
                        #  使用嵌入层处理数据
                        batch[self.embed_channel + '_embed'] = self.embedder(batch[self.embed_channel])
                    # 获取需要的参数
                    run_kwargs = collections.OrderedDict((key, batch[key]) for key in self.pass_keys)
                    if self.cat_channels:
                        # 如果需要连接通道
                        output = self.model(torch.cat(tuple(run_kwargs.values()), 1))
                    elif self.pass_as_kwargs:
                        # 如果参数作为关键字传递
                        output = self.model(**run_kwargs)
                    else:
                        # 正常传递参数
                        output = self.model(*run_kwargs.values())
                     # 计算损失
                    if mode == TorchMode.TRAIN or all([key in batch for key in self.gt_keys]):
                         # 获取真实值
                        loss_kwargs = collections.OrderedDict((key, batch[key]) for key in self.gt_keys)
                        if self.pass_as_kwargs:
                            # 使用关键字传递损失
                            loss = self.loss_fn(output, **loss_kwargs)
                        else:
                            # 位置参数传递损失
                            loss = self.loss_fn(output, *loss_kwargs.values())
                        if mode == TorchMode.TRAIN:
                            #反向传播
                            loss.backward()
                            # 更新优化器
                            self.optimizer.step()
                    else:
                        # 不计算损失
                        loss = None
                    print_str = f'Epoch id: {self.run_times[dataloader.dataset]}\t{did: 6d} / {datalen : 6d}\t'
                    if loss is not None:
                        print_str += f'Loss: {loss:8.04f}\t'
                    if self.accum_losses:
                        loss_cpu = loss.detach().cpu()  # 将损失移动到 CPU
                        self.run_losses[dataloader.dataset].append(loss.detach() * batch_len)
                        mean_loss = np.sum([l.item() for l in self.run_losses[dataloader.dataset]]) 
                        # print_str += f'Mean loss over epoch: {np.sum(self.run_losses[dataloader.dataset])/did: 8.04f}\t'
                        print_str += f'Mean loss over epoch: {mean_loss / did: 8.04f}\t'
                    extra = self.run_after_iter(batch, output, loss, mode, did, batch_id, len(dataloader), dataloader.dataset)
                    if extra is not None:
                        print_str += str(extra)
                    print(print_str)
                print_str = f'Epoch {self.run_times[dataloader.dataset]} for dataset {dataloader.dataset} and mode {mode} finished!\n'
                extra = self.run_after_epoch(dataloader.dataset, mode)
                if extra is not None:
                    print_str += str(extra)
                print(print_str)
                self.run_times[dataloader.dataset] += 1

    def run_after_iter(self, batch, output, loss, mode, did, batch_id, total_batches, dataset):
        pass

    def run_after_epoch(self, dataset, mode):
        pass

    def run_pre_epoch(self, dataset, mode):
        pass
