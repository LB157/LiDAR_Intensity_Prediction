import functools
import glob
import os.path as osp
import yaml
import attr
import numpy as np

import datatools.rays as rays
import otils as ot

npa = np.array  # pylint: disable=invalid-name

TYPES = [('Car', 1), ('Van', 1), ('Truck', 1), 
         ('Pedestrian', 2), ('Person_sitting', 2), ('Cyclist', 2), 
         ('Tram', 3), ('Misc', 3), ('DontCare', 3)]
LAB_MAP = {t.lower(): i for t, i in TYPES}


@attr.s
class Label:
    typ = attr.ib()
    truncated = attr.ib()
    occluded = attr.ib()
    alpha = attr.ib()
    bboxl = attr.ib()
    bboxt = attr.ib()
    bboxr = attr.ib()
    bboxb = attr.ib()
    height = attr.ib()
    width = attr.ib()
    length = attr.ib()
    locx = attr.ib()
    locy = attr.ib()
    locz = attr.ib()
    roty = attr.ib()



# 这个函数 get_label_inds 的作用是根据给定的点云（pcl）和标签（label）定义的边界框，确定哪些点属于这个边界框。
# 具体来说，它通过以下步骤实现其功能：
# 坐标转换：首先，将标签的三维位置转换为 NumPy 数组。
# 旋转和平移：对点云进行旋转（围绕 Y 轴）和平移，以便将其调整到与标签的坐标系统一致。
# 边界定义：为标签的高度、长度和宽度定义边界范围。边界范围是根据标签的属性计算出来的。
# 索引筛选：创建布尔索引，筛选出在这些边界范围内的点。最终，这个布尔数组指示了哪些点位于标签定义的3D边界框内。
def get_label_inds(pcl, label):
    center = np.array([label.locx, label.locy, label.locz])
    rpcl = ot.visual.rot_mat(npa([0, label.roty, 0]), rads=True).T @ (pcl - center[:, None])
    hbounds = [-label.height, 0]
    lbounds = [-label.length / 2, label.length / 2]
    wbounds = [-label.width / 2, label.width / 2]
    ind = (lbounds[0] <= rpcl[0, :]) & (rpcl[0, :] <= lbounds[1])
    ind &= (hbounds[0] <= rpcl[1, :]) & (rpcl[1, :] <= hbounds[1])
    ind &= (wbounds[0] <= rpcl[2, :]) & (rpcl[2, :] <= wbounds[1])
    return ind


def calib_loader(fname):
    with open(fname, 'rt') as f:
        result = {}
        for line in f:
            data = line.strip().split()
            if not data:
                continue
            key = data[0][:-1]
            result[key] = np.fromiter(map(float, data[1:]), dtype='<f8')
            if key[0] == 'P':
                result[key] = result[key].reshape((3, 4))
            if key[0] == 'R':
                tmp = result[key]
                result[key] = np.eye(4)
                result[key][:3, :3] = tmp.reshape((3, 3))
            if key[0] == 'T':
                tmp = result[key]
                result[key] = np.eye(4)
                result[key][:3, :] = tmp.reshape((3, 4))
    return result


def velo_loader(fname):
    return np.fromfile(fname, dtype='<f4').astype('<f8').reshape((-1, 4)).T


def label_loader(fname):
    with open(fname, 'rt') as f:
        result = []
        for line in f:
            data = line.strip().split()
            # 如果为最大标签（一般是 'DontCare'），则跳过这一行。
            if LAB_MAP[data[0].lower()] == max(LAB_MAP.values()):
                continue
            result.append(Label(LAB_MAP[data[0].lower()], *map(float, data[1:])))
    return result

#加载SemanticKITTI的标签
def label_semkitti_loader(fname):
    yaml_fname = '/home/public/liubo/lidar-intensity/python/datatools/semantickitti-all.yaml'
    with open(yaml_fname, 'r') as f:
        mapping = yaml.safe_load(f)
        learning_map = mapping['learning_map']
    original_labels=np.fromfile(fname, dtype='u4') & 0xFFFF
    labels = np.vectorize(learning_map.get)(original_labels)
    labels = np.where(labels is None, 0, labels) 
    return labels

class KittiDataset(ot.dataset.Dataset):
    def __init__(self, base_dir, odo_dataset=True, have_labels=True):
        num_files = ot.dataset.NumFiles(num_files=len(glob.glob(osp.join(base_dir, '09/image_2', '*.png'))))
        kwargs = {'width': 6, 'odo_dataset': odo_dataset, 'have_labels': have_labels}
        super().__init__(base_dir, num_files, KittiEntry, entry_kwargs=kwargs)


class KittiEntry(ot.dataset.DatasetEntry):
    #10个分类
    BINS = np.array([0.03577229, 0.1232794, 0.20273558, 0.24916231, 0.27905208, 0.2998364, 0.32104259, 0.34792211, 0.3898593, 0.70251575])
    EDGES = np.array([0.07952585, 0.16300749, 0.22594894, 0.2641072, 0.28944424, 0.31043949, 0.33448235, 0.3688907, 0.54618753])

    def __init__(self, parent, data_dir, data_id, width, autosave=True, odo_dataset=True, have_labels=True):
        super().__init__(parent, data_dir, data_id, width, autosave)
        self.odo_dataset = odo_dataset
        self.have_labels = have_labels
        self.calib_mat = self.calib_global['Tr'] if self.odo_dataset else self.calib['R0_rect'] @ self.calib['Tr_velo_to_cam']
        self.correct_calib = self.calib_global if self.odo_dataset else self.calib
        self.correct_label_velo = None if not self.have_labels else (self.label_semkitti if self.odo_dataset else self.label_velo)

    def _color_pcl_create(self):
        # 使用 np.concatenate 函数将以下三个数组按列拼接在一起：
        # 1. self.velo: 原始的点云数据（一般表示为 3D 坐标点）
        # 2. self.correct_label_velo.reshape((1, -1)): 将正确的标签数据（correct_label_velo）重塑为一行，确保与其他数组在拼接时维度一致。
        # 3. self.color_velo: 与点云数据相对应的颜色信息（通常用于可视化）。
        # 拼接后的数组将形成一个包含点云坐标、标签和颜色信息的综合数组 pcl。
        pcl = np.concatenate((self.velo, self.correct_label_velo.reshape((1, -1)), self.color_velo), axis=0)
        # 返回点云数据中颜色值大于0的部分，筛选出有效的点。
        # 这里 pcl[-1, :] > 0 的意思是取最后一行（即颜色信息）大于0的列索引，从而返回只有有效颜色的点云数据。
        return pcl[:, pcl[-1, :] > 0]
        #pcl维度特征为 x,y,z,intensity,label,r,g,b,color_mask


    def _label_objects_create(self):
        pcl_rect = ot.visual.fromhomo(self.calib_mat @ ot.visual.tohomo(self.velo[:3, :]))
        result = {}
        for i, label in enumerate(self.label):
            inds = get_label_inds(pcl_rect, label)
            name = f'{label.type}_{i:02d}'
            result[name] = np.concatenate((self.velo[:, inds], self.color_velo[:, inds]), axis=0)
            if result[name].shape[1] <= 1:
                del result[name]
        return result

    def _grid_create(self, params, shift=None):
        inten = self.velo[-1]
        # 根据强度信息使用边界（EDGES）查找所对应的箱（bins）。
        # np.searchsorted 返回 bins 中每个强度值应插入的位置索引。
        bins = np.searchsorted(self.EDGES, inten)

        # 计算强度与其对应的箱的中心（BINS）之间的距离。
        # 这个距离可以用于后续的分析或处理。
        dist = inten - self.BINS[bins]

        # 检查是否有标签信息（have_labels）
        if self.have_labels:
            # 如果有标签，将以下各部分按列拼接成一个点云（pcl）数组：
            # 1. self.velo: 原始的点云数据。
            # 2. bins: 每个点的箱索引，增加一维以便与其他数据拼接。
            # 3. dist: 强度与其对应箱中心之间的距离，也增加一维。
            # 4. self.correct_label_velo: 正确的标签数据，增加一维。
            # 5. self.color_velo: 点云的颜色信息。
            pcl = np.concatenate((self.velo, bins[None, ...], dist[None, ...], 
                                  self.correct_label_velo[None, ...], self.color_velo), axis=0)
        else:
            # 如果没有标签，仅拼接以下各部分：
            # 1. self.velo: 原始的点云数据。
            # 2. bins: 每个点的箱索引。
            # 3. dist: 强度与箱中心的距离。
            # 4. self.color_velo: 点云的颜色信息。
            pcl = np.concatenate((self.velo, bins[None, ...], dist[None, ...], self.color_velo), axis=0)
        return params.pcl2grid(pcl, camera_center=shift)

    def _lidar_pcl_create(self, name, params, shift=None):
        lidar_type = name.split('_')[0]
        #获取self.velo_grid
        grid = getattr(self, lidar_type + '_grid')
        # 调用 params.grid2pcl 方法，将网格数据转换为点云（pcl）并返回。
        # 此时可以选择是否传入相机中心的位移（shift），用于在转换时调整点云的位置信息。
        return params.grid2pcl(grid, camera_center=shift)

    def _color_velo_create(self):
         # 将激光雷达坐标转换为齐次坐标
        pcl_rect = self.calib_mat @ ot.visual.tohomo(self.velo[:3, :])
        # 使用相机 P2 矩阵将点云转换到第一个相机坐标系
        pcl2 = self.correct_calib['P2'] @ pcl_rect
        # 使用相机 P3 矩阵将点云转换到第二个相机坐标系
        pcl3 = self.correct_calib['P3'] @ pcl_rect
        # 创建布尔数组，标记第一个相机坐标系中有效的点
        where2 = pcl2[-1, :] > 0
         # 创建布尔数组，标记第二个相机坐标系中的有效点
        where3 = pcl3[-1, :] > 0
         # 将齐次坐标转换为非齐次坐标（去掉最后一个维度）
        pcl2 = ot.visual.fromhomo(pcl2)
        pcl3 = ot.visual.fromhomo(pcl3)
         # 对于第二个相机，检查点是否在图像边界内
        where2 &= (pcl2[0, :] >= 0) & (pcl2[1, :] >= 0) & (pcl2[0, :] < self.img2.shape[1]) & (pcl2[1, :] < self.img2.shape[0])
         # 对于第三个相机，检查点是否在图像边界内
        where3 &= (pcl3[0, :] >= 0) & (pcl3[1, :] >= 0) & (pcl3[0, :] < self.img3.shape[1]) & (pcl3[1, :] < self.img3.shape[0])
        # 初始化一个全零的点云结果数组，形状与 self.velo 相同
        result = np.zeros(self.velo.shape)
        # 将第一个相机的有效颜色信息加到结果点云中
        result[:3, where2] += self.img2[pcl2[1, where2].astype(np.int64), pcl2[0, where2].astype(np.int64), :].T
        result[3, where2] += 1
        # 将第二个相机的有效颜色信息加到结果点云中
        result[:3, where3] += self.img3[pcl3[1, where3].astype(np.int64), pcl3[0, where3].astype(np.int64), :].T
        result[3, where3] += 1
        result = ot.visual.fromhomo(result, return_all_dims=True)
        # print(f'result.shape: {result.shape} \n')
        # reslut的维度为4*n的   1，2，3是rgb信息，第4维是被用来标识每个点是否得到了颜色信息（通过加法来实现）。
        return result

    def _label_velo_create(self):
        pcl_rect = ot.visual.fromhomo(self.calib_mat @ ot.visual.tohomo(self.velo[:3, :]))
        result = np.zeros((pcl_rect.shape[1],))
        for label in self.label:
            result[get_label_inds(pcl_rect, label)] = label.type
        return result


   #_color_pcl_gs_create 函数的结果是将点云数据中的强度列替换为灰度值，而其他列的值保持不变。
    def _color_pcl_gs_create(self):
        # color_pcl: 9*n    x,y,z,intensity,label,r,g,b,color_mask
        pcl = npa(self.color_pcl)
        pcl[3] = ot.visual.rgb2gs(pcl[5:8])
        return pcl

    _velo_grid_create = functools.partial(_grid_create, params=rays.velodyne_params)
    _velo_pcl_create = functools.partial(_lidar_pcl_create, params=rays.velodyne_params,name='velodyne_default')

    label = ot.dataset.DataAttrib('{data_id:0{width}d}.txt', label_loader, 'label_2', deletable=False)
    label_semkitti = ot.dataset.DataAttrib('{data_id:0{width}d}.label', label_semkitti_loader, '09/labels', deletable=False)
    img2 = ot.dataset.DataAttrib('{data_id:0{width}d}.png', ot.io.img_load, '09/image_2', deletable=False)
    img3 = ot.dataset.DataAttrib('{data_id:0{width}d}.png', ot.io.img_load, '09/image_3', deletable=False)
    calib = ot.dataset.DataAttrib('{data_id:0{width}d}.txt', calib_loader, 'calib', deletable=False, wfable=False)
    calib_global = ot.dataset.DataAttrib('calib.txt', calib_loader, '09', deletable=False, wfable=False)
    velo = ot.dataset.DataAttrib('{data_id:0{width}d}.bin', velo_loader, '09/velodyne', deletable=False)
    color_velo = ot.dataset.DataAttrib('{data_id:0{width}d}.npy', np.load, 'color_velo/09', _color_velo_create, np.save)
    label_velo = ot.dataset.DataAttrib('{data_id:0{width}d}.npy', np.load, 'label_velo/09', _label_velo_create, np.save)
    color_pcl = ot.dataset.DataAttrib('{data_id:0{width}d}.npy', np.load, 'color_pcl/09', _color_pcl_create, np.save)
    color_pcl_gs_inten = ot.dataset.DataAttrib('{data_id:0{width}d}.npy', np.load, 'color_pcl_gs/09', _color_pcl_gs_create, np.save)
    label_objects = ot.dataset.DataAttrib(
        '{data_id:0{width}d}.npz', lambda fname: dict(np.load(fname)), 'label_objects/09', _label_objects_create, ot.io.np_savez
    )
    velodyne_grid = ot.dataset.DataAttrib('{data_id:0{width}d}.npy', np.load, ('pseudo-velodyne', 'grid/09'), _velo_grid_create, np.save)
    velodyne_pcl = ot.dataset.DataAttrib('{data_id:0{width}d}.npy', np.load, ('pseudo-velodyne', 'pcl/09'), _velo_pcl_create, np.save, ['name'])

# grid: 400: (72, 2084, 13)    depth,x,y,z,intensity,bins,dist,label,r,g,b,color_mask,mask
# pcl: 400: (11, 104662)    depth,x,y,z,intensity,label,r,g,b,color_mask,mask
# color_pcl: 400: (9, 20346)    x,y,z,intensity,label,r,g,b,color_mask
# color_pcl_gs_inten: 400: (9, 20346)  x,y,z,灰度,label,r,g,b,color_mask
# color_velo: 400: (4, 124799)     1，2，3是rgb信息，第4维是被用来标识每个点是否得到了颜色信息（通过加法来实现）。