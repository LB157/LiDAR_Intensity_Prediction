import numpy as np
import yaml


def label_semkitti_loader(fname):
    yaml_fname = '/home/public/liubo/lidar-intensity/python/datatools/semantickitti-all.yaml'
    with open(yaml_fname, 'r') as f:
        mapping = yaml.safe_load(f)
        learning_map = mapping['learning_map']
    original_labels=np.fromfile(fname, dtype='u4') & 0xFFFF
    labels = np.vectorize(learning_map.get)(original_labels)
    labels = np.where(labels is None, 0, labels) 
    return labels

def velo_loader(fname):
    return np.fromfile(fname, dtype='<f4').astype('<f8').reshape((-1, 4)).T

# 加载 .npy 文件
fname='/home/public/DataSetLarge/SemanticKitti/dataset/sequences/09/labels/000400.label'
# file_path = '/home/public/liubo/lidar-intensity/test/dataset/084816.npy'  # 假设你的文件名为 data.npy
# file_path = '/home/public/liubo/lidar-intensity/test/dataset/084816.npy'  # 假设你的文件名为 data.npy
# file_path = '/home/public/DataSetLarge/SemanticKitti/dataset/sequences/pseudo-velodyne/grid/09/000400.npy'  # 假设你的文件名为 data.npy
file_path = '/home/public/DataSetLarge/SemanticKitti/dataset/sequences/09/velodyne/000400.bin'  # 假设你的文件名为 data.npy
# label_semkitti_loader(fname)
# data = np.load(file_path)
# 打印第 5 维（索引为 4）的数据
data = velo_loader(file_path)

# fifth_dimension_data = data[:, :, 4]  # 这里 4 表示第 5 维的索引
# with open('fifth_dimension_data.txt', 'w') as f:
    # np.savetxt(f, fifth_dimension_data, fmt='%.6f')  # 使用 savetxt 将数据保存为 txt 文件，保留 6 位小数
# print(fifth_dimension_data)

# 输出数据的形状
# 获取最后一行的数据
last_row = data[-1, :]

# 获取最后一行的最大值和最小值
max_value = np.max(last_row)
min_value = np.min(last_row)

# 打印结果
print("最后一行的最大值:", max_value)
print("最后一行的最小值:", min_value)
print("数据的形状:", data.shape)