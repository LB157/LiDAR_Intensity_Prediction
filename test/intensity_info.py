# 设置序列路径（替换为实际路径）
# sequence_path = "/home/public/DataSetLarge/SemanticKitti/dataset/sequences/"
import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_point_cloud(filepath):
    """
    加载 SemanticKITTI 点云数据
    :param filepath: .bin 文件的路径
    :return: 点云的 numpy 数组 (N, 4)，包含 [x, y, z, intensity]
    """
    point_cloud = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
    return point_cloud

def process_file(filepath):
    """
    处理单个点云文件并提取强度值
    :param filepath: .bin 文件路径
    :return: 强度值数组
    """
    point_cloud = load_point_cloud(filepath)
    return point_cloud[:, 3]  # 返回强度值列

def get_all_intensities_parallel(base_path, sequences, max_workers=8):
    """
    使用线程池并行提取多个序列中所有点云的强度值
    :param base_path: 数据集的根目录，例如 "/path_to_semantic_kitti/"
    :param sequences: 需要处理的序列列表，例如 ["00", "01", ..., "10"]
    :param max_workers: 最大线程数
    :return: 包含所有强度值的 numpy 数组
    """
    intensity_values = []
    filepaths = []

    # 收集所有点云文件的路径
    for seq in sequences:
        sequence_path = os.path.join(base_path, seq, "velodyne")
        all_files = sorted(os.listdir(sequence_path))
        for file in all_files:
            if file.endswith('.bin'):
                filepaths.append(os.path.join(sequence_path, file))
    
    # 使用线程池并行处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, filepath): filepath for filepath in filepaths}
        for future in as_completed(future_to_file):
            try:
                intensities = future.result()  # 获取单个文件的强度值结果
                intensity_values.append(intensities)
            except Exception as e:
                print(f"Error processing file {future_to_file[future]}: {e}")
    
    # 将所有强度值拼接到一个数组中
    return np.concatenate(intensity_values)

def save_intensity_distribution(intensity_values, output_path, bins=100):
    """
    绘制并保存强度值的分布直方图
    :param intensity_values: 强度值数组
    :param output_path: 保存图片的路径
    :param bins: 直方图的分箱数量
    """
    plt.figure(figsize=(8, 6))
    plt.hist(intensity_values, bins=bins, range=(0, 1), color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Intensity Distribution in [0, 1]", fontsize=16)
    plt.xlabel("Intensity", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(output_path)  # 保存图片
    plt.close()  # 关闭图像以释放内存

# 主程序
if __name__ == "__main__":
    # 设置数据集路径
    base_path = "/path_to_semantic_kitti/"
    
    # 定义需要处理的序列
    sequences = [f"{i:02d}" for i in range(11)]  # 生成 ["00", "01", ..., "10"]
    
    # 提取所有序列的强度值（使用线程池）
    intensity_values = get_all_intensities_parallel(base_path, sequences, max_workers=8)
    
    # 检查强度值范围（可选）
    print(f"Min intensity: {intensity_values.min()}, Max intensity: {intensity_values.max()}")
    
    # 保存强度分布直方图
    output_path = "intensity_distribution.png"
    save_intensity_distribution(intensity_values, output_path)
    print(f"Intensity distribution saved to {output_path}")




# 主程序
if __name__ == "__main__":
    # 设置数据集路径
    base_path = "/home/public/DataSetLarge/SemanticKitti/dataset/sequences/"
    
    # 定义需要处理的序列
    sequences = [f"{i:02d}" for i in range(1)]  # 生成 ["00", "01", ..., "10"]
    
    # 提取所有序列的强度值
    intensity_values = get_all_intensities(base_path, sequences)
    
    # 检查强度值范围（可选）
    print(f"Min intensity: {intensity_values.min()}, Max intensity: {intensity_values.max()}")
    
    # 保存强度分布直方图
    test_data_local_path = '/home/public/liubo/lidar-intensity/test/dataset/'
    output_path=os.path.join(test_data_local_path, 'intensity_distribution.png')
    # output_path = "intensity_distribution.png"
    save_intensity_distribution(intensity_values, output_path)
    print(f"Intensity distribution saved to {output_path}")