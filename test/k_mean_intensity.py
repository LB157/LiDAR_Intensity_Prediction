import numpy as np
import os
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import json
def load_point_cloud(file_path):
    """ 加载点云数据，假设是 .bin 格式 """
    points = np.load(file_path).astype(np.float32)
    intensity=points[:,:,4]
    return intensity.flatten()

def kmeans_clustering(points, n_clusters):
    """ 对点云进行 K 均值聚类 """
    kmeans = KMeans(n_clusters=n_clusters)
    intensity_values = points.reshape(-1, 1) 
    labels = kmeans.fit_predict(intensity_values) 
    return labels

def calculate_intensity_stats(points, labels):
    """ 计算每个类簇的强度均值、最大值、最小值 """
    stats = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_intensities = points[labels == label]
        stats[label] = {
        'mean_intensity': np.mean(cluster_intensities),
        'max_intensity': np.max(cluster_intensities),
        'min_intensity': np.min(cluster_intensities)
        }

    return stats

def convert_to_native_types(data):
    if isinstance(data, dict):
        return {convert_to_native_types(key): convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # 将 NumPy 数组转换为列表
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)  # 转换为原生 float 类型
    elif isinstance(data, (np.int32)):
        return float(data)  # 转换为原生 float 类型
    else:
        return data  # 其他类型直接返回
        
def process_file(points, n_clusters):
    intensity_points = points.flatten()
    labels = kmeans_clustering(intensity_points, n_clusters)
    intensity_stats = calculate_intensity_stats(intensity_points, labels)
    return intensity_stats

def main(data_dir, n_clusters=5, n_jobs=-1):
    """ 主函数 """
    # 获取所有点云文件（序列文件 00 到 10）
    file_paths = []
    all_points = []
    for i in range(7): # 0 到 10
        seq_dir = os.path.join(data_dir, f'{i:02d}')
        if os.path.exists(seq_dir):
            for f in os.listdir(seq_dir):
                if f.endswith('.npy'): # 假设文件是 .bin 格式
                    file_path=os.path.join(seq_dir, f)
                    points=load_point_cloud(file_path)
                    all_points.append(points)
                    

    # 并行处理
    all_points = np.vstack(all_points)
    # results = Parallel(n_jobs=n_jobs)(
        # delayed(process_file)(all_points, n_clusters) 
    # )
    results = process_file(all_points, n_clusters)
    results_serializable = convert_to_native_types(results)
    with open('0_6results.txt', 'w') as f:
        json.dump(results_serializable, f, indent=4)
    

        
if __name__ == "__main__":
    data_directory = "/home/public/DataSetLarge/SemanticKitti/dataset/sequences/pseudo-velodyne/grid"
    nume_classes = 19
    main(data_directory, n_clusters=nume_classes)







# 以下是对每个文件中的digo的强度进行聚类，并计算每个类簇的强度均值、最大值、最小值。
# import numpy as np
# import os
# from sklearn.cluster import KMeans
# from joblib import Parallel, delayed

# def load_point_cloud(file_path):
#     """ 加载点云数据，假设是 .bin 格式 """
#     points = np.load(file_path).astype(np.float32)
#     intensity=points[:,:,4]
#     return intensity.flatten()

# def kmeans_clustering(points, n_clusters=5):
#     """ 对点云进行 K 均值聚类 """
#     kmeans = KMeans(n_clusters=n_clusters)
#     intensity_values = points.reshape(-1, 1) 
#     labels = kmeans.fit_predict(intensity_values) 
#     return labels

# def calculate_intensity_stats(points, labels):
#     """ 计算每个类簇的强度均值、最大值、最小值 """
#     stats = {}
#     unique_labels = np.unique(labels)

#     for label in unique_labels:
#         cluster_intensities = points[labels == label]
#         stats[label] = {
#         'mean_intensity': np.mean(cluster_intensities),
#         'max_intensity': np.max(cluster_intensities),
#         'min_intensity': np.min(cluster_intensities)
#         }

#     return stats

# def process_file(file_path, n_clusters=5):
#     """ 处理单个点云文件 """
#     print(f"Processing {file_path}...")
#     points = load_point_cloud(file_path)
#     labels = kmeans_clustering(points, n_clusters)
#     intensity_stats = calculate_intensity_stats(points, labels)
#     return intensity_stats

# def main(data_dir, n_clusters=5, n_jobs=-1):
#     """ 主函数 """
#     # 获取所有点云文件（序列文件 00 到 10）
#     file_paths = []
#     all_points = []
#     for i in range(1): # 0 到 10
#         seq_dir = os.path.join(data_dir, f'{i:02d}')
#         if os.path.exists(seq_dir):
#             for f in os.listdir(seq_dir):
#                 if f.endswith('.npy'): # 假设文件是 .bin 格式
#                     file_paths.append(os.path.join(seq_dir, f))
                    

#     # 并行处理
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(process_file)(file_path, n_clusters) for file_path in file_paths
#     )
#     # results = [process_file(file_path, n_clusters) for file_path in file_paths]

#     # 输出结果
#     for file_path, stats in zip(file_paths, results):
#         print(f"Finished processing {file_path}. Stats: {stats}")

# if __name__ == "__main__":
#     data_directory = "/home/public/DataSetLarge/SemanticKitti/dataset/sequences/pseudo-velodyne/grid"
#     nume_classes = 19
#     main(data_directory, n_clusters=nume_classes)