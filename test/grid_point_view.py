import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
import open3d as o3d

def load_point_cloud_from_bin(file_path):
    # Load point cloud data from a .bin file
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # We only need xyz coordinates


def semantickitti_label_yaml_loader():
    yaml_fname = '/home/public/liubo/lidar-intensity/python/datatools/semantickitti-all.yaml'
    with open(yaml_fname, 'r') as f:
        mapping = yaml.safe_load(f)
    return mapping
    
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


def npy_loder(fname):
    return np.load(fname)
# 加载 .npy 文件


def grid_label_load(fname):
    fdata=npy_loder(fname)
    label_7=fdata[:,:,7]
    return label_7

def vis_girds(fname):
    file_dir = os.path.dirname(fname)  # 获取文件路径
    file_name = os.path.basename(fname)  # 获取文件名，包含扩展名
    file_name_pre = file_name.split('.')[0]  # 去掉扩展名
        # 获取上级目录
    parent_dir = os.path.dirname(file_dir)
    
    label_grid=grid_label_load(fname)
    
    all_mapping=semantickitti_label_yaml_loader()
    mapping_inv=all_mapping['learning_map_inv']
    mapping_clolor=all_mapping['color_map'] #bgr
    output_image = np.zeros((72, 2084, 3), dtype=np.uint8)
    for label in np.unique(label_grid):
        if label in mapping_inv:  # 确保标签在映射范围内
            color_index = mapping_inv[label]  # 映射到对应的颜色索引
            mapping_color = mapping_clolor[color_index]  # 得到对应的颜色
            # 绘制
            output_image[label_grid == label] = mapping_color  # 映射到对应的颜色
    save_fname=os.path.join(file_dir,file_name_pre+'.png')
    save_image(output_image, save_fname)  # 保存图片


def save_image(output_image, save_filename):
    """
    Save the output image as a file.
    
    :param output_image: The image data to save (a NumPy array of shape (height, width, 3)).
    :param filename: The filename (including path) to save the image.
    """
    plt.imsave(save_filename, output_image)  # 使用 imsave 保存图像



def load_point_cloud_from_bin(file_path):
    # Load point cloud data from a .bin file
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud[:, :3]  # We only need xyz coordinates

def visualize_point_cloud(point_cloud):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])



def vis_point(bin_file_path):

    if os.path.exists(bin_file_path):
        point_cloud = load_point_cloud_from_bin(bin_file_path)
        visualize_point_cloud(point_cloud)
    else:
        print(f"File not found: {bin_file_path}")

if __name__ == '__main__':
    # grid：depth,x,y,z,intensity,bins,dist,label,r,g,b,color_mask,mask
    dataset_local_path = '/home/public/DataSetLarge/SemanticKitti/dataset/'
    test_data_local_path = '/home/public/liubo/lidar-intensity/test/dataset/'
    bin_file_path = os.path.join(test_data_local_path, '000400.bin')  # Update this with your actual file path
    vis_point(bin_file_path)
    

# fname='/home/public/DataSetLarge/SemanticKitti/dataset/sequences/09/labels/000400.label'
# file_path = '/home/public/liubo/lidar-intensity/test/dataset/084816.npy'  # 假设你的文件名为 data.npy
# file_path = '/home/public/liubo/lidar-intensity/test/dataset/084816.npy'  # 假设你的文件名为 data.npy
# file_path = '/home/public/DataSetLarge/SemanticKitti/dataset/sequences/pseudo-velodyne/grid/09/000400.npy'  # 假设你的文件名为 data.npy
# file_path = '/home/public/DataSetLarge/SemanticKitti/dataset/sequences/09/velodyne/000400.bin'  # 假设你的文件名为 data.npy
# label_semkitti_loader(fname)
# data = np.load(file_path)
# 打印第 5 维（索引为 4）的数据
# data = velo_loader(file_path)

# fifth_dimension_data = data[:, :, 4]  # 这里 4 表示第 5 维的索引

# 输出数据的形状

# fname=test_data_local_path+'000400.npy'
# grid_label_load(fname)
# vis_girds(fname)

