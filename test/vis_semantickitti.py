import numpy as np
# import open3d as ot
import cv2
import os
import sys
import matplotlib.pyplot as plt
import visual as visual
import yaml

class SemantickittiVisualizer:
    def __init__(self, velo_file, calib_file, img2_file, img3_file,label_file):
        self.velo = np.fromfile(velo_file, dtype='<f4').astype('<f8').reshape((-1, 4)).T
        self.calib_global = self.calib_loader(calib_file)
        self.calib_mat = self.calib_global['Tr']
        self.correct_calib = self.calib_global
        self.img2 = self.img_loader(img2_file)
        self.img3 = self.img_loader(img3_file)
        self.label=self.leable_loader(label_file)
        self.label_mapping=self.label_to_color()
        self.color_maping=self.label_mapping['color_map']
        
    def label_to_color(self):
        yaml_fname = '/home/public/liubo/lidar-intensity/python/datatools/semantickitti-all.yaml'
        with open(yaml_fname, 'r') as f:
            mapping = yaml.safe_load(f)
        return mapping
    
    def img_loader(self, img_file):
        img = cv2.imread(img_file)
        # cv2.imshow("projection", img)
        # cv2.waitKey(0)
        return img
    
    def leable_loader(self,label_file):
        original_labels=np.fromfile(label_file, dtype='u4') & 0xFFFF
        return original_labels
    
    def calib_loader(self,fname):
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

   
    
    def show_velo_on_img(self):
         # 将激光雷达坐标转换为齐次坐标
        pcl_rect = self.calib_mat @ visual.tohomo(self.velo[:3, :])
        # 使用相机 P2 矩阵将点云转换到第一个相机坐标系
        ori_pcl2_2drect = self.correct_calib['P2'] @ pcl_rect
        # 使用相机 P3 矩阵将点云转换到第二个相机坐标系
        ori_pcl3_2drect = self.correct_calib['P3'] @ pcl_rect
        # 创建布尔数组，标记第一个相机坐标系中有效的点
        where2 = ori_pcl2_2drect[-1, :] > 0
         # 创建布尔数组，标记第二个相机坐标系中的有效点
        where3 = ori_pcl3_2drect[-1, :] > 0
         # 将齐次坐标转换为非齐次坐标（去掉最后一个维度）
        ori_pcl2 = visual.fromhomo(ori_pcl2_2drect)
        ori_pcl3 = visual.fromhomo(ori_pcl3_2drect)
        
         # 对于第二个相机，检查点是否在图像边界内
        where2 &= (ori_pcl2[0, :] >= 0) & (ori_pcl2[1, :] >= 0) & (ori_pcl2[0, :] < self.img2.shape[1]) & (ori_pcl2[1, :] < self.img2.shape[0])
         # 对于第三个相机，检查点是否在图像边界内
        where3 &= (ori_pcl3[0, :] >= 0) & (ori_pcl3[1, :] >= 0) & (ori_pcl3[0, :] < self.img3.shape[1]) & (ori_pcl3[1, :] < self.img3.shape[0])
        # 初始化一个全零的点云结果数组，形状与 self.velo 相同
        # result = np.zeros(self.velo.shape)
        # # 将第一个相机的有效颜色信息加到结果点云中
        # result[:3, where2] += self.img2[pcl2[1, where2].astype(np.int64), pcl2[0, where2].astype(np.int64), :].T
        # result[3, where2] += 1
        # # 将第二个相机的有效颜色信息加到结果点云中
        # result[:3, where3] += self.img3[pcl3[1, where3].astype(np.int64), pcl3[0, where3].astype(np.int64), :].T
        # result[3, where3] += 1
        # result = ot.visual.fromhomo(result, return_all_dims=True)
        # # print(f'result.shape: {result.shape} \n')
        # # reslut的维度为4*n的   1，2，3是rgb信息，第4维是被用来标识每个点是否得到了颜色信息（通过加法来实现）。
        # return result
            # )
        pcl2_2d=ori_pcl2[:,where2]
        pcl3_2d=ori_pcl3[:,where3]
        pcl2_2drect=ori_pcl2_2drect[:,where2]
        pcl3_2drect=ori_pcl3_2drect[:,where3]

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        img2 = self.img2.copy()
        # cv2.imshow("projection", img2)

        for i in range(pcl2_2d.shape[1]):
            depth = pcl2_2drect[2, i]
            color = cmap[int(640.0 / depth), :]
            cv2.circle(
                img2,
                (int(np.round(pcl2_2d[0, i])), int(np.round(pcl2_2d[1, i]))),
                2,
                color=tuple(color),
                thickness=-1,
            )
        cv2.imshow("projection", img2)
        cv2.waitKey(0)
        return img2
    

    def show_velo_on_img_with_label(self):
                 # 将激光雷达坐标转换为齐次坐标
        pcl_rect = self.calib_mat @ visual.tohomo(self.velo[:3, :])
        # 使用相机 P2 矩阵将点云转换到第一个相机坐标系
        ori_pcl2_2drect = self.correct_calib['P2'] @ pcl_rect
        # 使用相机 P3 矩阵将点云转换到第二个相机坐标系
        ori_pcl3_2drect = self.correct_calib['P3'] @ pcl_rect
        # 创建布尔数组，标记第一个相机坐标系中有效的点
        where2 = ori_pcl2_2drect[-1, :] > 0
         # 创建布尔数组，标记第二个相机坐标系中的有效点
        where3 = ori_pcl3_2drect[-1, :] > 0
         # 将齐次坐标转换为非齐次坐标（去掉最后一个维度）
        ori_pcl2 = visual.fromhomo(ori_pcl2_2drect)
        ori_pcl3 = visual.fromhomo(ori_pcl3_2drect)
        
         # 对于第二个相机，检查点是否在图像边界内
        where2 &= (ori_pcl2[0, :] >= 0) & (ori_pcl2[1, :] >= 0) & (ori_pcl2[0, :] < self.img2.shape[1]) & (ori_pcl2[1, :] < self.img2.shape[0])
         # 对于第三个相机，检查点是否在图像边界内
        where3 &= (ori_pcl3[0, :] >= 0) & (ori_pcl3[1, :] >= 0) & (ori_pcl3[0, :] < self.img3.shape[1]) & (ori_pcl3[1, :] < self.img3.shape[0])
        pcl2_2d=ori_pcl2[:,where2]
        pcl3_2d=ori_pcl3[:,where3]
        pcl2_2drect=ori_pcl2_2drect[:,where2]
        pcl3_2drect=ori_pcl3_2drect[:,where3]
        pcl2_label=self.label[where2]
        black_background = np.zeros_like(self.img2)

        for i in range(pcl2_2d.shape[1]):
            depth = pcl2_2drect[2, i]
            bgr_color = self.color_maping[pcl2_label[i]]
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
            cv2.circle(
                black_background,
                (int(np.round(pcl2_2d[0, i])), int(np.round(pcl2_2d[1, i]))),
                2,
                color=tuple(rgb_color),
                thickness=-1,
            )
        cv2.imshow("projection", black_background)
        cv2.waitKey(0)
        return img2
    
    
    def show_velo_on_img_with_intensity(self):
                 # 将激光雷达坐标转换为齐次坐标
        pcl_rect = self.calib_mat @ visual.tohomo(self.velo[:3, :])
        # 使用相机 P2 矩阵将点云转换到第一个相机坐标系
        ori_pcl2_2drect = self.correct_calib['P2'] @ pcl_rect
        # 使用相机 P3 矩阵将点云转换到第二个相机坐标系
        ori_pcl3_2drect = self.correct_calib['P3'] @ pcl_rect
        # 创建布尔数组，标记第一个相机坐标系中有效的点
        where2 = ori_pcl2_2drect[-1, :] > 0
         # 创建布尔数组，标记第二个相机坐标系中的有效点
        where3 = ori_pcl3_2drect[-1, :] > 0
         # 将齐次坐标转换为非齐次坐标（去掉最后一个维度）
        ori_pcl2 = visual.fromhomo(ori_pcl2_2drect)
        ori_pcl3 = visual.fromhomo(ori_pcl3_2drect)
        
         # 对于第二个相机，检查点是否在图像边界内
        where2 &= (ori_pcl2[0, :] >= 0) & (ori_pcl2[1, :] >= 0) & (ori_pcl2[0, :] < self.img2.shape[1]) & (ori_pcl2[1, :] < self.img2.shape[0])
         # 对于第三个相机，检查点是否在图像边界内
        where3 &= (ori_pcl3[0, :] >= 0) & (ori_pcl3[1, :] >= 0) & (ori_pcl3[0, :] < self.img3.shape[1]) & (ori_pcl3[1, :] < self.img3.shape[0])
        pcl2_2d=ori_pcl2[:,where2]
        pcl3_2d=ori_pcl3[:,where3]
        pcl2_2drect=ori_pcl2_2drect[:,where2]
        pcl3_2drect=ori_pcl3_2drect[:,where3]
        pcl2_intensity=self.velo[3,where2]
        black_background = np.zeros_like(self.img2)
        cmap = plt.cm.get_cmap("hsv", 256) #YlGn
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        for i in range(pcl2_2d.shape[1]):
            color=cmap[int(pcl2_intensity[i]*255),:]
            cv2.circle(
                black_background,
                (int(np.round(pcl2_2d[0, i])), int(np.round(pcl2_2d[1, i]))),
                2,
                color=tuple(color),
                thickness=-1,
            )
        cv2.imshow("projection", black_background)
        cv2.waitKey(0)
        return img2
        
        
if __name__ == '__main__':
    dataset_local_dir=   base_path = "/home/public/DataSetLarge/SemanticKitti/dataset/sequences/"
    img2_file = os.path.join(dataset_local_dir, '09/image_2/000400.png')
    img3_file = os.path.join(dataset_local_dir, '09/image_3/000400.png')
    label_file = os.path.join(dataset_local_dir, '09/labels/000400.label')
    velo_file = os.path.join(dataset_local_dir, '09/velodyne/000400.bin')
    calib_file = os.path.join(dataset_local_dir, '09/calib.txt')
    visualizer = SemantickittiVisualizer(velo_file, calib_file, img2_file, img3_file,label_file)
    img2=visualizer.show_velo_on_img()
    # img2=visualizer.show_velo_on_img_with_label()
    # img2=visualizer.show_velo_on_img_with_intensity()