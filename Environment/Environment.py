import math
import statistics
import cv2
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation
from Utils.IO import *
from Utils.Algo import *
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

def pcd2d_to_3d(pcd_2d, num_rows=5):
    num_points = np.shape(pcd_2d)[0]
    pcd_3d = np.zeros((num_points * num_rows, 3))
    pcd_3d[:, 1:] = np.repeat(pcd_2d, num_rows, axis=0)
    x = np.linspace(-0.2, 0.2, num_rows).reshape((-1, 1))
    xx = np.repeat(x, num_points, axis=1)
    # weights_diag = np.diag(np.linspace(0.0001, -0.0001, num_rows))
    weights_diag = np.diag(np.linspace(0, 0, num_rows))
    idx = np.arange(num_points)
    idx_m = np.repeat(idx.reshape((-1, 1)).T, num_rows, axis=0)
    xx = xx + np.matmul(weights_diag, idx_m)
    pcd_3d[:, 0] = np.reshape(xx.T, (-1,))
    return pcd_3d

class Env_Type(Enum):
    Levelground = 0
    Upstair = 1
    DownStair = 2
    Upslope = 3
    Downslope = 4
    Obstacle = 5
    Unknown = 6

class Environment:
    def __init__(self):
        self.type_pred_from_nn = Env_Type.Levelground
        self.type_pred_buffer = np.zeros(10, dtype=np.uint64)
        self.pcd_2d = np.zeros([0, 2])
        self.pcd_thin = np.zeros([0, 2])
        self.img_binary = np.zeros((100, 100)).astype('uint8')
        self.R_world_imu = np.identity(3)
        self.R_world_camera = np.identity(3)
        self.R_world_body = np.identity(3)
        self.R_imu_camera = Rotation.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
    
    def pcd_to_binary_image(self, pcd, imu):
        eular = imu[0:3]
        # imu在世界坐标系下的姿态
        self.R_world_imu = Rotation.from_euler('xyz', [eular[0], eular[1], eular[2]],
                                               degrees=True).as_matrix()
        # camera 在世界坐标系下的姿态
        self.R_world_camera = np.matmul(self.R_world_imu, self.R_imu_camera)

        # body 在世界坐标系下的姿态
        """
        0 — — —>z
        |
        |               [][][][]
        V y             []
                        []
                  [][][][]
                  []
            [][][][]
        """
        R_body_imu = Rotation.from_euler('xyz', [eular[0] - 90, 0, 180], degrees=True).as_matrix()

        # camera 在body坐标系下的姿态
        R_body_camera = np.matmul(R_body_imu, self.R_imu_camera)
        
        # pcd 本来在camera坐标系下，现在旋转到body坐标系下
        pcd_body = np.matmul(R_body_camera, pcd.T).T
        
        #取水平方向 -0.1 to 0.1
        chosen_idx = np.logical_and(
            pcd_body[:,0] < 0.02, pcd_body[:, 0]> -0.01 
        )
        pcd_body = pcd_body[chosen_idx,:]
        
        #取深度z 0.05 to 1
        chosen_idx = np.logical_and(
            pcd_body[:, 2]>0.05, pcd_body[:, 2]<1
        )
        pcd_body = pcd_body[chosen_idx, :]

        chosen_y = pcd_body[:, 1]
        chosen_z = pcd_body[:, 2]


        # 为了方便调试，self.pcd3d x轴向前，y轴向左，z轴向上
        self.pcd_3d = np.zeros_like(pcd_body)
        self.pcd_3d[:,0] = pcd_body[:,2]
        self.pcd_3d[:,1] = -pcd_body[:,0]
        self.pcd_3d[:,2] = -pcd_body[:,1]
    

        self.img_binary = np.zeros((100, 100)).astype('uint8')
        if self.pcd_3d.any():
            self.pcd_2d = self.pcd_3d[:, 0::2]
            y_max = np.max(chosen_y)
            z_min = np.min(chosen_z)

            chosen_y = chosen_y+(0.99-y_max)
            chosen_z = chosen_z+(0.01-z_min)
            chosen_idx = np.logical_and(np.logical_and(chosen_y > 0, chosen_y < 1),
                                        np.logical_and(chosen_z > 0, chosen_z < 1))
            chosen_y = chosen_y[chosen_idx]
            chosen_z = chosen_z[chosen_idx]
            pixel_y = np.floor(100 * chosen_y).astype('int')
            pixel_z = np.floor(100 * chosen_z).astype('int')
            
            self.img_binary[pixel_y.tolist(),pixel_z.tolist()]=255
        else:
            self.pcd_2d = np.zeros([100, 2])
    
    def thin(self):
        nb1 = NearestNeighbors(n_neighbors=10, algorithm="auto")
        nb1.fit(self.pcd_2d)
        _, idx = nb1.kneighbors(self.pcd_2d)
        if np.shape(idx)[0] > 0:
            self.pcd_thin = np.mean(self.pcd_2d[idx, :], axis=1)
            xmin = np.min(self.pcd_thin[:,0])
            idx_chosen = np.where(self.pcd_thin[:, 0] - xmin < 1)[0]
            self.pcd_thin = self.pcd_thin[idx_chosen, :]
            mean_x = np.mean(self.pcd_thin[:, 0])
            sigma_x = np.std(self.pcd_thin[:, 0])
            mean_y = np.mean(self.pcd_thin[:, 1])
            sigma_y = np.std(self.pcd_thin[:, 1])
            idx_remove = np.logical_and(np.abs(self.pcd_thin[:, 0] - mean_x) > 3 * sigma_x,
                                    np.abs(self.pcd_thin[:, 1] - mean_y) > 3 * sigma_y)
            self.pcd_thin = np.delete(self.pcd_thin, idx_remove, axis=0)

    def elegant_img(self):
        img = np.zeros((500, 500, 3)).astype('uint8')
        img_binary_copy = np.copy(self.img_binary)
        img[:, :, 0] = cv2.resize(img_binary_copy, (500, 500))
        img[:, :, 1] = cv2.resize(img_binary_copy, (500, 500))
        img[:, :, 2] = cv2.resize(img_binary_copy, (500, 500))
        return img

        