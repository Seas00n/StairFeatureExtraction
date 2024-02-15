import sys

sys.path.append("/home/yuxuan/Project/MPV_2024/")
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from matplotlib.pyplot import MultipleLocator
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D

#实验数据被存储在moca_path中的Moca{idx_exp}.mat文件，变量名为SA1
idx_exp = 14
moca_path = "/media/yuxuan/My Passport/VIO_Experiment/vsMoca/{}/".format(idx_exp)
moca_file = moca_path + "Moca{}.mat".format(idx_exp)
moca_data = scio.loadmat(moca_file)["SA{}".format(idx_exp)]


'''
平滑处理：
1. interp_interval_replace(data, idx)对idx区间上的点根据区间起点终点的数据进行插值替换，适合小范围的数据丢失或者跳变
e.g.
smooth_idx = np.arange(1140, 1485)
toe = interp_interval(toe, smooth_idx)
2. interp_interval_shift(data, shift_idx, shift_vec, smooth_idx)对shift_idx上的点首先进行移动，再对smooth_idx上的点进行插值，适合数据形状正常但是区间上整体漂移的情况
e.g.
shift_idx = [980, 1080]
smooth_idx = np.arange(1140, 1485)
toe = interp_interval(toe, shift_idx,[-200,160,20],smooth_idx)
'''


def interp_interval_replace(data, idx):
    data_start = data[idx[0], :]
    data_end = data[idx[-1], :]
    x = np.array([idx[0], idx[-1]])
    xx = idx
    y = np.array([data_start, data_end])
    f = interpolate.interp1d(x, data[x, :], kind='linear', axis=0)
    data[idx] = f(xx)
    return data


def interp_interval_shift(data, idx_shift, shift_vec,idx_smooth):
    data[np.arange(idx_shift[0], idx_shift[1] + 1), 0] += shift_vec[0]
    data[np.arange(idx_shift[0], idx_shift[1] + 1), 1] += shift_vec[1]
    data[np.arange(idx_shift[0], idx_shift[1] + 1), 2] += shift_vec[2]
    x = np.hstack([idx_smooth[0], np.arange(idx_shift[0], idx_shift[1]), idx_smooth[1]])
    xx = np.arange(idx_smooth[0], idx_smooth[1])
    y = data[x, :]
    f = interpolate.interp1d(x, y, kind="linear", axis=0)
    data[xx] = f(xx)
    for i in range(3):
        data[xx, i] = gaussian_filter1d(data[xx, i], 3)
    return data


plot_3d = True
fig = plt.figure()

frames = moca_data[:, 0]
time = moca_data[:, 1]

idx_stair = np.arange(2, 2 + 3 * 8)
idx_cam = np.arange(26, 29)
idx_ankle = np.arange(29, 32)
idx_heel = np.arange(32, 35)
idx_toe = np.arange(35, 38)

# 2-25
stairs = np.mean(moca_data[:, idx_stair], axis=0)
stairs_x = stairs[0::3]
stairs_y = stairs[1::3]
stairs_z = stairs[2::3]

# 26-28
cam = moca_data[:, idx_cam]

# 29-31
ankle = moca_data[:, idx_ankle]

# 32:34
heel = moca_data[:, idx_heel]

# 35:37
toe = moca_data[:, idx_toe]



if plot_3d:
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([200, 1400])
    ax.set_ylim([-750, 750])
    ax.set_zlim([200, 1000])
    ax.plot3D(stairs_x, stairs_y, stairs_z)
    ax.plot3D(cam[:, 0], cam[:, 1], cam[:, 2])
    ax.plot3D(ankle[:, 0], ankle[:, 1], ankle[:, 2])
    ax.plot3D(heel[:, 0], heel[:, 1], heel[:, 2])
    ax.plot3D(toe[:, 0], toe[:, 1], toe[:, 2])
else:
    ax = fig.add_subplot()
    diff_toe_y = np.gradient(toe[:, 1])
    diff_ankle_y = np.gradient(ankle[:, 1])
    diff_heel_y = np.gradient(heel[:, 1])
    ax.plot(frames, ankle)
    ax.plot(frames, diff_ankle_y)
plt.show()

'''
后处理后的数据被替换为Moca{}_smooth.npy
'''
moca_smooth_path = moca_path + "Moca{}_smooth.npy".format(idx_exp)
input(moca_file+"将被替换为"+moca_smooth_path+"(回车确认):")
moca_data[:, idx_toe] = toe
moca_data[:, idx_heel] = heel
moca_data[:, idx_ankle] = ankle
np.save(moca_smooth_path, moca_data)