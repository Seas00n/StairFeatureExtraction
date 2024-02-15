import sys
import time
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
sys.path.append("//")

from Environment.Environment import *
from Environment.feature_extra_new import *
from Environment.alignment_knn import *

from Utils.IO import fifo_data_vec
from Utils.Plot import FastPlotCanvas

experiment_idx = 4
moca_align_file_path = "/home/yuxuan/Project/StairFeatureExtraction/MocaDataProcess/align_data/"
moca_data = np.load(moca_align_file_path+"Moca_align{}.npy".format(experiment_idx),allow_pickle=True)
idx_align = np.load(moca_align_file_path+"idx_align{}.npy".format(experiment_idx),allow_pickle=True)
idx_stair = np.arange(2, 2 + 3 * 8)
idx_cam = np.arange(26, 29)
idx_knee = np.arange(29, 32)
idx_ankle = np.arange(32, 35)
idx_heel = np.arange(35, 38)
idx_toe = np.arange(38, 41)

vio_save_path = "/media/yuxuan/My Passport/VIO_Experiment/vsMoca/{}/".format(experiment_idx)
result_save_path = "/media/yuxuan/My Passport/VIO_Experiment/Result2/"
cam_origin = moca_data[0, idx_cam]
stair = np.mean(moca_data[:, idx_stair], axis=0)
stairs_x = stair[0::3]
stairs_z = stair[2::3]
leg_pose = moca_data[:, idx_knee[0]:]
scio.savemat(vio_save_path + "appendix_{}.mat".format(experiment_idx),
             {"cam_origin": cam_origin,
              "stair_x": stairs_x,
                "stair_z": stairs_z,
              "leg_pose": leg_pose}
             )

camera_dx_buffer = []
camera_dy_buffer = []
camera_x_buffer = []
camera_y_buffer = []
camera_x_o3d_buffer = []
camera_y_o3d_buffer = []
camera_dx_o3d_buffer = []
camera_dy_o3d_buffer = []
kalman_x_buffer = []
kalman_y_buffer = []
time_buffer = []
pcd_os_buffer = [[], []]
alignment_flag_buffer = []
alignment_flag_o3d_buffer = []

if __name__ == "__main__":
    plt.ion()
    plt.show(block=False)
    fast_plot_ax = FastPlotCanvas()
    env = Environment()
    align_fail_time = 0
    align_fail_time_o3d = 0
    start_idx = idx_align[1]
    end_idx = idx_align[3]
    try:
        for i in range(start_idx, end_idx):
            print("----------------------------Frame[{}]------------------------".format(i))
            print("load binary image and pcd to process")
            pcd_data = np.load(vio_save_path + "{}_pcd.npy".format(i), allow_pickle=True)
            pcd_data = pcd_data[0:-1, :]
            imu_data = np.load(vio_save_path + "{}_imu.npy".format(i), allow_pickle=True)
            eular_angle = imu_data[7:10]
            time_data = np.load(vio_save_path + "{}_time.npy".format(i), allow_pickle=True)
            env.pcd_to_binary_image(pcd_data, eular_angle)
            env.thin()
            if i == start_idx:
                camera_dx_buffer.append(0)
                camera_dy_buffer.append(0)
                camera_x_buffer.append(0)
                camera_y_buffer.append(0)
                camera_dx_o3d_buffer.append(0)
                camera_dy_o3d_buffer.append(0)
                camera_x_o3d_buffer.append(0)
                camera_y_o3d_buffer.append(0)
                pcd_os = pcd_feature_extract_system(pcd_new=env.pcd_thin)
                env.type_pred_from_nn = Env_Type.Upstair.value
                pcd_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn)
                pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_os)
                time_buffer.append(time_data)
                alignment_flag_buffer.append(0)
                alignment_flag_o3d_buffer.append(0)
            else:
                pcd_pre_os = pcd_os_buffer[-1]  # type: pcd_feature_extract_system
                pcd_pre = pcd_pre_os.pcd_new
                pcd_new, pcd_new_os = env.pcd_thin, pcd_feature_extract_system(env.pcd_thin)
                env.type_pred_from_nn = Env_Type.Upstair.value
                pcd_new_os.get_fea(_print_=True, nn_class=env.type_pred_from_nn, ax=None)
                pcd_os_buffer = fifo_data_vec(pcd_os_buffer, pcd_new_os)

                fea_to_align_new, fea_to_align_pre, flag_fea_extrafail = align_fea(pcd_new=pcd_new_os,
                                                                                   pcd_pre=pcd_pre_os,
                                                                                   _print_=True)
                xmove, ymove = 0, 0
                flag_fea_alignfail = 1
                try:
                    if flag_fea_extrafail == 0:
                        t0 = datetime.datetime.now()
                        xmove, ymove = icp_knn(pcd_s=fea_to_align_pre,
                                               pcd_t=fea_to_align_new)
                        t1 = datetime.datetime.now()
                        print("#################=====FeatureAlign:{}=====#########".format(
                            (t1 - t0).total_seconds() * 1000))
                        flag_fea_alignfail = 0
                except Exception as e:
                    print("align method exception:{}".format(e))
                alignment_flag_buffer.append(flag_fea_alignfail)

                use_pre_move = True
                if flag_fea_alignfail == 1 or abs(xmove) > 0.05 or abs(ymove) > 0.05:  # 0.1 0.22
                    align_fail_time += 1
                    if use_pre_move:
                        if align_fail_time > 5 or env.type_pred_from_nn == 0:
                            xmove = 0
                            ymove = 0
                        else:
                            xmove_pre = camera_dx_buffer[-1]
                            ymove_pre = camera_dy_buffer[-1]
                            xmove = xmove_pre  # 0
                            ymove = ymove_pre  # 0
                            print("对齐失败，使用上一次的xmove_pre = {},ymove_pre = {}".format(xmove_pre, ymove_pre))
                    else:
                        xmove, ymove = 0, 0
                        print("对齐失败，xmove, ymove = 0")
                else:
                    align_fail_time = 0
                print("当前最终xmove = {}, ymove = {}".format(xmove, ymove))
                camera_dx_buffer.append(xmove)
                camera_dy_buffer.append(ymove)
                camera_x_buffer.append(camera_x_buffer[-1] + xmove)
                camera_y_buffer.append(camera_y_buffer[-1] + ymove)

                if len(camera_x_buffer)>2:
                    pcd_new_os.show_fast(fast_plot_ax, 'new', id=int(i), p_text=[0.2, 0.3],
                                             p_pcd=[camera_x_buffer[-1], camera_y_buffer[-1]], downsample=8)
                    # pcd_new_os.show_fast(fast_plot_ax, 'pre', id=int(i), p_text=[0.2, 0.3],
                    #                          p_pcd=[camera_x_buffer[-2], camera_y_buffer[-2]], downsample=8)
                    fast_plot_ax.set_camera_traj(np.array(camera_x_buffer), np.array(camera_y_buffer),
                                                     prediction_x=None, prediction_y=None)

                    fast_plot_ax.update_canvas()
                # time.sleep(0.1)
    except KeyboardInterrupt:
        pass
