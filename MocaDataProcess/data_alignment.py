import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import PIL
import os
import scipy.interpolate as scio

#先运行moca_postprocess程序，得到Moca_smooth.npy
# 1 4 8 9 10 11 12 14
idx_exp = 4
file_path = "/media/yuxuan/My Passport/VIO_Experiment/vsMoca/{}/".format(idx_exp)
moca_data = np.load(file_path + "Moca{}_smooth.npy".format(idx_exp))
save_path = "/home/yuxuan/Project/StairFeatureExtraction/MocaDataProcess/align_data/"

#按顺序读取的文件名被存储在timestr中
timeorder_file_path = "timestr{}.npy".format(idx_exp)
#对齐序号被存储在idx_align中
align_file_path = "idx_align{}.npy".format(idx_exp)
#对齐数据被存储在Moca_align中
moca_align_file_path = "Moca_align{}.npy".format(idx_exp)

plt.ion()


#加载动捕数据
time_moca = moca_data[:, 1]-moca_data[0, 1]
frame_moca = moca_data[:, 0]

idx_stair = np.arange(2, 2 + 3 * 8)
idx_cam = np.arange(26, 29)
idx_ankle = np.arange(29, 32)
idx_heel = np.arange(32, 35)
idx_toe = np.arange(35, 38)
ankle = moca_data[:, idx_ankle]
heel = moca_data[:, idx_heel]
toe = moca_data[:, idx_toe]

#加载实验数据
num_frame_vio = int((len(os.listdir(file_path)) - 2) / 3)



# timestr_unsort_buffer = []
# time_unsorted_buffer = []
# for file in os.listdir(file_path):
#     if file[-8:] == "_imu.npy":
#         timestr_unsort_buffer.append(file[0:-8])
#         time_unsorted_buffer.append(float(file[0:-8]))

# idx_read_order = np.argsort(np.array(time_unsorted_buffer))


# time_buffer = []
# timestr_order_buffer = []
# imu_buffer = []
# pcd_buffer = []

# for i in idx_read_order:
#     name = timestr_unsort_buffer[i]
#     timestr_order_buffer.append(name)
#     time_buffer.append(float(name))
#     imu_buffer.append(np.load(file_path+"{}_imu.npy".format(name)))
#     pcd_buffer.append(np.load(file_path+"{}_pcd.npy".format(name)))


# print("存储排序后的时间名")
# np.save(timeorder_file_path, timestr_order_buffer)

idx_read_order = np.arange(num_frame_vio)+1
time_buffer = []
imu_buffer = []
for i in idx_read_order:
    t = np.load(file_path+"{}_time.npy".format(int(i)),allow_pickle=True).tolist().timestamp()
    time_buffer.append(t)
    imu_buffer.append(np.load(file_path+"{}_imu.npy".format(int(i)),allow_pickle=True))

imu_data = np.array(imu_buffer)
time_vio = np.array(time_buffer)
time_vio -= time_vio[0]
frame_vio = np.arange(0, np.shape(imu_buffer)[0])
acc = np.linalg.norm(imu_data[:,1:4],axis=1)

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
idx_align = []
def on_press(event):
    global idx_align
    global ax1
    if len(idx_align) == 0:
        idx_align.append(int(event.xdata))
        print("动捕起点:{}".format(int(event.xdata)))
        ax1.scatter(event.xdata,event.ydata, linewidths=5)
    elif len(idx_align) == 1:
        idx_align.append(int(event.xdata))
        print("vio起点:{}".format(int(event.xdata)))
        ax2.scatter(event.xdata,event.ydata, linewidths=5)
    elif len(idx_align) == 2:
        idx_align.append(int(event.xdata))
        print("动捕终点:{}".format(int(event.xdata)))
        ax1.scatter(event.xdata,event.ydata, linewidths=5)
    elif len(idx_align) == 3:
        idx_align.append(int(event.xdata))
        print("vio终点:{}".format(int(event.xdata)))
        ax2.scatter(event.xdata,event.ydata, linewidths=5)    


ax1.plot(frame_moca, heel[:,2])
ax2.plot(frame_vio, acc)
fig.canvas.mpl_connect('button_press_event', on_press)

print("依次按下动捕起点，vio起点，动捕终点，vio终点：")
try:
    while(len(idx_align)<4):
        plt.pause(0.5)
except KeyboardInterrupt:
    plt.close()


time_vio_chosen = np.copy(time_vio[idx_align[1]:idx_align[3]]) - time_vio[idx_align[1]]
time_moca_chosen = np.copy(time_moca[idx_align[0]:idx_align[2]])-time_moca[idx_align[0]]
counter_vio = 0
counter_moca = 0
time_align_new = []
tv_idx_in_time_align = []
tm_idx_in_time_align = []
while counter_moca < np.shape(time_moca_chosen)[0] and counter_vio < np.shape(time_vio_chosen)[0]:
    tm = time_moca_chosen[counter_moca]
    tv = time_vio_chosen[counter_vio]
    if tm < tv:
        time_align_new.append(tm)
        counter_moca += 1
        tm_idx_in_time_align.append(len(time_align_new)-1)
    elif tm > tv:
        time_align_new.append(tv)
        counter_vio += 1
        tv_idx_in_time_align.append(len(time_align_new)-1)
    elif tm == tv:
        time_align_new.append(tv)
        counter_vio += 1
        counter_moca += 1
        tv_idx_in_time_align.append(len(time_align_new)-1)
        tm_idx_in_time_align.append(len(time_align_new)-1)
    

time_align_new = np.array(time_align_new)
tm_idx_in_time_align = np.array(tm_idx_in_time_align)
tv_idx_in_time_align = np.array(tv_idx_in_time_align)

if tm_idx_in_time_align[-1] > tv_idx_in_time_align[-1]:
    # vio先关
    chosen = np.where(tm_idx_in_time_align<=tv_idx_in_time_align[-1])[0]
    chosen = np.hstack([chosen, chosen[-1]+1])
    tm_idx_chosen_in_time_align = tm_idx_in_time_align[chosen]
    tv_idx_chosen_in_time_align = tv_idx_in_time_align
    idx_align[2] = np.shape(tm_idx_chosen_in_time_align)[0]+idx_align[0]
    idx_align[3] = np.shape(tv_idx_chosen_in_time_align)[0]+idx_align[1]   
elif tm_idx_in_time_align[-1] < tv_idx_in_time_align[-1]:
    # moca先关
    tv_idx_chosen_in_time_align = tv_idx_in_time_align[np.where(tv_idx_in_time_align<=tm_idx_in_time_align[-1])[0]]
    tm_idx_chosen_in_time_align = tm_idx_in_time_align
    idx_align[2] = np.shape(tm_idx_chosen_in_time_align)[0]+idx_align[0]
    idx_align[3] = np.shape(tv_idx_chosen_in_time_align)[0]+idx_align[1]



data_moca_chosen_in_time_align = np.copy(moca_data[idx_align[0]:idx_align[2],2:])
tx = time_align_new[tm_idx_chosen_in_time_align]
txx = time_align_new[tv_idx_chosen_in_time_align]

print("按回车对对齐后的动捕数据重新插值")

data_moca_new = []
for i in range(np.shape(data_moca_chosen_in_time_align)[1]):
    y = data_moca_chosen_in_time_align[:,i]
    fun = scio.interp1d(tx,y,'linear')
    yy = fun(txx)
    data_moca_new.append(yy)

data_moca_new = np.array(data_moca_new).T
frame = np.arange(0, np.shape(data_moca_new)[0])
data_moca_new = np.hstack([txx.reshape((-1,1)),data_moca_new])
data_moca_new = np.hstack([frame.reshape((-1,1)), data_moca_new])

x = np.arange(0,np.shape(time_align_new[tv_idx_chosen_in_time_align])[0])
ax3.plot(x,acc[idx_align[1]:idx_align[3]])
ax3.plot(x,data_moca_new[:,idx_heel[0]])
plt.show()


input("存储序号按回车：(Ctrl+Z取消)")
np.save(save_path+align_file_path, np.array(idx_align))
np.save(save_path+moca_align_file_path, np.array(data_moca_new))
print("对齐序号已存储")
print("动捕对齐数据已存储")
plt.close()