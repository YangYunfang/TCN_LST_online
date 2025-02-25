# 在Server基础上结合LSTM模型进行训练再将数据输出到Labview
import socket
import threading
import keyboard
import time
import sys
import math
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.optimizers import Adam
import h5py
import time
# def stair_function(x, y, data_list, terrain_kind):
#     if terrain_kind > 0:
#         if x < data_list[6, 0]:
#             terrain = y
#         elif data_list[6, 0] <= x < data_list[7, 0]:
#             terrain = y - data_list[6][1]
#         elif data_list[7, 0] <= x < data_list[8, 0]:
#             terrain = y - data_list[7, 1]
#         elif data_list[8, 0] <= x < data_list[9, 0]:
#             terrain = y - data_list[8, 1]
#         elif data_list[9, 0] <= x < data_list[10, 0]:
#             terrain = y - data_list[9, 1]
#         elif x >= data_list[10, 0]:
#             terrain = y - data_list[10, 1]
#     elif terrain_kind < 0:
#         if x > data_list[6, 0]:
#             terrain = y
#         elif data_list[7, 0] <= x < data_list[6, 0]:
#             terrain = y - data_list[6][1]
#         elif data_list[8, 0] <= x < data_list[7, 0]:
#             terrain = y - data_list[7, 1]
#         elif data_list[9, 0] <= x < data_list[8, 0]:
#             terrain = y - data_list[8, 1]
#         elif data_list[10, 0] <= x < data_list[9, 0]:
#             terrain = y - data_list[9, 1]
#         elif x < data_list[10, 0]:
#             terrain = y - data_list[10, 1]
#     return terrain
#
#
# def slope_function(x, y, data_list, terrain_kind):
#     slope, intercept = np.ployfit(data_list[6:9, 0], data_list[:, 1], deg=1)
#     value = slope * x + intercept
#     if terrain_kind > 0:
#         if x < data_list[6, 0]:
#             terrain = y
#         elif data_list[6, 0] <= x < data_list[10, 0]:
#             terrain = y - value
#         elif x >= data_list[10, 0]:
#             terrain = y - data_list[10, 1]
#     if terrain_kind < 0:
#         if x >= data_list[6, 0]:
#             terrain = y
#         elif data_list[10, 0] <= x < data_list[6, 0]:
#             terrain = y - value
#         elif x < data_list[10, 0]:
#             terrain = y - data_list[10, 1]
#     return terrain


def terrain_function(x, data_list, terrain_kind):
    if terrain_kind == 2:
        slope, intercept = np.ployfit(data_list[6:9, 0], data_list[:, 1], deg=1)
        value = slope * x + intercept
        if x < data_list[6, 0]:
            terrain = 0
        elif data_list[6, 0] <= x < data_list[10, 0]:
            terrain = value
        elif x >= data_list[10, 0]:
            terrain = data_list[10, 1]
    elif terrain_kind == -2:
        slope, intercept = np.ployfit(data_list[6:9, 0], data_list[:, 1], deg=1)
        value = slope * x + intercept
        if x >= data_list[6, 0]:
            terrain = 0
        elif data_list[10, 0] <= x < data_list[6, 0]:
            terrain = value
        elif x < data_list[10, 0]:
            terrain = data_list[10, 1]
    elif terrain_kind == 1:
        if x < data_list[6, 0]:
            terrain = 0
        elif data_list[6, 0] <= x < data_list[7, 0]:
            terrain = data_list[6, 1]
        elif data_list[7, 0] <= x < data_list[8, 0]:
            terrain = data_list[7, 1]
        elif data_list[8, 0] <= x < data_list[9, 0]:
            terrain = data_list[8, 1]
        elif data_list[9, 0] <= x < data_list[10, 0]:
            terrain = data_list[9, 1]
        elif x >= data_list[10, 0]:
            terrain = data_list[10, 1]
    elif terrain_kind == -1:
        if x > data_list[6, 0]:
            terrain = 0
        elif data_list[7, 0] <= x < data_list[6, 0]:
            terrain = data_list[6][1]
        elif data_list[8, 0] <= x < data_list[7, 0]:
            terrain = data_list[7, 1]
        elif data_list[9, 0] <= x < data_list[8, 0]:
            terrain = data_list[8, 1]
        elif data_list[10, 0] <= x < data_list[9, 0]:
            terrain = data_list[9, 1]
        elif x < data_list[10, 0]:
            terrain = data_list[10, 1]
    elif terrain_kind == 0:
        terrain = 0
    return terrain


def handle_client(client_socket, other_device_socket):
    # global Toe_Info  # 脚尖信息z轴的位置和速度
    global com_Info
    global Knee_Info
    global save_list
    # global vel_com
    # global send_data
    # global Fixed_Array
    # com = 0   # 质心原始位置

    origin = 0  # 起点位置设置
    number = 0
    time1 = 0
    try:
        while True:
            start_time = time.perf_counter()
            data = client_socket.recv(90).decode('utf-8')
            time2 = time.perf_counter()
            time_sec = time2 - time1
            # # 将字符串拆分为数组元素
            # data_list = [float(value.strip()) for value in data.replace(' ', ',').replace('\n', ',').split(',')
            #              if value.strip()]
            # 将字符串分割成行
            lines = data.strip().split('\n')
            # lines = lines[:-1]
            # 对每一行进行处理
            data_array = [list(map(float, line.split(','))) for line in lines]

            # 将列表转换为NumPy数组
            data_list = np.array(data_array)
            # print(data_list)
            if not data:
                break
            if number == 0:
                origin = data_list[4, 0]
            if Terrain == -1 or Terrain == -2:
                data_list[:, 0] = origin - data_list[:, 0]
            else:
                data_list[:, 0] = data_list[:, 0] - origin

            if not np.any(data_list == 0) or not number:
                # 除地形外其余特征
                Features[0, 0] = (data_list[0, 0] + data_list[1, 0]) / 2000  # 质心y位置
                com = (data_list[0, 1] + data_list[1, 1]) / 2  # 质心z原始高度
                com_terrain = terrain_function(Features[0, 0], data_list, Terrain)  # 质心对应的地形
                Features[0, 1] = (com - com_terrain) / Height  # 标准化后质心z位置
                knee_terrain = terrain_function(data_list[2, 0], data_list, Terrain)  # 膝关节对应的地形
                Features[0, 2:4] = [data_list[2, 0]/1000, (data_list[2, 1] - knee_terrain) / Height]
                vel_com = np.array(([Features[0, 0], com] - com_Info) * 100)
                Features[0, 4] = Features[0, 0] + vel_com[0, 0] * np.sqrt(Leg_Length / 9.8)
                Features[0, 5] = Features[0, 1] + vel_com[0, 1] * np.sqrt(Leg_Length / 9.8) / Height
                vel_knee = (data_list[2, :] - Knee_Info[0, :]) / 10
                Features[0, 6] = Mass * (0.5 * (np.sum(np.square(vel_knee)) - np.sum(np.square(Knee_Info[1, :]))) +
                                         9.8 * (data_list[2, 1] - Knee_Info[0, 1]) / 1000)

                # vel_toe = (data_list[4][1] - Toe_Info[0, 0]) / 10
                # a_toe = (vel_toe - Toe_Info[1, 0]) * 100
                Knee_Info = np.vstack((data_list[2, :], vel_knee))
                com_Info = np.array([[Features[0, 0], com]])
                # Toe_Info = np.vstack([data_list[4, 1], vel_toe])
                leg_angle = math.atan2(data_list[1, 0] - data_list[2, 0], data_list[1, 1] - data_list[2, 1])
                knee_angle = math.atan2(data_list[2, 0] - data_list[3, 0], data_list[2, 1] - data_list[3, 1])
                angle_leg = math.degrees(leg_angle)
                angle_knee = math.degrees(knee_angle)

                # 判断时相
                heel_height = data_list[4, 1] - terrain_function(data_list[4, 0] + 30, data_list, Terrain)

                # 模型特征计算
                if Terrain == 0:  # 水平行走
                    Features[0, 7] = min(data_list[4, 1], data_list[5, 1])

                elif Terrain == 1 or Terrain == -1:
                    Features[0, 7] = min(data_list[4, 1] - terrain_function(data_list[4, 0], data_list, Terrain),
                                         data_list[5, 1] - terrain_function(data_list[5, 0], data_list, Terrain))
                elif Terrain == 2 or Terrain == -2:
                    Features[0, 7] = min(data_list[4, 1] - terrain_function(data_list[4, 0], data_list, Terrain),
                                         data_list[5, 1] - terrain_function(data_list[5, 0], data_list, Terrain))

                # Fixed_Array = np.roll(Fixed_Array, shift=-1, axis=0)
                # Fixed_Array[-1, :] = Features
                features = (Features - muX) / sigmaX
                data = features[None, :, :]

                # 进行预测
                pred_model = model(data)[0, 0]
                pred = pred_model * sigmaY + muY
                # send_data = [pred-angle_leg, a_toe]
                data_array = np.array([pred - angle_leg, heel_height, angle_knee - angle_leg], dtype=np.float32)

                send_str = ';'.join(['{:.2f}'.format(value) for value in data_array])
                # send_str = ';'.join(['{:.2f}'.format(value) for value in send_data])
                send_fix = send_str.ljust(18).encode('utf-8')

            # 发送处理后的数据到其他设备
            # sys.stdout.flush()  # 强制刷新缓冲区

            other_device_socket.send(send_fix)
            # start_time = time.perf_counter()
            save_list.append(np.array([leg_angle, pred_model, data_list[2, 0], data_list[2, 1], data_list[3, 0],
                                       data_list[3, 1], features[0, 0], features[0, 1], features[0, 2], features[0, 3],
                                       features[0, 4], features[0, 5], features[0, 6], features[0, 7]]))
            # save_list.append(np.array([angle_leg, time_sec, data_list[2, 0],
            #                            data_list[2, 1], data_list[3, 0], data_list[3, 1]]))
            time1 = time2
            end_time = time.perf_counter()
            # print(start_time-end_time)
            number = number + 1
            print(start_time - end_time, number)

    except ConnectionAbortedError:
        print("连接中断")
    finally:
        save_data = np.stack(save_list)
        np.save('C:\\Users\\17543\\Desktop\\yangdata\\NPY\\00123night_1.npy', save_data)
        client_socket.close()


# 主函数
# 个人参数的更改
Mass = 75 * 5.7 / 100  # 体重/Kg, 男生5.7 女生：6.1，暂时不以百分数乘
Height = 1820  # 身高/mm
Leg_Length = 1.0  # 腿长/m
Terrain = 1  # 地形限制, 0:平地，1:上楼梯，-1:下楼梯, 2：上斜坡， -2：下斜坡

# 初始化
Features = np.zeros((1, 8))  # 脚尖信息z轴的位置和速度
# Toe_Info = np.zeros((2, 1))
com_Info = np.zeros((1, 2))
Knee_Info = np.zeros((2, 2))  # 速度和位置
send_fix = ''.ljust(18).encode('utf-8')
save_list = []

# 停止进程
stop = 0
# Fixed_Array = np.zeros((200, 8))

# 导入onnx模型， 简单predict预测不改变模型的参数, 要固定一个长度的数据，保留之前的数据
# 加载模型参数
# start_time = time.perf_counter()
# # sess = rt.InferenceSession("Lstm_Simplified.onnx")
# input_name = sess.get_inputs()[0].name

with h5py.File('LSTM_Params_V3.mat', 'r') as file:
    lstmInputWeights1 = np.array(file.get('lstmInputWeights1'))
    lstmRecurrentWeights1 = np.array(file.get('lstmRecurrentWeights1'))
    lstmBias1 = np.reshape(np.array(file.get('lstmBias1')), (400,))

    lstmInputWeights2 = np.array(file.get('lstmInputWeights2'))
    lstmRecurrentWeights2 = np.array(file.get('lstmRecurrentWeights2'))
    lstmBias2 = np.reshape(np.array(file.get('lstmBias2')), (400,))

    fcWeights = np.array(file.get('fcWeights'))
    fcBias = np.reshape(np.array(file.get('fcBias')), (1,))
# with h5py.File('Test_1.mat', 'r') as file:
#     # 读取double数据集
#     double_data1 = file['ans'][0:1, :]
#     double_data2 = file['ans'][100:101, :]

# 构建LSTM模型并设置权重
model = Sequential()

model.add(LSTM(100, batch_input_shape=(1, None, 8), return_sequences=True, stateful=True))
model.layers[0].set_weights((lstmInputWeights1, lstmRecurrentWeights1, lstmBias1))
model.add(Dropout(0.2))
model.add(LSTM(100, stateful=True))
model.layers[2].set_weights([lstmInputWeights2, lstmRecurrentWeights2, lstmBias2])
model.add(Dropout(0.2))
model.add(Dense(1))
model.layers[4].set_weights([fcWeights, fcBias])
# 编译模型
model.compile(optimizer='adam', loss='mse')

# 标准化参数
muY = 14.9002
sigmaY = 15.6916
muX = np.array([[2.0123, 0.5570, 2.0385, 0.2833, 2.1391, 0.5570, 5.6007e-05, 40.6117]])
sigmaX = np.array([[1.3809, 0.0248, 1.3863, 0.0240, 1.3915, 0.0383, 1.4690, 29.4007]])

# 创建TCP套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定地址和端口
server_address = ('0.0.0.0', 1234)
server_socket.bind(server_address)

# 监听连接
server_socket.listen(1)

print('等待连接...')

# 连接到 "other device"，只打开一次
other_device_address = ('127.0.0.1', 5678)
other_device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
other_device_socket.connect(other_device_address)

try:
    while True:
        # 接受连接
        client_socket, client_address = server_socket.accept()
        print('连接Labview成功')

        # 启动一个新线程来处理客户端连接
        client_handler = threading.Thread(target=handle_client,
                                          args=(client_socket, other_device_socket))
        client_handler.start()
except KeyboardInterrupt:
    print('服务中断')
finally:
    client_handler.join()
    server_socket.close()
    other_device_socket.close()
