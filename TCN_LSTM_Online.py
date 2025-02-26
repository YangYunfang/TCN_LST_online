import argparse
import time
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import hdf5storage
from generator_data import generator_data2


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, outputs, pre_len, num_channels, n_layers, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.pre_len = pre_len
        self.n_layers = n_layers
        self.hidden_size = num_channels[-2]
        self.hidden = nn.Linear(num_channels[-1], num_channels[-2])
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, n_layers, bias=True,
                            batch_first=True)  # output (batch_size, obs_len, hidden_size)
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-2], outputs)

    def forward(self, x):
        self.lstm.flatten_parameters()  # 解决warning问题
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)

        batch_size, obs_len, features_size = x.shape  # (batch_size, obs_len, features_size)
        xconcat = self.hidden(x)  # (batch_size, obs_len, hidden_size)
        H = torch.zeros(batch_size, obs_len - 1, self.hidden_size).to(device)  # (batch_size, obs_len-1, hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(
            device)  # (num_layers, batch_size, hidden_size)
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)  # (batch_size, 1, hidden_size)
            out, (ht, ct) = self.lstm(xt, (ht, ct))  # ht size (num_layers, batch_size, hidden_size)
            htt = ht[-1, :, :]  # (batch_size, hidden_size)
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)  # (batch_size, obs_len-1, hidden_size)

        x = self.linear(H)
        # return x[:, -self.pre_len:, :]
        return x[:, :, :]


def predict_result(model, args, device, scaler):  # 没有padding的测试集预测
    # 预测未知数据的功能
    pre_data_x, pre_data_y, scaler = generator_data2(args.choice, 'train')
    for i in range(len(pre_data_x)):
        tensor_pred = torch.FloatTensor(pre_data_x[i]).to(device)
        tensor_pred = tensor_pred.unsqueeze(0)  # 单次预测 , 滚动预测功能暂未开发后期补上
        # model = model
        # model.load_state_dict(torch.load('save_model_TZH.pth'))
        model.eval()  # 评估模式

        pred = model(tensor_pred)[0, :, 0]
        pred = pred.detach().cpu().numpy() * scaler[1] + scaler[0]
        label = pre_data_y[i][1:] * scaler[1] + scaler[0]
        # 绘制历史数据
        plt.plot(label, label='True')

        # 绘制预测数据
        plt.plot(pred, label='Prediction')

        # 添加标题和图例
        plt.title("True and Predictions")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='TCN-LSTM', help="模型持续更新")
    parser.add_argument('-window_size', type=int, default=126, help="时间窗口大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=1, help="预测未来数据长度")
    # data
    parser.add_argument('-choice', type=str, default='all', help="选取的顺序")
    parser.add_argument('-input_size', type=int, default=9, help='你的特征个数不算时间那一列')
    parser.add_argument('-feature', type=str, default='MS', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
    parser.add_argument('-model_dim', type=list, default=[64, 64, 64], help='这个地方是这个TCN卷积的关键部分,它代表了TCN的层数我这里输'
                                                                              '入list中包含三个元素那么我的TCN就是三层，这个根据你的数据复杂度来设置'
                                                                              '层数越多对应数据越复杂但是不要超过5层')

    # device
    parser.add_argument('-use_gpu', type=bool, default=True)
    parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")
    args = parser.parse_args()

    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    print("使用设备:", device)
    # train_loader, test_loader, valid_loader, scaler = generate_dataloader(args, device)
    scaler = np.array([14.73, 15.46])

    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size

    with open(f'fold_models_{args.choice}.pkl', 'rb') as f:
        fold_models = pickle.load(f)
    model = fold_models[1]
    print("Loaded one fold models from fold_models.pth")

    print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    predict_result(model, args, device, scaler)
    plt.show()

