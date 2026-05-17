import torch
# 神经网络层、损失函数、优化器等
import torch.nn as nn
import torch.optim as optim
# 计算机视觉工具包：数据集、数据预处理、模型
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 绘图工具：用于显示图片
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用Windows自带的黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号变成方框的问题
# 进度条工具：训练时显示进度
from tqdm import tqdm
# 数值计算
import numpy as np

# ===================== 2. 配置运行设备（自动选择GPU/CPU） =====================
# 如果电脑有NVIDIA显卡，用CUDA加速训练；否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"模型将在: {device} 上运行")
