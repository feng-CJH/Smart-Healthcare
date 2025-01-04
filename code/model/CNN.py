import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积，输入通道为1（灰度图像），输出通道为32，卷积核大小为3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 最大池化层
        # 第二层卷积，输入通道为32，输出通道为64，卷积核大小为3x3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 第三层卷积，输入通道为64，输出通道为128，卷积核大小为3x3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # 全连接层，将最后的特征图展平为一维向量
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # 224x224的图像经过池化后变为28x28
        self.fc2 = nn.Linear(256, 2)  # 2个输出，分别代表正常和肺炎

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 第一层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二层卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv3(x)))  # 第三层卷积 + 激活 + 池化
        x = x.view(-1, 128 * 28 * 28)  # 展平
        x = F.relu(self.fc1(x))  # 全连接层 + 激活
        x = self.fc2(x)  # 输出层
        return x