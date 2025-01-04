import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import ISBI_Loader  # 确保数据加载器正确实现
from model.CNN import SimpleCNN
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试模型
test_img_path = "E:/学校/智慧医疗/大作业/Smart-Healthcare/data/stage_2_test_images"  # 测试数据路径
test_loader = ISBI_Loader(test_img_path)  # 请确保加载器有train参数来区分训练和测试模式
train_loader = torch.utils.data.DataLoader(dataset=test_loader,
                                           batch_size=1,
                                           shuffle=False)
# 加载模型
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('simple_cnn_model.pth'))
model.eval()  # 设置模型为评估模式

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
correct = 0
total = 0

# 存储结果
results = []

with torch.no_grad():  # 不需要梯度计算
    print("begin")
    for images, _, image_paths in tqdm(test_loader):
        images = images.to(device)
        labels = torch.randint(0, 2, (images.size(0),)).to(images.device)  # 这里用随机标签替代，请替换为实际标签

        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        test_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 判断每张图片是否有肺炎
        probabilities = nn.Softmax(dim=1)(outputs)  # 获取输出概率
        for i in range(images.size(0)):
            if probabilities[i][1] >= 0.5:  # 如果预测为有肺炎
                results.append((image_paths, "有肺炎"))
            else:
                results.append((image_paths, "没有肺炎"))

# 计算平均损失和准确率
average_test_loss = test_loss / len(test_loader)
test_accuracy = correct / total

print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 打印结果
for img_path, label in results:
    print(f"Image: {img_path}, Prediction: {label}")