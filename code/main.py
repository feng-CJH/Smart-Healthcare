import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import ISBI_Loader  # 确保数据加载器正确实现
from model.CNN import SimpleCNN
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像路径
img_path = "E:/学校/智慧医疗/大作业/Smart-Healthcare/data/stage_2_train_images"

# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 加载数据
train_loader = ISBI_Loader(img_path)

train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                           batch_size=1,
                                           shuffle=False)
# 训练模型
num_epochs = 8
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, img_path in tqdm(train_loader):  # 确保train_loader返回图像和标签
        images = images.to(device)  # 转移到设备上
        labels = labels.to(device)  # 转移标签到设备上

        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # 计算总损失

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), 'simple_cnn_model.pth')

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='orange')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()