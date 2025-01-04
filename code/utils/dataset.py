import pydicom
import numpy as np
import os
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import pandas as pd
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.dcm'))
        self.labels_df = pd.read_csv('E:\学校\智慧医疗\大作业\Smart-Healthcare\data/stage_2_train_labels.csv')

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        search_patient_id = img_path.split("\\")[-1].split(".")[0]
        # 查找对应的Target值
        label = self.labels_df.loc[self.labels_df ['patientId'] == search_patient_id, 'Target']
        if label.empty:
            tag = 0
        else:
            tag = label.values[0]
        dcm_image = pydicom.dcmread(img_path)
        image_array = dcm_image.pixel_array
        # 归一化
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

        # 转换为PIL图像
        image = Image.fromarray((image_array * 255).astype(np.uint8))

        # 数据增强和调整大小
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整为224x224
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
        ])

        image = transform(image)
        # 这个部分如果是测试模型时，需要返回个路径作用于结果输出
        return image, tag, img_path



if __name__ == "__main__":
    isbi_dataset = ISBI_Loader(r"E:\学校\智慧医疗\大作业\Smart-Healthcare\data\stage_2_train_images")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=False)
    for image, labels, img_path in train_loader:
        print(image.shape)
