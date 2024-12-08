
# from datasets.dataset_synapse import RandomGenerator, random_rotate, random_rot_flip
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import cv2

class LiverImageDataset(Dataset):
    def __init__(self, root_dir=None, transform=None, output_size=(512, 512), list_dir=None, split=None):
        self.transform = transform
        self.output_size = output_size
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'liver_masks')
        # 如果指定了list_dir,就按照list_dir中的文件名来读取图像和掩码
        if list_dir is not None:
            if split is not None:
                self.images = [f.strip() for f in open(os.path.join(list_dir, split + '.txt')).readlines()]
        else:
            # 没有指定list_dir,就获取所有图像文件名，假设图像和掩码文件名是一一对应的
            self.images = [f for f in os.listdir(self.image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.images[idx])

        # 使用opencv读取图像和掩码
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        # 调整图像和掩码大小
        image = cv2.resize(image, self.output_size)
        mask = cv2.resize(mask, self.output_size)

        # 对mask进行二值化处理，将大于128的像素值设为1，小于等于128的像素值设为0
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)

        sample = {'image': image, 'label': mask}
        if self.transform:
            sample = self.transform(sample)
        # image[512,512] label[512,512]
        # 添加样本的 case_name（这里使用文件名作为 case_name，可根据实际需求调整）
        sample['case_name'] = self.images[idx].split('.')[0]
        return sample

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #
    #     img_name = os.path.join(self.image_dir, self.images[idx])
    #     mask_name = os.path.join(self.mask_dir, self.images[idx])
    #
    #     # 使用opencv读取图像和掩码
    #     image = cv2.imread(img_name, cv2.IMREAD_COLOR)  # 读取为3通道（RGB）
    #     mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    #
    #     # 调整图像和掩码大小
    #     image = cv2.resize(image, self.output_size)
    #     mask = cv2.resize(mask, self.output_size)
    #
    #     # 对mask进行二值化处理，将大于128的像素值设为1，小于等于128的像素值设为0
    #     _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
    #
    #     # 将mask转换为one-hot编码
    #     # 先初始化一个形状为 [2, 512, 512] 的零矩阵
    #     one_hot_mask = np.zeros((2, self.output_size[0], self.output_size[1]), dtype=np.float32)
    #     # 设置 liver 类别的通道为 1（即mask值为1的地方）
    #     one_hot_mask[0] = (mask == 0).astype(np.float32)  # 背景，mask值为0的地方
    #     one_hot_mask[1] = (mask == 1).astype(np.float32)  # Liver，mask值为1的地方
    #     # 转换为torch的张量
    #     one_hot_mask = torch.tensor(one_hot_mask)
    #
    #     # # 将image转为[3, 512, 512]的张量
    #     image = np.transpose(image, (2, 0, 1))  # 转换为[3, 512, 512]格式
    #     # image = torch.tensor(image, dtype=torch.float32)
    #
    #     sample = {'image': image, 'label': one_hot_mask}
    #     if self.transform:
    #         sample = self.transform(sample)
    #
    #     # image[3,512,512] label[2,512,512]
    #     # 添加样本的 case_name（这里使用文件名作为 case_name，可根据实际需求调整）
    #     sample['case_name'] = self.images[idx].split('.')[0]
    #     return sample