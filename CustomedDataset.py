import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as transforms
from skimage import exposure
from torch.utils.data import Dataset
import glob
from skimage import io, transform
import random

UltrasoundDataTransform = transforms.Compose([
    transforms.Lambda(lambd=lambda x: np.transpose(x, (2, 0, 1))),
    transforms.ToPILImage(),
    transforms.Resize([256, 256]),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees = 0, translate = (0.05, 0.05), scale = (0.95, 1.05)),
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) ])

UltrasoundDataTransform_b = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([256, 256]),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor()
])

UltrasoundDataTransform2 = transforms.Compose([
    transforms.Lambda(lambd=lambda x: np.transpose(x, (2, 0, 1))),
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

UltrasoundDataTransform2_b = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])

class TensorDatasetWithTransform(Dataset):
    def __init__(self, tensors, dataFolderPath, dataIndexes, transform, mode_flag):
        self.Tensors = tensors
        self.Transform = transform
        self.DataFolderPath = dataFolderPath
        self.DataIndexes = dataIndexes
        self.mode_flag = mode_flag

    def __getitem__(self, index):
        dataIndex = self.DataIndexes[index]
        if self.mode_flag=="B":
            file_dir_0_B = self.DataFolderPath + "/" + dataIndex + "/B/"+ dataIndex +".png"
            image_B = cv2.imread(file_dir_0_B, cv2.IMREAD_GRAYSCALE)
            image_B = cv2.resize(image_B, (256, 256))
            image_new = exposure.equalize_adapthist(image_B, clip_limit=0.015)* 255  
            image_new = torch.Tensor(image_new)/255
        elif self.mode_flag=="C":
            file_dir_0_C = self.DataFolderPath + "/" + dataIndex + "/C/"+ dataIndex +".png"
            image_C = cv2.imread(file_dir_0_C)  
            mask_dir_0_C = self.DataFolderPath + "/" + dataIndex + "/C/Roi.png"
            image_roi_C = cv2.imread(mask_dir_0_C, 0)  
            image_C = image_C
            image_C = exposure.equalize_adapthist(image_C, clip_limit=0.015) * 255  # ！！！
            image_C = torch.Tensor(image_C) / 255  # ！！！
            image_roi_C= exposure.equalize_adapthist(image_roi_C, clip_limit=0.015) * 255
            image_roi_C= torch.Tensor(image_roi_C) 
        else:
            image_new = np.zeros((5,224, 224))
            file_dir_0_B = self.DataFolderPath + "/" + dataIndex + "/B/"+ dataIndex +".png"
            image_B = cv2.imread(file_dir_0_B,0)  
            image_B = exposure.equalize_adapthist(image_B, clip_limit=0.015) * 255  # ！！！
            image_B = torch.Tensor(image_B) / 255

            file_dir_0_C = self.DataFolderPath + "/" + dataIndex + "/C/"+ dataIndex +".png"
            image_C = cv2.imread(file_dir_0_C) 
            mask_dir_0_C = self.DataFolderPath + "/" + dataIndex + "/C/Roi.png"
            image_roi_C = cv2.imread(mask_dir_0_C, 0)  
            image_C = exposure.equalize_adapthist(image_C, clip_limit=0.015) * 255  # ！！！
            image_C = torch.Tensor(image_C) / 255  # ！！！
            image_roi_C= exposure.equalize_adapthist(image_roi_C, clip_limit=0.015) * 255
            image_roi_C= torch.Tensor(image_roi_C) 
           

        label = self.Tensors[0][index]
        if self.Transform is not None:
            if self.mode_flag == "B+C":#混合模态
                image_new[0,:, :] = UltrasoundDataTransform_b(image_B)
                image_new[1:4,:, :] = UltrasoundDataTransform(image_C)
                image_new[4,:, :] = UltrasoundDataTransform_b(image_roi_C)
            elif self.mode_flag == "B":
                image_new= self.Transform(image_new)
            else:
                image_new[:-1,:,:] = self.Transform(image_C)
                image_new[3,:,:] = UltrasoundDataTransform_b(image_roi_C)

        return image_new, label, dataIndex#image_B，image_new
   
    def __len__(self):
        return self.Tensors[0].size(0)
