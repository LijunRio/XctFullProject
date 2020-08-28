# DataInterfaceForTest.py是给predict.py使用的数据接口
# DataInterfaceForTest.py is the data interface used by predict.py
# 该文件读入的是训练文件的路径存放csv文件，第一列是DRR的路径，第二列是CT的路径，第三列是对应CT的文件名
# 注意用write_csv.py脚本生成的csv文件是没有第三列的，这部分偷了个懒，是直接用excel提取文件名的
import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
import matplotlib.pyplot as plt


class MedReconDataset(Dataset):
    def __init__(self, data_list, transform_img=None, transform_ct=None):
        # 读取csv文件，获取img和ct和CT对应路径
        self.drr_list = []
        self.ct_list = []
        self.type = []

        for item in data_list:
            self.drr_list.append(item[0])
            self.ct_list.append(item[1])
            self.type.append(item[2])
        self.transform_img = transform_img
        self.transform_ct = transform_ct

    def __len__(self):
        return len(self.ct_list)

    # 装载数据，返回[img,label,ct_path, type_class]
    def __getitem__(self, idx):
        drr_path = self.drr_list[idx]
        ct_path = self.ct_list[idx]
        type_class = self.type[idx]

        # 对二维图像进行数据处理与扩增
        drr_img = Image.open(drr_path).convert('RGB')
        drr_tensor = self.transform_img(drr_img)

        # 对CT图像做数据处理
        ct_volume = sitk.ReadImage(ct_path)
        # 如果ct值没有缩放到0-1的话需要加入下面这句话
        ct_volume = sitk.Cast(sitk.RescaleIntensity(ct_volume), sitk.sitkUInt8) # cast ct volume to 0-1
        ct_array = sitk.GetArrayFromImage(ct_volume)
        # 此时的ct_array是 C, W, H 需要把输入格式变成 w h c与二维图像对齐
        ct_transpose = np.transpose(ct_array, (1, 2, 0))
        ct_tensor = self.transform_ct(ct_transpose)

        ct_tensor = torch.tensor(ct_tensor).to(drr_tensor.dtype)
        sample = {'image': drr_tensor, "ct_volume": ct_tensor, "ct_path": ct_path, "type": type_class}   # 新增训练时需要删除
        return sample
