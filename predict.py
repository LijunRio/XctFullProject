import torch
from test_net import resnet50
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from DataInterfaceForTest import MedReconDataset
import csv
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
import random
import os
import csv
from BothChangeNetwork import XctNet
from torch.backends import cudnn

cudnn.benchmark = False  # if benchmark=True, deterministic will be False
cudnn.deterministic = True
seed = 0
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(128),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ct_transform = transforms.Compose([
    transforms.ToTensor()
])


def GetDataList(path):
    data_list = []
    with open(path, "rt") as f:
        reader = csv.reader(f)
        for line in reader:
            data_list.append(line)
    return data_list


# test_path = "D:\\7_31data\\train\\test.csv"
# test_path = "D:\\Rio\\open_dataset\\open_dataset\\test.csv"
test_path = "/home/lijun/open_dataset/open_dataset/test2.csv"
output_path = "/home/lijun/open_dataset/all_test/BothChangeNetworkResult/"
path = '/home/lijun/open_dataset/all_test/BothChangeNetworkResultError.csv'
test_list = GetDataList(test_path)
test_dataset = MedReconDataset(test_list, transform_img=data_transform,
                               transform_ct=ct_transform)
bath_size = 1  # 因为要对应，智能设为1
test_loader = DataLoader(test_dataset, batch_size=bath_size,
                         shuffle=False, num_workers=0)

sample = test_dataset[0]
image = sample['image']
image = torch.unsqueeze(image, dim=0)
label = sample['ct_volume']
print(label.shape)

model = nn.DataParallel(XctNet())
model.cuda()
model_weitht_path = "./resnet34_0826.pth"
model.load_state_dict(torch.load(model_weitht_path))
model.eval()
# 预测过程过一定要torch.no_grad()不计算误差函数
with torch.no_grad():
    count = 0
    # for step, sample in enumerate(test_loader, start=0):
    with open(path, 'w', newline="") as f:
        writer = csv.writer(f)
        for sample in test_loader:
            img = sample['image']
            label = sample['ct_volume']
            type = str(sample['type'])
            print(type)
            img = img.cuda()
            label = label.cuda()
            # print(label.shape)
            # print(img.shape)

            outputs = model(img)
            out_squeeze = torch.squeeze(outputs)
            label_sequeeze = torch.squeeze(label)
            print("out:", out_squeeze.shape)
            mistake_error = (abs(out_squeeze - label_sequeeze).sum().item())

            label_sequeeze = label_sequeeze.cpu()
            result_array = np.array(label_sequeeze)

            # 保存输出结果
            outname = output_path + type + ".nii"
            result = sitk.GetImageFromArray(result_array)
            # 将错误率写入csv文件
            sitk.WriteImage(result, outname)
            writer.writerow([outname, mistake_error])
            print(count, " error: ", mistake_error)
            count += 1
            # if count ==158:
            #     break
