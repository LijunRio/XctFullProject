import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os
import torch.optim as optim
# from test_net import resnet50
import numpy as np
from dataset import MedReconDataset
from torch.utils.data import DataLoader
import csv
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
# from netchange import new_net
import random
from BothChangeNetwork import XctNet

writer = SummaryWriter()  # 用于可视化网络参数
# 设置GPU参数============================================================================
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


def GetDataList(path):
    data_list = []
    with open(path, "rt") as f:
        reader = csv.reader(f)
        for line in reader:
            data_list.append(line)
    return data_list

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'ct': transforms.Compose([
        transforms.ToTensor()
    ])
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_path = "/home/lijun/open_dataset/open_dataset/train.csv"
valid_path = "/home/lijun/open_dataset/open_dataset/valid.csv"
train_list = GetDataList(train_path)
valid_list = GetDataList(valid_path)

train_dataset = MedReconDataset(train_list, transform_img=image_transforms["train"],
                                transform_ct=image_transforms["ct"])
valid_dataset = MedReconDataset(valid_list, transform_img=image_transforms["valid"],
                                transform_ct=image_transforms["ct"])
train_num = len(train_dataset)
val_num = len(valid_dataset)
bath_size = 10  # 单个GPU估计batch size只能等于1
train_loader = DataLoader(train_dataset, batch_size=bath_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=bath_size, shuffle=False, num_workers=4)

net = nn.DataParallel(XctNet())
net.cuda()
summary(net, (3, 128, 128))
model_weight_path = "./resnet34-333f7ec4.pth"
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)

loss_function = nn.MSELoss(reduction='mean').cuda()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
min_error = 1000.0
save_path = './resnet34_0826.pth'
for epoch in range(8):
    net.train()
    running_loss = 0.0
    for step, sample in enumerate(train_loader, start=0):
        images = sample['image']
        labels = sample['ct_volume']
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        logits = net(images)
        loss = loss_function(logits, labels)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    error = 0.0
    with torch.no_grad():
        for val_data in valid_loader:
            val_images = val_data['image']
            val_labels = val_data['ct_volume']
            val_images = val_images.cuda()
            val_labels = val_labels.cuda()

            outputs = net(val_images)  # eval model only have last output layer
            out_squeeze = torch.squeeze(outputs)
            label_sequeeze = torch.squeeze(val_labels)
            # acc += abs(out_squeeze - label_sequeeze).sum().item()
            error += (abs(out_squeeze - label_sequeeze).sum().item())
            # predict_y = torch.max(outputs, dim=1)[1]  # 在输出的1维度上(因为0维度为batch)找出最大的值并返回他的index;[1]表示index
            # acc += (predict_y == val_labels).sum().item()  # 相同的地方返回true为1， 并所有求和就能得到正确率
        # val_accurate = acc / val_num
        val_error = error / val_num
        # writer.add_scalar("Accuracy/valid", val_accurate, epoch)
        writer.add_scalar("Error/valid", val_error, epoch)

        val_error_print = val_error / 1000
        if val_error_print < min_error:
            min_error = val_error_print
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  val_error: %.3f' %
              (epoch + 1, running_loss / step, val_error))
        print('[epoch %d] train_loss: %.3f  val_error_print: %.3f' %
              (epoch + 1, running_loss / step, val_error_print))
        print('[epoch %d] train_loss: %.3f  min_error: %.3f' %
              (epoch + 1, running_loss / step, min_error))


writer.flush()
