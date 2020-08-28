import csv
import os
from random import shuffle

# path = 'D:\\7_31data\\DRR\\data.csv'
path = '/home/lijun/open_dataset/open_dataset/data.csv'
data_list = []
with open(path, "rt") as f:
    reader = csv.reader(f)
    for line in reader:
        data_list.append(line)
# 打乱所有数据
shuffle(data_list)
length = len(data_list)
print(length)

valid_list = data_list[:int(0.1 * length)]
train_list = data_list[int(0.1 * length):int(0.9 * length)]
test_list = data_list[int(0.9 * length):length]
print(len(train_list), len(valid_list), len(test_list))

# train_path = "D:\\7_31data\\train\\train.csv"
# valid_path = "D:\\7_31data\\train\\valid.csv"
# test_path = "D:\\7_31data\\train\\test.csv"
train_path = "/home/lijun/open_dataset/open_dataset/train.csv"
valid_path = "/home/lijun/open_dataset/open_dataset/valid.csv"
test_path = "/home/lijun/open_dataset/open_dataset/test.csv"
with open(train_path, "w", newline="")as f:
    writer = csv.writer(f)
    for item in train_list:
        writer.writerow(item)

with open(valid_path, "w", newline="")as f:
    writer = csv.writer(f)
    for item in valid_list:
        writer.writerow(item)

with open(test_path, "w", newline="")as f:
    writer = csv.writer(f)
    for item in test_list:
        writer.writerow(item)