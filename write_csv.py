import csv
import os
import sys

# data1_drr = "D:\\7_31data\\DRR\\aug\\"
# data2_drr = "D:\\7_31data\\DRR\\aug2\\"
# data1_drr = "D:\\Rio\\open_dataset\\open_dataset\\aug_drr\\"
data1_drr = "/home/lijun/open_dataset/open_dataset/aug_drr/"
# data2_drr = "D:\\7_31data\\DRR\\aug2\\"
# test
list_name1 = os.listdir(data1_drr)
# list_name2 = os.listdir(data2_drr)
# path = 'D:\\7_31data\\DRR\\data.csv'
path = '/home/lijun/open_dataset/open_dataset/data.csv'
# ct_path = 'D:\\7_31data\\training\\ct\\'
ct_path = '/home/lijun/open_dataset/open_dataset/ct_nii/'

with open(path, 'w', newline="") as f:
    writer = csv.writer(f)
    for item in list_name1:
        flag = item.rfind('_')
        img_type = item[flag+1:-4]
        print(item, "   num: ", img_type)
        print(data1_drr + item, ct_path+img_type+".nii")
        writer.writerow([data1_drr + item, ct_path+img_type+".nii"])


# with open(path, 'a+', newline="") as f:
#     writer = csv.writer(f)
#     for item in list_name2:
#         if item[-6:-5] == "_":
#             img_type = item[-5:-4]
#         else:
#             img_type = item[-6:-4]
#         print(data2_drr + item, ct_path+img_type+".nii")
#         writer.writerow([data2_drr + item, ct_path+img_type+".nii"])
