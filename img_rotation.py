from PIL import Image, ImageEnhance
import numpy as np
import imutils
import os

data_path = 'D:\\7_31data\\DRR\\DRR\\NII_DRR2'
video_list = os.listdir(data_path)

save_path = 'D:\\7_31data\\DRR\\DRR\\NII_DRR_NEW'
if not os.path.exists(save_path):  # 文件夹不存在，则创建
    os.mkdir(save_path)

img_list = os.listdir(data_path)
for j in range(0, len(img_list)):
    # print(j)
    img_path = os.path.join(data_path, img_list[j])  # 图片文件
    if os.path.isfile(img_path):
        Img = Image.open(img_path)
        Img = Img.rotate(180)
        img_rgb = Img.convert("RGB")
        # array = np.array(img_rgb)
        # print(array.shape)
        save_rotate_path1 = os.path.join(save_path, img_list[j])
        img_rgb.save(save_rotate_path1)