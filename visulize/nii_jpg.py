import cv2
import SimpleITK as sitk
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

input_drr = "/home/lijun/open_dataset/all_test/BothChangeNetworkResult/['237'].nii"
output = "/home/lijun/open_dataset/all_test/img_seriese/"

img = sitk.ReadImage(input_drr)
img_arr = sitk.GetArrayFromImage(img)*500
# img = sitk.GetImageFromArray(img_arr)
# img_255 = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
# img_arr = sitk.GetArrayFromImage(img_255)
print(img_arr.max())

for i in tqdm(range(img_arr.shape[0])):
    img_name = str(output + str(i) + ".jpg")
    cv2.imwrite(img_name, img_arr[i])
