from PIL import Image, ImageEnhance
import numpy as np
import imutils
import os
import csv


# 1 尺寸：8
# 2 旋转 9
# 3 镜像 3
# 4 平移 9
# 5 亮度不变 9
# 6 色度不变 7
# 7 对比度不变 7
# 8 锐度 7

# 尺寸调整
def ImgResize(Img, ScaleFactor):
    ImgSize = Img.size  # 获得图像原始尺寸
    NewSize = [int(ImgSize[0] * ScaleFactor), int(ImgSize[1] * ScaleFactor)]  # 获得图像新尺寸，保持长宽比
    Img = Img.resize(NewSize)  # 利用PIL的函数进行图像resize，类似matlab的imresize函数
    return Img


def ImgResizeTo(Img, NewSize):
    Img = Img.resize(NewSize)  # 利用PIL的函数进行图像resize，类似matlab的imresize函数
    return Img


# 旋转
def ImgRotate(Img, Degree):
    return Img.rotate(Degree)  # 利用PIL的函数进行图像旋转，类似matlab imrotate函数


# 利用PIL的函数进行水平以及上下镜像
def ImgLRMirror(Img):
    return Img.transpose(Image.FLIP_LEFT_RIGHT)


def ImgTBMirror(Img):
    return Img.transpose(Image.FLIP_TOP_BOTTOM)


# 亮度,增强因子为1.0是原始图像
def BrightEnchance(Img, factor):
    enh_bri = ImageEnhance.Brightness(Img)
    image_brightened = enh_bri.enhance(factor)
    return image_brightened


# 色度,增强因子为1.0是原始图像
def ColorEnchance(Img, factor):
    enh_col = ImageEnhance.Color(Img)
    image_colored = enh_col.enhance(factor)
    return image_colored


# 对比度，增强因子为1.0是原始图片
def ContrastEnchance(Img, factor):
    enh_con = ImageEnhance.Contrast(Img)
    image_contrasted = enh_con.enhance(factor)
    return image_contrasted


# 锐度，增强因子为1.0是原始图片
def SharpEnchance(Img, factor):
    enh_sha = ImageEnhance.Sharpness(Img)
    image_sharped = enh_sha.enhance(factor)
    return image_sharped


# data_path = 'D:\\7_31data\\DRR\\DRR\\NII_DRR_NEW'
# data_path = 'D:\\7_31data\\training\\drr'
# data_path = 'D:\\Rio\\open_dataset\\open_dataset\\drr'
data_path = '/home/lijun/open_dataset/open_dataset/drr2'
video_list = os.listdir(data_path)

# save_path = 'D:\\7_31data\\DRR\\aug\\'
# save_path = 'D:\\7_31data\\DRR\\aug2\\'
# save_path = 'D:\\Rio\\open_dataset\\open_dataset\\aug_drr'
save_path = '/home/lijun/open_dataset/open_dataset/aug_drr'
if not os.path.exists(save_path):  # 文件夹不存在，则创建
    os.mkdir(save_path)

img_list = os.listdir(data_path)
for j in range(0, len(img_list)):
    print(img_list[j])
    img_path = os.path.join(data_path, img_list[j])  # 图片文件
    if os.path.isfile(img_path):
        Img = Image.open(img_path)
        # 1 尺寸
        # path = 'D:\\7_31data\\DRR\\data.csv'
        # with open(path, 'wb') as f:
        #     writer = csv.writer(f)
        scale_img1 = ImgResize(Img, 0.2)
        scale_img2 = ImgResize(Img, 0.4)
        scale_img3 = ImgResize(Img, 0.6)
        scale_img4 = ImgResize(Img, 0.8)
        scale_img5 = ImgResize(Img, 1)
        scale_img6 = ImgResize(Img, 1.2)
        scale_img7 = ImgResize(Img, 1.4)
        scale_img8 = ImgResize(Img, 1.5)
        save_scale_path1 = os.path.join(save_path, 'scale_img1_' + img_list[j])
        save_scale_path2 = os.path.join(save_path, 'scale_img2_' + img_list[j])
        save_scale_path3 = os.path.join(save_path, 'scale_img3_' + img_list[j])
        save_scale_path4 = os.path.join(save_path, 'scale_img4_' + img_list[j])
        save_scale_path5 = os.path.join(save_path, 'scale_img5_' + img_list[j])
        save_scale_path6 = os.path.join(save_path, 'scale_img6_' + img_list[j])
        save_scale_path7 = os.path.join(save_path, 'scale_img7_' + img_list[j])
        save_scale_path8 = os.path.join(save_path, 'scale_img8_' + img_list[j])
        scale_img1.save(save_scale_path1)
        scale_img2.save(save_scale_path2)
        scale_img3.save(save_scale_path3)
        scale_img4.save(save_scale_path4)
        scale_img5.save(save_scale_path5)
        scale_img6.save(save_scale_path6)
        scale_img7.save(save_scale_path7)
        scale_img8.save(save_scale_path8)


        # 2 旋转
        rotate_img1 = ImgRotate(Img, 10)
        rotate_img2 = ImgRotate(Img, 30)
        rotate_img3 = ImgRotate(Img, 45)
        rotate_img4 = ImgRotate(Img, 90)
        rotate_img5 = Img
        rotate_img6 = ImgRotate(Img, 120)
        rotate_img7 = ImgRotate(Img, 135)
        rotate_img8 = ImgRotate(Img, 170)
        rotate_img9 = ImgRotate(Img, 180)
        save_rotate_path1 = os.path.join(save_path, 'rotate_img1_' + img_list[j])
        save_rotate_path2 = os.path.join(save_path, 'rotate_img2_' + img_list[j])
        save_rotate_path3 = os.path.join(save_path, 'rotate_img3_' + img_list[j])
        save_rotate_path4 = os.path.join(save_path, 'rotate_img4_' + img_list[j])
        save_rotate_path5 = os.path.join(save_path, 'rotate_img5_' + img_list[j])
        save_rotate_path6 = os.path.join(save_path, 'rotate_img6_' + img_list[j])
        save_rotate_path7 = os.path.join(save_path, 'rotate_img7_' + img_list[j])
        save_rotate_path8 = os.path.join(save_path, 'rotate_img8_' + img_list[j])
        save_rotate_path9 = os.path.join(save_path, 'rotate_img9_' + img_list[j])
        rotate_img1.save(save_rotate_path1)
        rotate_img2.save(save_rotate_path2)
        rotate_img3.save(save_rotate_path3)
        rotate_img4.save(save_rotate_path4)
        rotate_img5.save(save_rotate_path5)
        rotate_img6.save(save_rotate_path6)
        rotate_img7.save(save_rotate_path7)
        rotate_img8.save(save_rotate_path8)
        rotate_img9.save(save_rotate_path9)

        # 镜像
        mirror_img1 = ImgLRMirror(Img)
        mirror_img2 = ImgTBMirror(Img)
        mirror_img3 = Img
        save_mirror_path1 = os.path.join(save_path, 'mirror_img1_' + img_list[j])
        save_mirror_path2 = os.path.join(save_path, 'mirror_img2_' + img_list[j])
        save_mirror_path3 = os.path.join(save_path, 'mirror_img3_' + img_list[j])
        mirror_img1.save(save_mirror_path1)
        mirror_img2.save(save_mirror_path2)
        mirror_img3.save(save_mirror_path3)

        # 平移
        array_img = np.array(Img)  # PIL.Image 转 numpy.array
        translation_img1 = imutils.translate(array_img, 60, 60)
        translation_img2 = imutils.translate(array_img, -60, 60)
        translation_img3 = imutils.translate(array_img, 60, -60)
        translation_img4 = imutils.translate(array_img, -60, -60)
        translation_img5 = array_img
        translation_img6 = imutils.translate(array_img, 30, 30)
        translation_img7 = imutils.translate(array_img, -30, 30)
        translation_img8 = imutils.translate(array_img, 30, -30)
        translation_img9 = imutils.translate(array_img, -30, -30)
        save_translation_path1 = os.path.join(save_path, 'translation_img1_' + img_list[j])
        save_translation_path2 = os.path.join(save_path, 'translation_img2_' + img_list[j])
        save_translation_path3 = os.path.join(save_path, 'translation_img3_' + img_list[j])
        save_translation_path4 = os.path.join(save_path, 'translation_img4_' + img_list[j])
        save_translation_path5 = os.path.join(save_path, 'translation_img5_' + img_list[j])
        save_translation_path6 = os.path.join(save_path, 'translation_img6_' + img_list[j])
        save_translation_path7 = os.path.join(save_path, 'translation_img7__' + img_list[j])
        save_translation_path8 = os.path.join(save_path, 'translation_img8_' + img_list[j])
        save_translation_path9 = os.path.join(save_path, 'translation_img9_' + img_list[j])
        translation_img1 = Image.fromarray(translation_img1)
        translation_img2 = Image.fromarray(translation_img2)
        translation_img3 = Image.fromarray(translation_img3)
        translation_img4 = Image.fromarray(translation_img4)
        translation_img5 = Image.fromarray(translation_img5)
        translation_img6 = Image.fromarray(translation_img6)
        translation_img7 = Image.fromarray(translation_img7)
        translation_img8 = Image.fromarray(translation_img8)
        translation_img9 = Image.fromarray(translation_img9)
        translation_img1.save(save_translation_path1)
        translation_img2.save(save_translation_path2)
        translation_img3.save(save_translation_path3)
        translation_img4.save(save_translation_path4)
        translation_img5.save(save_translation_path5)
        translation_img6.save(save_translation_path6)
        translation_img7.save(save_translation_path7)
        translation_img8.save(save_translation_path8)
        translation_img9.save(save_translation_path9)

        # 以下只对原图操作，亮度,增强因子为0.0将产生黑色图像；为1.0将保持原始图像
        bright_img1 = BrightEnchance(Img, 0.3)
        bright_img2 = BrightEnchance(Img, 0.4)
        bright_img3 = BrightEnchance(Img, 0.5)
        bright_img4 = BrightEnchance(Img, 0.6)
        bright_img5 = BrightEnchance(Img, 0.7)
        bright_img6 = BrightEnchance(Img, 0.8)
        bright_img7 = BrightEnchance(Img, 0.9)
        bright_img8 = BrightEnchance(Img, 0.9)
        bright_img9 = Img
        save_bright_path1 = os.path.join(save_path, 'bright_img1_' + img_list[j])
        save_bright_path2 = os.path.join(save_path, 'bright_img2_' + img_list[j])
        save_bright_path3 = os.path.join(save_path, 'bright_img3_' + img_list[j])
        save_bright_path4 = os.path.join(save_path, 'bright_img4_' + img_list[j])
        save_bright_path5 = os.path.join(save_path, 'bright_img5_' + img_list[j])
        save_bright_path6 = os.path.join(save_path, 'bright_img6_' + img_list[j])
        save_bright_path7 = os.path.join(save_path, 'bright_img7__' + img_list[j])
        save_bright_path8 = os.path.join(save_path, 'bright_img8_' + img_list[j])
        save_bright_path9 = os.path.join(save_path, 'bright_img9_' + img_list[j])
        bright_img1.save(save_bright_path1)
        bright_img2.save(save_bright_path2)
        bright_img3.save(save_bright_path3)
        bright_img4.save(save_bright_path4)
        bright_img5.save(save_bright_path5)
        bright_img6.save(save_bright_path6)
        bright_img7.save(save_bright_path7)
        bright_img8.save(save_bright_path8)
        bright_img9.save(save_bright_path9)

        # 色度,增强因子为1.0是原始图像，大于1增强，小于1减弱
        color_img1 = ColorEnchance(Img, 0.5)
        color_img2 = ColorEnchance(Img, 0.7)
        color_img3 = ColorEnchance(Img, 0.9)
        color_img4 = ColorEnchance(Img, 1.1)
        color_img5 = ColorEnchance(Img, 1.3)
        color_img6 = ColorEnchance(Img, 1.5)
        color_img7 = Img
        save_color_path1 = os.path.join(save_path, 'color_img1_' + img_list[j])
        save_color_path2 = os.path.join(save_path, 'color_img2_' + img_list[j])
        save_color_path3 = os.path.join(save_path, 'color_img3_' + img_list[j])
        save_color_path4 = os.path.join(save_path, 'color_img4_' + img_list[j])
        save_color_path5 = os.path.join(save_path, 'color_img5_' + img_list[j])
        save_color_path6 = os.path.join(save_path, 'color_img6_' + img_list[j])
        save_color_path7 = os.path.join(save_path, 'color_img7_' + img_list[j])
        color_img1.save(save_color_path1)
        color_img2.save(save_color_path2)
        color_img3.save(save_color_path3)
        color_img4.save(save_color_path4)
        color_img5.save(save_color_path5)
        color_img6.save(save_color_path6)
        color_img7.save(save_color_path7)

        # 对比度，增强因子为1.0是原始图片,大于1增强，小于1减弱
        contrast_img1 = ContrastEnchance(Img, 0.5)
        contrast_img2 = ContrastEnchance(Img, 0.7)
        contrast_img3 = ContrastEnchance(Img, 0.9)
        contrast_img4 = ContrastEnchance(Img, 1.1)
        contrast_img5 = ContrastEnchance(Img, 1.3)
        contrast_img6 = ContrastEnchance(Img, 1.5)
        contrast_img7 = Img
        save_contrast_path1 = os.path.join(save_path, 'contrast_img1_' + img_list[j])
        save_contrast_path2 = os.path.join(save_path, 'contrast_img2_' + img_list[j])
        save_contrast_path3 = os.path.join(save_path, 'contrast_img3_' + img_list[j])
        save_contrast_path4 = os.path.join(save_path, 'contrast_img4_' + img_list[j])
        save_contrast_path5 = os.path.join(save_path, 'contrast_img5_' + img_list[j])
        save_contrast_path6 = os.path.join(save_path, 'contrast_img6_' + img_list[j])
        save_contrast_path7 = os.path.join(save_path, 'contrast_img7_' + img_list[j])
        contrast_img1.save(save_contrast_path1)
        contrast_img2.save(save_contrast_path2)
        contrast_img3.save(save_contrast_path3)
        contrast_img4.save(save_contrast_path4)
        contrast_img5.save(save_contrast_path5)
        contrast_img6.save(save_contrast_path6)
        contrast_img7.save(save_contrast_path7)

        # 锐度，增强因子为1.0是原始图片,大于1增强，小于1减弱
        sharp_img1 = SharpEnchance(Img, 0.5)
        sharp_img2 = SharpEnchance(Img, 0.7)
        sharp_img3 = SharpEnchance(Img, 0.9)
        sharp_img4 = SharpEnchance(Img, 1.1)
        sharp_img5 = SharpEnchance(Img, 1.3)
        sharp_img6 = SharpEnchance(Img, 1.5)
        sharp_img7 = Img
        save_sharp_path1 = os.path.join(save_path, 'sharp_img1_' + img_list[j])
        save_sharp_path2 = os.path.join(save_path, 'sharp_img2_' + img_list[j])
        save_sharp_path3 = os.path.join(save_path, 'sharp_img3_' + img_list[j])
        save_sharp_path4 = os.path.join(save_path, 'sharp_img4_' + img_list[j])
        save_sharp_path5 = os.path.join(save_path, 'sharp_img5_' + img_list[j])
        save_sharp_path6 = os.path.join(save_path, 'sharp_img6_' + img_list[j])
        save_sharp_path7 = os.path.join(save_path, 'sharp_img7_' + img_list[j])
        sharp_img1.save(save_sharp_path1)
        sharp_img2.save(save_sharp_path2)
        sharp_img3.save(save_sharp_path3)
        sharp_img4.save(save_sharp_path4)
        sharp_img5.save(save_sharp_path5)
        sharp_img6.save(save_sharp_path6)
        sharp_img7.save(save_sharp_path7)
