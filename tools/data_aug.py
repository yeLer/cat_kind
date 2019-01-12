#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: data_aug.py
@time: 2019/1/12 11:42
@desc:
布偶猫:  101张  --->101*20=1020
孟买猫： 97张    --->97*20=1940
暹罗猫： 101张   --->101*20=1020
英国短毛猫： 85张  --->85*20=1700

共计：384张  --->5680
'''
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import argparse, os
from PIL import Image
# 进度条模块
from tqdm import tqdm

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,  # 浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
    height_shift_range=0.2,  # height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
    rescale=1. / 255,   # 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
    shear_range=0.2,      # 浮点数，剪切强度（逆时针方向的剪切变换角度）
    zoom_range=0.2,     # 浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    horizontal_flip=True,  # 布尔值，进行随机水平翻转
    vertical_flip=False,   # 布尔值，进行随机竖直翻转
    fill_mode='nearest',   # ‘constant’‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
    cval=0,     # 浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
    channel_shift_range=0,  # 随机通道转换的范围
)


def data_aug(img_path, save_to_dir, agu_num):
    img = load_img(img_path)
    # 获取被扩充图片的文件名部分，作为扩充结果图片的前缀
    save_prefix = os.path.basename(img_path).split('.')[0]
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=save_to_dir,
                              save_prefix=save_prefix, save_format='jpg'):
        i += 1
        # 保存agu_num张数据增强图片
        if i >= agu_num:
            break


# 读取文件夹下的图片，并进行数据增强，并将结果保存到dataAug文件夹下
def handle_muti_aug(options):
    src_images_dir = options.src_images_dir
    save_dir = options.save_dir
    list_name = list(os.listdir(src_images_dir))

    for name in list_name:
        if not os.path.exists(os.path.join(save_dir, name)):
            os.mkdir(os.path.join(save_dir, name))

    for i in range(len(list_name)):
        handle_name = os.path.join(src_images_dir, list_name[i] + '/')
        # tqdm()为数据增强添加进度条
        for jpgfile in tqdm(os.listdir(handle_name)):
            # 将被扩充的图片也保存到增强的文件夹下
            Image.open(handle_name+jpgfile).save(save_dir+'/'+list_name[i]+'/'+jpgfile)
            # 调用数据增强过程函数
            data_aug(handle_name+jpgfile, os.path.join(options.save_dir, list_name[i]), options.agu_num)


def main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_images_dir', type=str, default='../source_images/',
                        help="需要被增强的训练集的源图片路径")
    parser.add_argument("--agu_num", type=int, default=19,
                        help='每张训练图片需要被增强的数量，这里设置为19，加上本身的1张，每张图片共计变成20张')
    parser.add_argument("--save_dir", type=str, default='../dataAug', help='增强数据的保存位置')
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    options = main_args()
    handle_muti_aug(options)
