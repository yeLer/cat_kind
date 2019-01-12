#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: prepare_data.py
@time: 2019/1/12 12:37
@desc:
布偶猫:  101张
孟买猫： 97张
暹罗猫： 101张
英国短毛猫： 85张

共计：384张

统计分类完成后：
train: 344张
test: 40张
合计： 384张

training rate : 339/384 = 0.90
testing rate : 45/384 = 0.10
'''
import os
from PIL import Image
import argparse
from tqdm import tqdm


class PrepareData:
    def __init__(self, options):
        self.moudle_name = "prepare data"
        self.options = options
        self.src_images_dir = self.options.src_images_dir
        self.save_img_with = self.options.out_img_size[0]
        self.save_img_height = self.options.out_img_size[1]
        self.save_dir = self.options.save_dir

    # 统一图片类型
    def renameJPG(self, filePath, kind):
        '''
        :param filePath:图片文件的路径
        :param kind: 图片的label种类标签
        :return:
        '''
        images = os.listdir(filePath)
        for name in images:
            if (name.split('_')[0] in ['0', '1', '2', '3']):
                continue
            else:
                os.rename(filePath + name, filePath + kind + '_' + str(name).split('.')[0] + '.jpg')

    # 调用图片处理
    def handle_rename_covert(self):
        save_dir = self.save_dir
        # 调用统一图片类型
        list_name = list(os.listdir(self.src_images_dir))
        print(list_name)

        train_dir = os.path.join(save_dir, "train")
        test_dir = os.path.join(save_dir, "test")
        # 1.如果已经有存储文件夹，执行则退出
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            os.mkdir(train_dir)
            os.mkdir(test_dir)
        list_source = [x for x in os.listdir(self.src_images_dir)]
        # 2.获取所有图片总数
        count_imgs = 0
        for i in range(len(list_name)):
            count_imgs += len(os.listdir(os.path.join(self.src_images_dir, list_name[i])))
        count = 1
        # 3.开始遍历文件夹，并处理每张图片
        for i in range(len(list_name)):
            handle_name = os.path.join(self.src_images_dir, list_name[i] + '/')
            self.renameJPG(handle_name, str(i))
            # 调用统一图片格式
            img_src_dir = os.path.join(self.src_images_dir, list_source[i])
            for jpgfile in tqdm(os.listdir(handle_name)):
                img = Image.open(os.path.join(img_src_dir, jpgfile))
                try:
                    new_img = img.resize((self.save_img_with, self.save_img_height), Image.BILINEAR)
                    if (count >= int(count_imgs * self.options.split_rate)):
                        new_img.save(os.path.join(test_dir, os.path.basename(jpgfile)))
                    else:
                        new_img.save(os.path.join(train_dir, os.path.basename(jpgfile)))
                    count += 1
                except Exception as e:
                    print(e)


def main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_images_dir', type=str, default='../dataAug/',
                        help="训练集的源图片路径")
    parser.add_argument("--split_rate", type=int, default=0.9, help='将训练集二和测试集划分的比例，0.9表示训练集与占90%')
    parser.add_argument('--out_img_size', type=tuple, default=(224, 224),
                        help='保存图片的大小，如果使用简单网络结构参数大小为(100,100),如果使用resnet大小参数为(224,224)')
    parser.add_argument("--save_dir", type=str, default='../cat_data_resNet50', help='训练数据的保存位置')
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    # 获取参数对象
    options = main_args()
    # 获取类对象
    pd_obj = PrepareData(options)

    pd_obj.handle_rename_covert()
