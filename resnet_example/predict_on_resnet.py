#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: predict_on_resnet.py
@time: 2018/11/23 11:00
@desc:
'''
from resnet50 import ResNet50
import numpy as np
import os
from PIL import Image

cat_dict = {0: "孟买猫", 1: "布偶猫", 2: "暹罗猫", 3: "英国短毛猫"}


def pre_handle_picture(src_img_dir, pic_name):
    img = Image.open(os.path.join(src_img_dir, pic_name))
    new_img = img.resize((224, 224), Image.BILINEAR)
    new_img.save(os.path.join(src_img_dir, os.path.basename(pic_name)))


def covert_img_toarray(src_img_dir, filename):
    img = Image.open(os.path.join(src_img_dir, filename)).convert('RGB')
    return np.array(img)


def predict_multi(src_img_dir):
    model = ResNet50(weights='cat_kind')

    images = os.listdir(src_img_dir)
    for imageName in images:
        # 读入图片
        pre_handle_picture(src_img_dir, imageName)
        x_test = []
        x_test.append(covert_img_toarray(src_img_dir, imageName))
        x_test = np.array(x_test)

        x_test = x_test.astype('float32')
        x_test /= 255
        ground_truth = imageName.split('_')[0]
        y_pred1 = model.predict(x_test)
        classes = np.argmax(y_pred1, axis=1)[0]
        for key in cat_dict.keys():
            if classes == key:
                print("Ground Truth: {0},  class: {1},  name: {2}".format(ground_truth, classes, cat_dict.get(key)))


def predict_one(src_img_dir, pic_name):
    pre_handle_picture(src_img_dir, pic_name)
    x_test = []
    x_test.append(covert_img_toarray(src_img_dir, pic_name))
    x_test = np.array(x_test)

    x_test = x_test.astype('float32')
    x_test /= 255

    model = ResNet50(weights='cat_kind')
    y_pred1 = model.predict(x_test)
    classes = np.argmax(y_pred1, axis=1)[0]
    for key in cat_dict.keys():
        if classes == key:
            print("class: {0},  name: {1}".format(classes, cat_dict.get(key)))


if __name__ == "__main__":
    src_img_dir = '../predict_imgs'
    pic_name = '1_bom.jpg'
    # predict_one(src_img_dir, pic_name)  # 调用单张图片预测
    predict_multi(src_img_dir='../predict_imgs') #调用多张图片预测
