#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: train_on_resnet.py
@time: 2018/11/22 17:35
@desc:
数据增强前，迭代100次的结果
339/339 [==============================] - 0s 1ms/step - loss: 0.0218 - acc: 0.9941
tesing loss:0.918,Testing acc:0.866

数据增强后，迭代64次的结果
7499/7499 [==============================] - 86s 11ms/step - loss: 0.0037 - acc: 0.9989
Testing loss:0.0011459080708486314,Testing acc:0.9997332977730364
'''
# 导入必要的包
import os
from PIL import Image
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import argparse
# 导入网络模型
from resnet50 import ResNet50


def convert_image_array(filename, src_dir):
    img = Image.open(os.path.join(src_dir, filename)).convert('RGB')
    return np.array(img)


def prepare_data(train_or_test_dir):
    x_train_test = []
    # 将训练或者测试集图片转换为数组
    ima1 = os.listdir(train_or_test_dir)
    for i in ima1:
        x_train_test.append(convert_image_array(i, train_or_test_dir))
    x_train_test = np.array(x_train_test)
    # 根据文件名提取标签
    y_train_test = []
    for filename in ima1:
        y_train_test.append(int(filename.split('_')[0]))
    y_train_test = np.array(y_train_test)
    # 将标签转换格式
    y_train_test = np_utils.to_categorical(y_train_test)
    # 将特征点从0~255转换成0~1提高特征提取精度
    x_train_test = x_train_test.astype('float32')
    x_train_test /= 255
    # 返回训练和测试数据
    return x_train_test, y_train_test

def main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='../cat_data_resNet50/train',
                        help="the path to the training imgs")
    parser.add_argument('--test_dir', type=str, default='../cat_data_resNet50/test', help='the path to the testing imgs')
    parser.add_argument("--save_model", type=str, default='../models/cat_weight_resNet50.h5', help='the path and the model name')
    parser.add_argument("--batch_size", type=int, default=32, help='the training batch size of data')
    parser.add_argument("--epochs", type=int, default=64, help='the training epochs')
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    # 调用函数获取用户参数
    options = main_args()
    # 搭建卷积神经网络
    # Input size must be at least 197x197;
    model = ResNet50(weights=None, classes=4)
    # 选择在imagenet上进行微调
    # model = ResNet50(include_top=False, weights='imagenet', classes=4)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # 调用函数获取训练数据和标签
    x_train, y_train = prepare_data(options.train_dir)
    x_test, y_test = prepare_data(options.test_dir)
    model.fit(x_train, y_train, shuffle=True, batch_size=options.batch_size,
              epochs=options.epochs, validation_data=(x_test, y_test))
    save_model_path = os.path.dirname(options.save_model)
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    # 保存模型
    model.save_weights(options.save_model, overwrite=True)
    score = model.evaluate(x_test, y_test, batch_size=options.batch_size)
    print("Testing loss:{0},Testing acc:{1}".format(score[0], score[1]))
