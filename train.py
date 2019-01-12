#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: train.py
@time: 2018/11/18 21:48
@desc:
'''
# 导入必要的包
import os
from PIL import Image
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D
import argparse
import random


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


def train_model():
    '''
    # 搭建卷积神经网络
    :return:
    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # 完成模型的搭建后，我们需要使用.compile()方法来编译模型：
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./cat_kind_data/train',
                        help="the path to the training imgs")
    parser.add_argument('--test_dir', type=str, default='./cat_kind_data/test', help='the path to the testing imgs')
    parser.add_argument("--save_model", type=str, default='./models/cat_weight.h5', help='the path and the model name')
    parser.add_argument("--batch_size", type=int, default=10, help='the training batch size of data')
    parser.add_argument("--epochs", type=int, default=32, help='the training epochs')
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    # 调用函数获取用户参数
    options = main_args()
    # 调用函数获取模型
    model = train_model()
    # 调用函数获取训练数据和标签
    x_train, y_train = prepare_data(options.train_dir)
    x_test, y_test = prepare_data(options.train_dir)

    # 小批量数据可以将x_train, y_train一次性载入内存进行训练
    # 训练数据上按batch进行一定次数的迭代来训练网络,这里表示每次读入10张图片作为一个批量大小，数据集循环迭代32次
    model.fit(x_train, y_train, shuffle=True, batch_size=options.batch_size, epochs=options.epochs)

    # # 大批量数据可以通过train_on_batch进行训练，下面代码还需要整理，加循环操作
    # # train_on_batch不支持shuffle,这里手动混合
    # cc = list(zip(x_train, y_train))
    # random.shuffle(cc)
    # x_train[:], y_train[:] = zip(*cc)
    # del cc
    # model.train_on_batch(x_train, y_train)

    # 保存训练完成的模型文件
    save_model = options.save_model
    save_model_path = os.path.dirname(save_model)
    save_model_name = os.path.basename(save_model)
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    model.save_weights(save_model, overwrite=True)

    # 使用一行代码对我们的模型进行评估，看看模型的指标是否满足我们的要求
    score = model.evaluate(x_test, y_test, batch_size=10)
    print("Testing loss:{0},Testing acc:{1}".format(score[0], score[1]))
