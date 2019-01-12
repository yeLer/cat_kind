#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: predict.py
@time: 2018/11/19 8:57
@desc:利用训练保存的模型进行图片的预测
target = ['布偶猫', '孟买猫', '暹罗猫', '英国短毛猫']
label = ['1','0','2','3']
'''
# 导入必要的包
from PIL import Image
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D


# os.path.basename(), 返回path最后的文件名
# eg:
# path = 'D:\CSDN'
# os.path.basename(path) = CSDN

def pre_handle_picture(file_name):
    img = Image.open(file_name).resize((100, 100), Image.BILINEAR)
    img_RGB = img.convert('RGB')
    return np.array(img_RGB)


def tets_nets():
    '''
    # 搭建卷积神经网络,此网络需要与训练网络结构一致
    :return: 返回网络模型
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
    # 将网络展平
    model.add(Flatten())
    # 加256个神经元的全连接
    model.add(Dense(256, activation='relu'))
    # 按照50%丢失输出
    model.add(Dropout(0.5))
    # 最后一层4个神经元的输出
    model.add(Dense(4, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def main_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_img', type=str, default='./predict_imgs/1_bom.jpg',
                        help="your choice to the img，you just need to change the name of img")
    parser.add_argument('--model', type=str, default='./models/cat_weight.h5', help='the model to predict result')
    options = parser.parse_args()
    return options


if __name__ == "__main__":
    options = main_args()
    x_test = []
    x_test.append(pre_handle_picture(options.test_img))
    x_test = np.array(x_test)

    x_test = x_test.astype('float32')
    x_test /= 255

    model = tets_nets()
    # 加载权重文件
    model.load_weights(options.model)

    # [0] 表示取第一个识别结果
    classes = model.predict_classes(x_test)[0]

    print(classes)
