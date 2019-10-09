# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 16:19
# @Author  : liujunfeng
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications import ResNet50, VGG16, InceptionV3
from keras.applications.vgg16 import preprocess_input, decode_predictions
# from utils import make_parallel
import os
import pickle
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback
from tqdm import tqdm
from keras import backend as K
from sklearn.model_selection import train_test_split


def get_params_count(model):
    # 获取可训练参数量
    trainable = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    return trainable, non_trainable

path =  "data/train"
label = np.array([0] * 1250 + [1] * 1250)
data = np.zeros((2500, 224, 224, 3), dtype=np.uint8)

for i in tqdm(range(1250)):
    img = cv2.imread(path + '/cat.' + str(i) + '.jpg')
    # print(img)
    img = img[:, :, ::-1]
    img = cv2.resize(img, (224, 224))
    data[i] = img

for i in tqdm(range(1250)):
    img = cv2.imread(path + '/dog.' + str(i) + '.jpg')
    img = img[:, :, ::-1]
    img = cv2.resize(img, (224, 224))
    data[i + 1250] = img

print('Training Data Size = %.2f GB' % (sys.getsizeof(data) / 1024 ** 3))

# 二 数据集分割
X_train, X_val, y_train, y_val = train_test_split(data, label, shuffle=True, test_size=0.2, random_state=42)

# 三 创建模型
base_model = VGG16(include_top=False, weights='imagenet')
for layers in base_model.layers:
    layers.trainable = False
y = GlobalAveragePooling2D()(base_model.output)
y = Dropout(0.25)(y)
y = Dense(1, activation='sigmoid')(y)
model = Model(inputs=base_model.input, outputs=y)
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

weights_history = []
get_weights_cb = LambdaCallback(on_batch_end=lambda batch,
                                logs: weights_history.append(model.layers[-1].get_weights()[0]))

print('Trainable Parameters: ', get_params_count(model)[0])

# 四 模型训练
history = model.fit(x=X_train, y=y_train,
                    batch_size=16,
                    epochs=1,
                    validation_data=(X_val, y_val),
                    callbacks=[get_weights_cb])




# 五 模型保存
with open('weights_history2.p', 'wb') as f:
    pickle.dump(weights_history, f)
with open('weights_history2.p', 'rb') as f:
    weights_history = pickle.load(f)


target = data[1][:, :, ::-1]
out_base = base_model.predict(np.expand_dims(target, axis=0))
out_base = out_base[0]
print(out_base.shape)