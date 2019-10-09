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
import seaborn as sns
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
# history = model.fit(x=X_train, y=y_train,
#                     batch_size=16,
#                     epochs=1,
#                     validation_data=(X_val, y_val),
#                     callbacks=[get_weights_cb])




# 五 模型保存
# with open('weights_history2.p', 'wb') as f:
#     pickle.dump(weights_history, f)
with open('weights_history2.p', 'rb') as f:
    weights_history = pickle.load(f)


target = data[1][:, :, ::-1]
out_base = base_model.predict(np.expand_dims(target, axis=0))
out_base = out_base[0]
print(out_base.shape)

# 根据卷积层输出特征图集和模型某一参数状态计算预测概率（为了简单省略了bias计算）
def predict_on_weights(out_base, weights):
    gap = np.average(out_base, axis=(0, 1))
    logit = np.dot(gap, np.squeeze(weights))
    return 1 / (1 +  np.e ** (-logit))

predict_on_weights(out_base, weights_history[42])


plt.figure(figsize=(15, 0.5))
band = np.array([list(np.arange(0, 255, 10))] * 1)
sns.heatmap(band, annot=True, fmt="d", cmap='jet', cbar=False)
plt.show()


def getCAM(image, feature_maps, weights, display=False):
    predict = predict_on_weights(feature_maps, weights)

    # Weighted Feature Map
    cam = (predict - 0.5) * np.matmul(feature_maps, weights)
    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    # Resize as image size
    cam_resize = cv2.resize(cam, (224, 224))
    # Format as CV_8UC1 (as applyColorMap required)
    cam_resize = 255 * cam_resize
    cam_resize = cam_resize.astype(np.uint8)
    # Get Heatmap
    heatmap = cv2.applyColorMap(cam_resize, cv2.COLORMAP_JET)
    # Zero out
    heatmap[np.where(cam_resize <= 100)] = 0

    out = cv2.addWeighted(src1=image, alpha=0.8, src2=heatmap, beta=0.4, gamma=0)
    out = cv2.resize(out, dsize=(400, 400))

    if predict < 0.5:
        text = 'cat %.2f%%' % (100 - predict * 100)
    else:
        text = 'dog %.2f%%' % (predict * 100)

    cv2.putText(out, text, (210, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)
    if display:
        plt.figure(figsize=(7, 7))
        plt.imshow(out[:, :, ::-1])
        plt.show()
    return out

print(len(weights_history))
getCAM(image=target, feature_maps=out_base, weights=weights_history[124], display=True)


def batch_CAM(weights):
    idx = 0
    result = None
    for j in range(4):
        for i in range(4):
            idx += 1
            src = data[idx][:, :, ::-1]
            out_base = base_model.predict(np.expand_dims(src, axis=0))
            out_base = out_base[0]
            out = getCAM(image=src, feature_maps=out_base, weights=weights)
            out = cv2.resize(out, dsize=(300, 300))
            if i > 0:
                canvas = np.concatenate((canvas, out), axis=1)
            else:
                canvas = out
        if j > 0:
            result = np.concatenate((result, canvas), axis=0)
        else:
            result = canvas
    return result

plt.figure(figsize=(15, 15))
plt.imshow(batch_CAM(weights_history[111])[:, :, ::-1])
plt.show()
print(batch_CAM(weights_history[0]).shape)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_batch.mp4',0x00000021, 20.0, (1200, 1200))

for weight in tqdm(weights_history):
    img = batch_CAM(weight)
    out.write(img)
#     cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output4.mp4',0x00000021, 20.0, (400, 400))

for weight in weights_history:
    img = batch_CAM(weight)
    out.write(img)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()