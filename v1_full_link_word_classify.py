#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:52:02 2021

@author: nijiahui
"""
import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import pickle
import math
from tensorflow.keras.preprocessing.sequence import pad_sequences
from classify_utils import cut_word
from classify_utils import word2int

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# 定义TensorFlow配置
config = ConfigProto()
# 配置GPU内存分配方式，按需增长，很关键
config.gpu_options.allow_growth = True
# 在创建session的时候把config作为参数传进去
session = InteractiveSession(config=config)

# 打印深度学习包版本
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


# 一,特征工程,处理数据
# [csv数据文本地址]
csv_file_path = "./data/v7_correct_data.csv"
# [词典路径]
word_dict_path = "./word_dict.pkl"
# [训练集 / 测试集数据 比例]
proportion_data = 0.3

# 1,读取.训练-测试数据
# df_data = pd.read_csv(csv_file_path,header=None,encoding='gbk')
df_data = pd.read_csv(csv_file_path,header=0,encoding="GBK")
df_data.rename(columns={0:'text',1:'label'},inplace=True)

# 打乱dataframe数据,并重新定义索引
df_data = df_data.sample(frac=1).reset_index(drop=True)

# 获得目标label
labels = np.array(df_data["label"])

# # 2,构建词典
# if os.path.exists(word_dict_path):
#     with open(word_dict_path, 'rb') as f:
#         word_dict = pickle.load(f)
# else:
#     title_list = df_data["text"].tolist()
#     word_dict = cut_word.generate_word_dict(title_list,cutword_method="jieba",generate_pkl=True)

title_list = df_data["text"].tolist()
word_dict = cut_word.generate_word_dict(title_list,cutword_method="jieba",generate_pkl=True)

# 词典总数量
vocab_size = len(word_dict)
    
# 3,保留少量负样本数据
def random_batch(x, y, batch_size=32):
    idx = np.random.randint(0, len(x), size=batch_size)
    return x[idx], y[idx]

# 4,词嵌入-将中文转化为英文
word2int_list = word2int.word2int(df_data,word_dict)
word2int_list = np.array(word2int_list)

print("共得到%s条词嵌入后数据"%len(word2int_list))


# 5,生成 训练集,测试集数据
# 分割数据点
split_number = int(len(df_data) * proportion_data)
# 获得 训练数据,训练标签
train_data = word2int_list[:-split_number]
train_label = labels[:-split_number]
# 获得 测试数据,测试标签
test_data = word2int_list[-split_number:]
test_label = labels[-split_number:]

print("训练集",train_data)

# 6,对训练集,测试集 数据做预处理
max_length = 40

# 1) 对训练集数据做预处理
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, # list of list
    value = 0,    # 超出最大值的部分需填充的数据
    padding = 'post',   # post:在后填充; pre:在前填充
    maxlen = max_length)    # 处理段落的最大值 -若超出则阶段;若不足则填充;

# 2) 对测试集数据做预处理
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, # list of list
    value = 0,# 超出最大值的部分需填充的数据
    padding = 'post', # post:在后填充; pre:在前填充
    maxlen = max_length)

# 3) 打印处理后的几条数据
print(train_data[0])
print(test_data[0])

# 二,定义模型
# 1,模型参数
embedding_dim = 16
batch_size = 128

# 2,定义模型
# ###############################################################################
# # [全链接模型]
# # --------
# # 1,使用 MaxPooling + l2  --200--> 0.96
# # 2,使用 AveragePooling - l2  --200--> 0.96

# # --------
model = keras.models.Sequential([
    # 1,定义一个矩阵:define matrix:[vocab_size,embedding_dim]
    # 2,[max_length * embedding_dim]
    # 3,[batch_size * max_length * embedding_dim]
    keras.layers.Embedding(vocab_size,embedding_dim,trainable=True,
                            input_length=max_length),
    # [batch_size * max_length * embedding_dim]
    # --> batch_size * embedding_dim
    keras.layers.GlobalAveragePooling1D(),
    # keras.layers.GlobalMaxPooling1D(),
    # keras.layers.Flatten(),  # 可以使用展平,后接入全连接层
    
    # keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    # keras.layers.Dense(64, activation="relu",kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(21, activation="softmax")
])
# ###############################################################################
# [深度神经网络]

###############################################################################
# # [单向RNN]
# model = keras.models.Sequential([
#     # 1. define matrix: [vocab_size, embedding_dim]
#     # 2. [1,2,3,4..], max_length * embedding_dim
#     # 3. batch_size * max_length * embedding_dim
#     keras.layers.Embedding(vocab_size, embedding_dim,trainable=True,
#                             input_length = max_length),
#     keras.layers.SimpleRNN(units = 64, return_sequences = False),
#     keras.layers.Dense(64, activation = 'relu'),
#     keras.layers.Dense(21, activation='softmax'),
# ])

# ###############################################################################
# # [双层双向RNN]  
# # ---- 1,使用双层双向RNN --200--> 0.93
# model = keras.models.Sequential([
#     # 1. define matrix: [vocab_size, embedding_dim]
#     # 2. [1,2,3,4..], max_length * embedding_dim
#     # 3. batch_size * max_length * embedding_dim
#     keras.layers.Embedding(vocab_size, embedding_dim,
#                             input_length = max_length),
#     keras.layers.Bidirectional(
#         keras.layers.SimpleRNN(
#             units = 64, return_sequences = True)),
#     keras.layers.Bidirectional(
#         keras.layers.SimpleRNN(
#             units = 128, return_sequences = False)),
#     keras.layers.Dense(64, activation = 'relu'),
#     keras.layers.Dense(21, activation='softmax'),
# ])

# ###############################################################################
# # # [双层双向LSTM] 
# # ---- 1,使用双层双向LSTM --200--> 0.94
# # ---- 2,使用单层层双向LSTM --200--> 0.94
# model = keras.models.Sequential([
#     # 1. define matrix: [vocab_size, embedding_dim]
#     # 2. [1,2,3,4..], max_length * embedding_dim
#     # 3. batch_size * max_length * embedding_dim
#     keras.layers.Embedding(vocab_size, embedding_dim,
#                             input_length = max_length),
#     keras.layers.Bidirectional(
#         keras.layers.LSTM(
#             units = 64, return_sequences = True)),
#     keras.layers.Bidirectional(
#         keras.layers.LSTM(
#             units = 64, return_sequences = False)),
#     keras.layers.Dense(64, activation = 'relu'),
#     keras.layers.Dense(21, activation='softmax'),
# ])

# # ###############################################################################
# # # [双层双向LSTM] 
# # ---- 1,使用双层双向LSTM --200--> 0.94
# # ---- 2,使用单层层双向LSTM --200--> 0.94
# inputs = keras.Input(shape=(None,))
# lstm = keras.models.Sequential([
#     # 1. define matrix: [vocab_size, embedding_dim]
#     # 2. [1,2,3,4..], max_length * embedding_dim
#     # 3. batch_size * max_length * embedding_dim
#     keras.layers.Embedding(vocab_size, embedding_dim),
#     keras.layers.LSTM(units = 128, return_sequences = False),
#     # keras.layers.Dense(64, activation = 'relu'),
#     keras.layers.Dense(21, activation='softmax'),
# ])
# output = lstm(inputs)
# model = keras.Model(inputs, output)
# ###############################################################################
# 打印模型概况
print(model.summary())
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 三,训练模型
epochs = 30
epochs = 200

# 1,回调函数
# Tensorboard, earlystopping, ModelCheckpoint
logdir = './weights/v1/callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,"fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),# TensorBoard-终端输入“tensorboard --logdir=callbacks”查看图结构
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only = True,
                                    save_weights_only = False
                                    ),# 保存最好的模型结果
    keras.callbacks.EarlyStopping(patience=50, min_delta=1e-4),# 当连续“patience”次增益小于“min_delta”时提前结束
]

# 2,训练模型
history = model.fit(train_data,train_label,
                    epochs = epochs,
                    batch_size = batch_size,
                    validation_split = 0.2,
                    callbacks = callbacks)

loss_max_value = max(history.history["loss"])

# 四,打印模型训练曲线
def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_'+label] = history.history['val_'+label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()
    
plot_learning_curves(history, 'accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 0, loss_max_value)

# 五,估计器预测测试数据集准确率
evaluate_rate = model.evaluate(
    test_data, test_label,
    batch_size = batch_size,
    verbose = 0
    ) 
 
print("估计器预测测试数据集的 损失和准确率分别为:%s,%s"%tuple(evaluate_rate))
              


# # 六,
# input_example_batch = test_data
# target_example_batch = test_label

# example_batch_predictions = model(input_example_batch)
# print(example_batch_predictions.shape)

# # input_example_batch = test_data[:100]
# # target_example_batch = test_label[:100]
# # sample_indices = tf.random.categorical(
# #     logits = example_batch_predictions[:1], num_samples = 1)
# # sample_indices = tf.squeeze(sample_indices, axis = -1)


# res_predictions = model(test_data)


# pre_res = np.argmax(example_batch_predictions, axis=-1)

# from sklearn.metrics import classification_report

# # classification_report()
# report = classification_report(target_example_batch,pre_res)

# print(report)
# ###############################################################################
# input_example_batch = test_data[15:16,:]
# target_example_batch = test_label[15:16,:]
# example_batch_predictions = loaded_model(input_example_batch)
# pre_res = np.argmax(example_batch_predictions, axis=-1)

# # ###############################################################################
# input_example_batch = test_data
# target_example_batch = test_label


# # loaded_model = keras.models.load_model(output_model_file)
# loaded_model = model

# example_batch_predictions = loaded_model(input_example_batch)

# pre_res = np.argmax(example_batch_predictions, axis=-1)
# # 


# target_list = target_example_batch.tolist()
# pre_list = pre_res.tolist()

# fail_list = list()

# for i ,j in enumerate(zip(target_list,pre_list)):
#     if j[0] != j[1]:
#         fail_list.append((i,j))

# len(fail_list)



# ###############################################################################


# # # 保存模型
# 




# import shutil

# if os.path.exists('./model/1'):
#     shutil.rmtree('./model/1')

# export_path = './model/1'

# # Fetch the Keras session and save the model
# with tf.keras.backend.get_session() as sess:
#     tf.saved_model.simple_save(
#         sess,
#         export_path,
#         inputs={'inputs': model.input},
#         outputs={t.name:t for t in model.outputs})




              
              
              
              
              
            