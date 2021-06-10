import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import Sequence
import math
import os
import sys
import pickle
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as mpl 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, Dense, GlobalMaxPool1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers

# 打印深度学习包版本
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# 定义TensorFlow配置
config = ConfigProto()
# 配置GPU内存分配方式，按需增长，很关键
config.gpu_options.allow_growth = True
# 在创建session的时候把config作为参数传进去
session = InteractiveSession(config=config)


# 一,常量定义
data_path = 'data/v7_correct_data.csv'
save_weight_path = "callbacks/best_model.weights"
save_model_path = "callbacks/fashion_mnist_model.h5"
badcase_path = "badcase/test.csv"
dict_word_pkl_path = "callbacks/dict_word.pkl"

# 二,变量定义
# epochs = 40
epochs = 60
batch_size = 32

# 三,训练数据,验证数据,测试数据
# 实例化 分词器 Tokenizer
tokenizer = Tokenizer(filters='', char_level=True)

# df = pd.read_csv(data_path,header=0, names=['text', 'label'], encoding='utf-8')
df = pd.read_csv(data_path,header=0, names=['text', 'label'], encoding='gbk')
df = df.sample(frac=1)

# 将 data 分词
tokenizer.fit_on_texts(df['text'])
x_data = tokenizer.texts_to_sequences(df['text'])
y_data = df['label']

# 获取 - 训练数据,验证数据,测试数据
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.3,
                                                    shuffle=True, random_state=1)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,test_size=0.3,
                                                    shuffle=True, random_state=1)
# 获取词典数量
dict_word = tokenizer.word_index
vocab_size = len(dict_word)

# 将数据写入pickle文件
if not os.path.exists(dict_word_pkl_path):
    with open(dict_word_pkl_path,'wb') as f:
        pickle.dump(tokenizer, f)

print("训练数据样本：\n",x_train[0])
print("训练数据样本分布：\n", y_train.value_counts(),"共%d条"%(len(y_train)))
print("验证数据样本分布：\n", y_valid.value_counts(),"共%d条"%(len(y_valid)))
print("测试数据样本分布：\n", y_test.value_counts(),"共%d条"%(len(y_test)))

# 四,定义数据加载类
class DataLoader(Sequence):
    def __init__(self, x, y, batch_size=32):
        super(DataLoader, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        # return math.ceil(len(self.x) / self.batch_size)
        return math.floor(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        batch_x = self.x[index * self.batch_size: (index + 1) * self.batch_size]
        batch_y = self.y[index * self.batch_size: (index + 1) * self.batch_size]
        return pad_sequences(batch_x, padding='post', truncating='post'), batch_y

# 五,定义回调类
class EvalCallback(Callback):
    x_data = pad_sequences(x_data, padding='post', truncating='post')

    def __init__(self,):
        self.acc = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_sparse_categorical_accuracy'] > self.acc:
            self.acc = logs['val_sparse_categorical_accuracy']
            # model.save_weights(save_weight_path)
            model.save(save_model_path)

        # y_pred = model.predict(self.x_test, batch_size=64)

    def on_train_end(self, logs=None):
        y_pred = model.predict(self.x_data, batch_size=64)
        text = tokenizer.sequences_to_texts(x_data)
        text = [each.replace(" ", "") for each in text]
        with open(badcase_path, 'w') as fd:
            for each in zip(text, y_data, y_pred.argmax(-1)):
                fd.write("%s,%s,%s\n" % each)
        print(classification_report(y_data, y_pred.argmax(-1), labels=range(0, 21)))
        print(confusion_matrix(y_data, y_pred.argmax(-1)))

# 六,定义模型
inputs = keras.Input(shape=(None,))
output = Embedding(vocab_size + 1, 100)(inputs)
# lstm_layer1 = LSTM(128, return_state=False, return_sequences=True)

output = keras.layers.Bidirectional(
    LSTM(128, return_state=False, return_sequences=False)
    )(output)
output = Dense(64, activation="relu",kernel_regularizer=regularizers.l2(0.01))(output)
output = Dense(21, activation='softmax')(output)

model = Model(inputs, output)
model.summary()



# 七,配置模型
model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer=Adam(2e-3),
              metrics=[SparseCategoricalAccuracy()])

# 八,加载已有模型
if os.path.exists(save_model_path):
    model = keras.models.load_model(save_model_path)
    print("==== load weights!!! ====")

# 九,训练模型
history = model.fit(DataLoader(x_train, y_train),
          epochs=epochs,
          validation_data=DataLoader(x_valid, y_valid),
          callbacks=[EvalCallback()])

loss_max_value = max(history.history["loss"])

# 十,打印模型训练曲线
def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_'+label] = history.history['val_'+label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()
    
plot_learning_curves(history, 'sparse_categorical_accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 0, loss_max_value)

# 五,估计器预测测试数据集准确率
x_test = pd.DataFrame(x_test)
evaluate_rate = model.evaluate(
    x_test, y_test,
    batch_size = batch_size,
    verbose = 1)
 
print("估计器预测测试数据集的 损失和准确率分别为:%s,%s"%tuple(evaluate_rate))


