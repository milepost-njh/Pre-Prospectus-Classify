import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import Sequence
import math
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, Dense, GlobalMaxPool1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import Callback

tokenizer = Tokenizer(filters='', char_level=True)

df = pd.read_csv('data/v2_split_data.csv',
                 header=None, names=['text', 'label'], encoding='gbk')

tokenizer.fit_on_texts(df['text'])
x_train = tokenizer.texts_to_sequences(df['text'])
y_train = df['label']

vocab_size = len(tokenizer.word_index)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, shuffle=True, random_state=1)
print(x_train[0])
print("训练数据样本分布：\n", y_train.value_counts())


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


class EvalCallback(Callback):
    x_test = pad_sequences(x_test, padding='post', truncating='post')

    def __init__(self):
        self.acc = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_sparse_categorical_accuracy'] > self.acc:
            self.acc = logs['val_sparse_categorical_accuracy']
            model.save_weights('best_model/best_model.weights')

        y_pred = model.predict(self.x_test, batch_size=64)
        # print(classification_report(y_test, y_pred.argmax(-1), labels=range(0, 21)))
        # print(confusion_matrix(y_test, y_pred.argmax(-1)))

    def on_train_end(self, logs=None):
        y_pred = model.predict(self.x_test, batch_size=64)
        text = tokenizer.sequences_to_texts(x_test)
        text = [each.replace(" ", "") for each in text]
        with open("badcase/test.csv", 'w') as fd:
            for each in zip(text, y_test, y_pred.argmax(-1)):
                fd.write("%s,%s,%s\n" % each)
        print(classification_report(y_test, y_pred.argmax(-1), labels=range(0, 21)))
        print(confusion_matrix(y_test, y_pred.argmax(-1)))

inputs = keras.Input(shape=(None,))
output = Embedding(vocab_size + 1, 100)(inputs)
# lstm_layer1 = LSTM(128, return_state=False, return_sequences=True)
output = LSTM(128, return_state=False, return_sequences=False)(output)
output = Dense(21, activation='softmax')(output)

model = Model(inputs, output)
model.summary()

if os.path.exists('best_model/best_model.weights'):
    model.load_weights('best_model/best_model.weights')
    print("==== load weights!!! ====")

model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer=Adam(2e-3),
              metrics=[SparseCategoricalAccuracy()])

model.fit(DataLoader(x_train, y_train),
          epochs=40,
          validation_data=DataLoader(x_test, y_test),
          callbacks=[EvalCallback()])

# EvalCallback().on_epoch_end(1)

