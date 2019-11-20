#!/usr/bin/env python3

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import numpy as np
import random
import utils

input_dim = 1                # 入力データの次元数：実数値1個なので1を指定
output_dim = 1               # 出力データの次元数：同上
num_hidden_units = 128       # 隠れ層のユニット数
batch_size = 300             # ミニバッチサイズ
num_of_training_epochs = 100 # 学習エポック数
learning_rate = 0.001        # 学習率
num_training_samples = 1000  # 学習データのサンプル数

# データを作成
def create_data(nb_of_samples):
    # 長さを対数正規分布に従って決める
    leng = np.around(np.random.lognormal(np.log(5.0), 0.5, (nb_of_samples, 1))).astype("int")
    max_sequence_len = leng.max()
    # 乱数で {0.0, 1.0} の列を生成する
    X = np.random.randint(0, 2, (nb_of_samples, max_sequence_len)).astype("float32")
    # 長さを超えた部分を-1.0に置き換える
    X[np.arange(max_sequence_len).reshape((1, -1)) >= leng] = -1.0
    # 各行の-1.0を除いた総和を正解ラベルとする
    t = np.ma.array(X, mask=(X == -1.0)).sum(axis=1)
    # LSTMに与える入力は (サンプル, 時刻, 特徴量の次元) の3次元になる。
    return X.reshape((nb_of_samples, max_sequence_len, 1)), t

# 乱数シードを固定値で初期化
random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

X, t = create_data(num_training_samples)

# モデル構築
model = Sequential()
# パディングの値を指定してMaskingレイヤーを作成する
model.add(Masking(
    input_shape=(None, input_dim),
    mask_value=-1.0))
model.add(LSTM(
    num_hidden_units,
    return_sequences=False))
model.add(Dense(output_dim))
model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate),metrics=['accuracy'])
model.summary()

# 学習
history=model.fit(
    X, t,
    batch_size=batch_size,
    epochs=num_of_training_epochs,
    validation_split=0.1
)
utils.showGrapth(history)
# 予測
# 任意の長さの入力を受け付ける
test = np.array([1, 1, 1, 0, 1, 0, 1]).astype("float32")
# (a) 長さを変えずに入力
print(model.predict(test.reshape((1, -1, 1))))                                                    # [[4.9639335]]
# (b) 後ろに適当な数の-1.0を追加して入力
print(model.predict(np.pad(test, (0, 10), "constant", constant_values=-1.0).reshape((1, -1, 1)))) # [[4.9639335]]