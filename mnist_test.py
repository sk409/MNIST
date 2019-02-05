import keras.backend as K
import numpy as np
import pickle
from keras.datasets import mnist

import skml_config
from models import load_model, Sequential
from util import convert_to_one_hot

num_classes = 10
img_rows = 28
img_cols = 28

(train_x, train_t), (test_x, test_t) = mnist.load_data()

test_x = np.reshape(test_x, (len(test_x), 1, img_rows, img_cols)).astype(skml_config.config.i_type)
test_t = convert_to_one_hot(test_t, num_classes)

model = load_model("models/best_model/mnist_model.pkl")


min_index = 0
max_index = len(test_t)
while True:
    print("===================================")
    print("何番目のデータを読み込みますか?")
    print("{}~{}の範囲で入力してください。".format(min_index, max_index))
    index = input()
    if not index.isdecimal():
        print("数値を入力してください。")
        continue
    index = int(index)
    if index < min_index or max_index < index:
        print("{}~{}の範囲で入力してください。".format(min_index, max_index))
        continue
    correct = np.argmax(test_t[index])
    print("*************************")
    print("正解は{}です。".format(correct))
    x = test_x[index : index + 1]
    prediction = np.argmax(model.predict(x)[0])
    print("モデルの予測は{}です。".format(prediction))
    if correct == prediction:
        print("成功しました!!")
    else:
        print("失敗しました...")
    print("*************************")
    