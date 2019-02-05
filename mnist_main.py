import numpy as np
import pickle

from datetime import datetime
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

import skml_config
from layers import Affine, BatchNormalization, Convolution, GlobalAveragePooling, MaxPooling, ReLU, SoftmaxCrossEntropy
from models import load_model, Sequential
from optimizers import Adam, SGD
from util import convert_to_one_hot
from weight_initializers import Xavier

num_classes = 10
save_path = "models/mnist_model.pkl"

img_rows = 28
img_cols = 28
input_shape = (1, img_rows, img_cols)
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = np.reshape(train_x, (len(train_x), 1, img_rows, img_cols)).astype(skml_config.config.i_type)
train_y = convert_to_one_hot(train_y, num_classes)
test_x = np.reshape(test_x, (len(test_x), 1, img_rows, img_cols)).astype(skml_config.config.i_type)
test_y = convert_to_one_hot(test_y, num_classes)

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y)


filters = 64
model = Sequential()
model.add(Convolution(filters, 3, input_shape=input_shape))
model.add(BatchNormalization())
model.add(ReLU())
model.add(MaxPooling(2))
model.add(Convolution(filters, 3))
model.add(BatchNormalization())
model.add(ReLU())
model.add(GlobalAveragePooling())
model.add(Affine(num_classes))
model.compile(SoftmaxCrossEntropy(), Adam())

train_batch_size = 100
valid_batch_size = 1
print("訓練開始: {}".format(datetime.now().strftime("%Y/%m/%d %H:%M")))
model.fit(train_x, train_y, train_batch_size, 20, validation_data=(valid_batch_size, valid_x, valid_y), validation_steps=1)
print("訓練終了: {}".format(datetime.now().strftime("%Y/%m/%d %H:%M")))

model.save(save_path)

loss, acc = model.evaluate(test_x, test_y)
print("Test loss: {}".format(loss))
print("Test acc: {}".format(acc))