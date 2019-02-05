import copy
import numpy as np
import pickle

from datetime import datetime
from math import ceil

import skml_config


def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def deepcopy_model(model):
    return copy.deepcopy(model)

class Sequential:

    def __init__(self):
        self.layers = []
        self.params = []
        self.grads = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss_layer, optimizer):
        self.loss_layer = loss_layer
        self.optimizer = optimizer
        input_shape = None
        for layer in self.layers:
            input_shape = layer.compile(input_shape)
            if layer.has_params():
                self.params += layer.params
                self.grads += layer.grads

    def predict(self, x, train=False):
        y = x
        for layer in self.layers:
            assert y.dtype == skml_config.config.f_type or y.dtype == skml_config.config.i_type
            y = layer.forward(y, train)
        return y

    def fit(self, x, y, batch_size, epochs, verbose=1, shuffle=True, initial_epoch=0, validation_data=None, validation_steps=-1, callbacks=None):
        train_data_size = len(x)
        iterations = ceil(train_data_size / batch_size)
        if validation_data is not None:
            valid_batch_size, valid_x, valid_y = validation_data
            valid_data_size = len(valid_x)
            valid_iterations = ceil(valid_data_size / valid_batch_size)
            valid_data_indices = np.arange(valid_data_size)
        for epoch in range(initial_epoch + 1, epochs + 1):
            loss = 0
            accuracy = 0
            train_data_indices = np.random.permutation(train_data_size) if shuffle else np.arange(train_data_size)
            for iteration in range(iterations):
                batch_start_index = iteration * batch_size
                batch_end_index = min(train_data_size, (iteration + 1) * batch_size)
                batch_indices = train_data_indices[batch_start_index : batch_end_index]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                batch_loss, batch_accuracy = self.forward(batch_x, batch_y)
                loss += batch_loss
                accuracy += batch_accuracy
                self.backward()
                self.optimizer.update(self.params, self.grads)
                if 0 < verbose and iteration % verbose == 0:
                    proportion = iteration / iterations 
                    bar_length = 10
                    completed = int(proportion * bar_length)
                    print("\rEpoch-{}  |{}{}| {:.2f}%".format(epoch, "#"*completed, "-"*(bar_length - completed), proportion*100), end="")
            if 0 < validation_steps and epoch % validation_steps == 0:
                print("\r", end="")
                loss /= iterations
                accuracy /= iterations
                if validation_data is None:
                    print("Epoch-{}  train-loss: {:.5f}  train_accuracy: {:.5f}  ||-{}".format(epoch, loss, accuracy, datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
                else:
                    valid_loss = 0
                    valid_accuracy = 0
                    for iteration in range(valid_iterations):
                        batch_start_index = iteration * valid_batch_size
                        batch_end_index = min(valid_data_size, (iteration + 1) * valid_batch_size)
                        batch_indices = valid_data_indices[batch_start_index : batch_end_index]
                        batch_x = valid_x[batch_indices]
                        batch_y = valid_y[batch_indices]
                        batch_loss, batch_accuracy = self.forward(batch_x, batch_y, False)
                        valid_loss += batch_loss
                        valid_accuracy += batch_accuracy
                    valid_loss /= valid_iterations
                    valid_accuracy /= valid_iterations
                    print("Epoch-{}  train-loss: {:.5f}  valid-loss: {:.5f}  train_accuracy: {:.5f}  valid_accuracy: {:.5f}    ||-{}".format(epoch, loss, valid_loss, accuracy, valid_accuracy, datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
            if callbacks is not None:
                for callback in callbacks:
                    callback(epoch)
    
    def forward(self, x, t, train=True):
        y = self.predict(x, train)
        assert y.dtype == skml_config.config.f_type or y.dtype == skml_config.config.i_type
        accuracy = (np.argmax(y, axis=1) == np.argmax(t, axis=1)).sum() / len(y)
        loss = self.loss_layer.forward(y, t)
        return loss, accuracy

    def backward(self):
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def evaluate(self, x, t):
        loss = 0.0
        acc = 0.0
        for i in range(len(x)):
            l, a = self.forward(x[i:i+1], t[i:i+1], False)
            loss += l
            acc += a
        return loss / len(x), acc / len(x)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)