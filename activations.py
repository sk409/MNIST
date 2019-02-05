import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    assert x.ndim == 2
    x -= np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    sum_exp = np.sum(exp, axis=1, keepdims=True)
    return exp / sum_exp