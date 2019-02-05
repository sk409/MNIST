import numpy as np


def mean_squared_error(y, t):
    batch_size = y.shape[0]
    diff = y - t
    return 0.5 * np.sum(diff**2) / batch_size

def huber_loss(y, t, delta):
    a = np.abs(y - t)
    loss = np.where(a <= delta, 0.5 * a**2, delta * (a - 0.5 * delta))
    return loss

def cross_entropy_error(y, t, eps=1e-7):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + eps)) / batch_size