import copy
import functools
import numpy as np
import operator

import skml_config

from activations import *
from losses import cross_entropy_error, mean_squared_error, huber_loss
from util import filter_out_size, im2col, col2im, UnigramSampler
from weight_initializers import He

class ReLU:

    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.mask = None

    def has_params(self):
        return False

    def compile(self, input_shape):
        if input_shape == None:
            assert self.input_shape is not None
            input_shape = self.input_shape
        return input_shape

    def forward(self, x, train=True):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Sigmoid:

    def __init__(self, input_shape=None):
        self.input_shape = input_shape

    def has_params(self):
        return False

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        return input_shape

    def forward(self, x, train=True):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1.0 - self.out)



class Tanh:

    def __init__(self, input_shape=None):
        self.input_shape = input_shape

    def has_params(self):
        return False

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        return self.input_shape

    def forward(self, x, train=True):
        self.y = np.tanh(x)
        return self.y

    def backward(self, dout):
        return dout * (1 - self.y**2)


class Flatten:

    def __init__(self, input_shape=None):
        self.input_shape = input_shape

    def has_params(self):
        return False

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        out_size = functools.reduce(operator.mul, self.input_shape)
        return (out_size,)

    def forward(self, x, train=True):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Dropout:
   
    def __init__(self, dropout_ratio=0.5, input_shape=None):
        self.dropout_ratio = dropout_ratio
        self.input_shape = input_shape

    def has_params(self):
        return False

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        return self.input_shape

    def forward(self, x, train=True):
        if train:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Affine:

    def __init__(self, out_size, weight_initializer=He(), input_shape=None):
        self.out_size = out_size
        self.weight_initializer = weight_initializer
        self.input_shape = input_shape
        self.w = None
        self.b = None
        self.x = None
        self.params = None
        self.grads = None

    def has_params(self):
        return True

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        assert len(self.input_shape) == 1
        self.w = self.weight_initializer(self.input_shape, (self.input_shape[0], self.out_size))
        self.b = np.zeros(self.out_size, skml_config.config.f_type)
        self.params = [self.w, self.b]
        self.grads = [np.zeros_like(self.w, skml_config.config.f_type), np.zeros_like(self.b, skml_config.config.f_type)]
        return (self.out_size,)

    def forward(self, x, train=True):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        dw = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dw
        self.grads[1][...] = db
        return np.dot(dout, self.w.T)


class Convolution:

    def __init__(self, out_channel, kernel, stride=1, padding=0, weight_initializer=He(), input_shape=None):
        self.out_channel = out_channel
        self.padding = padding
        self.weight_initializer = weight_initializer
        self.input_shape = input_shape
        if type(kernel) == tuple:
            assert len(kernel) == 2
            self.fh, self.fw = kernel
        else:
            assert type(kernel) == int
            self.fh = kernel
            self.fw = kernel
        if type(stride) == tuple:
            assert len(stride) == 2
            self.stride_h, self.stride_w = stride
        else:
            assert type(stride) == int
            self.stride_h = stride
            self.stride_w = stride


    def has_params(self):
        return True
    
    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        assert len(self.input_shape) == 3
        ic, h, w = self.input_shape
        oc, fw, fh = self.out_channel, self.fw, self.fh
        out_h, out_w = filter_out_size(h, w, fh, fw, self.stride_h, self.stride_w, self.padding)
        self.w = self.weight_initializer(self.input_shape, (oc, ic, fh, fw))
        self.b = np.zeros(oc, skml_config.config.f_type)
        self.params = [self.w, self.b]
        self.grads = [np.zeros_like(self.w, skml_config.config.f_type), np.zeros_like(self.b, skml_config.config.f_type)]
        return (oc, out_h, out_w)

    def forward(self, x, train=True):
        oc, _, fh, fw = self.w.shape
        n, _, h, w = x.shape
        out_h, out_w = filter_out_size(h, w, fh, fw, self.stride_h, self.stride_w, self.padding)
        col = im2col(x, fh, fw, self.stride_h, self.stride_w, self.padding)
        col_w = self.w.reshape(oc, -1).T
        out = np.dot(col, col_w) + self.b
        out = out.reshape(n, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.x = x
        self.col = col
        self.col_w = col_w
        return out
        
    def backward(self, dout):
        oc, c, fh, fw = self.w.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, oc)
        db = np.sum(dout, axis=0)
        dW = np.dot(self.col.T, dout).transpose(1, 0).reshape(oc, c, fh, fw)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        dcol = np.dot(dout, self.col_w.T)
        return col2im(dcol, self.x.shape, fh, fw, self.stride_h, self.stride_w, self.padding)



class ResidualBlock:

    def __init__(self, *layers, input_shape=None):
        self.layers = layers
        self.input_shape = input_shape
        self.params = []
        self.grads = []

    def has_params(self):
        return True

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        output_shape = self.input_shape
        for layer in self.layers:
            output_shape = layer.compile(output_shape)
            if layer.has_params():
                self.params += layer.params
                self.grads += layer.grads
        assert len(output_shape) == 3
        self.shortcut_path = None
        if self.input_shape != output_shape:
            stride_h = int((self.input_shape[1] - 1) / (output_shape[1] - 1))
            stride_w = int((self.input_shape[2] - 1) / (output_shape[2] - 1))
            self.shortcut_path = Convolution(output_shape[0], 1, (stride_h, stride_w))
            shortcut_shape = self.shortcut_path.compile(self.input_shape)
            assert output_shape == shortcut_shape
            self.params += self.shortcut_path.params
            self.grads += self.shortcut_path.grads
        return output_shape


    def forward(self, x, train=True):
        out = x
        shortcut = x
        for layer in self.layers:
            out = layer.forward(out, train)
        if self.shortcut_path is not None:
            shortcut = self.shortcut_path.forward(x, train)
        return out + shortcut

    def backward(self, dout):
        dx = dout
        dshortcut = dout
        for layer in reversed(self.layers):
            dx = layer.backward(dx)
        if self.shortcut_path is not None:
            dshortcut = self.shortcut_path.backward(dshortcut)
        return dx + dshortcut



class BatchNormalization:

    def __init__(self, initial_gamma=1.0, initial_beta=0.0, momentum=0.99, running_mean=None, running_var=None, input_shape=None, eps=10e-7):
        assert 0.0 <= momentum <= 1.0
        self.initial_gamma = initial_gamma
        self.initial_beta = initial_beta
        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var
        self.input_shape = input_shape
        self.eps = eps

    def has_params(self):
        return True

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        if len(self.input_shape) == 1:
            d, = self.input_shape
        else:
            assert len(self.input_shape) == 3
            c, h, w = self.input_shape
            d = c * h * w
        if self.running_mean is None:
            self.running_mean = np.zeros(d, skml_config.config.f_type)
        else:
            assert len(self.running_mean) == d
        if self.running_var is None:
            self.running_var = np.ones(d, skml_config.config.f_type)
        else:
            assert len(self.running_var) == d
        self.gamma = np.array([self.initial_gamma] * d, skml_config.config.f_type)
        self.beta = np.array([self.initial_beta] * d, skml_config.config.f_type)
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma, skml_config.config.f_type), np.zeros_like(self.beta, skml_config.config.f_type)]
        return self.input_shape

    def forward(self, x, train=True):
        if x.ndim != 2:
            n, *_ = x.shape
            x = x.reshape(n, -1)
        out = self.__forward(x, train)
        return out.reshape((x.shape[0], *self.input_shape))
            
    def __forward(self, x, train):
        if train:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + self.eps)
            xn = xc / std
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + self.eps)))
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            n, *_ = dout.shape
            dout = dout.reshape(n, -1)
        dx = self.__backward(dout)
        dx = dx.reshape((dout.shape[0], *self.input_shape))
        return dx

    def __backward(self, dout):
        batch_size = dout.shape[0]
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / batch_size
        self.grads[0][...] = dgamma
        self.grads[1][...] = dbeta
        return dx


#######################
#######################
# 未テスト
#######################
#######################
class Embedding:

    def __init__(self, w, input_shape=None):
        self.params = [w]
        self.grads = [np.zeros_like(w)]
        self.input_shape = input_shape

    def has_params(self):
        return True

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        assert len(input_shape) == 1
        w, = self.params
        _, *output_shape = w.shape
        return (len(input_shape), *output_shape)

    def forward(self, idx, train=True):
        w, = self.params
        self.idx = idx
        return w[idx]

    def backward(self, dout):
        self.grads[...] = 0
        np.add.at(self.grads, self.idx, dout)


#######################
#######################
# 未テスト
#######################
#######################
class EmbeddingDot:

    def __init__(self, w, input_shape=None):
        self.embed = Embedding(w, input_shape)
        self.params = self.embed.params
        self.grads = self.embed.grads
    
    def forward(self, x, idx, train=True):
        target_w = self.embed.forward(idx)
        self.x = x
        self.target_w = target_w
        return np.sum(x * target_w, axis=1)

    def backward(self, dout):
        dout = np.reshape(dout, (dout.shape[0], 1))
        dw = dout * self.x
        self.embed.backward(dw)
        return dout * self.target_w


class MaxPooling:

    def __init__(self, kernel, stride=None, padding=0, input_shape=None):
        self.padding = padding
        self.input_shape = input_shape
        if type(kernel) == tuple:
            assert len(kernel) == 2
            self.fh, self.fw = kernel
        else:
            assert type(kernel) == int
            self.fh = kernel
            self.fw = kernel
        if stride is None:
            self.stride_h = self.fh
            self.stride_w = self.fw
        elif type(stride) == tuple:
            assert len(stride) == 2
            self.stride_h, self.stride_w = stride
        else:
            assert type(stride) == int
            self.stride_h = stride
            self.stride_w = stride
            

    def has_params(self):
        return False

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        assert len(self.input_shape) == 3
        c, h, w = input_shape
        out_h, out_w = filter_out_size(h, w, self.fh, self.fw, self.stride_h, self.stride_w, self.padding)
        return (c, out_h, out_w)

    def forward(self, x, train=True):
        n, c, h, w = x.shape
        self.batch_size = n
        out_h, out_w = filter_out_size(h, w, self.fh, self.fw, self.stride_h, self.stride_w, self.padding)
        col = im2col(x, self.fh, self.fw, self.stride_h, self.stride_w, self.padding)
        col = col.reshape(-1, self.fh*self.fw)
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(n, out_h, out_w, c).transpose(0, 3, 1, 2)
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.fh * self.fw
        dmax = np.zeros((dout.size, pool_size), skml_config.config.f_type)
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, (self.batch_size, *self.input_shape), self.fh, self.fw, self.stride_h, self.stride_w, self.padding)
        return dx


class AveragePooling:

    def __init__(self, kernel, stride=None, padding=0, input_shape=None):
        self.padding = padding
        self.input_shape = input_shape
        if type(kernel) == tuple:
            assert len(kernel) == 2
            self.fh, self.fw = kernel
        else:
            assert type(kernel) == int
            self.fh = kernel
            self.fw = kernel
        if stride is None:
            self.stride_h = self.fh
            self.stride_w = self.fw
        elif type(stride) == tuple:
            assert len(stride) == 2
            self.stride_h, self.stride_w = stride
        else:
            assert type(stride) == int
            self.stride_h = stride
            self.stride_w = stride


    def has_params(self):
        return False

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        assert len(self.input_shape) == 3
        c, h, w = input_shape
        out_h, out_w = filter_out_size(h, w, self.fh, self.fw, self.stride_h, self.stride_w, self.padding)
        return (c, out_h, out_w)

    def forward(self, x, train=True):
        n, c, h, w = x.shape
        self.batch_size = n
        out_h, out_w = filter_out_size(h, w, self.fh, self.fw, self.stride_h, self.stride_w, self.padding)
        col = im2col(x, self.fh, self.fw, self.stride_h, self.stride_w, self.padding)
        col = col.reshape(-1, self.fh*self.fw)
        out = np.mean(col, axis=1)
        out = out.reshape(n, out_h, out_w, c).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.fh * self.fw
        davg = dout / pool_size
        davg = np.repeat(davg, pool_size)
        davg = davg.reshape(dout.shape + (pool_size,)) 
        dcol = davg.reshape(davg.shape[0] * davg.shape[1] * davg.shape[2], -1)
        return col2im(dcol, (self.batch_size, *self.input_shape), self.fh, self.fw, self.stride_h, self.stride_w, self.padding)


class GlobalAveragePooling:

    def __init__(self, input_shape=None):
        self.input_shape = input_shape

    def has_params(self):
        return False

    def compile(self, input_shape):
        if input_shape is None:
            assert self.input_shape is not None
        else:
            self.input_shape = input_shape
        assert len(self.input_shape) == 3
        c = self.input_shape[0]
        return (c,)

    def forward(self, x, train=True):
        self.batch_size = x.shape[0]
        n = x.shape[0]
        c = x.shape[1]
        out = x.reshape(n, c, -1)
        return np.mean(out, axis=2)

    def backward(self, dout):
        n = self.batch_size
        c, h, w = self.input_shape
        dgbl = dout / (h * w)
        dgbl = np.repeat(dout, h * w)
        dgbl = dgbl.reshape(n, c, h, w)
        return dgbl



class MeanSquaredError:

    def forward(self, y, t):
        self.y = y
        self.t = t
        return mean_squared_error(self.y, self.t)

    def backward(self, dout=1):
        return dout * (self.y - self.t)




class HuberLoss:

    def __init__(self, delta):
        assert 0 < delta
        self.delta = delta

    def forward(self, y, t):
        batch_size = y.shape[0]
        self.diff = y - t
        self.abs = np.abs(self.diff)
        loss = np.where(self.abs <= self.delta, 0.5 * self.diff**2, self.delta * (self.abs - 0.5 * self.delta))
        return np.sum(loss) / batch_size

    def backward(self, dout=1):
        return dout * np.where(self.abs <= self.delta, self.diff, np.where(self.diff < 0, -self.delta, self.delta))




########################
# 未テスト
########################
class SigmoidCrossEntropy:

    def forward(self, x, t):
        self.y = sigmoid(x)
        self.t = t
        return cross_entropy_error(self.y, t)

    def backward(self, dout=1):
        return dout * (self.y - self.t)



class SoftmaxCrossEntropy:

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t
        return cross_entropy_error(self.y, t)

    def backward(self, dout=1):
        return dout * (self.y - self.t)


#######################
#######################
# 未テスト
#######################
#######################
class NegativeSamplingLoss:

    def __init__(self, w, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, sample_size, power)
        self.loss_layers = [SigmoidCrossEntropy() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(w) for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        score = self.embed_dot_layers[0].forward(x, target)
        correct_label = np.ones(batch_size, skml_config.config.i_type)
        loss = self.loss_layers[0].forward(score, correct_label)
        negative_label = np.zeros(batch_size, skml_config.config.i_type)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(x, negative_target)
            loss += self.loss_layers[i + i].forward(score, negative_label)
        return loss

    def backward(self, dout=1):
        dx = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dx += l1.backward(dscore)
        return dx