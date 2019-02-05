import functools
import numpy as np
import operator


import skml_config


class RandomInitializer:

    def __call__(self, input_shape, output_shape):
        return np.random.randn(*output_shape).astype(skml_config.config.f_type)

class Xavier:

    def __call__(self, input_shape, output_shape):
        assert len(input_shape) == 1 or len(input_shape) == 3
        input_size = functools.reduce(operator.mul, input_shape)
        return np.random.randn(*output_shape).astype(skml_config.config.f_type) / np.sqrt(input_size)


class He:

    def __call__(self, input_shape, output_shape):
        assert len(input_shape) == 1 or len(input_shape) == 3
        input_size = functools.reduce(operator.mul, input_shape)
        return np.random.randn(*output_shape).astype(skml_config.config.f_type) / np.sqrt(2 / input_size)