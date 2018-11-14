import numpy as np

from tensorflow.python.keras.models import Input, Model
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.utils.vis_utils import plot_model

from modules import *


def HD_Net(input_shape, classes, levels, modules, filters, dilations, scales):
    input = Input(shape=input_shape)

    x = []
    x_pre = input

    for n_level, n_module, n_filter, dilation, scale in zip(np.arange(levels), modules, filters, dilations, scales):
        x_pre, output = hierarchical_layer(n_level, n_filter, classes, n_module, dilation, scale)(x_pre)
        print('level:', n_level, 'output:',output.shape, 'x_pre:', x_pre.shape)
        x.append(output)

    x = concatenate(x, axis=1)
    x = Lambda(fusion(32, classes))(x)
    model = Model(input, x)

    return  model


if __name__=='__main__':
    input_shape = (3, 8, 120, 120)
    classes = 10
    levels = 3
    modules = [3, 3, 3]
    filters = [32, 48, 72]
    dilations = [[1, 3, 5], [1, 2, 4], [1, 2, 2]]
    scales = [1, 2, 4]
    model = HD_Net(input_shape, classes, levels, modules, filters, dilations, scales)
    plot_model(model, 'model.png', show_shapes=True)