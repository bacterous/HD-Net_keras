from tensorflow.python.keras.layers import Activation, BatchNormalization
from tensorflow.python.keras.layers.convolutional import Conv3D, MaxPooling3D, Conv3DTranspose
from tensorflow.python.keras.layers.merge import add, concatenate
from tensorflow.python.keras import backend as K

DATA_FORMAT = 'channels_first'


def building_block(filters, dilation=1, activate=True):
    """
    3*3 convolution -> Batch normalization -> Relu
    """
    def layer(x):
        x = Conv3D(filters, 3, padding='same', dilation_rate=dilation, data_format=DATA_FORMAT)(x)
        x = BatchNormalization()(x)
        if activate:
            x = Activation('relu')(x)
        return x

    return layer


def shortcut(filters, use_bn=False):
    """
    input -> 1*1 conv -> BN -> output
    """
    def layer(x):
        x = Conv3D(filters, 1, padding='same', data_format=DATA_FORMAT)(x)
        x = BatchNormalization()(x)
        return x

    return layer


def dilated_res_block(filters, dilation=1):
    """
    input -> building_block -> building_block -> + -> output
          ↘ -----------(shortcut)------------- ↗
    """
    def left(x):
        x = building_block(filters, dilation)(x)
        x = building_block(filters, dilation, activate=False)(x)
        return x

    def right(x):
        x = shortcut(filters)(x)
        return x

    def layer(x):
        x = add([left(x), right(x)])
        x = Activation('relu')(x)
        return x

    return layer


def hierarchical_dilated_module(filters, modules, dilation):
    """
    input -> dilated_res_block -> dilated_res_block -> dilated_res_block -> concat -> output
          ↘ -------> ↓ ------------------> ↓ -----------------> ↓ ------- ↗
    """
    def layer(x):
        output = []
        output.append(x)

        for i in range(modules):
            x = dilated_res_block(filters, dilation[i])(output[i])
            x = dilated_res_block(filters, dilation[i])(x)
            output.append(x)

        return concatenate(output, axis=1)

    return layer


def head(filters, dilation=1):
    """
    input -> building_block -> dilated_res_block -> output
    """
    def layer(x):
        x = building_block(filters, dilation)(x)
        x = dilated_res_block(filters, dilation)(x)

        return x

    return layer


def down(filters, dilation=1):
    """
    input -> max_pool -> head -> output
    """
    def layer(x):
        x = MaxPooling3D(2, data_format=DATA_FORMAT)(x)
        x = head(filters, dilation)(x)

        return x

    return layer


def tail(filters, classes, scale=1, dilation=1):
    """
    input -> building_block -> (up sampling ->) building_block -> 1*1 conv -> n_classes output
    """
    def layer(x):
        x = building_block(filters, dilation)(x)
        if scale > 1:
            x = Conv3DTranspose(filters, kernel_size=scale, strides=scale, data_format=DATA_FORMAT)(x)
        x = building_block(int(filters/2), dilation)(x)
        x = Conv3D(classes, 1, data_format=DATA_FORMAT)(x)

        return x

    return layer


def fusion(filters, classes):
    """
          ↗  mean ↘
    input -> raw  -> concat -> building_block -> 1*1 conv -> output
          ↘  max  ↗
    """
    def layer(x):
        x_mean = K.expand_dims(K.mean(x, axis=1), axis=1)
        x_max = K.expand_dims(K.max(x, axis=1), axis=1)
        x = concatenate([x, x_mean, x_max], axis=1)
        x = building_block(filters)(x)
        x = Conv3D(classes, 1, data_format=DATA_FORMAT)(x)

        return x

    return layer


def hierarchical_layer(level, filters, classes, modules, dilation, scale):
    def layer(x):
        if level==0:
            x_pre = head(filters, dilation)(x)
        else:
            x_pre = down(filters, dilation)(x)
        x = hierarchical_dilated_module(filters, modules, dilation)(x_pre)
        x = tail(64, classes, scale)(x)

        return x_pre, x

    return layer










