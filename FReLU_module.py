import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D, Lambda


def FReLU(inputs, kernel_size=3):
    # T(x)の部分
    x = DepthwiseConv2D(kernel_size, strides=(
        1, 1), padding='same', depthwise_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    # max(x, T(x))の部分
    x = tf.maximum(inputs, x)
    return x
