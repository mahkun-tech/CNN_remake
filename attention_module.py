import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

rrelu = tfa.activations.rrelu


def ChannelAttention_Module(input: tf.keras.Model, ratio=8):
    channel = input.shape[-1]

    shared_dense_one = tf.keras.layers.Dense(channel // ratio,
                                             activation=rrelu,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
    shared_dense_two = tf.keras.layers.Dense(channel,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input)
    avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input)
    max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    x = tf.keras.layers.Add()([avg_pool, max_pool])
    x = tf.keras.layers.Activation('sigmoid')(x)

    return tf.keras.layers.multiply([input, x])


def SpatialAttention_Module(input: tf.keras.Model, kernel_size=3):
    avg_pool = tf.keras.layers.Lambda(
        lambda x: K.mean(x, axis=3, keepdims=True))(input)
    max_pool = tf.keras.layers.Lambda(
        lambda x: K.max(x, axis=3, keepdims=True))(input)
    x = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    for i in [64, 32, 16]:
        x = tf.keras.layers.Conv2D(filters=i,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   padding='same',
                                   activation=rrelu,
                                   kernel_initializer='he_normal',
                                   use_bias=False)(x)
    x = tf.keras.layers.Conv2D(filters=1,
                               kernel_size=kernel_size,
                               strides=1,
                               padding='same',
                               activation='sigmoid',
                               kernel_initializer='he_normal',
                               use_bias=False)(x)

    return tf.keras.layers.multiply([input, x])
