from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers.merge import add
from keras.engine import InputSpec
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import BatchNormalization
from keras import regularizers

import tensorflow as tf

reg_weights = 0.00001

def bn_relu():
    def bn_relu_func(x):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    return bn_relu_func

def res_conv(nb_filter, nb_row, nb_col, stride=(1, 1)):
    def _res_func(x):
        identity = x

        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(x)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Conv2D(nb_filter, (nb_row, nb_col), strides=stride, padding='same', kernel_regularizer=regularizers.l2(reg_weights))(a)
        y = BatchNormalization()(a)

        return add([identity, y])

    return _res_func


def dconv_bn_nolinear(nb_filter, nb_row, nb_col, stride=(2, 2), activation="relu"):
    def _dconv_bn(x):
        x = UnPooling2D(size=stride)(x)
        x = ReflectionPadding2D(padding=(int(nb_row/2), int(nb_col/2)))(x)
        x = Conv2D(nb_filter, (nb_row, nb_col), padding='valid', kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    return _dconv_bn


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), dim_ordering='default', **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.padding = padding
        if isinstance(padding, dict):
            if set(padding.keys()) <= {'top_pad', 'bottom_pad', 'left_pad', 'right_pad'}:
                self.top_pad = padding.get('top_pad', 0)
                self.bottom_pad = padding.get('bottom_pad', 0)
                self.left_pad = padding.get('left_pad', 0)
                self.right_pad = padding.get('right_pad', 0)
            else:
                raise ValueError('Unexpected key found in `padding` dictionary. '
                                 'Keys have to be in {"top_pad", "bottom_pad", '
                                 '"left_pad", "right_pad"}.'
                                 'Found: ' + str(padding.keys()))
        else:
            padding = tuple(padding)
            if len(padding) == 2:
                self.top_pad = padding[0]
                self.bottom_pad = padding[0]
                self.left_pad = padding[1]
                self.right_pad = padding[1]
            elif len(padding) == 4:
                self.top_pad = padding[0]
                self.bottom_pad = padding[1]
                self.left_pad = padding[2]
                self.right_pad = padding[3]
            else:
                raise TypeError('`padding` should be tuple of int '
                                'of length 2 or 4, or dict. '
                                'Found: ' + str(padding))

        if dim_ordering not in {'tf'}:
            raise ValueError('dim_ordering must be in {tf}.')
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def call(self, x, mask=None):
        top_pad = self.top_pad
        bottom_pad = self.bottom_pad
        left_pad = self.left_pad
        right_pad = self.right_pad

        paddings = [[0, 0], [left_pad, right_pad], [top_pad, bottom_pad], [0, 0]]

        return tf.pad(x, paddings, mode='REFLECT', name=None)

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'tf':
            rows = input_shape[1] + self.top_pad + self.bottom_pad if input_shape[1] is not None else None
            cols = input_shape[2] + self.left_pad + self.right_pad if input_shape[2] is not None else None

            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UnPooling2D(UpSampling2D):
    def __init__(self, size=(2, 2)):
        super(UnPooling2D, self).__init__(size)

    def call(self, x, mask=None):
        shapes = x.get_shape().as_list()
        w = self.size[0] * shapes[1]
        h = self.size[1] * shapes[2]
        return tf.image.resize_nearest_neighbor(x, (w, h))


class InstanceNormalize(Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalize, self).__init__(**kwargs)
        self.epsilon = 1e-3

    def call(self, x, mask=None):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, self.epsilon)))

    def compute_output_shape(self, input_shape):
        return input_shape

def create_vae(input_shape):
    # Encoder
    input = Input(shape=input_shape, name='image')

    enc1_conv = Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.00001))(input)
    enc1_bn_relu = bn_relu()(enc1_conv)
    
    enc2_conv = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.00001))(enc1_bn_relu)
    enc2_bn_relu = bn_relu()(enc2_conv)
    
    enc3_conv = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.00001))(enc2_bn_relu)
    enc3_bn_relu = bn_relu()(enc3_conv)
    
    enc4_conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.00001))(enc3_bn_relu)
    enc4_bn_relu = bn_relu()(enc4_conv)
    
    enc5_conv = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.00001))(enc4_bn_relu)
    enc5_bn_relu = bn_relu()(enc5_conv)
    
    enc6_conv = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.00001))(enc5_bn_relu)
    enc6_bn_relu = bn_relu()(enc6_conv)


    x0 = res_conv(128, 3, 3)(enc6_bn_relu)
    x1 = res_conv(128, 3, 3)(x0)
    x2 = res_conv(128, 3, 3)(x1)
    x3 = res_conv(128, 3, 3)(x2)
    x4 = res_conv(128, 3, 3)(x3)
    
    dec6 = res_conv(128, 3, 3)(x4)

    merge6 = concatenate([enc6_bn_relu, dec6], axis=3)
    dec5 = dconv_bn_nolinear(128, 3, 3, stride=(1, 1))(merge6)
    merge5 = concatenate([enc5_bn_relu, dec5], axis=3)
    dec4 = dconv_bn_nolinear(128, 3, 3, stride=(2, 2))(merge5)
    merge4 = concatenate([enc4_bn_relu, dec4], axis=3)
    dec3 = dconv_bn_nolinear(64, 3, 3, stride=(1, 1))(merge4)
    merge3 = concatenate([enc3_bn_relu, dec3], axis=3)
    dec2 = dconv_bn_nolinear(64, 3, 3, stride=(2, 2))(merge3)
    merge2 = concatenate([enc2_bn_relu, dec2], axis=3)
    dec1 = dconv_bn_nolinear(32, 3, 3, stride=(1, 1))(merge2)
    merge1 = concatenate([enc1_bn_relu, dec1], axis=3)
    dec0 = dconv_bn_nolinear(16, 3, 3, stride=(2, 2))(merge1)
    
    output = [Conv2D(1, (3, 3), padding='same', activation=None)(dec0)]

    # Full net
    vae_model = Model(input, output)

    return vae_model