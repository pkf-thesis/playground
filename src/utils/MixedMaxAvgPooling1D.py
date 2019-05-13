import tensorflow as tf
from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from keras.layers import MaxPooling1D, AveragePooling1D, multiply, add, conv_utils


class MixedMaxAvgPooling1D(Layer):

    def __init__(self, name, alpha, pool_size=2, strides=None,
                 padding='valid', data_format='channels_last', **kwargs):
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 1, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = data_format
        self.input_spec = InputSpec(ndim=3)
        self.alpha = alpha
        self.name = name
        super(MixedMaxAvgPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.alpha == -1:
            self.alpha = self.add_weight(name=self.name, shape=(1,), initializer='zeros')
        super(MixedMaxAvgPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        dummy_axis = 2 if self.data_format == 'channels_last' else 3
        max = MaxPooling1D(pool_size=3, strides=3)(inputs)
        avg = AveragePooling1D(pool_size=3, strides=3)(inputs)
        outputs = add([tf.multiply(max, self.alpha), tf.multiply(avg, 1 - self.alpha)])
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            steps = input_shape[2]
            features = input_shape[1]
        else:
            steps = input_shape[1]
            features = input_shape[2]
        length = conv_utils.conv_output_length(steps,
                                               self.pool_size[0],
                                               self.padding,
                                               self.strides[0])
        if self.data_format == 'channels_first':
            return (input_shape[0], features, length)
        else:
            return (input_shape[0], length, features)
