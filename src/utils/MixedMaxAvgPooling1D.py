import tensorflow as tf
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.initializers import  RandomUniform
from keras.layers import MaxPooling1D, AveragePooling1D, multiply, add, conv_utils


class MixedMaxAvgPooling1D(Layer):

    def __init__(self, name, alpha, method, input_dim, pool_size=2, strides=None,
                 padding='valid', data_format='channels_last', **kwargs):
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 1, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = data_format
        self.input_spec = InputSpec(ndim=3)
        self.alpha = alpha
        self.method = method
        self.name = name
        self.input_dim = input_dim
        super(MixedMaxAvgPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.alpha is None:
            if 'region' not in self.method:
                self.alpha = self.add_weight(name=self.name, shape=(1,),
                                             initializer=RandomUniform(minval=0.0, maxval=1.0))
            else:
                self.alpha = self.add_weight(name=self.name, shape=(self.input_dim/3, 1),
                                             initializer=RandomUniform(minval=0.0, maxval=1.0))
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
