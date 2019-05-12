import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import MaxPooling1D, AveragePooling1D, multiply, add


class MixedMaxAvgPooling1D(Layer):

    def __init__(self, alpha, **kwargs):
        if alpha == -1:
            self.alpha = K.variable(0.0)
        super(MixedMaxAvgPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MixedMaxAvgPooling1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        tensor, activation = x
        x1 = MaxPooling1D(pool_size=3, strides=3)(tensor)
        x2 = AveragePooling1D(pool_size=3, strides=3)(tensor)
        outputs = add([tf.multiply(x1, self.alpha), x2])
        return outputs, self.alpha

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], 1, input_shape[2]), self.alpha]
