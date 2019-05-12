import tensorflow as tf
from keras.layers import MaxPooling1D, AveragePooling1D, add, K, RepeatVector



# define mixed max-average pooling layer
def mixed_pooling(inputs, alpha, size=2):
    """Mixed pooling operation, nonresponsive
       Combine max pooling and average pooling in fixed proportion specified by alpha a:
        f mixed (x) = a * f max(x) + (1-a) * f avg(x)
        arguments:
          inputs: tensor of shape [batch size, height, width, channels]
          size: an integer, width and height of the pooling filter
          alpha: the scalar mixing proportion of range [0,1]
        return:
          outputs: tensor of shape [batch_size, height//size, width//size, channels]
    """
    if alpha == -1:
        alpha = tf.Variable(initial_value=0.0, trainable=True)
    x1 = MaxPooling1D(pool_size=size, strides=3)(inputs)
    x2 = AveragePooling1D(pool_size=size, strides=3)(inputs)
    outputs = add([tf.multiply(x1, alpha), x2])
    #return x1
    return [alpha, outputs]
