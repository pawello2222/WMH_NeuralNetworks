import tensorflow as tf
from tensorflow.contrib import rnn


def RNN(x, weights, biases, input_size, num_hidden):
    x = tf.unstack(x, input_size, 1)
    # tf.reshape(X, [-1, 784])

    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights) + biases
