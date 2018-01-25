from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import model

learning_rate = 0.03
training_steps = 1000
batch_size = 128
display_step = 200

num_input = 28
num_hidden = 64
num_classes = 10


def load_data():
    return input_data.read_data_sets("/tmp/data/", one_hot=True)


def load_model():
    x = tf.placeholder(tf.float32, [None, num_input, num_input])
    y = tf.placeholder(tf.float32, [None, num_classes])

    weights = tf.Variable(tf.random_normal([num_hidden, num_classes]))
    biases = tf.Variable(tf.random_normal([num_classes]))

    logits = model.RNN(x, weights, biases, num_input, num_hidden)
    prediction = tf.nn.softmax(logits)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return x, y, train_op, accuracy


def train(data):
    x, y, train_op, accuracy = load_model()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for step in range(1, training_steps + 1):
            batch_x, batch_y = data.train.next_batch(batch_size)
            batch_x = batch_x.reshape((batch_size, num_input, num_input))

            sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0 or step == 1:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                print("Step " + str(step) + ", Training Accuracy = " + "{:.3f}".format(acc))

        print("Training complete!")

        test_size = batch_size*10
        test_data = data.test.images[:test_size].reshape((-1, num_input, num_input))
        test_labels = data.test.labels[:test_size]
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_labels}))


if __name__ == "__main__":
    train(load_data())
