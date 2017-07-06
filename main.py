# -*- coding: utf-8 -*-
# @Author: Zhaoyin Zhang
# @Date:2017/7/3 16:10 
# @Contact: 940942500@qq.com
import tensorflow as tf
from q1_classifier import SoftmaxModel
from sklearn.datasets import load_iris

class Config(object):
    """Holds model hyperparams and data information.

     The config class is used to store various hyperparameters and dataset
     information parameters. Model objects are passed a Config() object at
     instantiation.
     """

    iris = load_iris()
    input_data = iris.data
    input_labels = iris.target

    batch_size = 50
    # n_samples = 1024
    n_features = 4
    n_classes = 4



    # You may adjust the max_epochs to ensure convergence.
    max_epochs = 50
    # You may adjust this learning rate to ensure convergence.
    lr = 0.3


def test_SoftmaxModel():
    """Train softmax model for a number of steps."""

    config = Config()
    with tf.Graph().as_default():
        model = SoftmaxModel(config)
        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        # Run the Op to initialize the variables.
        init = tf.global_variables_initializer()
        sess.run(init)

        losses = model.fit(sess, model.input_data, model.input_labels)

    # If ops are implemented correctly, the average loss should fall close to zero
    # rapidly.
    assert losses[-1] < 1.1
    print("Basic (non-exhaustive) classifier tests pass\n")
if __name__ == '__main__':
    test_SoftmaxModel()
