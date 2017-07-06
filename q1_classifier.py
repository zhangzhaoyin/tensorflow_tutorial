# -*- coding: utf-8 -*-
# @Author: Zhaoyin Zhang
# @Date:2017/7/5 16:33 
# @Contact: 940942500@qq.com
import tensorflow as tf
import numpy as np
from model import Model
from utils import data_iterator
import time
import sys

class SoftmaxModel(Model):
    """Implements a Softmax classifier with cross-entropy loss."""

    def load_data(self):
        """Creates a synthetic dataset and stores it in memory."""
        # np.random.seed(1234)
        # self.input_data = np.random.rand(
        #     self.config.n_samples, self.config.n_features)
        # self.input_labels = np.ones((self.config.n_samples,), dtype=np.int32)

        self.input_data  = self.config.input_data
        self.input_labels = self.config.input_labels

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors.

            These placeholders are used as inputs by the rest of the model building
            code and will be fed data during training.

            Adds following nodes to the computational graph

            input_placeholder: Input placeholder tensor of shape
                               (batch_size, n_features), type tf.float32
            labels_placeholder: Labels placeholder tensor of shape
                               (batch_size, n_classes), type tf.int32

            Add these placeholders to self as the instance variables

              self.input_placeholder
              self.labels_placeholder

            (Don't change the variable names)
            """
        self.input_placeholder = tf.placeholder(
            tf.float32, shape = (self.config.batch_size, self.config.n_features)
        )

        self.labels_placeholder = tf.placeholder(
            tf.float32, shape = (self.config.batch_size, self.config.n_classes)
        )

    def create_feed_dict(self, input_batch, label_batch):
        """Creates the feed_dict for softmax classifier.

            A feed_dict takes the form of:

            feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
            }

            If label_batch is None, then no labels are added to feed_dict.

            Hint: The keys for the feed_dict should match the placeholder tensors
                  created in add_placeholders.

            Args:
              input_batch: A batch of input data.
              label_batch: A batch of label data.
            Returns:
              feed_dict: The feed dictionary mapping from placeholders to values.
            """
        feed_dict = {
            self.input_placeholder : input_batch,
            self.labels_placeholder: label_batch,
        }

        return feed_dict

    def add_training_op(self,loss):
        """Sets up the training Ops.

            Creates an optimizer and applies the gradients to all trainable variables.
            The Op returned by this function is what must be passed to the
            `sess.run()` call to cause the model to train. See

            https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

            for more information.

            Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
                  Calling optimizer.minimize() will return a train_op object.

            Args:
              loss: Loss tensor, from cross_entropy_loss.
            Returns:
              train_op: The Op for training.
            """

        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        global_step = tf.Variable(0, name="global", trainable=False)
        train_op = optimizer.minimize(loss,global_step)

        return train_op

    def add_model(self, input_data):
        """Adds a linear-layer plus a softmax transformation

            The core transformation for this model which transforms a batch of input
            data into a batch of predictions. In this case, the mathematical
            transformation effected is

            y = softmax(xW + b)

            Hint: Make sure to create tf.Variables as needed. Also, make sure to use
                  tf.name_scope to ensure that your name spaces are clean.
            Hint: For this simple use-case, it's sufficient to initialize both weights W
                  and biases b with zeros.

            Args:
              input_data: A tensor of shape (batch_size, n_features).
            Returns:
              out: A tensor of shape (batch_size, n_classes)
            """

        n_features, n_classes = self.config.n_features, self.config.n_classes
        with tf.name_scope("softmax_linear"):
            weights = tf.Variable(
                tf.zeros([n_features, n_classes]),
            name = "weights")
            biases = tf.Variable(tf.zeros([n_classes]),name="biases")
            logits = tf.matmul(input_data, weights) + biases
            out  = tf.nn.softmax(logits)
        return out



    def add_loss_op(self, pred):

        """Adds cross_entropy_loss ops to the computational graph.

           Hint: Use the cross_entropy_loss function we defined. This should be a very
                 short function.
           Args:
             pred: A tensor of shape (batch_size, n_classes)
           Returns:
             loss: A 0-d tensor (scalar)

           """
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder))

        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))

        # loss = -tf.reduce_sum(self.labels_placeholder * tf.log(pred))


        # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder))
        # tf.add_to_collection('total_loss', cross_entropy)
        # loss = tf.add_n(tf.get_collection('total_loss'))
        # ### END YOUR CODE
        # return los


        # loss = self.cross_entropy_loss(self.labels_placeholder, pred)
        return loss

    def run_epoch(self,sess, input_data, input_labels,shuffle=True, verbose=True):
        """Runs an epoch of training.

            Trains the model for one-epoch.

            Args:
              sess: tf.Session() object
              input_data: np.ndarray of shape (n_samples, n_features)
              input_labels: np.ndarray of shape (n_samples, n_classes)
            Returns:
              average_loss: scalar. Average minibatch loss of model on epoch.
            """
        # And then after everything is built, start the training loop.

        orig_X, orig_y = input_data, input_labels
        total_loss = []
        total_correct_examples = 0
        total_processed_examples = 0
        total_steps = len(orig_X) / self.config.batch_size

        for step, (input_batch, label_batch) in enumerate(data_iterator(orig_X, orig_y,
                      batch_size = self.config.batch_size,
                      label_size = self.config.n_classes,shuffle=shuffle)):

            feed_dict = self.create_feed_dict(input_batch, label_batch)

            loss, total_correct, _ = sess.run(
                [self.loss, self.correct_predictions, self.train_op],
                feed_dict=feed_dict)
            total_processed_examples += len(input_batch)
            total_correct_examples += total_correct
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()

        return np.mean(total_loss), total_correct_examples / float(total_processed_examples)


    def fit(self, sess, input_data, input_labels):
        losses = []
        for epoch in range(0,self.config.max_epochs):

            start_time = time.time()
            train_loss, train_acc = self.run_epoch(sess, input_data, input_labels,verbose=False)
            duration = time.time() - start_time
            print('Epoch %d: loss = %.2f,acc = %.2f (%.3f sec)' % (epoch, train_loss,train_acc, duration))

            # print('Training loss: {}'.format(train_loss))
            # print('Training acc: {}'.format(train_acc))
            losses.append(train_loss)

        return losses

    def __init__(self,config):

        """Initializes the model.

            Args:
              config: A model configuration object of type Config
        """
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.pred = self.add_model(self.input_placeholder)

        self.loss = self.add_loss_op(self.pred)

        self.predictions = tf.nn.softmax(self.pred)
        one_hot_prediction = tf.argmax(self.predictions, 1)
        correct_prediction = tf.equal(
            tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
        self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))

        self.train_op = self.add_training_op(self.loss)








