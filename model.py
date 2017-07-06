# -*- coding: utf-8 -*-
# @Author: Zhaoyin Zhang
# @Date:2017/7/3 16:10 
# @Contact: 940942500@qq.com


"""
改类主要是定义模型的结构
"""


class Model(object):

    def load_data(self):

        """加载数据到内存中

        """

        raise NotImplementedError("Each model must re-implement this method.")

    def add_placeholders(self):

        """

        """

        raise NotImplementedError("Each model must re-implement this method.")

    def create_feed_dict(self, input_batch, label_batch):

        """

        """
        raise NotImplementedError("Each model must re-implement this method.")

    def add_model(self, input_data):

        """

        """
        raise NotImplementedError("Each model must re-implement this method.")

    def add_loss_op(self, pred):

        """

        """
        raise NotImplementedError("Each model must re-implement this method.")

    def run_epoch(self,sess, input_data, input_labels):

        """

        """
        raise NotImplementedError("Each model must re-implement this method.")

    def fit(self, sess, input_data, input_labels):

        """
        anf  eni   et er  et e

        """
        raise NotImplementedError("Each model must re-implement this method.")

    def predict(self, sess, input_data, input_labels):

        """

        """
        raise NotImplementedError("Each model must re-implement this method.")

class LanguageModel(Model):

    def add_embedding(self):

        """

        """
        raise NotImplementedError("Each model must re-implement this method.")





