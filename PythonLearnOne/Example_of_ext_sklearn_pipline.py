# -*- coding: utf-8 -*-
# @Time   : 2017/3/12
# @Author : XL ZHONG
# @File   : Example_of_ext_sklearn_pipline.py



'''
接收一个numpy数组，根据其均值将其离散化，任何高于均值的特征值替换为1，小于或等于均值的替换为0
这是一个创建自己的转换器的例子！
'''
from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array
import numpy as np

class MeanDiscrete(TransformerMixin):

  #计算出数据集的均值，用内部变量保存该值。
  def fit(self, X, y=None):
        X = as_float_array(X)
        self.mean = np.mean(X, axis=0)
        #返回self，确保在转换器中能够进行链式调用（例如调用transformer.fit(X).transform(X)）
        return self

    def transform(self, X):
        X = as_float_array(X)
        assert X.shape[1] == self.mean.shape[0]
        return X > self.mean #注意这里是判断语句，如果为真则返回1，如果是假则为0，因而实现了上面的功能