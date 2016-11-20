# coding=utf-8
# @author:xiaolin
# @file:ClassBPNeturalNetworks.py
# @time:2016/11/18 15:19

'''
network.py
~~~~~~~~~~~~~~~~~~~~~~
这个模型的宗旨是根据随机梯度下降法制作一个前馈式神经网络
梯度的计算是通过逆向（BP）传播的方法。
'''

import random
import numpy as np


class BPNeturalNetworks(object):
    '''
    这是基于多层感知器的神经网络，其中的内核函数为 sigmod()
    '''
    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes  # 数组型[30,20,1]

        # 赋予权重值一个随机数。
        # ----------------------
        # 神经网络的 b：一个一维数组
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # 神经网络中的 权重值 W ：一个多维数组
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        '''
        :param a: 前馈函数的输入值
        :return: 迭代计算的输入值，作为下一层的输入
        '''
        '''返回输入a对应的网络'''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        这里是基于损失函数的神经网络学习过程的主函数！

        使用基于mini batch 的梯度下降法训练神经网络。
        :param training_data: 格式为：（x，y）的形式，x为训练数据，y为标签
        :param epochs: 训练的回合数
        :param mini_batch_size: 用于梯度下降法的mini batch 的样本数量
        :param eta: 训练的步长
        :param test_data: 测试数据
        :return:
        '''

        if test_data: n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):

            # 以行的形式将训练数据打乱随机排序
            random.shuffle(training_data)

            # 生成一定大小的 batch 块，后面的操作都是基于 batch
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            # 对于 batch 中的每一个数据都执行更新权重的操作
            for mini_batch in mini_batches:
                self.SGD_mini_batch(mini_batch, eta)

            if test_data:
                print('循环{0}的准确率：{1}/{2}'.format(j, self.evaluate(test_data), n_test))
            else:
                print('循环 {0} 结束'.format(j))

    def SGD_mini_batch(self, mini_batch, eta):
        '''
        更新训练网络中的 权重 和 偏移 的值，通过梯度下降法，逆向传递一个最小样本数，
        :param mini_batch: 输入的数据块
        :param eta: 梯度下降的步长
        :return:

        example:
        ----------
        nabla:求导算子的名称

        output=F(z)=1/(1+exp(-z))=1/(1+exp(-(w*x+b)))

        W'= W+delta_W = W-(eta/len(mini_batch))*nabla_W
        B'= B+delta_B = B-(eta/len(mini_batch))*nabla_B

        '''

        # 初始化 并迭代更新 权值
        # ---------------------------------------------------
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        for x, y in mini_batch:

            # 迭代更新权重
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #    W'= W+delta_W = W-(eta/len(mini_batch))*nabla_W
        #    B'= B+delta_B = B-(eta/len(mini_batch))*nabla_B

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
        返回梯度值（nable_b,nable_w）表示C-x的梯度值，可以看做是cost函数对w,b的求导结果.
        :param x: 单独一个样本的特征数据。
        :param y: 单独一个样本的标签。
        :return:
        '''

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前馈函数
        # x=F(z)=1/(1+exp(-z))=1/(1+exp(-(w*x+b)))
        # 上一层网络的输出作为下一层的输入
        activation = x # 一个样本的全部特征数据
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)


        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """测试数据的正确性"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """返回cost值，也就是计算出的值和想要得到的结果的值"""
        return (output_activations - y)


#### Miscellaneous functions
def sigmoid(z):
    """sigmoid方法"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """sigmoid的求导."""
    return sigmoid(z) * (1 - sigmoid(z))
