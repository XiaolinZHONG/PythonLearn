
# -*- coding: utf-8 -*-
# @Time   : 2017/5/30
# @Author : XL ZHONG
# @File   : ClassTreeTools.py
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class TreeTools(BaseEstimator,TransformerMixin):
    '''
    基于树的工具：决策树分 BIN （最优分 BIN 法）
    '''
    def __init__(self):
        self

    def fit(self,X,y=None):


        from sklearn import tree

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=10, max_leaf_nodes=10)
        clf.fit(X,y)
        count_leaf=0
        for i in clf.tree_.children_left:
            if i==_tree.TREE_LEAF:
                count_leaf+=1
        count_leaf


        threshold = clf.tree_.threshold  #所有的节点全部是<=
        # threshold=np.sort(threshold)[count_leaf:]
        # #这种方式不太好，如果有小于-2的排序就会出问题。
        #  -2的数目和叶子数相同 后面需要排除掉-2的值
        count=0
        for i in threshold:
            if i==-2: count+=1

        new_threshold=list(filter(lambda x: x!=-2,threshold))

        if count>count_leaf:new_threshold+=[-2]

        new_threshold_2=np.sort(new_threshold)

        return new_threshold_2


    def transform(self,x,threshold):
        '''
        这里调用了 numpy 的 内置函数 searchsorted来实现。

        :param x:
        :param threshold:
        :return:
        '''

        thres_index = np.asarray(threshold).searchsorted(x, side='right')  # 注意这里的 right 表示的是<=
        thres_index = thres_index.ravel()
        x_new = []
        for i in range(len(x)):
            if thres_index[i] + 1 <= len(threshold):
                x_new.append(threshold[thres_index[i]])
            else:
                x_new.append(x.max())
        return x_new

    def fit_transform(self, X, y=None, **fit_params):

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=10, max_leaf_nodes=10)
        clf.fit(X,y)
        count_leaf=0
        for i in clf.tree_.children_left:
            if i==_tree.TREE_LEAF:
                count_leaf+=1
        count_leaf


        threshold = clf.tree_.threshold  #所有的节点全部是<=
        # threshold=np.sort(threshold)[count_leaf:]
        # #这种方式不太好，如果有小于-2的排序就会出问题。
        #  -2的数目和叶子数相同 后面需要排除掉-2的值
        count=0
        for i in threshold:
            if i==-2: count+=1

        new_threshold=list(filter(lambda x: x!=-2,threshold))

        if count>count_leaf:new_threshold+=[-2]

        new_threshold_2=np.sort(new_threshold)

        thres_index = np.asarray(new_threshold_2).searchsorted(X, side='right')  # 注意这里的 right 表示的是<=
        thres_index = thres_index.ravel()
        x_new = []
        for i in range(len(X)):
            if thres_index[i] + 1 <= len(new_threshold_2):
                x_new.append(new_threshold_2[thres_index[i]])
            else:
                x_new.append(X.max())#这里后面可以修改
        return x_new # 返回的数值是右侧的等于的值

















