# -*- coding: utf-8 -*-
# @Time   : 2017/5/30
# @Author : XL ZHONG
# @File   : ClassTreeTools.py

class TreeTools:
    '''
    基于树的工具：决策树分 BIN （最优分 BIN 法）
    '''
    def __init__(self):
        self

    def tree_split_thres(self,x,y):

        from sklearn.tree import DecisionTreeClassifier, _tree
        import numpy as np
        from sklearn import tree

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=10, max_leaf_nodes=10)
        clf.fit(x,y)
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


    def tree_discrete(self,x,threshold,quantitle=False):
        '''
        这里调用了 Pandas 的cut 函数，如果是使用上面给出的分 BIN 的 阈值则直接调用的是 cut 函数
        :param x: 
        :param threshold: 
        :param quantitle: 如果使用的是分位数的数组来进行离散化
        :return: 
        '''

        import pandas as pd
        if quantitle==False:
            x_new=pd.cut(x,threshold)
        else:
            x_new=pd.qcut(x,threshold)

        return x_new









