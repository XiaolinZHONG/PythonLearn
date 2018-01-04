# coding=utf-8
# @author:xiaolin
# @file:ClassDataPreprocess.py
# @time:2016/11/17 11:11

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import learning_curve


class DataPreprocess:
    '''
    类功能：
    对DataFrame数据探索处理: 影响因子排序，相关性分析，学习曲线，分类报告，数据分布特点
    '''
    import numpy as np
    def importancePlot(self, dataflag, flag="label", figure=True):
        '''
        输入 dataframe 绘制影响因子排序及图
        :param flag: 标签名称
        :param figure: 是否画图
        :return: 提示建议选取特征数目，仅供参考。
        '''

        label = dataflag[flag]
        data = dataflag.drop(flag, axis=1)
        data1 = np.array(data)
        label = np.array(label).ravel()

        model = ExtraTreesClassifier()
        model.fit(data1, label)

        importance = model.feature_importances_
        std = np.std([importance for tree in model.estimators_], axis=0)
        indices = np.argsort(importance)[::-1]

        featurename = list(data.columns[indices])

        # Print the feature ranking
        print("Feature ranking:")
        importa = pd.DataFrame(
            {'特征权重': list(importance[indices]), '特征名称': featurename})
        print(importa)

        modelnew = SelectFromModel(model, prefit=True)

        print('建议选取的特征数目:', modelnew.transform(data1).shape[1])

        # Plot the feature importances of the forest
        if figure == True:
            plt.figure()
            plt.title("Feature importances")
            plt.bar(range(data1.shape[1]), importance[indices],
                    color="g", yerr=std[indices], align="center")
            plt.xticks(range(data1.shape[1]), indices, rotation=90)
            plt.xlim([-1, data1.shape[1]])
            plt.grid(True)
            plt.show()

    def corrAnaly(self, data):

        corr = data.corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(20, 20))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(110, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, linewidths=.5,
                    cbar_kws={"shrink": .6}, annot=True, annot_kws={"size": 8})
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        sns.plt.show()
        return corr

    def learnCurve(self, modelEstimator, title, data, label, cv=None, train_sizes=np.linspace(0.1, 1.0, 5)):
        '''
        :param estimator: the model/algorithem you choose
        :param title: plot title
        :param x: train data numpy array style
        :param y: target data vector
        :param xlim: axes x lim
        :param ylim: axes y lim
        :param cv:
        :return: the figure
        '''

        train_sizes, train_scores, test_scores = \
            learning_curve(modelEstimator, data, label, cv=cv, train_sizes=train_sizes)

        '''this is the key score function'''
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color='b')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color='g')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label='training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='cross valid score')
        plt.xlabel('training examples')
        plt.ylabel('score')
        plt.legend(loc='best')
        plt.grid('on')
        plt.title(title)
        plt.show()

    def classifyReport(self, label_tst, pre):
        '''
        交叉验证数据分类报告，0为负样本，1为正样本
        :param label_tst: 标签
        :param pre: 分类结果可能性
        :return:
        '''
        count_tn, count_fp, count_fn, count_tp = 0, 0, 0, 0

        for i in range(len(label_tst)):
            if label_tst[i] == 0:
                if pre[i] < 0.5:
                    count_tn += 1
                else:
                    count_fp += 1
            else:
                if pre[i] < 0.5:
                    count_fn += 1
                else:
                    count_tp += 1

        print('Total:', len(label_tst))
        print('FP被分为好的坏:', count_fp, 'TN正确分类的坏:', count_tn, '坏正确率：',
              round(float(count_tn) / float((count_fp + count_tn)), 3))
        print('FN被分为坏的好:', count_fn, 'TP正确分类的好:', count_tp, '好正确率：',
              round(float(count_tp) / float((count_fn + count_tp)), 3))

    def datadescribe(self,data,miss_value=None):

        def valuecount(x):
            count = 0
            for i in x:
                if i == -900 or i == -999 or i == None or i == "NULL" or i == "Null" or np.isnan(i):
                    count = count + 1
            return count / len(x)

        # 计算缺失率
        a = pd.DataFrame(data.apply(lambda x: valuecount(x), axis=0), columns=['miss_rate'])

        # 计算分位数
        data = data.dropna(how='any')
        b = data.describe([0.25, 0.5, 0.75, 0.9, 0.95]).T

        print("Calculating the percentile, please wait patiently !")

        result = pd.concat([b, a], axis=1, join='inner')
        result = result.T
        return result

from sklearn.pipeline import FeatureUnion, _fit_one_transformer, _fit_transform_one, _transform_one
from sklearn.externals.joblib import Parallel, delayed
from scipy import sparse
import numpy as np


class FeatureUnionExt(FeatureUnion):
    '''
    通过继承父类的方法来实现，对不同列的数据进行不同的处理。
    原始的FeatureUnion只支持对整个DF/Array 进行处理操作。

    这个算法是以 pipeline 的流水线操作形式

    原始父类的例子：
    ------------
    FeatureUnion(transform_list)
    where "transform_list"=[("pca",PCA),("K",Kbest)]
    就是对整个数据集先进行 PCA 降维，在进行 特征选取。注意这两个操作时并行的！
    不是像pipeline那样串行操作。

    新类主要是通过传入要处理的列的列表，然后单独提取相应的列（可以是多列）做相应的处理
    （这些处理是并行的），然后再合并处理后的数据，得到相应的处理后的数据.

    注意：
    ----
    这种重写以 array 或sparse matrix 形式进行。但是支持 DATA FRAME 形式！
    因为父类中对 dataframe 会做进一步的处理为可处理的形式。


    例子：
    ----
    # 新建计算缺失值的对象
    step1 = ('Imputer', Imputer())
    # 新建将部分特征矩阵进行定性特征编码的对象
    step2_1 = ('OneHotEncoder', OneHotEncoder(sparse=False))
    # 新建将部分特征矩阵进行对数函数转换的对象
    step2_2 = ('ToLog', FunctionTransformer(log1p))
    # 新建将部分特征矩阵进行二值化类的对象
    step2_3 = ('ToBinary', Binarizer())
    # 新建部分并行处理对象，返回值为每个并行工作的输出的合并
    step2 = ('FeatureUnionExt', FeatureUnionExt(transformer_list=[step2_1, step2_2, step2_3], idx_list=[[0], [1, 2, 3], [4]]))
    # 新建无量纲化对象
    step3 = ('MinMaxScaler', MinMaxScaler())
    # 新建卡方校验选择特征的对象
    step4 = ('SelectKBest', SelectKBest(chi2, k=3))
    # 新建PCA降维的对象
    step5 = ('PCA', PCA(n_components=2))
    # 新建逻辑回归的对象，其为待训练的模型作为流水线的最后一步
    step6 = ('LogisticRegression', LogisticRegression(penalty='l2'))
    # 新建流水线处理对象
    # 参数steps为需要流水线处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
    pipeline = Pipeline(steps=[step1, step2, step3, step4, step5, step6])

    df=pd.read_csv()
    pipeline.fit_transform(df)

    '''

    def __init__(self, transformer_list, idx_list, n_jobs=1, transformer_weights=None):
        '''
        :param transformer_list: 做transform操作的方法的列表
        :param idx_list: 做transform操作的列（注意是列数）的列表
        :param n_jobs:
        :param transformer_weights:
        '''
        self.idx_list = idx_list  # 新添加的参数用来存储列名或列序列号（注意从0开始）
        FeatureUnion.__init__(self, transformer_list=transformer_list,
                              n_jobs=n_jobs, transformer_weights=transformer_weights)

    # fit重构
    def fit(self, X, y=None):

        # 生成一个 transform list 包含 三部分,处理方法名+处理方法+列名/列号 :
        # [("pca",PCA,0),("K",Kbest,"colmuns_name")]
        transformer_idx_list = list(map(lambda trans, idx: (trans[0], trans[1], idx),
                                        self.transformer_list, self.idx_list))
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, X[:, idx], y) for name, trans, idx in transformer_idx_list)
        self._update_transformer_list(transformers)
        return self

    # fit_transform重构
    def fit_transform(self, X, y=None, **fit_params):
        transformer_idx_list = list(map(lambda trans, idx: (trans[0], trans[1], idx), self.transformer_list,
                                   self.idx_list))
        result = Parallel(n_jobs=self.n_jobs)(

            delayed(_fit_transform_one)(trans, name, X[:, idx], y,
                                        self.transformer_weights, **fit_params)
            for name, trans, idx in transformer_idx_list)

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    # transform重构
    def transform(self, X):
        transformer_idx_list = list(map(lambda trans, idx: (trans[0], trans[1], idx), self.transformer_list,
                                   self.idx_list))

        Xs = Parallel(n_jobs=self.n_jobs)(

            delayed(_transform_one)(trans, name, X[:, idx], self.transformer_weights)
            for name, trans, idx in transformer_idx_list)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

# if __name__ == '__main__':
#
