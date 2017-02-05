# coding=utf-8
# @author:xiaolin
# @file:Temptstfile.py
# @time:2016/11/16 20:58

# from PythonLearnOne.ClassScanTransfer import ScanTransfer
#
# s = ScanTransfer("D:/data/moneydata/TST", "D:/hahaha.txt")
# print(s.__doc__)


# def test_var_args(farg, *args,**kwargs):
#     print ("formal arg:", farg)
#     for arg in args:
#         print ("another arg:", arg)
#     print(kwargs['l'])
#     # l=kwargs['l']
#     # print(l)
#
# test_var_args(1)

##################################################
from PythonLearnOne.ClassSortAlgorithms import SortAlgorithms


class tst(SortAlgorithms):
    def __init__(self, tst, data):
        self.tst = tst
        self.data = data
        _value = []


import numpy as np

b = [[3, 2, 1], [1, 2], [2, 2, 2]]
i = [1, 2]
from PythonLearnOne.ClassPythonTools import PythonTools


@PythonTools.timethis
def t(i):
    if i in b:
        print(b.index(i))


t(i)
# import os
#
# path = os.path.expanduser(r"~/Desktop/dataAnalysis/Ctrip/train_data_09.csv")
# import pandas as pd
#
# df = pd.read_csv(path, sep=",")
#
# print(df.head())

''' python 默认的路径是 /USER/
    SPARK 默认的路径是项目的文件夹
'''

from PythonLearnOne.ClassPythonTools import PythonTools

@PythonTools.timethis
def analysis(trainaddress,testaddress,nullvalue=None):
    reader = pd.read_csv(trainaddress, sep = ',', iterator = True)
    chunks = []
    loop = True
    i = 1
    while loop:
        try:
            chunk = reader.get_chunk(30000)
            chunks.append(chunk)
            print (i)
            i += 1
        except StopIteration:
            loop = False
            print ('Iteration is stopped.')
    data = pd.concat(chunks, ignore_index = True)
    reader2 = pd.read_csv(trainaddress, sep=',', iterator=True)
    chunks2 = []
    loop2 = True
    i2 = 1
    while loop2:
        try:
            chunk2 = reader2.get_chunk(30000)
            chunks2.append(chunk2)
            print(i2)
            i2 += 1
        except StopIteration:
            loop2 = False
            print('Iteration is stopped.')
    data2 = pd.concat(chunks2, ignore_index=True)

    def nullcount(x):
        return sum(x.isnull())/len(x)
    def valuecount(x,value):
        count=0
        for i in x:
            if i==value:
                count=count+1
        return count/len(x)


    if nullvalue:
        a = pd.DataFrame(data.apply(lambda x:valuecount(x,nullvalue), axis=0), columns=['rate'])
        print(a.head())
        a2 = pd.DataFrame(data2.apply(lambda x: valuecount(x, nullvalue), axis=0), columns=['rate'])
        print(a2.head())
    else:
        a = pd.DataFrame(data.apply(nullcount, axis=0), columns=['rate'])
        a2 = pd.DataFrame(data2.apply(nullcount, axis=0), columns=['rate'])

    b = data.describe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).T
    b2 = data2.describe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).T

    result=pd.DataFrame()
    print(b.ix[:,2])
    for s in range(14):
        result= pd.concat([result,b.ix[:,s],b2.ix[:,s]],axis=1,join='inner')
    result=pd.concat([result,a,a2],axis=1,join='inner')
    result.to_csv("D:/data/report.csv")
    # print("it's ok")
    return print("it's ok")

