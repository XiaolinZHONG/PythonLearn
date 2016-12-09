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

    def __init__(self,tst,data):
        self.tst=tst
        self.data=data
        _value= []



import numpy as np

b=[[3,2,1],[1,2],[2,2,2]]
i=[1,2]
from PythonLearnOne.ClassPythonTools import PythonTools

@PythonTools.timethis
def t(i):
    if i in b:
        print(b.index(i))

t(i)