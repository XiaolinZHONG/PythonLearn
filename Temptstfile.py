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

import pandas as pd
import numpy as np
df=pd.read_csv("D:/project_csm/bigtable_2B_xc_new.csv",sep=",")
print(df.head())
print(df.ix[:6,2])
print(np.array(df.ix[:5,2]).ravel())