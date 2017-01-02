# -*- coding: utf-8 -*-
# @Time   : 2016/12/19
# @Author : XL ZHONG
# @File   : twoSigma.py

import pandas as pd


path = "/Users/xiaolin/Desktop/dataAnalysis/kaggle/twoSigma/train.h5"
sotre = pd.HDFStore(path=path, mode="r")
print (sotre)