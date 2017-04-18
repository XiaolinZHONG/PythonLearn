#coding=utf-8
#@author:xiaolin
#@file:20170228.py
#@time:2017/2/28 13:00

import pandas as pd
import numpy as np
from sklearn import tree
from IPython.display import Image
import pydotplus


data=pd.read_csv("D:/data/bigtable_2B.csv",sep=",")
data=data.dropna(how="any")
x=data[["uid_age"]]
y=data["uid_flag"]
clf=tree.DecisionTreeClassifier(criterion="entropy",max_depth=10,max_leaf_nodes=10)
clf.fit(x,y)
dot_data=tree.export_graphviz(clf,out_file="string")


b=0
for i in range(10):
    mean = i
    if i-b<0.1:
        b=(mean+i)/2
    else:
        b=i

