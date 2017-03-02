#coding=utf-8
#@author:xiaolin
#@file:ReportTest.py
#@time:2017/2/23 20:12

import pandas as pd
from PythonLearnOne.ClassPythonTools import PythonTools
import numpy as np

@PythonTools.timethis
def statisticAnalysis(trainaddress,testaddress=None,save_address="D:/StatisticAnalysisReport.CSV"):

    #读取数据
    reader = pd.read_csv(trainaddress, sep = ',', iterator = True)
    chunks = []
    loop = True
    i = 1
    while loop:
        try:
            chunk = reader.get_chunk(3000000)
            chunks.append(chunk)
            print (i)
            i += 1
        except StopIteration:
            loop = False
            print ('Iteration is stopped.')
    data = pd.concat(chunks, ignore_index = True)
    if testaddress:
        reader2 = pd.read_csv(testaddress, sep=',', iterator=True)
        chunks2 = []
        loop2 = True
        i2 = 1
        while loop2:
            try:
                chunk2 = reader2.get_chunk(3000000)
                chunks2.append(chunk2)
                print(i2)
                i2 += 1
            except StopIteration:
                loop2 = False
                print('Iteration is stopped.')
        data2 = pd.concat(chunks2, ignore_index=True)

    def valuecount(x):
        count=0
        for i in x:
            if i==-900 or i==-999 or i==None or i=="NULL" or i=="Null" or np.isnan(i):
                count=count+1
        return count/len(x)


    #计算缺失率
    a = pd.DataFrame(data.apply(lambda x:valuecount(x), axis=0), columns=['rate_train'])
    if testaddress:
        a2 = pd.DataFrame(data2.apply(lambda x: valuecount(x), axis=0), columns=['rate_test'])

    #计算分位数
    data=data.dropna(how='any')
    b = data.describe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).T
    b.rename(columns=lambda x: str(x) + "_train", inplace=True)
    if testaddress:
        data2=data2.dropna(how="any")
        b2 = data2.describe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).T
        b2.rename(columns=lambda x: str(x) + "_test", inplace=True)

    print("Calculating the percentile, please wait patiently !")



    #合并数据并保存
    test = pd.DataFrame()
    if testaddress:
        for i in range(14):
            test=pd.concat([test,b.ix[:,i],b2.ix[:,i]],axis=1)
        result=pd.concat([test,a,a2],axis=1,join='inner')
        result.to_csv(save_address)
    else:
        for i in range(14):
            test = pd.concat([test, b.ix[:, i]], axis=1)
        result = pd.concat([test, a], axis=1, join='inner')
        result.to_csv(save_address)

    return print("it's ok")

if __name__ == '__main__':

    # trainaddress="D:/data/flashodermoney3.csv"
    # testaddress="D:/data/flashodermoney3.csv"
    # save_address="D:/data/report.csv"
    # statisticAnalysis(trainaddress)
