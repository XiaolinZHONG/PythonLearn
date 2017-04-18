#coding=utf-8
#@author:xiaolin
#@file:ReportTest.py
#@time:2017/2/23 20:12

import pandas as pd
# from PythonLearnOne.ClassPythonTools import PythonTools
import numpy as np

# @PythonTools.timethis
def statisticAnalysis(trainaddress,testaddress=None,save_address="D:/StatisticAnalysisReport.CSV" ,columnnames=None):

    #读取数据
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
    print(data.columns)
    if columnnames:
        data=data[columnnames]

    if testaddress:
        reader2 = pd.read_csv(testaddress, sep=',', iterator=True)
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
        print(data.columns)
        if columnnames:
            data2=data2[columnnames]

    def valuecount(x):
        count=0
        for i in x:
            if i==-900 or i==-999 or i==None or i=="NULL" or i=="Null" or np.isnan(i):
                count=count+1
        return count/len(x)


    #计算缺失率
    a = pd.DataFrame(data.apply(lambda x:valuecount(x), axis=0), columns=['缺失率_train'])
    if testaddress:
        a2 = pd.DataFrame(data2.apply(lambda x: valuecount(x), axis=0), columns=['缺失率_test'])

    #计算分位数
    data=data.dropna(how='any')
    b = data.describe([0.25,0.75,0.9,0.95,0.99]).T
    b.rename(columns=lambda x: str(x) + "_train", inplace=True)
    if testaddress:
        data2=data2.dropna(how="any")
        b2 = data2.describe([0.25,0.75,0.9,0.95,0.99]).T
        b2.rename(columns=lambda x: str(x) + "_test", inplace=True)

    print("Calculating the percentile, please wait patiently !")



    #合并数据并保存
    test = pd.DataFrame()
    if testaddress:
        for i in range(10):
            test=pd.concat([test,b.ix[:,i],b2.ix[:,i]],axis=1)
        result=pd.concat([test,a,a2],axis=1,join='inner')
        result=result.T
        result.to_csv(save_address)
    else:
        for i in range(10):
            test = pd.concat([test, b.ix[:, i]], axis=1)
        result = pd.concat([test, a], axis=1, join='inner')
        result = result.T
        result.to_csv(save_address)

    return print("it's ok")

if __name__ == '__main__':

    trainaddress="D:/data/credit/train_sample.csv"
    testaddress="D:/data/credit/test_sample.csv"
    save_address="D:/data/reportHWH.csv"
    columns=['rcy_band','qbmm_max_12m','payhabitsum_ltr_3_12m',
             'train_uid_amt_msn','uid_newamt_conris_cnt','uid_newamt_conred_cnt',
             'payhabitsum_countsum_6m','highamt_uid_ord_rat1y','highamt_uid_ord_ris1y',
             'payhabitsum_ltr_6_12m','uidip_newamt_sum2y','uid_newamt_ris1y',
             'uidip_newamt_sum1yx6m','uid_newamt_sum2y','flt_uidip_ord_cnt1y',
             'uid_ord_cnt1y','uid_ordtype_ris1yx6m','train_uidip_ord_dmnsmin']
    statisticAnalysis(trainaddress,testaddress,save_address,columns)