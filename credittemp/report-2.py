#coding=utf-8
#@author:xiaolin
#@file:ReportTest.py
#@time:2017/2/23 20:12

import pandas as pd
# from PythonLearnOne.ClassPythonTools import PythonTools
import numpy as np

# @PythonTools.timethis
def statisticAnalysis2(trainaddress,testaddress=None,save_address="D:/StatisticAnalysisReport.CSV" ,columnnames=None):

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
    # print(data.columns)
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
        # print(data.columns)
        if columnnames:
            data2=data2[columnnames]

    def valuecount(x):
        count=0
        for i in x:
            if i==-900 or i==None or i=="NULL" or i=="Null" or np.isnan(i):
                count=count+1
        return count/len(x)


    #计算缺失率
    a = pd.DataFrame(data.apply(lambda x:valuecount(x), axis=0), columns=['Loss rate train'])
    if testaddress:
        a2 = pd.DataFrame(data2.apply(lambda x: valuecount(x), axis=0), columns=['Loss rate test'])

    #计算分位数
    data=data.dropna(how='any')
    b = data.describe([0.25,0.5,0.75,0.9,0.95,0.99]).T
    b.rename(columns=lambda x: str(x) + " train", inplace=True)
    if testaddress:
        data2=data2.dropna(how="any")
        b2 = data2.describe([0.25,0.5,0.75,0.9,0.95,0.99]).T
        b2.rename(columns=lambda x: str(x) + " test", inplace=True)

    print("Calculating the percentile, please wait patiently !")



    #合并数据并保存
    temp=pd.DataFrame({'----------':'--------------'},index=b.index)
    if testaddress:
        result=pd.concat([b,a,temp,b2,a2],axis=1,join='inner')
        result=result.T
        result.to_csv(save_address)
    else:
        result = pd.concat([b, a], axis=1, join='inner')
        result = result.T
        result.to_csv(save_address)

    return print("it's ok")

if __name__ == '__main__':

    trainaddress="D:/trainsample.csv"
    testaddress="D:/testsample.csv"
    save_address="D:/report.csv"
    columns=["rcy_band_email",
"htl_uidip_ord_ris1y",
"ns_uidcid_ord_rat2y",
"ns_uid_ord_dmnsmax",
"lowamt_uid_ord_dmnsmax",
"allcom_ratio_ord_24m",
"htl_uid_newamt_msx2y",
"highamt_uid_ord_rat2y",
"uid_newamt_madmax3m",
"htl_star_uid_ord_rat1y",
"complaint_ratio_ord_12m",
"uid_ordtype_cnt2y",
"htl_star_uidcid_amt_msx2y",
"uid_ord_dmnsmax",
"complaint_msx",
"train_uid_ord_msn",
"train_uidip_ord_msx2y",
"c_uidcid_ord_cnt2y",
"htl_uid_ord_madmax3m",
"ns_uidcid_ord_avg2y",
"uid_newamt_msx2y",
"flt_uidcid_newamt_msx2y",
"uid_newamt_conred_cnt",
"c_uidip_amt_sum6m",
"c_uidip_ord_cnt3m",
"htl_uid_ord_max2y"]
    statisticAnalysis2(trainaddress,testaddress,save_address,columns)