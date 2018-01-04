# -*- coding: utf-8 -*-
# @Time   : 2017/4/18
# @Author : XL ZHONG
# @File   : time_shift.py

import pandas as pd
import datetime
import numpy as np

# 定义读取行为数据的方法
def read_action_data(address):
    reader = pd.read_csv(address, sep = ',', iterator = True,parse_dates=['time'])
    chunks = []
    loop = True
    i = 1
    while loop:
        try:
            chunk = reader.get_chunk(30000000)
            chunks.append(chunk)
            print ('reading data',i)
            i += 1
        except StopIteration:
            loop = False
            print ('Iteration is stopped.')
    data = pd.concat(chunks, ignore_index = True)

    print("data shape:",data.shape,"\n","DONE")
    return data


address02="~/Desktop/dataAnalysis/JDATA/JData/JData_Action_201602.csv"
address03="~/Desktop/dataAnalysis/JDATA/JData/JData_Action_201603.csv"
address04="~/Desktop/dataAnalysis/JDATA/JData/JData_Action_201604.csv"
df_02=read_action_data(address02)
df_03=read_action_data(address03)
df_04=read_action_data(address04)
df_action=pd.concat([df_02,df_03,df_04],axis=0,ignore_index=True)

df_02=df_action

def make_samples(df_02):

    #提取好样本 type=4

    df_02_buy=df_02.loc[df_02['type']==4]

    print('the buy action numbers:',df_02_buy.shape[0])

    delta_time=datetime.timedelta(days=-7)
    df_02_buy.is_copy = False
    print(df_02_buy.dtypes)
    df_02_buy["start_time"]=df_02_buy['time']+delta_time

    df_buy=df_02_buy[df_02_buy['start_time']>'2016-02-01']

    df_02_new=pd.merge(df_02,df_buy[["user_id","sku_id","start_time"]],
                       how="left",on=["user_id","sku_id"]).drop_duplicates()

    df_02_buy_action=df_02_new.loc[df_02_new['time'] >= df_02_new['start_time']].dropna(subset=["start_time"])

    #df_02_new 时间要重铸

    df_02_new['time']=df_02_new["time"].apply(str)

    df_02_not_buy = df_02_new[
        ~df_02_new[["user_id", "sku_id","time", "model_id", "type", "cate", "brand"]].isin(
            df_02_buy[["user_id", "sku_id"]])]

    # df_02_not_buy_id=df_02_not_buy[df_02_not_buy['time']>='2016-02-07'][["user_id", "sku_id"]].drop_duplicates().sample(n=500000)  #随机取50万 id


    n=df_02_buy_action.shape[0]*2
    df_02_not_buy_tmp=df_02_not_buy[['user_id','sku_id','time']].sample(n)
    df_02_not_buy_tmp['time']=df_02_not_buy_tmp['time'].astype('datetime64[ns]')
    print(df_02_not_buy_tmp.dtypes)
    # df_02_not_buy=pd.merge(df_02_not_buy_id,df_02_not_buy,how="left",on=["user_id","sku_id"]).drop_duplicates()
    df_02_not_buy_tmp=df_02_not_buy_tmp.groupby(['user_id','sku_id'])["time"].max().reset_index()
    df_02_not_buy_tmp['start_time']=df_02_not_buy_tmp['time']+datetime.timedelta(days=-7)
    df_02_not_buy_action = df_02_not_buy_tmp.loc[df_02_not_buy_tmp['time'] >= df_02_not_buy_tmp['start_time']]

    df_02_not_buy_final= pd.merge(df_02_not_buy_action, df_02_not_buy[["user_id", "sku_id",  "model_id", "type", "cate", "brand"]],
                         how="left", on=["user_id", "sku_id"]).drop_duplicates()

    return df_02_buy_action,df_02_not_buy_final

df_buy,df_not_buy=make_samples(df_02)
df_buy.to_csv("~/Desktop/dataAnalysis/JDATA/JData/JData_Action_buy.csv")
df_not_buy.to_csv("~/Desktop/dataAnalysis/JDATA/JData/JData_Action_not_buy.csv")
print (df_buy.head(),df_buy.shape)
print(df_not_buy.head(),df_not_buy.shape)



