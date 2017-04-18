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
    print("DONE")
    return data


address02="~/Desktop/dataAnalysis/JDATA/JData/JData_Action_201602.csv"

# 得到2月份的行为数据
df_02=read_action_data(address02)

#提取好样本 type=4

df_02_buy=df_02.loc[df_02['type']==4]
print('the buy action numbers:',df_02_buy.shape[0])
delta_time=datetime.timedelta(days=-7)
df_02_buy["start_time"]=df_02_buy['time']+delta_time

