# -*- coding: utf-8 -*-
# @Time   : 2017/5/16
# @Author : XL ZHONG
# @File   : pandas_tools.py

import pandas as pd
import numpy as np

class Pandas_tools:


    def __init__(self):
        self

    def read_data(self, address,sep=','):
        reader = pd.read_csv(address, sep=sep, iterator=True, parse_dates=True)
        chunks = []
        loop = True
        i = 1
        while loop:
            try:
                chunk = reader.get_chunk(30000000)
                chunks.append(chunk)
                print(i)
                i += 1
            except StopIteration:
                loop = False
                print('Iteration is stopped.')
        data = pd.concat(chunks, ignore_index=True)
        print("DONE")
        return data

    def time_shift(self,df,sort_a="uid",sort_b="time"):

        '''
        首先根据输入的dataframe 中的 uid 和 time 做排序，这样同一个 UID 对应几个 time，通过计算同一个 UID 下
        对应的各个时间做 shift 
        :param df:  
        :param sort_a: 输入的标识位
        :param sort_b: 输入的时间变量
        :return: 只保留了最初和最后时间的 uid 和对应的时间。
        
        examples: 
        +---+-----+          +----+-------+
        |uid| time|          |uid |  23243|
        +---+-----+          +----+-------+
        |123| 2011|      =>  |123 |    1  |
        +---+-----+          +----+-------+
        |123| 2012|
        +---+-----+
        '''
        count = df[sort_a].value_counts() #统计 df 中的每一个元素对应的出现频率

        df.sort_values([sort_a, sort_b]) # 对 uid time  排序

        for i in range(count.shape[0]):
            value = count.index[i]
            shift_t = count.values[i] - 1 #偏移量，一般是依据出现
            df.ix[df[sort_a] == value, "pre"] = df[df[sort_a] == value][sort_b].shift(shift_t)
            df.ix[df[sort_a] == value, "days"] = (df['time'] - df['pre']).dt.days
        # print(df)
        df = df.dropna(how="any") #这里只保存有值的，本来不是最大一天的都会有空置

        return df



 if __name__ == '__main__':
     Pandas_tools