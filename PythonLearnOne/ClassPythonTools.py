# coding=utf-8
# @author:xiaolin
# @file:ClassPythonTools.py
# @time:2016/12/9 15:09

class PythonTools():
    '''装饰器'''


    def timethis(func):
        import datetime
        from functools import wraps
        @wraps(func)
        def warapper(*args,**kwargs):
            start=datetime.datetime.now()
            result=func(*args,**kwargs)
            end=datetime.datetime.now()
            print(func.__name__,"time cost:",(end-start).microseconds/1000,"ms")
            return result
        return warapper

# @timethis
# # 装饰器对后面的方法进行处理。
# def checkPares(n):
#     '''
#     括号配对
#     :param text:
#     :return:
#     '''
#
#     while n>0:
#         n-=1
#
# checkPares(10000000)