# coding=utf-8
# @author:xiaolin
# @file:ClassScanTransfer.py
# @time:2016/11/16 17:55
import os
import pandas as pd
import csv


class ScanTransfer(object): # 使用object 是便于后面的方法调用自己内部的方法
    '''
    类功能：
    ------------
    遍历输入路径(filesPath)下的文件夹中的所有文件，
    输出结果到保存路径(savePathName)下。
    ------------
    基于版本：Python 3.x
    '''

    def __init__(self, filesPath, savePathName):
        '''
        初始化
        :param filesPath: 遍历的路径
        :param savePath: 保存的路径
        '''
        self.filesPath = filesPath
        self.savePath = savePathName

    def scanning(self):
        '''
        遍历输入的输入路径下的文件，返回文件列表
        :param filesPath: 需要遍历的路径
        :return: 路径下的文件列表
        '''
        filelist = []
        for root, dirs, item in os.walk(self.filesPath):
            for f in item:
                file = root + str("/") + f
                filelist.append(file)
        print("你输入的路径下有 %d 个文件" % (len(filelist)))
        return filelist

    def readCsvSave(self, filelist, headerNum=0, dropIndex=-1, selectcol=None):
        '''
        输入文件列表，返回一个特定格式的文件到给定路径下
        :param filelist: 文件清单
        :param headerNum: 数据的column的名称在第几行，默认为第一行
        :param dropIndex: 数据需要丢弃的后面几行？默认不丢弃
        :return: 写入数据到给定的目录下
        '''
        csvfile = open(self.savePath, 'w+', encoding="ISO-8859-1")

        # 注册一个切分格式，默认是可以不写的这句代码
        csv.register_dialect('csv', delimiter=',')

        # 默认格式时excel,后面表示结尾的标志，我们这里是回车\n

        writer = csv.writer(csvfile, dialect='csv', lineterminator='\n')
        for (i, f) in enumerate(filelist):

            # 注意:pandas不支持直接的中文编码！
            datadf0 = pd.read_csv(f, header=headerNum, encoding="ISO-8859-1")
            datadf1 = datadf0.drop(datadf0.index[dropIndex:-1])
            print("第 %d 个文件,共有 %d 行" % (i + 1, datadf1.shape[0]))
            if selectcol:
                rows = datadf1.ix[:, selectcol].values.tolist()
            else:
                rows = datadf1.values.tolist()
            writer.writerows(rows)
        csvfile.close()

    def readExcelSave(self, filelist, headerNum=0, dropIndex=-1, selectcol=None):
        '''
        输入文件列表，返回一个特定格式的文件到给定路径下
        :param filelist: 文件清单
        :param headerNum: 数据的column的名称在第几行，默认为第一行
        :param dropIndex: 数据需要丢弃的后面几行？默认不丢弃
        :return: 写入数据到给定的目录下
        '''
        csvfile = open(self.savePath, 'w+', encoding="ISO-8859-1")
        writer = csv.writer(csvfile, lineterminator='\n')
        for (i, f) in enumerate(filelist):

            # 注意:pandas不支持直接的中文编码UTF-8！
            datadf0 = pd.read_excel(f, header=headerNum, encoding="ISO-8859-1")
            datadf1 = datadf0.drop(datadf0.index[dropIndex:-1])
            print("第 %d 个文件,共有 %d 行" % (i, datadf1.shape[0]))
            if selectcol:
                rows = datadf1.ix[:, selectcol].values
            else:
                rows = datadf1.values
            writer.writerows(rows)
        csvfile.close()


if __name__ == '__main__':
    # 创建这个类的第一个属性
    s = ScanTransfer("D:/data/moneydata/TST", "D:/hahaha.txt")
    # 显示类的说明
    print(s.__doc__)
    # 类的实例化，扫描路径下的文件
    a = s.scanning()
    # print(a)
    s.readCsvSave(a, 4, -4, (2, 4, 11))
