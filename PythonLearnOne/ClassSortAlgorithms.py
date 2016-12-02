# coding=utf-8
# @author:xiaolin
# @file:ClassSortAlgorithms.py
# @time:2016/12/1 16:32

class SortAlgorithms:
    '''
    各种排序算法
    '''

    def __init__(self, listA):
        self.listA = listA


    def insertSort(self):

        # 例如从第二个数据开始，对比前面的数据做排序，
        # 第三个数据进来和前面的两个做比较，后面依次如此。
        for i in range(1, len(self.listA)):
            # 每次会通过 while 来遍历i以下的全部数据做排序
            value = self.listA[i]
            j = i - 1
            # 遍历 i 以下的数据
            while j >= 0:
                if self.listA[j] > value:
                    self.listA[j + 1] = self.listA[j]
                    self.listA[j] = value
                j -= 1
        return self.listA

    def quickSort(self, leftNum, rightNum):
        '''
        把最左边的数值当做key 用来比较.
        先从右边递减比较和key值得大小，如果＞key,则下标递减。如果＜key,则把这个值放在key的位置。
        接下来，从左边递加比较，如果＜key，则下标递加，如果＞key,那么把这个值放到上面移动的位置。
        因为前面每次移动的都是rightNum 的数值，所以用key最后填充到该位置。
        :param leftNum: list的开始位置 ，默认为0
        :param rightNum: list 的结束为止，默认为list长度减1
        :return: 排序好的 list

        改进：key值得选取可以采用，最左边，最右边，中间的三个值中的中间值
             当递归到较小样本数时（10），采用插入排序
        '''
        if leftNum >= rightNum:
            return self.listA
        key = self.listA[leftNum]
        low = leftNum
        high = rightNum
        while leftNum < rightNum:
            if self.listA[rightNum] >= key:
                rightNum -= 1
            self.listA[leftNum] = self.listA[rightNum]
            if self.listA[leftNum] <= key:
                leftNum += 1
            self.listA[rightNum] = self.listA[leftNum]
        self.listA[rightNum] = key
        SortAlgorithms.quickSort(self, low, leftNum - 1)
        SortAlgorithms.quickSort(self, leftNum + 1, high)
        return self.listA


if __name__ == '__main__':
    sorttst = SortAlgorithms([13, 5, 8, 3, 2, 123, 23, 34])
    print(sorttst.quickSort(0, 7))
