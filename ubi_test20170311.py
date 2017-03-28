# coding = utf-8
'''
    该代码是一个测试版本，出险概率间隔是0.01，用于估算CDF分布情况，
'''

import xlrd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 下面的函数用于将概率保留两位小数位
def round_local(x):
    return round(x,2)

# 下面的函数计算出险概率
def sigmoid(h):
    return 1.0 / (1.0 + np.exp(-h))

# 下面的函数用于设置画图时能够显示汉字
def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# datemode = 0,使用1900为基础的时间戳；
# datemode = 1,使用1904为基础的时间戳
def dateMap(excelDate):
    return xlrd.xldate.xldate_as_datetime(excelDate, 0)

def loadData(xlsFileName):
    sheet_index  = 1  # 风险因子数据所在的页
    x_rows_index = 1  # 风险因子数据起始行

    # 打开文件
    workbook = xlrd.open_workbook(xlsFileName)

    # 根据sheet索引或者名称获取sheet内容
    sheet1 = workbook.sheet_by_index(sheet_index)  # sheet索引从0开始
    print('该页基本信息（页名，行数，列数）', sheet1.name, sheet1.nrows, sheet1.ncols)

    # 读取所有行,并将数据从字符串转化为浮点数,map完后要转成list，否则会报错
    ubiData = []
    for ii in range(x_rows_index, sheet1.nrows):
        ubiData.append(list(map(float, sheet1.row_values(ii))))

    ubiData = np.array(ubiData)
    ubiDataType = ubiData.shape
    print('UBI原始样本值的大小：', ubiDataType)
    X = ubiData[:, [0,1,3,5,7,9,11,12,13,14,15]]
    y = ubiData[:, ubiDataType[1] - 1]

    # 返回训练集数据
    return X, y

if __name__ == '__main__':
    X, y = loadData('e:/python/data/20170309嘉兴人保数据.xlsx')
    # 进行Logistic学习，也就是训练train
    # X,y以矩阵的方式传入
    clf = LogisticRegression()
    clf.fit(X, y)

    # 预测概率
    preb_proba = list(clf.predict_proba(X)[:,1])
    preb_proba_round = list(map(round_local, preb_proba))# 为便于获得CDF，保留小数点后两位
    # for ii in range(len(preb_proba)):
    #     print(preb_proba[ii])
    #     print(preb_proba_round[ii])

    # print(np.array(num_proba_space)/3071)


    print('样本值总数：', len(X))
    print('样本均值：', np.mean(X, axis=0)) # 求每一列的均值
    print('各因子标准偏差：', np.std(X, axis=0)) # 求每一列的均值
    print('出险概率均值：', np.mean(preb_proba_round))  # 50%的概率分布点
    print('出险概率中位数：', np.median(preb_proba_round))  # 50%的概率分布点
    print('出险概率众数：', stats.mode(preb_proba_round))  # 出现次数最多的值

    set_ch( )
    # plt.hist(preb_proba_round, len(preb_proba_round), normed=True, histtype='step', cumulative=-1) # CCDF 互补累计概率分布函数
    plt.hist(preb_proba_round, len(preb_proba_round), normed=True, histtype='step',cumulative=True,color='g') # CDF 累计概率分布函数
 
    plt.title('出险概率以0.01为间隔的CCDF')
    plt.ylabel('Prob(X<x)')
    plt.xlabel('出险概率(0.01为间隔)')
    plt.xticks(np.linspace(0, 0.70, 36))
    plt.yticks(np.linspace(0, 1.0, 21))
    plt.grid(True)
    # plt.legend()
    plt.show()
