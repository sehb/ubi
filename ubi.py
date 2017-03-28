# coding = utf-8
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
    sheet_index = 1  # 风险因子数据所在的页
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
    X = ubiData[:, [0, 1, 3, 5, 7, 9, 11, 12, 13, 14, 15]]
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
    preb_proba = clf.predict_proba(X)[:, 1]
    # for ii in range(len(preb_proba)):
    #     print(preb_proba[ii])

    # 将计算的结果按照0.01的间隔进行划分
    axis_x = np.linspace(0, 0.7, 71)
    # print(axis_x)

    # 按照0.01的间隔进行统计

    num_proba_space = []
    num_proba = []
    for ii in axis_x:
        num_proba.append(np.sum(preb_proba >= ii))
    for ii in range(1, len(num_proba)):
        num_proba_space.append(num_proba[ii - 1] - num_proba[ii])

    num_proba_space1 = np.array(num_proba_space) / 3071

    num_proba_space = []
    num_proba = []
    for ii in axis_x:
        num_proba.append(np.sum(preb_proba[320:] >= ii))
    for ii in range(1, len(num_proba)):
        num_proba_space.append(num_proba[ii - 1] - num_proba[ii])

    num_proba_space2 = np.array(num_proba_space) / (3071 - 320)

    num_proba_space = []
    num_proba = []
    for ii in axis_x:
        num_proba.append(np.sum(preb_proba[0:320] >= ii))
    for ii in range(1, len(num_proba)):
        num_proba_space.append(num_proba[ii - 1] - num_proba[ii])

    num_proba_space3 = np.array(num_proba_space) / 320

    # print(np.array(num_proba_space)/3071)


    set_ch()
    # # plt.plot(axis_x[0:len(axis_x)-1],num_proba_space1,'b*')
    # plt.plot(axis_x[0:len(axis_x)-1],num_proba_space1,'r')

    # # plt.plot(axis_x[0:len(axis_x)-1],num_proba_space1,'b-')
    # plt.plot(axis_x[0:len(axis_x)-1],num_proba_space2,'g')

    # # plt.plot(axis_x[0:len(axis_x)-1],num_proba_space1,'b+')
    # plt.plot(axis_x[0:len(axis_x)-1],num_proba_space3,'b')

    # num_proba_space4 = (num_proba_space3*320)/(num_proba_space1*3071)
    # num_proba_space5 = (num_proba_space2*(3071-320))/(num_proba_space1*3071)
    # plt.plot(axis_x[0:len(axis_x)-1],num_proba_space4,'b*')
    # plt.plot(axis_x[0:len(axis_x)-1],num_proba_space5,'g+')

    plt.title('出险概率以0.01为间隔')
    plt.ylabel('同一间隔中出险和未出险所占比例')
    plt.xlabel('出险概率0.01为间隔')
    plt.xticks(np.linspace(0, 0.70, 36))
    plt.yticks(np.linspace(0, 1.0, 21))
    plt.grid()
    plt.show()
