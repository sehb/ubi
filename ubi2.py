# coding = utf-8
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB


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
    sheet_index = 3  # 风险因子数据所在的页
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
    X = ubiData[:, 0:ubiDataType[1] - 1]
    y = ubiData[:, ubiDataType[1] - 1]

    # 返回训练集的大小
    return X, y

def loadTestData(xlsFileName):
    sheet_index = 3  # 风险因子数据所在的页
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
    X = ubiData[:, 0:ubiDataType[1] - 1]
    y = ubiData[:, ubiDataType[1] - 1]

    # 返回训练集的大小
    return X, y

if __name__ == '__main__':
    # set_ch() #设置中文显示
    X, y = loadData('e:/python/data/20170307嘉兴人保数据.xlsx')

    # # 下面的代码用于抽取参数重要性
    # model = ExtraTreesClassifier()
    # model.fit(X, y)
    # # display the relative importance of each attribute
    # print('参数重要性：', model.feature_importances_)

    # # 对数据进行预处理
    # # normalize the data attributes
    # normalized_X = preprocessing.normalize(X)
    # # standardize the data attributes
    # standardized_X = preprocessing.scale(X)
    # X = StandardScaler().fit_transform(X)

    model = GaussianNB()
    model.fit(X, y)
    print(model)
    # make predictions
    expected = y
    predicted = model.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    
    # 评分函数，将返回一个小于1的得分，可能会小于0
    score = model.score(X, y)

    print('模型得分：',score)
