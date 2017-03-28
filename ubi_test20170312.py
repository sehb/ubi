# coding = utf-8
'''
    该代码是一个测试版本，用于估算各个因子的重要性，
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
    # print('该页基本信息（页名，行数，列数）', sheet1.name, sheet1.nrows, sheet1.ncols)

    # 读取所有行,并将数据从字符串转化为浮点数,map完后要转成list，否则会报错
    ubiData = []
    for ii in range(x_rows_index, sheet1.nrows):
        ubiData.append(list(map(float, sheet1.row_values(ii))))

    ubiData = np.array(ubiData)
    ubiDataType = ubiData.shape
    # print('UBI原始样本值的大小：', ubiDataType)
    X = ubiData[:, [0,1,3,5,7,9,11,12,13,14,15]]
    y = ubiData[:, ubiDataType[1] - 1]

    # X1 = ubiData[:, np.newaxis,[0,1,3,5,7,9,11,12,13,14,15]]
    # print(X1)
    print(X)


    # 返回训练集数据
    return X, y

if __name__ == '__main__':
    X, y = loadData('e:/python/data/20170309嘉兴人保数据.xlsx')
    # 进行Logistic学习，也就是训练train
    # X,y以矩阵的方式传入
    #导入sklearn的ExtraTreesClassifier和SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    #基于树模型进行模型选择
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    #特征重要性(数值越高特征越重要)
    print(clf.feature_importances_)
    #选择特征重要性为1.25倍均值的特征
    model = SelectFromModel(clf, threshold='1.1*mean',prefit=True)
    #返回所选的特征
    X_trees = model.transform(X)
    print(X_trees)
    print(X_trees.shape)


    #导入sklearn库中的SelectKBest和chi2
    from sklearn.feature_selection import SelectKBest ,chi2
    #选择相关性最高的前5个特征
    X_chi2 = SelectKBest(chi2, k=5).fit_transform(X, y)
    print(X_chi2.shape)

    #导入数据预处理库
    from sklearn import preprocessing
    #范围0-1缩放标准化
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaler=min_max_scaler.fit_transform(X)
    #查看特征的维度
    print(X_scaler.shape)

    #导入sklearn库中的VarianceThreshold
    from sklearn.feature_selection import VarianceThreshold
    #设置方差的阈值为0.8
    sel = VarianceThreshold(threshold=.08)
    #选择方差大于0.8的特征
    X_sel=sel.fit_transform(X_scaler)
    print(X_sel.shape)
