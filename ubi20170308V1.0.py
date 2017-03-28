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
from sklearn.metrics import precision_recall_curve, roc_curve, auc  
from sklearn.metrics import classification_report  

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
    X = ubiData[:, [0,1,3,5,7,9,11,12,13,14,15]]
    y = ubiData[:, ubiDataType[1] - 1]

    # 返回训练集的大小
    return X, y

if __name__ == '__main__':
    # set_ch() #设置中文显示
    X, y = loadData('e:/python/data/20170309嘉兴人保数据.xlsx')
    # print(X)
    # print(y)

    # 下面的代码用于抽取参数重要性
    # model = ExtraTreesClassifier()
    # model.fit(X, y)
    # # # display the relative importance of each attribute
    # print('参数重要性：', model.feature_importances_)

    # # 对数据进行预处理
    # # normalize the data attributes
    # normalized_X = preprocessing.normalize(X)
    # # standardize the data attributes
    # standardized_X = preprocessing.scale(X)
    # X = StandardScaler().fit_transform(X)

    # 进行Logistic学习，也就是训练train
    # X,y以矩阵的方式传入
    clf = LogisticRegression()
    clf.fit(X, y)
    # print(clf)

    # 得到训练之后的系数
    # print('模型参数:', clf.coef_)
    print('Ravel训练后得到的参数值:', clf.coef_.ravel())  # 多维数组转化为一维数组
    print('单位因子的增加对发生比的影响：', np.exp(clf.coef_.ravel()))  # 多维数组转化为一维数组

    # print('系数之和：', np.sum(clf.coef_.ravel()))

    print('截距：', clf.intercept_)

    # 对输入的因子进行粉线预测,返回预测值
    # sample = (np.array([39,1,1.6136,7.9091,0.1364,0.0455,85.7544,148.6852,0.09658,34.61,8.3837])).reshape(1,-1)
    # print(sample)
    # prob = clf.predict(sample)
    # print(prob)
    # prob = clf.predict(X)
    # for ii in range(1,len(prob)):
    #     if(prob[ii] != 0 ):
    #         print(prob[ii])
    #         print(ii)
    # print(prob)
    preb_proba = clf.predict_proba(X)[:,1]
    # for ii in range(len(preb_proba)):
    #     print(preb_proba[ii])

    print('总体均值',np.mean(preb_proba))
    preb_proba_mean = preb_proba-np.mean(preb_proba)
    for ii in range(len(preb_proba_mean)):
        print(preb_proba_mean[ii])

    print('出险的平均概率：',np.mean(preb_proba[0:319]))
    print('最大出险的概率：',np.max(preb_proba[0:319:]))
    print('出险概率中位数：',np.median(preb_proba[0:319:]))
    print('出险概率众数：',stats.mode(preb_proba[0:319:]))
    print('出险概率>0.11的个数：',np.sum(preb_proba[0:319]>0.12))
    print('出险的平均概率标准偏差：',np.std(preb_proba[0:319]))


    print('非出险的平均概率：',np.mean(preb_proba[320:]))
    print('最大非出险的概率：',np.max(preb_proba[320:]))
    print('非出险概率中位数：',np.median(preb_proba[320:]))
    print('非出险概率众数：',stats.mode(preb_proba[320:]))
    print('非出险概率>0.11的个数：',np.sum(preb_proba[320:]>0.12))
    print('非出险的平均概率标准偏差：',np.std(preb_proba[320:]))

    print('最大出险概率：', np.max(preb_proba))
    max_index = np.where( preb_proba == np.max(preb_proba))
    print('最大出险概率对应的UBI因子', X[max_index[0][0],:])

    print('最小出险概率：', np.max(preb_proba))
    min_index = np.where( preb_proba == np.min(preb_proba))
    print('最小出险概率对应的UBI因子', X[min_index[0][0],:])

    # 下面的代码验证了概率公式的有效性
    # coef = clf.coef_.ravel()
    # print(X[0,:])
    # print(X[0,:]*coef)
    # h = np.sum(X[0,:]*coef)+clf.intercept_
    # print(h)
    # print(sigmoid(h))


    # prob = preb_proba > 0.17


    # print(metrics.classification_report(y, prob))
    # print(metrics.confusion_matrix(y, prob))

    # 评分函数，将返回一个小于1的得分，可能会小于0
    # 这里的得分没有任何的意义，概率才是最重要的
    # score = clf.score(X, y)

    # print('模型得分：',score)

    # #准确率与召回率  
    # answer = clf.predict_proba(x_test)[:,1]  
    # precision, recall, thresholds = precision_recall_curve(y_test, answer)      
    # report = answer > 0.5  
    # print(classification_report(y_test, report, target_names = ['neg', 'pos']))  
    # print("average precision:", average/testNum)  
    # print("time spent:", time.time() - start_time)  
  
    # plot_pr(0.5, precision, recall, "pos")
