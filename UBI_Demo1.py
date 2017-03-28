# coding = utf-8
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats


def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# datemode = 0,使用1900为基础的时间戳；
# datemode = 1,使用1904为基础的时间戳
def dateMap(excelDate):
    return xlrd.xldate.xldate_as_datetime(excelDate, 0)


def ubiFunc(xlsFileName):
    sheet_index = 0
    comp_cols_indx = 4  # 赔偿金额所在例表

    g4_cols_indx = 13  # 四急所在列表，14：汇总；13：急转弯；12：急刹车；11：急减速；10：急加速

    OBD_Dind_indx = 15  # OBD绑定日期

    # 打开文件
    workbook = xlrd.open_workbook(xlsFileName)

    # 根据sheet索引或者名称获取sheet内容
    sheet1 = workbook.sheet_by_index(sheet_index)  # sheet索引从0开始
    # sheet1 = workbook.sheet_by_name('sheet1')

    # # sheet的名称，行数，列数
    # print (sheet1.name,sheet1.nrows,sheet1.ncols)

    # # 获取整行和整列的值（数组）
    # rows = sheet1.row_values(3) # 获取第四行内容
    # cols = sheet1.col_values(2) # 获取第三列内容
    # compensation = sheet1.col_values(comp_cols_indx)[1:sheet1.nrows-1] # 获取赔偿金额

    g4Count = np.array(sheet1.col_values(g4_cols_indx)[1:sheet1.nrows - 1])

    print(g4Count)
    # OBDdate = xlrd.xldate_as_tuple(sheet1.cell(OBD_Dind_indx,1).value,0)

    # 将绑定OBD的时间导出来，
    OBDdate = list(map(dateMap, sheet1.col_values(OBD_Dind_indx)[1:sheet1.nrows - 1]))
    OBDdateArray = np.array(OBDdate)  # 转换成numpy，便于后续的数组计算
    # print(OBDdateArray)

    statisticDate = datetime(year=2016, month=8, day=24)
    stallDaysArray = statisticDate - OBDdateArray

    stallDays = []
    for indx in stallDaysArray:
        stallDays.append(indx.days + 1)  # 这里面要加一天，否则出险nan的现象

    g4CountEveryDay = np.abs(g4Count / np.array(stallDays))

    print(g4CountEveryDay)

    # g4CountEveryDay = np.abs(g4Count // np.array(stallDays)) # 采用整除的方式计算日均

    # plt.hist(g4CountEveryDay,50, normed=True)  # 转换成概率后的结果
    # plt.hist(g4CountEveryDay, 70, normed=True, cumulative=True) # CDF 累计概率分布函数
    plt.hist(g4CountEveryDay, 50, normed=True, cumulative=True) # CDF 累计概率分布函数
    # plt.hist(g4CountEveryDay, 80)
    # plt.axis([0, 80, 0, 100])
    plt.grid(True)

    plt.ylabel('急转弯 累积分布函数')
    plt.xlabel('日均急转弯')
    # plt.xticks(range(0, 4, 1))
    # plt.yticks(np.linspace(0.0, 0.2, 5))

    print('样本值总数：', len(g4CountEveryDay))
    print('日均4急均值：', np.mean(g4CountEveryDay))
    print('日均4急标准偏差：', np.std(g4CountEveryDay))
    print('日均4急中位数：', np.median(g4CountEveryDay))  # 50%的概率分布点
    print('日均4急众数：', stats.mode(g4CountEveryDay))  # 出现次数最多的值

    plt.show()


if __name__ == '__main__':
    # ubiFunc('e:/python/UBI_824.xlsx')
    set_ch() #设置中文显示
    ubiFunc('e:/python/data/出险情况调查表-保单号（8月24日）.xlsx')
