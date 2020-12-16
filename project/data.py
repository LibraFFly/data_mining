from random import randrange

import pandas as pd
from csv import reader


# 读取csv文件
def load_csv(data_x, label_y):
    df1 = pd.read_csv(data_x)
    df2 = pd.read_csv(label_y)
    df_merge = pd.merge(df1, df2, on='index', how='inner')
    df_merge.to_csv(r'./data/merge.csv')

    dataSet = []
    with open('./data/merge.csv', 'r') as file:
        reader_csv = reader(file)
        for row in reader_csv:
            if not row:
                continue
            dataSet.append(row)
    return dataSet


# 将数据集dataSet分成n_flods份,用于交叉验证
def cross_validation_split(dataSet, n_folds):
    dataSet_split = []
    dataSet_copy = dataSet
    fold_size = int(len(dataSet_copy) / n_folds)

    for i in range(0, n_folds):
        fold = []
        for j in range(0, fold_size):

            # 交叉验证的数据集不能有重复行，因此需要弹出来
            index = randrange(len(dataSet_copy))
            fold.append(dataSet_copy.pop(index))
        dataSet_split.append(fold)
    return dataSet_split
