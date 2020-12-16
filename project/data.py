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
