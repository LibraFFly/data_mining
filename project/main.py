# -*- coding: utf-8 -*-

from random import seed


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):  # 评估算法性能，返回模型得分
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:  # 每次循环从folds从取出一个fold作为测试集，其余作为训练集，遍历整个folds，实现交叉验证
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])  # 将多个fold列表组合成一个train_set列表
        test_set = list()
        for row in fold:  # fold表示从原始数据集dataset提取出来的测试集
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Test the random forest algorithm
seed(1)  # 每一次执行本文件时都能产生同一个随机数
# load and prepare data
filename = 'sonar-all-data.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(0, len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
# str_column_to_int(dataset, len(dataset[0])-1)  ##将最后一列表示标签的值转换为Int类型0,1(可以不用转换，标签可以为str型)
# evaluate algorithm
n_folds = 5  # 分成5份数据，进行交叉验证
# max_depth = 10 #递归十次
max_depth = 20  # 调参（自己修改） #决策树深度不能太深，不然容易导致过拟合
min_size = 1
sample_size = 1.0
# n_features = int(sqrt(len(dataset[0])-1))
n_features = 15  # 调参（自己修改） #准确性与多样性之间的权衡
for n_trees in [1, 10, 20]:  # 理论上树是越多越好
    scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))