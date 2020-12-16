# -*- coding: utf-8 -*-
# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange

from math import sqrt
from math import log




# Make a prediction with a decision tree
def predict(node, row):  # 预测模型分类结果
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):  # isinstance是Python中的一个内建函数。是用来判断一个对象是否是一个已知的类型。
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]  # 使用多个决策树trees对测试集test的第row行进行预测，再使用简单投票法判断出该行所属分类
    return max(set(predictions), key=predictions.count)


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):  # 创建数据集的随机子样本
    sample = list()
    n_sample = round(len(dataset) * ratio)  # round() 方法返回浮点数x的四舍五入值。
    while len(sample) < n_sample:
        index = randrange(len(dataset))  # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，此则自助采样法。从而保证每棵决策树训练集的差异性
        sample.append(dataset[index])
    return sample


# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):  # n_trees表示决策树的数量
        sample = subsample(train, sample_size)  # 随机采样保证了每棵决策树训练集的差异性
        tree = build_tree(sample, max_depth, min_size, n_features)  # 建立一个决策树
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)


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