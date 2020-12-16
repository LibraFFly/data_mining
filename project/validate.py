from random import randrange


# 将数据集dataSet分成n_flods份,用于交叉验证
def cross_validation_split(dataSet, n_folds):
    dataSet_split = []
    dataSet_copy = dataSet
    fold_size = dataSet / n_folds
    for i in range(n_folds):
        fold = list()  # 每次循环fold清零，防止重复导入dataset_split
        while len(fold) < fold_size:
            index = randrange(len(dataSet_copy))
            fold.append(dataSet_copy.pop(index))
        dataSet_split.append(fold)
    return dataSet_split  # 由dataset分割出的n_folds个数据构成的列表，为了用于交叉验证


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):  # 导入实际值和预测值，计算精确度
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0