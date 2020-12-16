from random import randrange


# 计算预测精度
def accuracy_metric(actual, predictions):
    correct = 0.
    for i in range(0, len(actual)):
        if actual[i] == predictions[i]:
            correct += 1
    return correct / len(actual) * 100.0


# 模型评估

