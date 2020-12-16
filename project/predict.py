

# 单棵决策树预测单行数据结果
def tree_predict(test_data_row, node):
    res = []
    # 该节点选择的分裂特征和特征值
    index = node['index']
    value = node['value']

    # 小于特征值，去左子树
    if test_data_row[index] < value:

        # 字典结构的节点才能进一步分裂，否则表示预测结果值
        if isinstance(node['left'], dict):
            return tree_predict(test_data_row, node['left'])
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return tree_predict(test_data_row, node['right'])
        else:
            return node['right']


# 随机森林预测测试集结果
def forest_predict(forest, test_data):
    res_label = []
    for row in test_data:

        # 随机森林预测单行结果
        predictions = [tree_predict(row, tree) for tree in forest]
        # 少数服从多数
        y_label = max(set(predictions), key=predictions.count)
        res_label.append(y_label)
    return res_label


