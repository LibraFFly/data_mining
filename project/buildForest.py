from random import randrange


# cart,通过gini指数来建立决策树
# 从下到上实现随机森林的建立


# 首先要根据选定的特征和特征值来划分数据集
def dataSet_split_forGini(index, value, dataSet):
    left = []
    right = []
    for row in dataSet:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# 计算gini指数
def compute_gini(childs, ):
    left, right = childs
    sum_size = len(left) + len(right)
    p_minus = 0.
    p_one = 0.
    for row in left:
        if row[-1] == -1:
            p_minus += 1
        elif row[-1] == 1:
            p_one += 1

    first_add = len(left) / sum_size * (1 - (pow(p_minus / len(left), 2) + pow(p_one / len(left), 2)))
    p_minus = 0.
    p_one = 0.
    for row in right:
        if row[-1] == -1:
            p_minus += 1
        elif row[-1] == 1:
            p_one += 1

    second_add = len(right) / sum_size * (1 - (pow(p_minus / len(right), 2) + pow(p_one / len(right), 2)))

    return first_add + second_add


# 通过最小的gini指数，寻找最优特征和特征值，分割方法
# 需要对随机选取的每个特征的每个特征值进行遍历
def get_split_forNode(dataSet, n_features):
    features = []
    # 随机选取n个特征
    for i in range(0, n_features):

        index = randrange(1, 23)
        if index not in features:
            features.append(index)

    min_gini = 99999
    cart_index, cart_value, cart_childs = 0, 0, ([], [])
    for index in features:
        for row in dataSet:
            childs = dataSet_split_forGini(index, row[index], dataSet)
            gini = compute_gini(childs)
            if gini < min_gini:
                # 对于分裂节点记录其分裂的特征，特征值，还有其孩子节点
                cart_index, cart_value, cart_childs = index, row[index], childs

    node = {'index': cart_index, 'value': cart_value, 'childs': cart_childs}
    return node


def get_res_label(dataSet):
    all_label = [row[-1] for row in dataSet]
    # max指定key，表示遍历第一个参数中的所有数据，将其作为key的方法的参数
    label = max(set(all_label), key=all_label.count)
    return label


# 从某一个节点开始递归分割
def recur_split(node, max_depth, depth, n_features, min_size):
    left, right = node['childs']

    # 当节点按照gini划分数据集后没有分裂，就可以给出目标标签了
    if len(left) == 0 or len(right) == 0:
        node['left'] = node['right'] = get_res_label(left + right)
        return

    # 可以限定递归层数，太多容易过拟合
    if depth > max_depth:
        node['left'] = get_res_label(left)
        node['right'] = get_res_label(right)
        return

    # 左孩子节点继续分裂
    # 规定数据集中只剩下min_size个数据后，就不能分割
    if len(left) <= min_size:
        node['left'] = get_res_label(left)
    else:
        node['left'] = get_split_forNode(left, n_features)
        recur_split(node['left'], max_depth, depth + 1, n_features, min_size)

    # 右孩子节点继续分裂
    # 同理
    if len(right) <= min_size:
        node['right'] = get_res_label(right)
    else:
        node['right'] = get_split_forNode(right, n_features)
        recur_split(node['right'], max_depth, depth + 1, n_features, min_size)


# 建立单棵决策树,返回根节点
def build_dec_tree(dataSet, n_features, max_depth, min_size):
    root = get_split_forNode(dataSet, n_features)
    recur_split(root, max_depth, 2, n_features, min_size)
    return root


# 建立随机森林，返回根节点表
def build_rf(tree_num, train_data, n_features, max_depth, min_size):
    forest = []
    for i in range(0, tree_num):
        forest.append(build_dec_tree(train_data, n_features, max_depth, min_size))
    return forest


