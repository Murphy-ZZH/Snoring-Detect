import numpy as np

def onehot_code(labels, num_classes):
    """
    将标签列表转化为 One-Hot 编码
    :param labels: 标签列表或数组
    :param num_classes: 类别数
    :return: One-Hot 编码后的结果
    """
    # 使用 np.eye 来生成类别矩阵
    return np.eye(num_classes)[labels]
