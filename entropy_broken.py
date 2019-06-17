import numpy as np
import math


def entropy(y_values: np.ndarray) -> float:
    """
    Calculate the entropy of the random variable y.
    :param y_values: The y-values to calculate the entropy of.
    :return: The entropy value, H(y).
    """
    total = len(y_values)
    _, counts = np.unique(y_values, return_counts=True)
    counts = [c / total for c in counts]
    ps = [p * math.log(p) for p in counts]
    return sum(ps)


def conditional_entropy(attr_values: np.ndarray, y_values: np.ndarray) -> float:
    """
    Calculate the conditional entropy (i.e. P(y_values|attr_values)) of the variable attr and y.
    :param attr_values: The attribute values of a single variable, corresponding to the y values.
    :param y_values: The y values.
    :return: The conditional entropy value.
    """
    total = 0
    uniq_attrs, uniq_count = np.unique(attr_values, return_counts=True)
    for idx, attr in enumerate(uniq_attrs):
        p = uniq_count[idx] / len(attr_values)
        mask = attr_values == attr
        cur_ys = y_values[mask]
        total += p * entropy(cur_ys)
    return -total * (1 / len(uniq_attrs))


def info_gain(attr_values: np.ndarray, y_values: np.ndarray) -> float:
    """
    Calculate the information gain of an attribute.
    :param attr_values: The values for the attribute that correspond to each row in the y_values variable.
    :param y_values: The y values.
    :return: The information gain, as calculated.
    """
    e = entropy(y_values)
    ce = conditional_entropy(attr_values, y_values)
    return e - ce


if __name__ == '__main__':
    data = np.array([
        [1, 1, '+'],
        [1, 0, '+'],
        [1, 1, '+'],
        [1, 0, '+'],
        [0, 1, '+'],
        [0, 0, '-'],
        [0, 1, '-'],
        [0, 0, '-']
    ])
    print(conditional_entropy(data[:, 0], data[:, 2]))
    # for i in range(2):
    #     ig = info_gain(data[:, i], data[:, 2])
    #     print(f'Information gain for feature number {i}:', ig)
    # # print(entropy(np.array(['+', '+', '+', '+', '+', '-', '-', '-'])))
