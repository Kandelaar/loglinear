import numpy as np


def scatter_add(vector, indexs):
    result = 0
    for index in indexs:
        assert index < len(vector), indexs
        result += vector[index]
    return result


def vec_exp(v):
    return np.exp(v)


def softmax(v):
    return np.exp(v) / np.sum(np.exp(v))


def vec_add(v1, v2):
    return np.array(v1) + np.array(v2)


def vec_minor(v1, v2):
    return np.array(v1) - np.array(v2)


def num_mul(n, v):
    return n * np.array(v)


def dot(v1, v2):
    return np.dot(v1, v2)


def mat_minor(m1, m2):
    return np.array(m1) * np.array(m2)


def mat_add(m1, m2):
    return np.array(m1) + np.array(m2)


def num_mul_mat(n, m):
    return n * np.array(m)


def argmax(v):
    return np.argmax(v)


def get_matrix(a, b):
    return np.zeros((a, b))
