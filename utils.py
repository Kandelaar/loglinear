from math import exp


def scatter_add(vector, indexs):
    result = 0
    for index in indexs:
        assert index < len(vector), indexs
        result += vector[index]
    return result


def vec_exp(v):
    return [exp(val) for val in v]


def softmax(v):
    temp = vec_exp(v)
    norm = sum(temp)
    return [val / norm for val in temp]


def vec_add(v1, v2):
    assert len(v1) == len(v2)
    return [val + v2[i] for i, val in enumerate(v1)]


def vec_minor(v1, v2):
    assert len(v1) == len(v2)
    return [val - v2[i] for i, val in enumerate(v1)]


def num_mul(n, v):
    return [n * val for val in v]


def dot(v1, v2):
    assert len(v1) == len(v2)
    res = 0
    for i in range(len(v1)):
        res += v1[i] * v2[i]
    return res


def mat_minor(m1, m2):
    assert len(m1) == len(m2)
    assert len(m1[0]) == len(m2[0])
    return [vec_minor(v, m2[i]) for i, v in enumerate(m1)]


def mat_add(m1, m2):
    assert len(m1) == len(m2)
    assert len(m1[0]) == len(m2[0])
    return [vec_add(v, m2[i]) for i, v in enumerate(m1)]


def num_mul_mat(n, m):
    return [num_mul(n, v) for v in m]


def argmax(v):
    return v.index(max(v))


def get_matrix(a, b):
    return [[0.0 for _ in range(b)] for _ in range(a)]
