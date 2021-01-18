import math


def vec_add(a: tuple, b: tuple):
    return a[0] + b[0], a[1] + b[1]


def vec_sub(a: tuple, b: tuple):
    return a[0] - b[0], a[1] - b[1]


def vec_abs(a: tuple):
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


def vec_dis(a: tuple, b: tuple):
    return vec_abs(vec_sub(a, b))


def vec_times(t, a: tuple):
    return t * a[0], t * a[1]


def vec_dot(a: tuple, b: tuple):
    return a[0] * b[0] + a[1] * b[1]


def vec_rev(a: tuple):
    return -a[0], -a[1]


def vec_normal(a: tuple, b: tuple, direct: tuple, rev=False):
    c = (a[1] - b[1], a[0] - b[0])
    if vec_dot(c, vec_sub(direct, a)) < 0:
        if not rev:
            c = vec_rev(c)
    else:
        if rev:
            c = vec_rev(c)
    return vec_times(1 / vec_abs(c), c)


def vec_to_int(a: tuple):
    return int(a[0]), int(a[1])


def vec_insert(a: tuple, b: tuple, n):
    lst = []
    d = 1 / (n + 1)
    for i in range(n):
        t = d * (i + 1)
        lst.append(vec_add(vec_times(1 - t, a), vec_times(t, b)))
    return lst
