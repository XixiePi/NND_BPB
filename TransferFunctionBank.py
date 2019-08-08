# In this file, all concreat implementation of transfer function is provided.
from math import pi, sin, exp, cos


def get_cal(fun_name):
    if fun_name == 'logsig':
        return logsig
    elif fun_name == 'purelin':
        return purelin
    else:
        raise ValueError(fun_name)


def get_dri(fun_name):
    if fun_name == 'logsig':
        return logsig_dri
    elif fun_name == 'purelin':
        return purelin_dri
    else:
        raise ValueError(fun_name)


def logsig(x):
    return 1 / (1 + exp(-x))


def logsig_dri(x):
    return exp(-x) / (1 + exp(-x)) ** 2


def purelin(x):
    return x


def purelin_dri(x):
    return 1


'''
def tansig(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


def tansigS(n):
    return (sym.exp(n) - sym.exp(-n)) / (sym.exp(n) + sym.exp(-n))


def sqr(x):
    return x ** 3


def sqrS(n):
    return n ** 3


def times(x):
    return x


def timesS(n):
    return n
'''