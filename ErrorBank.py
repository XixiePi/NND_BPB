# In this file, all concreat implementation of transfer function is provided.
from math import pi, sin, exp, cos


def get_error_cal(fun_name):
    if fun_name == 'MSE':
        return mse
    elif fun_name == 'default':
        return mse
    else:
        raise ValueError(fun_name)


def get_error_dri(fun_name):
    if fun_name == 'MSE':
        return mse_dri
    elif fun_name == 'default':
        return mse_dri
    else:
        raise ValueError(fun_name)


def mse(t, a):
    return (t - a) ** 2


def mse_dri(t, a):
    # the derivative is respect to a
    return -2 * (t - a)
