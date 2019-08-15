# This class sibling the transfer function and it's derivative function.

import TransferFunctionBank as tfb


class TransferFunction:
    # Every transfer function has three properties: 1. name; 2. the function; 3. it's derivative

    def __init__(self, name, neural_takes_function, start_in_tf_domain, end_in__ft_domain, start_in_cal_domain, end_in_cal_domain):
        self.name = name
        self.cal = tfb.get_cal(name)
        self.dri = tfb.get_dri(name)
        self.neural_takes_function = neural_takes_function
        self.start_in_tf_domain = start_in_tf_domain
        self.start_in_cal_domain = start_in_cal_domain
        self.end_in_tf_domain = end_in__ft_domain
        self.end_in_cal_domain = end_in_cal_domain
