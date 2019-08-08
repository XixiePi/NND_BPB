# This class sibling the transfer function and it's derivative function.

import TransferFunctionBank as tfb


class TransferFunction:
    # Every transfer function has three properties: 1. name; 2. the function; 3. it's derivative
    name = ""
    cal = []
    dri = []
    function_on_a = []
    start_in_a = []
    end_in_a = []
    start_in_a_temp = []
    end_in_a_temp = []

    def __init__(self, name, function_on_a, start_in_a, end_in_a, start_in_a_temp, end_in_a_temp):
        self.name = name
        self.cal = tfb.get_cal(name)
        self.dri = tfb.get_dri(name)
        self.function_on_a = function_on_a
        self.start_in_a = start_in_a
        self.start_in_a_temp = start_in_a_temp
        self.end_in_a = end_in_a
        self.end_in_a_temp = end_in_a_temp
