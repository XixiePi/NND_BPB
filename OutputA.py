import numpy as np


class OutputA:
    # name
    name = ""
    # a list to store all the shape of output
    output_shape = []
    # a list to store all the output, the input is the initial output, i.e. output of layer 0
    output = []

    def __init__(self, name, output_shape=[]):
        self.name = name
        if not len(output_shape) == 0:
            self.output_shape = output_shape
            self.output = np.zeros(output_shape)

    def set_output(self, output):
        self.output = output
