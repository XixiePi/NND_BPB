import numpy as np


class OutputA:

    def __init__(self, name, shape=[]):
        self.name = name
        self.shape = []
        self.value = []
        if not len(shape) == 0:
            self.shape = shape
            self.value = np.zeros(shape)


    def set_value(self, output):
        if not output.shape == self.shape:
            raise ValueError(output.shape)
        self.value = output
