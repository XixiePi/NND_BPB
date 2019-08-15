import numpy as np
from math import sin, pi


class Dataset:
    def __init__(self, name, dataset_path=None, p_list=None, t_list=None):
        self.name = name
        if  dataset_path is not None:
            self.dataset_path = dataset_path
        if p_list is not None and t_list is not None:
            self.p_list = p_list
            self.t_list = t_list
            self.length = 0
            self.t_p_list = []
            self.t_t_list = []
            self.v_p_list = []
            self.v_t_list = []

    def set_train_validation(self, portion_of_train=0, portion_of_validation=0):
        self.t_p_list = self.p_list
        self.t_t_list = self.t_list
        self.v_p_list = self.p_list
        self.v_t_list = self.t_list
