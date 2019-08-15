import numpy as np
from math import sin, pi
import Dataset as ds

class DataIterator:
    def __init__(self, dataset=None, p_list=None, t_list=None, batch_size=1, iteration_method="loop"):

        if dataset is not None and (p_list is not None and t_list is not None):
            raise Exception("Dataset cannot be set at the same time of p_list or t_list.")

        if dataset is not None:
            self.dataset = dataset
            self.p_list = dataset.p_list
            self.t_list = dataset.t_list
            self.length = len(self.p_list)

        if p_list is not None and t_list is not None:
            self.p_list = p_list
            self.t_list = t_list
            self.length = len(self.p_list)

        self.batch_size = batch_size
        self.idx = 0
        self.iter_num = 0
        self.iter_num_adds_up = False
        self.iteration_method = None
        self.is_shuffle = False
        self.next_idx = 0


    # TODO: do we need to distinguish we need to shuffle data every loop or only the first one.
    #  In this implementation, shuffle_data means shuffle data at each loop, if there is only one loop.
    def iteration_starter(self, iteration_method="loop", shuffle_data=False, strat_idx=0):
        if self.p_list is None:
            self.p_list = self.dataset.p_list
        if self.t_list is None:
            self.t_list = self.dataset.t_list
        self.set_iter_method(iteration_method)
        self.iter_num = 0
        self.idx = strat_idx
        self.next_idx = strat_idx
        self.is_shuffle = shuffle_data

    def next(self):
        return self.iteration_method()

    # TODO: there should be a way to change the way the data get iterated,
    #  There should be at least 2 aspects of iteration,
    #  1). Loop or Single
    #  2). forward or Backward
    def set_iter_method(self, iteration_method):
        if iteration_method == "loop":
            self.iteration_method = self.loop_iteration_next
        if iteration_method == "single":
            self.iteration_method = self.single_iteration_next

    # TODO：the data iteration needs adds up the section for batch_numbers
    def loop_iteration_next(self):
        item_p = []
        itme_t = []
        self.idx = self.next_idx
        self.iter_num_adds_up = False
        if self.idx > self.length:
            raise ValueError(self.idx)
        if self.idx == self.length:
            self.iter_num = self.iter_num + 1
            self.next_idx = 0
            self.idx = self.next_idx
        self.next_idx = self.idx + 1
        if self.next_idx == self.length:
            self.iter_num_adds_up = True
        return [self.p_list[self.idx], self.t_list[self.idx]]

    def single_iteration_next(self):
        self.idx = self.next_idx
        if self.idx > self.length:
            raise ValueError(self.idx)
        if self.idx == self.length:
            return ["end", "end"]
        self.next_idx = self.next_idx + 1
        return [self.p_list[self.idx], self.t_list[self.idx]]

    # TODO: we need a method to fitch the particular section of data, to run the test maybe.
    def particular_section_fetcher(self):
        pass
