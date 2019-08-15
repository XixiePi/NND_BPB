import NeuralNetwork as nn
import ErrorBank as eb
import numpy as np

class Train:
    def __init__(self, network, training_data_iterator=None, validating_data_iterator=None, batch_size=1, learning_rate=0.1, error_method="MSE", error_condition=None, pass_train_sample_condition=None,ae_analyser=None):
        self.network = network
        # TODO: The training data should be a class, rather than a structure.
        self.training_data_iterator = training_data_iterator
        if validating_data_iterator is not None:
            self.validating_data_iterator = validating_data_iterator
        # TODO: THE Batch_size in training session only related to how the error is calculated
        #  it is not how relate to the memory efficacy.
        self.batch_size = batch_size

        self.error = None
        self.get_error = eb.get_error_cal(error_method)
        self.get_error_dir = eb.get_error_dri(error_method)
        self.learning_rate = learning_rate
        self.iter_num = 0
        if ae_analyser is not None:
            self.ae_analyser = ae_analyser
        else:
            self.ae_analyser = self.default_ae_analyser

        if error_condition is not None:
            self.error_condition = error_condition
        else:
            self.error_condition = self.default_error_condition

        if pass_train_sample_condition is not None:
            self.pass_train_sample_condition = pass_train_sample_condition
        else:
            self.pass_train_sample_condition = self.default_pass_train_sample_condition

        self.is_continue_train = self.default_is_continue_train


    def train(self):
        self.training_data_iterator.iteration_starter()
        while self.is_continue_train():
            [p, t] = self.training_data_iterator.next()
            a = self.forward(p)
            e = self.get_error(t=t, a=a)

            # To check if the this p-t pair gonna use for training.
            if not self.pass_train_sample_condition(e=e,t=t, a=a):
                # Each iteration, the error is got individually with record
                e_dir = self.get_error_dir(t=t, a=a)
                # In the backward process
                # It is possible to get the delta_w at the backward process.
                self.backward(e_dir)
                # Each iteration, the delta w is noted, however, not processed.
                # TODO: the only different between different batch number is when and how to update the wights, the weights update only deal with the counting and update weight.
                self.weights_update()
            self.evaluate()

    def evaluate(self):
        # need to set the evaluate method
        if self.training_data_iterator.iter_num_adds_up:
            self.validating_data_iterator.iteration_starter(iteration_method="single")
            e_list = []
            a_list = []
            while True:
                [p, t] = self.validating_data_iterator.next()
                if p is "end" and t is "end":
                    break
                a_list.append(self.forward(p))
                # Each iteration, the error is got individually with record.
                e_list.append(t - a_list[-1])
            # How to deal with the e is like more individually stuff, how to make this works out for every one?
            if self.ae_analyser is not None:
                self.ae_analyser(e_list= e_list,a_list=a_list)

    def set_pass_train_sample_condition(self,pass_train_sample_condition):
        self.pass_train_sample_condition = pass_train_sample_condition

    def set_error_condition(self,error_condtion):
        self.error_condition = error_condtion

    def set_ae_analyser(self, ae_analyser):
        self.ae_analyser = ae_analyser

    def set_is_continue_train(self,is_continue):
        self.is_continue_train = is_continue

    def forward(self, p):
        self.network.forward_with_df(p)
        return self.network.get_network_output()

    def backward(self,e_dir):
        self.network.backward(e_dir)

    def weights_update(self):
        if self.training_data_iterator.next_idx % self.batch_size == 0:
            for layer in self.network.layer_list:
                if (not layer.delta_w) or (not layer.delta_b):
                    continue
                delta_w = sum(layer.delta_w)
                delta_b = sum(layer.delta_b)

                # after each iteration the delta_w and delta_b will be clear out
                layer.delta_w = []
                layer.delta_b = []

                w_updated = layer.w - self.learning_rate * delta_w
                b_updated = layer.b - self.learning_rate * delta_b

                layer.set_weights(w_updated)
                layer.set_bias(b_updated)
            pass

    def is_accomplish_goal(self):
        pass

    def is_continue_train(self):
        return True

    def default_ae_analyser(self):
        pass

    def default_error_condition(self):
        return True

    def default_pass_train_sample_condition(self):
        return False

    def default_is_continue_train(self):
        return True