import numpy as np
import TrasferFunction as tf


class NeuralLayer:

    def __init__(self, name, input_shape=[], output_shape=[], function_name_list=[], weights=[], bias=[]):
        # W[a*b]p[b*1]+b[a*1] = out[a*1]
        # in this case output_shape[0] = a; input_shape[0] = b

        self.name = name
        self.layer_index = -1

        if len(input_shape) != 0:
            self.input_shape = input_shape

        if len(output_shape) != 0:
            self.output_shape = output_shape
            self.neural_number = output_shape[0]

            self.n = np.zeros(output_shape)
            # dF is a dialog matrix and each diagonal element is the derivative of the transfer function substitute by n
            self.dF = np.zeros((self.neural_number, self.neural_number))
            self.b = np.zeros(output_shape)
            if len(bias) == 0:
                self.set_bias(np.random.rand(output_shape[0], 1) - 0.5)
            else:
                self.set_bias(bias)

        if len(input_shape) != 0 and len(output_shape) != 0:
            self.w = np.zeros((output_shape[0], input_shape[0]))
            if len(weights) == 0:
                self.set_weights(np.random.rand(output_shape[0], input_shape[0]) - 0.5)
            else:
                self.set_weights(weights)

        self.function_list = []
        if len(function_name_list) != 0:
            # step1: set the size for element:
            # step2: initialize the weights and bia
            # each neural should have his own transfer function.
            self.get_function_list(function_name_list)
            self.dF = np.zeros((self.neural_number, self.neural_number))

        self.input_flow_list = []
        self.output_flow_list = []
        self.delta_w = []
        self.delta_b = []
        self.output = None
        self.s = None

    def set_input_index(self, input_index):
        self.input_index = input_index

    def get_weights(self):
        return self.w

    def get_weights_shape(self):
        return self.w.shape

    def set_weights(self, weights):
        if self.w.shape != weights.shape:
            raise Exception("weights shape didn't match.")
        self.w = weights

    def set_bias(self, bias):
        if self.b.shape != bias.shape:
            raise Exception("bias shape didn't match.")
        self.b = bias

    def forward(self, p):
        self.forward_get_n(p)
        return self.forward_get_a()

    def forward_with_df(self, p):
        self.forward_get_n(p)
        return self.forward_get_a_and_df()

    def forward_get_n(self, p):
        weights_shape = self.w.shape
        p_shape = p.shape
        bias_shape = self.b.shape
        w_p_shape = (weights_shape[0], p_shape[1])
        if w_p_shape != bias_shape:
            raise ValueError(bias_shape)
        self.n = self.w @ p + self.b

    def cal_transfer_on_matrix(self, f, x):
        return np.reshape(np.array(list(map(f, x))), x.shape)

    def forward_get_a(self):
        a = np.zeros(self.n.shape)
        for function in self.function_list:
            # 1. get the chunk of n
            n_temp = self.n[function.neural_takes_function]
            # 2. calculate a
            a_temp = self.cal_transfer_on_matrix(function.cal, n_temp)
            # 3. put the a into the right place
            for idx, (str_a, ed_a, str_at, ed_at) in enumerate(
                    zip(function.start_in_tf_domain, function.end_in_tf_domain, function.start_in_cal_domain, function.end_in_cal_domain)):
                a[str_a:ed_a, :] = a_temp[str_at:ed_at, :]
        return a

    def forward_get_a_and_df(self):
        a = np.zeros(self.n.shape)
        df_1d = np.zeros((self.neural_number, 1))
        for function in self.function_list:
            # 1. get the chunk of n
            n_temp = self.n[function.neural_takes_function]
            # 2. calculate a
            a_temp = self.cal_transfer_on_matrix(function.cal, n_temp)
            # 3. calculate df
            df_temp = self.cal_transfer_on_matrix(function.dri, n_temp)
            # 3. put the a into the right place
            for idx, (str_a, ed_a, str_at, ed_at) in enumerate(
                    zip(function.start_in_tf_domain, function.end_in_tf_domain, function.start_in_cal_domain, function.end_in_cal_domain)):
                a[str_a:ed_a, :] = a_temp[str_at:ed_at, :]
                df_1d[str_a:ed_a, :] = df_temp[str_at:ed_at, :]
        for idx, df in enumerate(df_1d):
            self.dF[idx][idx] = df
        return a

    def get_function_list(self, function_name_list):
        function_name_list = self.decode_function_name_list(function_name_list)
        self.neural_number = len(function_name_list)
        if len(function_name_list) != self.n.shape[0]:
            raise ValueError(function_name_list)
        self.function_list = []
        uniq_function_name_list = np.unique(function_name_list)
        for uniq_function_name in uniq_function_name_list:
            neural_takes_function = [i for i, name in enumerate(function_name_list) if name == uniq_function_name]
            start_in_tf_domain, end_in_tf_domain, start_in_cal_domain, end_in_cal_domain = self.get_function_domian(neural_takes_function)
            self.function_list.append(
                tf.TransferFunction(uniq_function_name, neural_takes_function, start_in_tf_domain, end_in_tf_domain, start_in_cal_domain,
                                    end_in_cal_domain))

    def decode_function_name_list(self, function_name_list):
        function_names = [" " for i in range(self.output_shape[0])]
        for idx, function_name in enumerate(function_name_list):
            name_elements = function_name.split(":")
            if len(name_elements) == 1:
                if not function_names[idx] == " ":
                    raise ValueError(function_names[idx])
                function_names[idx] = name_elements[0]
            elif name_elements[1] == "all":
                if len(function_name_list) != 1:
                    raise ValueError(function_name_list)
                for i in range(self.output_shape[0]):
                    function_names[i] = name_elements[0]
            else:
                sb_index = function_name.find("[")
                if sb_index == -1:
                    raise ValueError(function_name)
                source_range = function_name[sb_index + 1:-1]
                [range_start, range_end] = source_range.split(":")
                for i in range(int(range_start), int(range_end)):
                    if not function_names[i] == " ":
                        raise ValueError(function_names[idx])
                    function_names[i] = name_elements[0]
        if " " in function_names:
            raise ValueError(function_name_list)
        return function_names

    def get_a(self):
        return self.a

    def add_output_index(self, from_layer, from_range, to_layer, to_range):
        # The output index should stack in a way that every output has it's own range. There should be a one to one link, between layers
        # flow_info contains from where to where, which section to which section
        # 1. find the index of the layer.
        list.output_index.append(from_layer, from_range, to_layer, to_range)
        # 2. insert the info
        pass




    def get_function_domian(self, neural_takes_function):
        start_in_tf_domain = []
        end_in_tf_domain = []

        start_in_cal_domain = []
        end_in_cal_domain = []

        count = -1
        for idx_on_cal, idx_on_tf in enumerate(neural_takes_function):

            if len(neural_takes_function) == 1:
                start_in_tf_domain.append(idx_on_tf)
                start_in_cal_domain.append(idx_on_cal)
                end_in_tf_domain.append(idx_on_tf + 1)
                end_in_cal_domain.append(idx_on_cal + 1)
                continue

            if idx_on_tf == count and idx_on_tf == len(neural_takes_function) - 1:
                end_in_tf_domain.append(idx_on_tf + 1)
                end_in_cal_domain.append(idx_on_cal + 1)
                continue

            if idx_on_tf > count and len(start_in_tf_domain) == len(end_in_tf_domain):
                start_in_tf_domain.append(idx_on_tf)
                start_in_cal_domain.append(idx_on_cal)
                count = idx_on_tf + 1
                continue

            if idx_on_tf > count and len(start_in_tf_domain) > len(end_in_tf_domain):
                end_in_tf_domain.append(idx_on_tf + 1)
                end_in_cal_domain.append(idx_on_cal + 1)
                count = idx_on_tf + 1
                continue

            if idx_on_tf == count:
                count = count + 1
                continue

            if idx_on_tf < count:
                raise ValueError(count)

        if not (len(start_in_tf_domain) == len(end_in_tf_domain) == len(start_in_cal_domain) == len(end_in_cal_domain)):
            raise Exception("Length in the function domain didn't match")

        return start_in_tf_domain, end_in_tf_domain, start_in_cal_domain, end_in_cal_domain
