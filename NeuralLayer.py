import numpy as np
import TrasferFunction as tf


class NeuralLayer:
    # It should have a neural prospective lists
    name = ""
    function_list = []
    weights = 0
    bias = 0
    n = 0
    input_index = []
    input_shape = []
    output_shape = []

    def __init__(self, name, input_shape = [], output_shape = [], function_name_list = [], weights = [], bias = []):
        # W[a*b]p[b*1]+b[a*1] = out[a*1]
        # in this case output_shape[0] = a; input_shape[0] = b

        self.name = name

        if len(input_shape) != 0:
            self.input_shape = input_shape

        if len(output_shape) != 0:
            self.output_shape = output_shape

        if len(function_name_list) != 0:
            # step1: set the size for element:
            self.n = np.zeros(output_shape)
            self.weights = np.zeros((output_shape[0], input_shape[0]))
            self.bias = np.zeros(output_shape)

            # step2: initialize the weights and bias
            if len(weights) == 0:
                self.weights = self.set_weights(np.random.rand(output_shape[0], input_shape[0]) - 0.5)
            else:
                self.set_weights(weights)
            if len(bias) == 0:
                self.bias = self.set_bias(np.random.rand(output_shape[0], 1) - 0.5)
            else:
                self.set_bias(bias)

            # each neural should have his own functionList
            self.get_function_list(function_name_list)

    def set_input_index(self, input_index):
        self.input_index = input_index

    def get_weights(self):
        return self.weights

    def get_weights_shape(self):
        return self.weights.shape

    def set_weights(self,weights):
        if self.weights.shape != weights.shape:
            raise Exception("weights shape didn't match.")
        self.weights = weights

    def set_bias(self,bias):
        if self.bias.shape != bias.shape:
            raise Exception("bias shape didn't match.")
        self.bias = bias

    def forward(self, p):
        self.forward_get_n(p)
        return self.forward_get_a()

    def forward_get_n(self, p):
        weights_shape = self.weights.shape
        p_shape = p.shape
        bias_shape = self.bias.shape
        w_p_shape = (weights_shape[0],p_shape[1])
        if w_p_shape != bias_shape:
            raise ValueError(bias_shape)
        self.n = self.weights @ p + self.bias

    def cal_transfer_on_matrix(self, f, x):
        return np.reshape(np.array(list(map(f, x))), x.shape)

    def forward_get_a(self):
        a = np.zeros(self.n.shape)
        for function in self.function_list:
            # 1. get the chunk of n
            n_temp = self.n[function.function_on_a]
            # 2. calculate a
            a_temp = self.cal_transfer_on_matrix(function.cal,n_temp)
            # 3. put the a into the right place
            for idx, (str_a, ed_a, str_at, ed_at) in enumerate(zip(function.start_in_a, function.end_in_a, function.start_in_a_temp, function.end_in_a_temp)):
                a[str_a:ed_a, :] = a_temp[str_at:ed_at, :]
        return a

    def get_function_list(self,function_name_list):
        function_name_list = self.decode_function_name_list(function_name_list)
        if len(function_name_list) != self.n.shape[0]:
            raise ValueError(function_name_list)
        self.function_list = []
        uniq_function_name_list = np.unique(function_name_list)
        for uniq_function_name in uniq_function_name_list:
            function_on_a = [i for i, name in enumerate(function_name_list) if name == uniq_function_name]
            start_in_a, end_in_a, start_in_a_temp, end_in_a_temp = self.get_function_domian(function_on_a)
            self.function_list.append(tf.TransferFunction(uniq_function_name, function_on_a, start_in_a, end_in_a, start_in_a_temp, end_in_a_temp))

    def decode_function_name_list(self,function_name_list):
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
                for i in range(int(range_start),int(range_end)):
                    if not function_names[i] == " ":
                        raise ValueError(function_names[idx])
                    function_names[i] = name_elements[0]
        if " " in function_names:
            raise ValueError(function_name_list)
        return function_names

    def get_a(self):
        return self.a

    def get_function_domian(self, function_on_a):
        start_in_a = []
        end_in_a = []

        start_in_a_temp = []
        end_in_a_temp = []

        count = -1
        for idx_on_a_temp, idx_on_a in enumerate(function_on_a):

            if len(function_on_a) == 1:
                start_in_a.append(idx_on_a)
                start_in_a_temp.append(idx_on_a_temp)
                end_in_a.append(idx_on_a + 1)
                end_in_a_temp.append(idx_on_a_temp + 1)
                continue

            if idx_on_a == count and idx_on_a == len(function_on_a)-1:
                end_in_a.append(idx_on_a + 1)
                end_in_a_temp.append(idx_on_a_temp + 1)
                continue

            if idx_on_a > count and len(start_in_a) == len(end_in_a):
                start_in_a.append(idx_on_a)
                start_in_a_temp.append(idx_on_a_temp)
                count = idx_on_a + 1
                continue

            if idx_on_a > count and len(start_in_a) > len(end_in_a):
                end_in_a.append(idx_on_a + 1)
                end_in_a_temp.append(idx_on_a_temp + 1)
                count = idx_on_a + 1
                continue

            if idx_on_a == count:
                count = count + 1
                continue

            if idx_on_a < count:
                raise ValueError(count)

        if not (len(start_in_a) == len(end_in_a) == len(start_in_a_temp) == len(end_in_a_temp)):
            raise Exception("Length in the function domain didn't match")

        return start_in_a, end_in_a, start_in_a_temp, end_in_a_temp
