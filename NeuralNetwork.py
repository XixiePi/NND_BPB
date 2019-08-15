import NeuralLayer as nl
import LayerRouteIndex as lri
import OutputA as oa
import numpy as np
import FlowNode as fn


class NeuralNetwork:

    def __init__(self, name: str, network_input_shape: tuple):
        # set network name
        self.network_name = name
        self.layer_list = []
        self.layer_name_list  = []
        self.network_input_shape = network_input_shape
        # set the input
        layer_system_input = nl.NeuralLayer("input", output_shape=network_input_shape)
        self.network_input = oa.OutputA(layer_system_input.name, network_input_shape)
        layer_system_input.output = oa.OutputA(layer_system_input.name, network_input_shape)
        self.add_to_layer_list(layer_system_input.name, layer_system_input)
        self.network_output = []
        self.network_output_shape = []
        self.network_inspector_sources = []
        self.network_inspector_output = []
        self.network_inspector_index = None
        self.is_train = False


        # set the output
        # self.network_output = oa.OutputA("output")

    def set_network_inspector_index(self, inspector_source: str):
        self.network_inspector_sources = inspector_source.split(",")
        self.network_inspector_index = self.set_flow_node(inspector_source)

    def set_network_output_index(self, output_source:str, output_shape):
        layer_system_output = nl.NeuralLayer("output", input_shape=output_shape, output_shape=output_shape, weights=np.ones(output_shape))
        self.network_output_shape = output_shape
        self.set_flow_node(neural_layer=layer_system_output, input_sources=output_source, layer_index=len(self.layer_list))
        layer_system_output.output = oa.OutputA(layer_system_output.name, layer_system_output.output_shape)
        layer_system_output.layer_index = self.add_to_layer_list(layer_name=layer_system_output.name,
                                                           neural_layer = layer_system_output)
        self.network_output = layer_system_output.output

    def get_network_output(self):
        return self.layer_list[-1].output.value

    def set_network_output(self, network_output):
        self.network_output = network_output

    def set_network_inspector_output(self,network_inspector_output):
        self.network_inspector_output = network_inspector_output

    def print_network_output(self):
        print("\nNetwork output", self.get_network_output())

    def print_network_inspector_output(self):
        print("\nInspector")
        for source, data in zip(self.network_inspector_sources, self.network_inspector_output):
            print("source:\t", source)
            print("data:\t", data)

    def forward(self, input):
        # iteration through all the layers
        for idx, (layer, layer_name) in enumerate(zip(self.layer_list, self.layer_name_list)):
            # the first layer in the list is the input layer
            if layer_name == "input":
                self.layer_output_list[0].output = input
                continue
            input_array = self.get_input_array(layer.input_index)
            self.layer_output_list[idx].output = layer.forward(input_array)
        # get the output for the network
        self.set_network_output(self.get_input_array(self.network_output_index))
        self.network_inspector_output = self.get_input_array(self.network_inspector_index)

    def forward_with_df(self, input):
        # iteration through all the layers
        for idx, (layer, layer_name) in enumerate(zip(self.layer_list, self.layer_name_list)):
            # the first layer in the list is the input layer
            if layer_name == "input":
                # self.layer_output_list[0].output = input
                self.layer_list[0].output.set_value(input)
                continue
            input_array = self.get_input_array(layer_idx=layer.layer_index, input_shape =layer.input_shape, input_flow_index_list=layer.input_flow_list)
            if layer_name == "output":
                # self.layer_output_list[idx].output = input_array
                self.layer_list[idx].output.set_value(input_array)
                continue
            # self.layer_output_list[idx].output = layer.forward_with_df(input_array)
            self.layer_list[idx].output.set_value(layer.forward_with_df(input_array))
        # get the output for the network
        # self.set_network_output(self.get_input_array(self.network_output_index))
        if self.network_inspector_index is not None:
            self.network_inspector_output = self.get_input_array(self.network_inspector_index)

    def backward(self, e_dir):
        # TODO is there away to make this backward into layers.
        for idx, (layer, layer_name) in enumerate(zip(reversed(self.layer_list), reversed(self.layer_name_list))):
            # There will be some different layers need to be deal with.
            # 1. The output layer will not have the weights, used for dispensing the error only, the -2(t-a) part is it's S,
            # for the convince for calculating, The W matrix of output layer will be sited to diagonal matrix fill with 1.
            # the output layer is unified.
            if layer.name == "output":
                layer.s = e_dir
                continue
            if layer.name == "input":
                continue
            # 2. The normal layers, which need to iterated normally, at this point, we didn't consider one output is used twice by different input.
            # which means each output can calculate the S individually.
            layer.s = np.zeros((layer.neural_number, self.network_output_shape[0]))
            for out_put_flow in layer.output_flow_list:
                # 1. update S
                related_df = layer.dF[out_put_flow.from_range_start:out_put_flow.from_range_end,:]
                related_next_w = self.layer_list[out_put_flow.to_layer_index].w[:,out_put_flow.from_range_start:out_put_flow.from_range_end]
                related_next_s = self.layer_list[out_put_flow.to_layer_index].s

                layer.s[out_put_flow.from_range_start:out_put_flow.from_range_end] = related_df @ np.transpose(related_next_w) @ related_next_s

            input_array = self.get_input_array(layer_idx=layer.layer_index, input_shape=layer.input_shape,
                                               input_flow_index_list=layer.input_flow_list)
            w_change = layer.s @ np.transpose(input_array)
            b_change = layer.s
            layer.delta_w.append(w_change)
            layer.delta_b.append(b_change)

    def forward_simple(self):
        pass

    def get_input_array(self, layer_idx, input_shape, input_flow_index_list):
        input_array = np.zeros(input_shape)
        for input_flow_index in input_flow_index_list:
            # TODO: need add a layer index to verify the flow_node
            if layer_idx != input_flow_index.to_layer_index:
                raise ValueError(input_flow_index.to_layer_index)

            input_array[input_flow_index.to_range_start: input_flow_index.to_range_end] \
                = self.layer_list[input_flow_index.from_layer_index].output. \
                      value[input_flow_index.from_range_start: input_flow_index.from_range_end]

        return input_array

    def add_to_layer_list(self, layer_name, neural_layer):
        self.layer_name_list.append(layer_name)
        self.layer_list.append(neural_layer)
        return len(self.layer_list) - 1

    def add_layer(self, neural_layer, input_sources: str):
        predict_layer_index = len(self.layer_list)
        self.set_flow_node(neural_layer, input_sources, predict_layer_index)
        neural_layer.output = oa.OutputA(neural_layer.name,neural_layer.output_shape)
        neural_layer.layer_index = self.add_to_layer_list(layer_name=neural_layer.name,
                                                           neural_layer = neural_layer)
        if predict_layer_index != neural_layer.layer_index:
            raise ValueError(neural_layer.layer_index)

    def set_flow_node(self, neural_layer, input_sources: str, layer_index):
        # check up the input_index
        # the input input_index should be "layer1, layer2[3:4]" or "layer1, layer2" or "layer1"
        input_sources_list = input_sources.split(',')
        layer_range_start = 0

        for input_source in input_sources_list:
            # 1. decode the source infos and verify it
            [from_layer_index, from_range_start, from_range_end] = self.decode_source(input_source)

            # 2. make the flow node
            to_layer_index = layer_index
            to_range_start = layer_range_start
            to_range_end = layer_range_start + (from_range_end - from_range_start)

            flow_node = fn.FlowNode(from_layer_index, from_range_start, from_range_end, to_layer_index, to_range_start, to_range_end)

            # 3. update both layer index
            neural_layer.input_flow_list.append(flow_node)
            self.layer_list[from_layer_index].output_flow_list.append(flow_node)
            layer_range_start = to_range_end

    def decode_source(self, input_source):

        # 1. decode input_source into different part.
        sb_index = input_source.find("[")
        if not sb_index == -1:
            source_layer = input_source[0:sb_index]
            source_range = input_source[sb_index + 1:-1]
            [from_range_start, from_range_end] = source_range.split(":")
            from_range_start = int(from_range_start)
            from_range_end = int(from_range_end)
        else:
            source_layer = input_source
            from_range_start = 0
            from_range_end = -1

        # 2. check if the layer is there
        layer_source_index = self.layer_name_list.index(source_layer)
        if layer_source_index == -1:
            raise Exception("Source Layer not exist:", source_layer)

        # 3. check the range is correct or not
        if from_range_start < 0:
            raise ValueError(from_range_start)

        if from_range_end == -1:
            from_range_end = self.layer_list[layer_source_index].output.shape[0]

        if self.layer_list[layer_source_index].output.shape[0] < from_range_end - 1:
            raise Exception("Source Layer's output shape:", self.layer_output_list[layer_source_index].output_shape,
                            "Request index:", from_range_end)

        return layer_source_index, from_range_start, from_range_end
