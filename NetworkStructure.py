import NeuralLayer as nl
import InputIndex as ini
import OutputA as oa
import numpy as np


class NetworkStructure:
    network_name = ""
    network_input = []
    network_output = []
    network_input_shape = 0
    network_output_index = []
    # a list to store all the NeuralLayer
    layer_list = []
    # a list to store all the OutputA
    layer_output_list = []
    # a list to store all the List
    layer_name_list = []

    # for digging the net work
    network_inspector_sources = []
    network_inspector_output = []
    network_inspector_index = []

    def __init__(self, name: str, network_input_shape: tuple):
        # set network name
        self.network_name = name
        self.network_input_shape = network_input_shape
        # set the input
        layer_system_input = nl.NeuralLayer("input", output_shape=network_input_shape)
        self.network_input = oa.OutputA(layer_system_input.name, network_input_shape)
        self.add_to_layer_list(layer_system_input.name, layer_system_input,self.network_input)

        # set the output
        self.network_output = oa.OutputA("output")

    def set_network_inspector_index(self, inspector_source: str):
        self.network_inspector_sources = inspector_source.split(",")
        self.network_inspector_index = self.get_input_index(inspector_source)

    def set_network_output_index(self, output_source:str):
        self.network_output_index = self.get_input_index(output_source)

    def get_network_output(self):
        return self.network_output

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

    def get_input_array(self,input_index):
        input_array = []
        for [input_layer_index, input_range] in zip(input_index.input_layer_index_list,
                                                      input_index.input_range_list):
            chuck = self.layer_output_list[input_layer_index].output
            if input_range[1] == -1:
                input_range[1] = len(chuck)
            input_array.append(chuck[input_range[0]:input_range[1]])
            pass
        all_inputs = np.array(input_array)
        element_size = len(all_inputs[0][0])
        element_number = int(all_inputs.size / element_size)
        return all_inputs.reshape((element_number, element_size))

    def add_to_layer_list(self, layer_name, neural_layer, layer_output):
        self.layer_name_list.append(layer_name)
        self.layer_list.append(neural_layer)
        self.layer_output_list.append(layer_output)

    def add_layer(self, neural_layer, input_sources: str):
        neural_layer.input_index = self.get_input_index(input_sources)
        out_put = oa.OutputA(neural_layer.name,neural_layer.output_shape)
        self.add_to_layer_list(layer_name=neural_layer.name, neural_layer = neural_layer, layer_output=out_put)

    def get_input_index(self, input_sources:str):
        # check up the input_index
        # the input input_index should be "layer1, layer2[3:4]" or "layer1, layer2" or "layer1"
        input_sources_list = input_sources.split(',')
        input_layer_index_list = []
        input_range_list = []
        for input_source in input_sources_list:
            sb_index = input_source.find("[")
            if not sb_index == -1:
                source_layer = input_source[0:sb_index]
                source_range = input_source[sb_index+1:-1]
                [range_start, range_end] = source_range.split(":")
            else:
                source_layer = input_source
                range_start = 0
                range_end = -1

            layer_source_index = self.layer_name_list.index(source_layer)

            if layer_source_index == -1:
                raise Exception("Source Layer not exist:", source_layer)

            if range_start.__class__ == str:
                range_start = int(range_start)
                range_end = int(range_end)
                if self.layer_output_list[layer_source_index].output_shape[0] < range_end - 1:
                    raise Exception("Source Layer's output shape:", self.layer_output_list[layer_source_index].output_shape, "Request index:", range_end)
            input_layer_index_list.append(layer_source_index)
            input_range_list.append([range_start, range_end])
        input_index = ini.InputIndex(input_layer_index_list,input_range_list)
        return input_index
