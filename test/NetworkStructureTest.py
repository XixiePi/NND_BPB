import NeuralLayer as nl
import numpy as np
import NetworkStructure as ns

function_name_list = ["logsig:all"]
weights = np.array([[-0.27], [-0.41]])
bias = np.array([[-0.48], [-0.13]])
layer1 = nl.NeuralLayer(name="layer1",
                        input_shape=(1, 1),
                        output_shape=(2, 1),
                        function_name_list=function_name_list,
                        weights=weights,
                        bias=bias)

function_name_list = ["purelin:all"]
weights = np.array([[0.09, -0.17]])
bias = np.array([[0.48]])
layer2 = nl.NeuralLayer(name="layer2",
                        input_shape=(2, 1),
                        output_shape=(1, 1),
                        function_name_list=function_name_list,
                        weights=weights,
                        bias=bias)

network = ns.NetworkStructure(name="TestNetwork",
                              network_input_shape=(1, 1))

network.add_layer(neural_layer=layer1, input_sources="input")
network.add_layer(neural_layer=layer2, input_sources="layer1")
network.set_network_output_index(output_source="layer2")

network.set_network_inspector_index(inspector_source="input,layer1[1:2],layer2")

network_input = np.ones((1, 1))
network.forward(network_input)

network.print_network_output()
network.print_network_inspector_output()
