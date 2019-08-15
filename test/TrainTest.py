import Dataset as ds
import DataIterator as di
import Train as tr
import NeuralLayer as nl
import NeuralNetwork as nn
import numpy as np
from math import sin, pi





# prepare the data

p_list = []
t_list = []

for i in np.arange(0, 1):
    p_list.append(np.ones((1, 1)))
    t_list.append(np.array(1 + sin(pi / 4 * p_list[-1])).reshape((1, 1)))

dataset = ds.Dataset(name="test_dataset",p_list=p_list, t_list=t_list)

dataset.set_train_validation()

data_iterator = di.DataIterator(dataset=dataset)

# prepaer the way to analalize the data


# prepare the network
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

network = nn.NeuralNetwork(name="TestNetwork",
                           network_input_shape=(1, 1))

network.add_layer(neural_layer=layer1, input_sources="input")
network.add_layer(neural_layer=layer2, input_sources="layer1")
network.set_network_output_index(output_source="layer2", output_shape=(1, 1))

train = tr.Train(network=network, training_data_iterator=data_iterator)
train.train()


