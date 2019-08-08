
import NeuralLayer as nl
import numpy as np

input = np.ones((1,1))
output = np.zeros((2,1))
function_name_list = ["logsig","logsig"]

weights = np.array([[-0.27],[-0.41]])
bias = np.array([[-0.48],[-0.13]])
layer1 = nl.NeuralLayer("layer1",input.shape, output.shape, function_name_list,weights=weights,bias=bias)

out = layer1.forward(input)
print("out",out)

pass