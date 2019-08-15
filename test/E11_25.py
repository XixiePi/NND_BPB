import Dataset as ds
import DataIterator as di
import Train as tr
import NeuralLayer as nl
import NeuralNetwork as nn
import numpy as np
from math import sin,pi
import time


def error_condition(e):
    if np.sum(np.abs(e)) < 0.1:
        return True
    else:
        return False


def pass_train_sample_condition(e, t=None, a=None):
    return error_condition(t-a)


def is_continue_training():
    return False


def ae_analyser(e_list = [], a_list = []):
    correct_list = np.zeros(len(e_list))
    for idx, e in enumerate(e_list):
        if error_condition(e):
            correct_list[idx] = 1
        else:
            correct_list[idx] = 0

    correct_num = np.sum(correct_list)
    print("iter", train.training_data_iterator.iter_num, "e_sum", np.sum(np.abs(e_list))/len(e_list), "The num of correct:", correct_num)
    if correct_num == 80:
        train.set_is_continue_train(is_continue_training)


# prepare the data
p_list = []
t_list = []

for i in np.arange(-2, 2, 0.05):
    p_list.append(np.ones((1, 1)) * i)
    t_list.append(np.array(1 + sin(pi / 2 * p_list[-1])).reshape((1, 1)))

dataset = ds.Dataset(name="test_dataset",p_list=p_list,t_list=t_list)

dataset.set_train_validation()

data_iterator_train = di.DataIterator(p_list=dataset.t_p_list, t_list=dataset.t_t_list)
data_iterator_validation = di.DataIterator(p_list=dataset.v_p_list, t_list=dataset.v_t_list)


# prepare the network

# w_1 =np.array([[-0.09104102820309223], [-0.399086798236191], [0.3211462734889774], [-0.4602921429442325], [0.05184418428439452], [-0.2620641521572118], [-0.36586442105522243], [0.014148131598951896], [0.1891237515429619], [-0.46535909020346344]])
# b_1 =np.array([[0.19095008243589895], [0.27493869530434034], [-0.05854348977864221], [0.13166427147620008], [-0.09062146820845385], [-0.07194262195336587], [-0.21136703816952473], [-0.14952980393534077], [0.36356643977727154], [0.3803619177266502]])
function_name_list = ["logsig:all"]
layer1 = nl.NeuralLayer(name="layer1",
                        input_shape=(1, 1),
                        output_shape=(10, 1),
                        function_name_list=function_name_list)
                     #   weights=w_1, bias=b_1)

# w_2 = np.array([[-0.06554594287406135, 0.170058715314241, -0.25250030047206007, 0.47065927974824406, -0.3255926976041674, -0.49114342908144215, -0.4069472223278854, -0.44980964088312847, -0.3896824039585912, 0.2151336384314939]])
# b_2 = np.array([[0.3909413048749609]])
function_name_list = ["purelin:all"]
layer2 = nl.NeuralLayer(name="layer2",
                        input_shape=(10, 1),
                        output_shape=(1, 1),
                        function_name_list=function_name_list)
                        #weights=w_2, bias=b_2)

network = nn.NeuralNetwork(name="TestNetwork",
                           network_input_shape=(1, 1))

network.add_layer(neural_layer=layer1, input_sources="input")
network.add_layer(neural_layer=layer2, input_sources="layer1")
network.set_network_output_index(output_source="layer2", output_shape=(1, 1))

train = tr.Train(network=network,
                 training_data_iterator=data_iterator_train, validating_data_iterator=data_iterator_validation,
                 error_condition=error_condition,learning_rate=0.1,pass_train_sample_condition=pass_train_sample_condition, ae_analyser=ae_analyser)

stat_time = time.time()
train.train()
end_time = time.time()
print("Time Cost", end_time-stat_time)
