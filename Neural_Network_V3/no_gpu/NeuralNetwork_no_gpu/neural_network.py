import os

import NeuralNetwork_no_gpu.utils
from NeuralNetwork_no_gpu.batch import Batch
from NeuralNetwork_no_gpu.layer import Layer
import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:
    number_of_layers = 0
    layers = []

    def __init__(self, name: str, bias_momentum: float, weight_momentum: float, test_batch: Batch = None,
                 train_dataset: list = None):
        self._name: str = name
        self._bias_momentum: float = bias_momentum
        self._weight_momentum: float = weight_momentum
        self._save_net: bool = True
        self.test_batch: Batch = test_batch
        self.train_dataset: list = train_dataset
        self.__mistakes_log__: list = []

        self._graph_location: str = os.getcwd()
        self._create_graph: bool = True
        self._present_graph: bool = False
        self.graph_name = f'{self._name}_graph'
        self.create_log = True
        self.log_name = f'{self._name}_log'

    def add_layer(self, size: int, activation_func: str) -> None:
        """
        Creates a new fully-connected layer and adds it to the neural network.
        :param size: The new layer's size.
        :param activation_func: The new layer's activation function's name.
        :return:
        """
        layer_index = len(self.layers)
        prev_size = size if layer_index == 0 else self.layers[layer_index - 1].size
        self.layers.append(
            Layer(index=layer_index, size=size, prev_size=prev_size, activation_func_name=activation_func))
        self.number_of_layers += 1

    def run(self, inp: np.array) -> np.array:
        """
        Runs the neural network on a given input.
        :param inp: The net's input.
        :return: The net's output.
        """
        for layer_idx in range(self.number_of_layers):
            inp = self.layers[layer_idx].run(inp)
        return inp

    def test(self, b, l):
        st = ''
        for i in range(l):
            st += str(b.images[i]) + str(self.run(b.images[i])) + '\n'
        return st

    def graph(self) -> None:
        """
        Generates a mistake value for input graph with the 'mistake_value' attribute, and saves it in the location
        specified in the 'graph_location' attribute.
        :return: None.
        """
        x = [x for x in range(len(self.__mistakes_log__))]
        y = self.__mistakes_log__
        plt.plot(x, y)
        plt.title(self._name)
        plt.savefig(f'{self._graph_location}\\{self.graph_name}')
        if self._present_graph:
            plt.show()

    def lets_be_loggin_shit(self, time: float = 0, batch_index: int = 0, number_of_slaves: int = 0):
        ms = utils.batch_mistake_value(self, batch=self.test_batch)
        self.__mistakes_log__.append(ms)
        if min(self.__mistakes_log__) == ms:
            self.save()
        self.graph()
        with open(os.path.join(os.getcwd(), f'{self._name}_log.txt'), 'a') as f:
            f.write(f'Time = {time}, i = {batch_index} MV: {ms}, SLVS: {number_of_slaves}\n')

    def save(self) -> None:
        """
        Saves the neural network's representation in a .txt file. so it could be restored later.
        :return: None.
        """
        location = os.getcwd()
        with open(rf'{location}\{self._name}.txt', 'w') as f:
            f.write(repr(self))

    def change_momentum(self, type: str = 'bias', new_val: float = 0.0) -> bool:
        """
        Changes a net momentum to a new value (if everything is legal).
        :param type: Change the Bias momentum or the Weight momentum (can be 'bias' or 'weight').
        :param new_val: The new momentum value (has to be able to be converted to float).
        :return: If the operation was successful.
        """
        try:
            new_val = float(new_val)
        except:
            return False
        if type == 'bias':
            self._bias_momentum = new_val
            return True
        elif type == 'weight':
            self._weight_momentum = new_val
            return True
        return False

    def __repr__(self):
        splitter = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        net = str(self) + splitter

        for layer in self.layers:
            net += repr(layer) + '\n' + splitter
        return net

    def __str__(self):
        st = f'-----NAME: {self._name}-----' + '\n\n'
        st += f'Weight momentum = {self._weight_momentum} | Bias momentum = {self._bias_momentum}\n'
        st += f'Training batch size = {self.train_dataset[0].size} | Testing batch size = {self.test_batch.size}\n' \
              f'Last mistake value = {self.__mistakes_log__[len(self.__mistakes_log__) - 1] if self.__mistakes_log__ else "~"}\n'
        for layer in self.layers:
            st += str(layer) + '\n'
        return st

    def __eq__(self, other):
        return type(self) == type(other) and self._bias_momentum == other._bias_momentum and \
               self._weight_momentum == other._weight_momentum and self.number_of_layers == other.number_of_layers and \
               self.layers == other.layers


def restore_net(location, name) -> NeuralNetwork:
    with open(rf'{location}\{name}.txt', 'r') as f:
        net = f.read()
    net = net.split('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    net_param = net[0]
    net_param = net_param.split('\n')
    net_name = net_param[0].replace('-----', '').replace('NAME: ', '')

    net_momentums = net_param[2].split(' | ')
    w_momentum = float(net_momentums[0].strip('Weight momentum = '))
    b_momentum = float(net_momentums[1].strip('Bias momentum = '))

    net.remove(net[0])
    net.remove(net[len(net) - 1])

    prev_size = 0
    layers = []
    for layer in net:
        layer = layer.split('\n|')
        biases = layer[1]
        weights = layer[2]
        layer = layer[0]
        layer = layer.split('\n')
        index = int(layer[0].strip('INDEX: '))
        activation_function = layer[1].strip('ACTIVATION FUNCTION: ')
        weights = utils.str2arr(weights.strip('WEIGHTS: '))
        biases = utils.str2arr(biases.strip('BIASES: '))

        l = Layer(index=index, biases=biases, weights=weights, size=len(biases), prev_size=prev_size,
                  activation_func_name=activation_function, existing_parameters=True)

        prev_size = len(biases)
        layers.append(l)

    net = NeuralNetwork(name=net_name, bias_momentum=b_momentum, weight_momentum=w_momentum)
    net.number_of_layers = len(layers)
    net.layers = layers

    return net
