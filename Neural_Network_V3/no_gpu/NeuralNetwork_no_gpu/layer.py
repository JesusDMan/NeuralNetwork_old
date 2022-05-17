# Fully connected layer
import pickle
import numpy as np
import NeuralNetwork_no_gpu.utils
import sys


class Layer:

    def __init__(self, index=0, size=0, prev_size=0, activation_func_name='weaker sigmoid', existing_parameters=False,
                 weights=None,
                 biases=None):
        self.index = index
        self.size = size
        self.prev_size = prev_size
        self.activation_function = find_activation_function(activation_func_name)
        self.activation_function_name = activation_func_name
        if not existing_parameters:
            self.weights = np.apply_along_axis(arr=(np.random.uniform(low=-2, high=2, size=(self.size, self.prev_size)).astype(np.float64)), func1d=noice_round, axis=1)
            self.biases = np.apply_along_axis(arr=(np.random.uniform(low=-1, high=1, size=self.size).astype(np.float64)), func1d=noice_round, axis=0)
        else:
            self.biases = biases
            self.weights = weights

    def run(self, inp: np.array) -> np.array:
        return np.apply_along_axis(arr=(self.weights.dot(inp) + self.biases), func1d=self.activation_function, axis=0)

    def train_layer(self, net):
        wm = net._weight_momentum
        bm = net._bias_momentum
        b = net.train_dataset[net.batch_idx]
        for current_neuron_idx in range(self.size):
            # weights
            for prev_neuron_idx in range(self.prev_size):
                # val_0 = utils.batch_mistake_value(net, batch=b)
                self.weights[current_neuron_idx, prev_neuron_idx] -= wm
                val_1 = utils.batch_mistake_value(net, batch=b)
                self.weights[current_neuron_idx, prev_neuron_idx] += 2 * wm
                val_2 = utils.batch_mistake_value(net, batch=b)
                if val_1 < val_2:
                    self.weights[current_neuron_idx, prev_neuron_idx] -= 2 * wm

            # biases
            self.biases[current_neuron_idx] -= bm
            val_1 = utils.batch_mistake_value(net, batch=b)
            self.biases[current_neuron_idx] += 2 * bm
            val_2 = utils.batch_mistake_value(net, batch=b)
            if val_1 < val_2:
                # val_0
                self.biases[current_neuron_idx] -= 2 * bm

    def __repr__(self):
        np.set_printoptions(threshold=sys.maxsize)
        return f'INDEX: {self.index}\nACTIVATION FUNCTION: {self.activation_function_name}\n' \
               f'|BIASES: {utils.arr2str(self.biases)}\n|WEIGHTS: {utils.arr2str(self.weights)}'

    def __str__(self):
        return f'Layer {self.index}: size = {self.size}, activation function = {self.activation_function_name}'

    def __bytes__(self):
        return pickle.dumps(self.biases) + bytes('~~~~~~~~~', 'UTF8') + pickle.dumps(self.weights)

    def __eq__(self, other):
        return type(self) == type(other) and self.size == other.size and \
               self.activation_function_name == other.activation_function_name and \
               False not in (self.biases == other.biases) and False not in (self.weights == other.weights)


# ======================================================================================================================
# Activation functions

def find_activation_function(activation_function):
    if activation_function == 'strong sigmoid': return strong_sigmoid
    if activation_function == 'weak sigmoid': return weak_sigmoid
    if activation_function == 'weaker sigmoid': return weaker_sigmoid
    if activation_function == 'weaker sigmoidX2': return weaker_sigmoidX2
    if activation_function == 'tanh': return tanh
    if activation_function == 'weak tanh': return weak_tanh
    if activation_function == 'large weak tanh': return large_weak_tanh
    if activation_function == 'weaker tanh': return weaker_tanh
    if activation_function == 'reLu': return reLu
    if activation_function == 'reg': return reg


def reg(x): return x


def strong_sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def weaker_sigmoidX2(x): return 4 / (1 + 0.9 ** x) - 2


def weak_sigmoid(x):
    return 2 / (1 + np.e ** (-0.5 * x)) - 1


def weaker_sigmoid(x):
    return 2 / (1 + np.e ** (-0.2 * x)) - 1


def tanh(x):
    return np.tanh(x)


def weak_tanh(x):
    return np.tanh(0.25 * x)


def large_weak_tanh(x):
    return np.tanh(0.25 * x) * 2


def weaker_tanh(x):
    return np.tanh(0.1 * x)


def reLu(x):
    return np.maximum(-1, x)

def noice_round(x):
    return np.round(x*10000)/(10000)
