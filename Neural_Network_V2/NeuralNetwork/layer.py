# Fully connected layer
import pickle
# import numpy as np
from NeuralNetwork import utils
from torch import Tensor, rand
from torch.nn import SiLU, Softmax, Softmin, Sigmoid, ReLU, ReLU6, LeakyReLU, Tanh


class Layer:
    def __init__(self, index: int = 0, size: int = 0, prev_size: int = 0, activation_func_name: str = 'chill sigmoid',
                 existing_parameters: bool = False, weights: Tensor = None, biases: Tensor = None):
        self.index = index
        self.size = size
        self.prev_size = prev_size
        self.activation_function: callable = find_activation_function(activation_func_name)
        self.activation_function_name = activation_func_name
        if not existing_parameters:
            self.weights: Tensor = rand(self.size, self.prev_size) * 2
            self.biases: Tensor = rand(self.size).unsqueeze(1)
        else:
            self.biases = biases
            self.weights = weights

    def run(self, inp: Tensor) -> Tensor:
        return self.activation_function(self.weights.mm(inp) + self.biases)

    def number_of_parameters(self) -> int:
        sum = 0
        sum += len(self.biases)
        shape = self.weights.shape
        sum += shape[0] * shape[1]
        return sum

    def __repr__(self):
        # np.set_printoptions(threshold=sys.maxsize)
        return f'INDEX: {self.index}\nACTIVATION FUNCTION: {self.activation_function_name}\n' \
               f'|BIASES: {utils.tensor2str(self.biases)}\n|WEIGHTS: {utils.tensor2str(self.weights)}'

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
    if activation_function == 'sigmoid': return Sigmoid()
    if activation_function == 'chill sigmoid': return lambda x: Sigmoid()(x/70)
    if activation_function == 'SiLU': return SiLU()
    if activation_function == 'chill SiLU': return lambda x: SiLU()(x*0.1)
    if activation_function == 'reLu': return ReLU()
    if activation_function == 'softmax': return Softmax(dim=0)
    if activation_function == 'softmax': return Softmin(dim=0)
    if activation_function == 'reLu6': return ReLU6()
    if activation_function == 'leaky reLu': return LeakyReLU()
    if activation_function == 'tanh': return Tanh()
    if activation_function == 'pass': return lambda x: x


def noice_round(x):
    return round(x * 10000) / (10000)


class InputLayer(Layer):
    def __init__(self, index: int = 0, size: int = 0, prev_size: int = 0, activation_func_name: str = 'weaker sigmoid',
                 existing_parameters: bool = False, weights=None, biases=None):
        if existing_parameters:
            super().__init__(index, size, prev_size, activation_func_name, existing_parameters, weights, biases)
        else:
            weights: Tensor = (rand(size) * 2).unsqueeze(1)
            biases: Tensor = rand(size).unsqueeze(1)
            super().__init__(index, size, prev_size, activation_func_name, existing_parameters=True, weights=weights,
                             biases=biases)


    def run(self, inp: Tensor) -> Tensor:
        return self.activation_function(self.weights * inp + self.biases)