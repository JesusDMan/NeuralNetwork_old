import matplotlib.pyplot as plt

import numpy as np
from  NeuralNetwork_no_gpu.batch import Batch
from NeuralNetwork_no_gpu import neural_network
import socket
import json


def build_dataset(dataset_size, batch_size, input_size):
    dataset = []
    for i in range(dataset_size):
        dataset.append(Batch(size=batch_size, input_size=input_size))
    return dataset

def number2str(l):
    l = l.replace('[[', '[')
    l = l.replace(']]', ']').replace(']\n', '+').replace('\n', '').replace('+', ']\n')
    return l

def smart_write(fp, msg):
    with open(fp, 'r') as f_read:
        f_read = f_read.read()
    with open(fp, 'w') as f_write:
        f_write.write(f_read + '\n' + msg)


def batch_mistake_value(net, batch: Batch) -> float:
    mistake_sum = 0
    for i in range(batch.size):
        mistake_sum += mistake_value(batch.labels[i], net.run(batch.images[i]))
    return mistake_sum / batch.size


def mistake_value(expectation: np.array, prediction: np.array) -> float:
    mistake_sum = 0
    for i in range(len(prediction)):
        mistake_sum += np.absolute(prediction[i] - expectation[i])
    return mistake_sum / len(prediction)


def test(net, b, l):
    st = ''
    for i in range(l):
        st += (b.images[i], net.run(b.images[i]))
    return st


def graph(thing, name):
    x = [x for x in range(len(thing))]
    y = thing
    plt.plot(x, y)

    plt.title(name)

    plt.show()

def arr2str(arr: np.array) -> str:
    if arr.ndim == 1:
        return json.dumps(list(arr))
    return json.dumps([list(l) for l in arr.astype(float)])

def str2arr(arr: str) -> np.array:
    lst = json.loads(arr)
    l = [np.array(line) for line in lst]
    return np.array(l)
