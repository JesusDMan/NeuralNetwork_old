import random

import matplotlib.pyplot as plt

import numpy as np
import torch

from NeuralNetwork.batch import Batch
# from NeuralNetwork import neural_network
import socket
import json
from torch import Tensor
import pandas as pd


def build_sales_pred_dataset(batch_size, input_size, data_location):
    dataset = []
    df = pd.read_csv(data_location).fillna(-1)

    chunk_size = batch_size
    for _chunk_idx, _from in enumerate(range(0, 5000, chunk_size)):
        print(f'Chunk {_chunk_idx}...', end='')
        chunk = df[_from: _from + chunk_size]
        print(f' - done | {chunk.shape}')
        dataset.append(Batch(size=batch_size, input_size=input_size, data=chunk))
    return dataset

def build_iris_dataset(batch_size, input_size):
    from sklearn.datasets import load_iris
    data = load_iris()

    dataset = []
    for _batch_idx, _from in enumerate(range(0, 150, batch_size)):
        print(f'Batch {_batch_idx}...', end='')
        batch = Batch(size=batch_size, input_size=input_size, create_data=False)
        batch.images = [Tensor(x).unsqueeze(1) for x in data.data[_from:_from+batch_size]]
        batch.labels = [Tensor([x]).unsqueeze(1) for x in data.target[_from:_from+batch_size]]
        # print(f'From: {_from}\nTo: {_from+batch_size}\nData: {data.data[_from:_from+batch_size]}\n'
        #       f'Images: {batch.images}\nLabels:{batch.labels}')
        # input()
        dataset.append(batch)
    return dataset

def build_test_batch(batch_size, input_size, data_location):
    df = pd.read_csv(data_location).fillna(-1)
    rand_idx = random.randint(0, len(df)-batch_size)
    return Batch(size=batch_size, input_size=input_size, data=df[rand_idx:rand_idx+batch_size])


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
    inputs = batch.images
    outputs = batch.labels

    for i in range(len(inputs)):
        mistake_sum += (net.run(inputs[i]) - outputs[i])

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


def calc_progress(p1, net):
    sum1 = 0
    sum2 = 0
    for i in range(p1[1]):
        sum1 += net.layers[i].number_of_parameters()
    if p1[0] == 'W':
        for i in range(p1[2] + 1):
            sum1 += net.layers[p1[1]].size

        sum1 += p1[3]
    else:
        sum1 += p1[2]

    print(sum1)
    # for i in range(p2[1]):
    #     sum2 += net.layers[i].number_of_parameters()
    # if p2[0] == 'W':
    #     for i in range(p2[2]):
    #         sum2 += net.layers[p2[1]].weights().size
    #
    # sum2 += p2[3]
    #

    # ('W', l3, prvl10, currl2)
    # ('W', l3, prvl10, currl2)


def arr2str(arr: np.array) -> str:
    if arr.ndim == 1:
        return json.dumps(list(arr))
    return json.dumps([list(l) for l in arr])

def str2arr(arr: str) -> np.array:
    lst = json.loads(arr)
    l = [np.array(line) for line in lst]
    return np.array(l)


def str2tensor(t: str) -> Tensor:
    lst = json.loads(t)
    l = [torch.FloatTensor(line) for line in lst]
    return torch.stack(l)


def tensor2str(t: Tensor) -> str:
    if t.ndim == 1:
        return json.dumps(list(t))
    return json.dumps([l.tolist() for l in t])
