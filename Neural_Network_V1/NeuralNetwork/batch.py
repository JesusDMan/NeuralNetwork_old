import json
import random
import pickle
import os
from torch import Tensor
import pandas as pd
from sklearn.datasets import load_iris

class Batch:

    def __init__(self, size: int = 0, input_size: int = 0, create_data: bool = True, data = None):
        self.size: int = size
        self.input_size: int = input_size
        self.paths: list = []
        self.create_data = create_data
        self.data = data
        if self.create_data:
            self.images, self.labels = self.create_batch()

    def create_batch(self) -> tuple:
        batch_images = []
        batch_labels = []

        for _att in self.data.values:
            batch_images.append(Tensor(_att[0:len(_att)-1]/10).unsqueeze(1))
            batch_labels.append(Tensor([_att[len(_att)-1]/1000]).unsqueeze(1))


        return batch_images, batch_labels

    def __str__(self):
        return str(self.images) + '\n' + str(self.labels) + '\n'

    def __repr__(self):
        return str(self)

    def __bytes__(self):
        return bytes(self.images), bytes(self.labels)

