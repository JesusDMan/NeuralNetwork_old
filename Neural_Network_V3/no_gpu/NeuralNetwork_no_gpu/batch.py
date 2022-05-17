import random
import numpy as np
import pickle

class Batch:

    def __init__(self, size: int, input_size: int):
        self.size: int = size
        self.input_size: int = input_size
        self.images, self.labels = self.create_batch()

    def create_batch(self) -> tuple:
        batch_images = []
        batch_labels = []
        for j in range(self.size):
            image = []
            label = [0] * self.input_size
            max_ = maxi = -99
            for i in range(self.input_size):
                val = int(random.random() * 20 - 10)
                while val in image:
                    val = int(random.random() * 20 - 10)
                image.append(val)
                if val > max_:
                    max_ = val
                    maxi = i
            label[maxi] = 1
            batch_images.append(np.array(image))
            batch_labels.append(np.array(label))
        return batch_images, batch_labels

    def __str__(self):
        return str(self.images) + '\n' + str(self.labels) + '\n'

    def __repr__(self):
        return str(self)

    def __bytes__(self):
        return bytes(self.images), bytes(self.labels)
