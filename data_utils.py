import numpy as np
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensor import Tensor
import random


def load_data(name):
    
    np.random.seed(42)
    
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        labels = {0: "0",
          1: "1",
          2: "2",
          3: "3",
          4: "4",
          5: "5",
          6: "6",
          7: "7",
          8: "8",
          9: "9"}

    elif name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        labels = {0: "T-shirt/top",
          1: "Trouser",
          2: "Pullover",
          3: "Dress",
          4: "Coat",
          5: "Sandal",
          6: "Shirt",
          7: "Sneaker",
          8: "Bag",
          9: "Ankle boot"}
    else:
        raise ValueError(f"dataset : {name} is not supported !!")
    
    
    n_classes = int(np.unique(y_train).size)
    print("n classes : ", n_classes)
    flattened_dim = np.prod(x_train.shape[1:])
    
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train, y_train = x_train[indices], y_train[indices]

    n_valid_samples = int(np.ceil(x_train.shape[0] * 0.1))

    x_valid, y_valid = x_train[:n_valid_samples], y_train[:n_valid_samples]
    x_train, y_train = x_train[n_valid_samples:], y_train[n_valid_samples:]


    trainset = FashionMnistDataset(x_train, y_train)
    validset = FashionMnistDataset(x_valid, y_valid)
    evalset = FashionMnistDataset(x_test, y_test)

    return trainset, validset, evalset, n_classes, flattened_dim, labels


 
class FashionMnistDataset:
    def __init__(self, data, labels):
        N, _, _ = data.shape
        self.data = data.reshape(N, -1)
        self.labels = labels
        self.indices = np.arange(N)
    
    def shuffle(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.data[real_idx], self.labels[real_idx]

class FashionMnistDataloader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = True

    
    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            self.dataset.shuffle()
        
        N_batches = int(np.ceil(len(self.dataset) / self.batch_size))
        self.batches = np.array_split(self.dataset.indices, N_batches)
        print(f"Initialized dataloader with {len(self.batches)} batches")
        return self
    
    def normalize(self, x):
        x = (x - np.mean(x, axis=0, keepdims=True))/(np.std(x, axis=0, keepdims=True) + 1e-10)
        x_min = np.min(x, axis=0, keepdims=True)
        x_max = np.max(x, axis=0, keepdims=True)
        x = ((x-x_min)/(x_max-x_min + 1e-10) * 2) - 1
        return x

    def __next__(self):
        if self.current_batch >= len(self.batches):
            raise StopIteration
    
        batch_indices = self.batches[self.current_batch]
        x, y = self.dataset.data[batch_indices], self.dataset.labels[batch_indices]  
        x = self.normalize(x)
        self.current_batch += 1   
        return Tensor(x).float(), Tensor(y).long()
