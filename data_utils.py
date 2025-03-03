import numpy as np
from tensorflow.keras.datasets import fashion_mnist, mnist  # Ensure correct import
from tensor import Tensor


def load_data(name):
    
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"dataset : {name} is not supported !!")
    
    
    # x_train = x_train[:100]
    # y_train = y_train[:100]
    
    # print(x_train)
    # print(y_train)
    
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train, y_train = x_train[indices], y_train[indices]
    # x_valid, y_valid = x_train, y_train

    n_valid_samples = int(np.ceil(x_train.shape[0] * 0.1))

    x_valid, y_valid = x_train[:n_valid_samples], y_train[:n_valid_samples]
    x_train, y_train = x_train[n_valid_samples:], y_train[n_valid_samples:]

    trainset = FashionMnistDataset(x_train, y_train)
    validset = FashionMnistDataset(x_valid, y_valid)
    evalset = FashionMnistDataset(x_test, y_test)

    return trainset, validset, evalset


 
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

        if shuffle:
            dataset.shuffle()
        
        N_batches = int(np.ceil(len(dataset) / batch_size))
        self.batches = np.array_split(dataset.indices, N_batches)
    
    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        self.current_batch = 0
        print(f"Initialized dataloader with {len(self.batches)} batches")
        return self

    def __next__(self):
        if self.current_batch >= len(self.batches):
            raise StopIteration
    
        batch_indices = self.batches[self.current_batch]
        x, y = self.dataset.data[batch_indices], self.dataset.labels[batch_indices]  
        self.current_batch += 1   
        return Tensor(x).float(), Tensor(y).long()
