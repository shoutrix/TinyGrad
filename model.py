import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn




# real data test
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


B, _, _ = x_train.shape
x_train = x_train.reshape(B, -1)

x_train = x_train[:100]
y_train = y_train[:100]

print(x_train)
print(y_train)




linear1 = Linear(784, 1024)
non_linearity1 = Tanh()

linear2 = Linear(1024, 256)
non_linearity2 = Tanh()

linear3 = Linear(256, 10)
loss_fn = CrossEntropyLoss()



print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = Tensor(x_train).float()
y_train = Tensor(y_train).long()
print("input_shape : ", x_train.shape, y_train.shape)

def grad_norm(param):
    print(param.name, param.grad.shape, param.grad.data.max(), param.grad.data.min())
    print("Ufff NAN : ", np.isnan(param.grad.data).any())
    param.grad.data = np.linalg.norm(param.grad.data, ord=2) * 2.0


class SGD:
    def __init__(self, params, lr, weight_decay):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        
    def step(self):
        for param in self.params:
            if param.grad is not None and param.requires_grad:
                np.add.at(param.data, slice(None), -self.lr * (param.grad.data + self.weight_decay * param.data))

    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                # print(param.shape, param.grad.shape)
                # print(type(param.grad), type(param.grad.data))
                param.grad.zero_()
                # print(type(param.grad), type(param.grad.data))


params = [linear1.weight, linear1.bias, linear2.weight, linear2.bias, linear3.weight, linear3.bias]
optimizer = SGD(params=params, lr=1e-3, weight_decay=5e-4)


for i in range(1000):
    # print(f"\n\nstep : {i}")
    start_time = time.time()
    out = linear1.forward(x_train)
    out = non_linearity1(out)
    out = linear2.forward(out)
    out = non_linearity2(out)
    out = linear3.forward(out)
    
    loss = loss_fn(out, y_train)
        
    # print("linear1 weight grad mean : ", linear1.weight.grad)
    # print("linear2 weight grad mean : ", linear2.bias.grad.mean())
    # print("linear1 bias grad mean : ", linear1.weight.grad)
    # print("linear2 bias grad mean : ", linear2.bias.grad.mean())
        
    loss_data = loss.item()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    end_time = time.time()
    throughput = 100/(end_time - start_time)
        
    if i % 100 == 0:
        print(f"\n\nstep : {i} | loss : {loss_data} | throughput : {throughput}")
        

out = linear1.forward(x_train)
out = non_linearity1(out)
out = linear2.forward(out)
out = non_linearity2(out)
out = linear3.forward(out)

# softy
exp_ = np.exp(out.data)
prob = exp_ / exp_.sum(axis=1, keepdims=True)

predicted = np.argmax(prob, axis=1)
print(predicted)