import numpy as np

class BaseOptimizer:
    def __init__(self):
        pass
    
    
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
                param.grad.zero_()

class MomentumSGD:
    def __init__(self, params, lr, weight_decay=0.0, momentum=0.9):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.velocity = {param: np.zeros_like(param.data) for param in self.params}

    def step(self):
        for param in self.params:
            if param.grad is not None and param.requires_grad:
                np.add.at(param.grad.data, slice(None), self.weight_decay * param.data)
                np.add.at(self.velocity[param], slice(None), self.momentum * self.velocity[param] + param.grad.data - self.velocity[param])
                np.add.at(param.data, slice(None), -self.lr * self.velocity[param])

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
