import numpy as np
from tensor import Tensor, BaseBackward


def add_nonlinearity(name):
    assert name in ["sigmoid", "tanh", "ReLU"]
    if name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "ReLU":
        return ReLU()


class init:
    @staticmethod
    def random_(tensor):
        tensor.data[:] = np.random.randn(*tensor.data.shape)
        
    @staticmethod
    def uniform_(tensor, a=0.0, b=1.0):
        tensor.data[:] = np.random.uniform(a, b, tensor.data.shape)

    @staticmethod
    def zeros_(tensor):
        tensor.data[:] = np.zeros_like(tensor.data)

    @staticmethod
    def kaiming_uniform_(tensor, mode):
        bound = np.sqrt(6 / mode)
        tensor.data[:] = np.random.uniform(-bound, bound, tensor.data.shape)

    @staticmethod
    def xavier_uniform_(tensor):
        print("xavier init")
        fan_in, fan_out = tensor.data.shape
        bound = np.sqrt(6 / (fan_in + fan_out))
        tensor.data[:] = np.random.uniform(-bound, bound, tensor.data.shape)



class Linear:
    def __init__(self, fan_in, fan_out, weight_init):
        self.weight = Tensor(np.empty((fan_in, fan_out)), requires_grad=True, name="Linear_Weight")
        self.bias = Tensor(np.empty((1, fan_out)), requires_grad=True, name="Linear_Bias")
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.weight_init = weight_init
        self.init_params()

    def init_params(self):
        if self.weight_init == "random":
            init.random_(self.weight)
        elif self.weight_init == "Xavier":
            init.xavier_uniform_(self.weight)
        elif self.weight_init == "kaiming":
            init.kaiming_uniform_(self.weight, mode=self.fan_out)
        init.zeros_(self.bias)

    def __call__(self, x):
        value = x @ self.weight + self.bias
        return value

    def __repr__(self):
        return f"Linear(fan_in={self.fan_in}, fan_out={self.fan_out})"

class SigmoidBackward:
    def __init__(self, x):
        self.x = x
        
    def __repr__(self):
        return "<SigmoidBackward>"

    def __call__(self, upstream_grad):
        value = 1 / (1 + np.exp(-self.x.data))
        local_grad = value * (1 - value)
        self.x.grad = upstream_grad * local_grad

class Sigmoid:
    def __init__(self):
        self.cache = None        

    def __call__(self, x):
        sigmoid_value = 1 / (1 + np.exp(-x.data))
        out = Tensor(sigmoid_value, requires_grad=x.requires_grad)
        if x.requires_grad:
            out._grad_fn = SigmoidBackward(x)
            out._prev = {x}
        return out

class ReLUBackward:
    def __init__(self, x):
        self.x = x
        
    def __repr__(self):
        return "<ReLUBackward>"

    def __call__(self, upstream_grad):
        local_grad = (self.x.data > 0).astype(self.x.data.dtype)
        self.x.grad = upstream_grad * local_grad


class ReLU:
    def __init__(self):
        self.cache = None
    
    def __call__(self, x):
        value = np.maximum(0, x.data)
        out = Tensor(value, requires_grad=x.requires_grad)
        if x.requires_grad:
            out._grad_fn = ReLUBackward(x)
            out._prev = {x}
        return out
        


class TanhBackward:
    def __init__(self, x):
        self.x = x
        
    def __repr__(self):
        return "<TanhBackward>"

    def __call__(self, upstream_grad):
        local_grad = 1 - np.tanh(self.x.data)**2
        self.x.grad = upstream_grad * local_grad

class Tanh:
    def __init__(self):
        self.cache = None        

    def __call__(self, x):
        tanh_value = np.tanh(x.data)
        out = Tensor(tanh_value, requires_grad=x.requires_grad)
        if x.requires_grad:
            out._grad_fn = TanhBackward(x)
            out._prev = {x}
        return out

class MSELoss:
    @staticmethod
    def __call__(self, pred, target):
        batch_size = pred.shape[0]
        diff_ = (pred - target) ** 2
        loss = diff_.sum() / batch_size
        return loss
            
class CrossEntropyLossBackward(BaseBackward):
    def __call__(self, upstream_grad):
        self.init_grad()
        
        pred = self.sources[0]
        target = self.sources[1]   
        prob = self.sources[2]   

        scaled_upstream_grad = upstream_grad / pred.shape[0]

        pred_grad = prob.copy()
        pred_grad[np.arange(pred.shape[0]), target.data] -= 1
        np.add.at(pred.grad.data, slice(None), pred_grad * scaled_upstream_grad)

class CrossEntropyLoss:
    @staticmethod
    def __call__(pred, target):  
                      
        exp_ = np.exp(pred.data)
        prob = exp_ / exp_.sum(axis=1, keepdims=True)
        
        prob_filtered = prob[np.arange(target.shape[0]), target.data]
        mean_neg_log_likelihood = -np.log(prob_filtered).mean()

        loss = Tensor(mean_neg_log_likelihood, requires_grad=True, name="loss")
        loss._prev = {pred}
        loss._grad_fn = CrossEntropyLossBackward(pred, target, prob)
        return loss
