import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn



np.random.seed(42)
torch.manual_seed(42)

class BaseBackward:
    def __init__(self, *sources):
        self.sources = sources
    
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __call__(self, upstream_grad):
        raise NotImplementedError("Not Implemented")
    
    def validate_grad(self):
        for source in self.sources:
            if isinstance(source, Tensor) and source.grad is None and source.requires_grad:
                source.grad = Tensor(np.zeros_like(source.data))

class AddBackward(BaseBackward):
    def __call__(self, upstream_grad):
        self.validate_grad()
        
        if self.sources[0].requires_grad:
            expanded_dims = np.where(np.array(self.sources[0].shape) == 1)[0]  # for broadcasting
            if len(expanded_dims) > 0:
                np.add.at(self.sources[0].grad.data, slice(None), upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True).data)
            else:
                np.add.at(self.sources[0].grad.data, slice(None), upstream_grad.data)
        
        if self.sources[1].requires_grad:
            expanded_dims = np.where(np.array(self.sources[1].shape) == 1)[0]
            if len(expanded_dims) > 0:
                np.add.at(self.sources[1].grad.data, slice(None), upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True).data)
            else:
                np.add.at(self.sources[1].grad.data, slice(None), upstream_grad.data)


class SubBackward(BaseBackward):
    def __call__(self, upstream_grad):
        self.validate_grad()
        
        if self.sources[0].requires_grad:
            expanded_dims = np.where(np.array(self.sources[0].shape) == 1)[0]
            if len(expanded_dims) > 0:
                np.add.at(self.sources[0].grad.data, slice(None), upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True).data)
            else:
                np.add.at(self.sources[0].grad.data, slice(None), upstream_grad.data)
        
        if self.sources[1].requires_grad:
            expanded_dims = np.where(np.array(self.sources[1].shape) == 1)[0]
            if len(expanded_dims) > 0:
                np.add.at(self.sources[1].grad.data, slice(None), -upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True).data)
            else:
                np.add.at(self.sources[1].grad.data, slice(None), -upstream_grad.data)

            
class MulBackward(BaseBackward):
    def __call__(self, upstream_grad):
        self.validate_grad()
        
        if self.sources[0].requires_grad:
            expanded_dims = np.where(np.array(self.sources[0].shape) == 1)[0]
            if len(expanded_dims) > 0:
                np.add.at(self.sources[0].grad.data, slice(None), (upstream_grad * self.sources[1]).sum(axis=tuple(expanded_dims), keepdims=True).data)
            else:
                np.add.at(self.sources[0].grad.data, slice(None), (upstream_grad * self.sources[1]).data)

        if self.sources[1].requires_grad:
            expanded_dims = np.where(np.array(self.sources[1].shape) == 1)[0]
            if len(expanded_dims) > 0:np.add.at(self.sources[1].grad.data, slice(None), (upstream_grad * self.sources[0]).sum(axis=tuple(expanded_dims), keepdims=True).data)
            else:
                np.add.at(self.sources[1].grad.data, slice(None), (upstream_grad * self.sources[0]).data)


class DivBackward(BaseBackward):
    def __init__(self, source, scalar):
        super().__init__(source)
        self.scalar = scalar

    def __call__(self, upstream_grad):
        self.validate_grad()
        if self.sources[0].requires_grad:
            np.add.at(self.sources[0].grad.data, slice(None), (upstream_grad * (1 / self.scalar)).data)

class SumBackward(BaseBackward):
    def __init__(self, source, axis=None, keepdims=False):
        super().__init__(source)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, upstream_grad):
        self.validate_grad()
        if self.sources[0].requires_grad:
            grad_shape = np.ones_like(self.sources[0].data.shape)
            if self.axis is not None:
                grad_shape = np.array(self.sources[0].data.shape)
                grad_shape[self.axis] = 1
            np.add.at(self.sources[0].grad.data, slice(None), upstream_grad.data.reshape(grad_shape))

class MeanBackward(BaseBackward):
    def __init__(self, source, axis=None, keepdims=False):
        super().__init__(source)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, upstream_grad):
        self.validate_grad()
        if self.sources[0].requires_grad:
            grad_shape = np.ones_like(self.sources[0].data.shape)
            if self.axis is not None:
                scale = np.prod([self.sources[0].data.shape[ax] for ax in np.atleast_1d(self.axis)])
                grad_shape = np.array(self.sources[0].data.shape)
                grad_shape[np.atleast_1d(self.axis)] = 1
            else:
                scale = np.prod(self.sources[0].data.shape)
            np.add.at(self.sources[0].grad.data, slice(None), (upstream_grad.data.reshape(grad_shape) * (1 / scale)))



class MatmulBackward(BaseBackward):
    def __call__(self, upstream_grad):
        self.validate_grad()
        
        if self.sources[0].requires_grad:
            np.add.at(self.sources[0].grad.data, slice(None), (upstream_grad @ self.sources[1].transpose(1, 0)).data)
        
        if self.sources[1].requires_grad:
            np.add.at(self.sources[1].grad.data, slice(None), (self.sources[0].transpose(1, 0) @ upstream_grad).data)


class PowerBackward(BaseBackward):
    def __init__(self, source, power):
        super().__init__(source)
        self.power = power

    def __call__(self, upstream_grad):
        self.validate_grad()
        if self.sources[0].requires_grad:
            grad_value = upstream_grad.data * (self.power * np.power(self.sources[0].data, self.power - 1))
            np.add.at(self.sources[0].grad.data, slice(None), grad_value)


class TransposeBackward(BaseBackward):
    def __init__(self, source, axes=None):
        super().__init__(source)
        self.axes = axes
        self.inverse_axes = np.argsort(axes) if axes is not None else None

    def __call__(self, upstream_grad):
        self.validate_grad()
        if self.sources[0].requires_grad:
            grad_value = np.transpose(upstream_grad.data, self.inverse_axes)
            np.add.at(self.sources[0].grad.data, slice(None), grad_value)



class Tensor:
    def __init__(self, data, requires_grad=False, name=None):
        if isinstance(data, (list, tuple, int, float)):
            data = np.array(data, dtype=np.float32)
        
        self.data = data
        self.shape = data.shape
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self._prev = set()
        self.name = name
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._grad_fn = AddBackward(self, other)
        out._prev = {self, other}
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._grad_fn = SubBackward(self, other)
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._grad_fn = MulBackward(self, other)
        out._prev = {self, other}
        return out

    def __truediv__(self, value):
        assert isinstance(value, (int, float, np.number)), "Only scalar division is supported"
        out = Tensor(self.data / value, requires_grad=self.requires_grad)
        out._grad_fn = DivBackward(self, value)
        out._prev = {self}
        return out


    def __matmul__(self, other):
        # assert isinstance(other, Tensor)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._grad_fn = MatmulBackward(self, other)
        out._prev = {self, other}
        return out  
    
    
    def __pow__(self, pow):
        if isinstance(pow, Tensor):
            pow = pow.data
        out = Tensor(np.power(self.data, pow), requires_grad = self.requires_grad)
        out._grad_fn = PowerBackward(self, pow)
        out._prev = {self}
        return out

    def transpose(self, *axes):
        if not axes:
            axes = tuple(range(len(self.shape) - 1, -1, -1))
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad)
        out._grad_fn = TransposeBackward(self, axes)
        out._prev = {self}
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if self.requires_grad:
            out._grad_fn = SumBackward(self, axis, keepdims)
            out._prev = {self}
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if self.requires_grad:
            out._grad_fn = MeanBackward(self, axis, keepdims)
            out._prev = {self}
        return out

    def no_grad(self):
        self._grad_fn = None
        self.requires_grad = False
        self.grad = None
    
    def zero_(self):
        self.data = np.zeros_like(self.data)
        
        
    def item(self):
        if self.data.size != 1:
            raise ValueError("Only a single-element tensor can be converted to a Python scalar.")
        return self.data

        
    def backward(self, grad=None):
        
        if grad is None:
            grad = np.ones_like(self.data)
        
        if self.grad is None: # for grad accumulation
            self.grad = np.zeros_like(self.data)
        
        self.grad += grad

        stack = []
        visited = set()
        def build_topo_order(tensor):
            if tensor in visited:
                return
            visited.add(tensor)
            for prev in tensor._prev:
                build_topo_order(prev)
            stack.append(tensor)
                
        build_topo_order(self)
            
        for tensor in reversed(stack):
            if tensor.grad is not None and tensor._grad_fn:
                tensor._grad_fn(tensor.grad)
                
        
    def float(self):
        self.data = self.data.astype(np.float32)
        return self
    
    def int(self):
        self.data = self.data.astype(np.int32)
        return self
    
    def long(self):
        self.data = self.data.astype(np.int64)
        return self

    def __repr__(self):
            return f"Tensor(name={self.name}, {self.data}, requires_grad={self.requires_grad}), grad_fn={self._grad_fn}"


class init:
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


class Linear:
    def __init__(self, fan_in, fan_out):
        self.weight = Tensor(np.empty((fan_in, fan_out)), requires_grad=True, name="Linear_Weight")
        self.bias = Tensor(np.empty((1, fan_out)), requires_grad=True, name="Linear_Bias")
        self.init_params(fan_in)

    def init_params(self, fan_in):
        init.kaiming_uniform_(self.weight, mode=fan_in)
        init.zeros_(self.bias)

    def forward(self, x):
        value = x @ self.weight + self.bias
        # print(f"linear out: ", value)
        return value

    def __repr__(self):
        return f"Linear(weight={self.weight}, bias={self.bias})"

class SigmoidBackward:
    def __init__(self, x):
        self.x = x
        
    def __repr__(self):
        return "<SigmoidBackward>"

    def __call__(self, upstream_grad):
        value = 1 / (1 + np.exp(-self.x.data))
        # print("sigmoid backward upstream grad : ", upstream_grad)
        local_grad = value * (1 - value)
        # print("sigmoid backward local grad : ", local_grad)
        self.x.grad = upstream_grad * local_grad


class TanhBackward:
    def __init__(self, x):
        self.x = x
        
    def __repr__(self):
        return "<TanhBackward>"

    def __call__(self, upstream_grad):
        local_grad = 1 - np.tanh(self.x.data)**2
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
        self.validate_grad()
        
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
        
        # print("pred : ", pred)
              
        exp_ = np.exp(pred.data)
        prob = exp_ / exp_.sum(axis=1, keepdims=True)
        
        # print("cross entropy loss : ", target.shape[0], target.data.dtype)
        prob_filtered = prob[np.arange(target.shape[0]), target.data]
        mean_neg_log_likelihood = -np.log(prob_filtered).mean()

        loss = Tensor(mean_neg_log_likelihood, requires_grad=True, name="loss")
        loss._prev = {pred}
        loss._grad_fn = CrossEntropyLossBackward(pred, target, prob)
        return loss



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

# import code; code.interact(local=locals())
# x_test = x_test.reshape(, -1)


linear1 = Linear(784, 1024)
non_linearity1 = Tanh()

linear2 = Linear(1024, 256)
non_linearity2 = Tanh()

linear3 = Linear(256, 10)
loss_fn = CrossEntropyLoss()

# x = np.zeros((500, 16))
# indices = np.random.choice(x.size, 1000, replace=False)
# x.flat[indices]=1
# y = np.zeros((500))
# y_indices = np.any(x[:, :8]==1, axis=1)

# y[y_indices] = 1

# x_eval = x[::3, :]
# y_eval = y[::3]
# x_train = np.delete(x, np.arange(0, len(x), 3), axis=0)
# y_train = np.delete(y, np.arange(0, len(y), 3))

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = Tensor(x_train).float()
y_train = Tensor(y_train).long()
print("input_shape : ", x_train.shape, y_train.shape)

def grad_norm(param):
    print(param.name, param.grad.shape, param.grad.data.max(), param.grad.data.min())
    print("Ufff NAN : ", np.isnan(param.grad.data).any())
    param.grad.data = np.linalg.norm(param.grad.data, ord=2) * 2.0


params = [linear1.weight, linear1.bias, linear2.weight, linear2.bias]
lr = 1e-3


for i in range(1000):
    # print(f"\n\nstep : {i}")
    start_time = time.time()
    for param in params:
        # print(param)
        if param.grad is not None:
            param.grad.zero_()
        # print(param.grad)
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
    for param in params:
        # grad_norm(param)
        param.data = param.data - lr * param.grad.data
    
    end_time = time.time()
    throughput = 100/(end_time - start_time)
        
    if i % 100 == 0:
        print(f"\n\nstep : {i} | loss : {loss_data} | throughput : {throughput}")

# print("x_eval shape : ", x_test.shape)

# print("linear 1 weight shape : ", linear1.weight.shape)
# x_eval = Tensor(x_test).float()
# out1 = linear1.forward(x_eval)
# out1 = sig1(out1)
# eval_out = linear2.forward(out1)
# pred = np.argmax(eval_out.data, axis=1)
# print("pred : ", pred)
# print("target : ", y_test)
# print(pred.shape, y_test.shape)

# acc = (pred==y_test).sum() / len(pred)
# print("accuracy : ", acc)





# linear1 = nn.Linear(16, 1024)
# sig1 = nn.Tanh()
# linear2 = nn.Linear(1024, 2)
# loss_fn = nn.CrossEntropyLoss()

# x = np.zeros((500, 16))
# indices = np.random.choice(x.size, 750, replace=False)
# x.flat[indices]=1
# y = np.zeros((500))
# y_indices = np.any(x[:, :8]==1, axis=1)

# print(y_indices)

# y[y_indices] = 1

# x_eval = x[::3, :]
# y_eval = y[::3]
# x_train = np.delete(x, np.arange(0, len(x), 3), axis=0)
# y_train = np.delete(y, np.arange(0, len(y), 3))

# print(x_train.shape, y_train.shape, x_eval.shape, y_eval.shape)

# x_train = torch.from_numpy(x_train).float()
# y_train = torch.from_numpy(y_train).long()

# # print("input_shape : ", x_train.shape, y_train.shape, x_train.dtype, y_train.dtype)

# optimizer = torch.optim.SGD([linear1.weight, linear1.bias, linear2.weight, linear2.bias], lr=1e-3)

# loss_list = []
# for i in range(1):
#     start_time = time.time()
#     optimizer.zero_grad()
#     out1 = sig1(linear1(x_train))
#     out2 = linear2(out1)
#     # print(out2.shape, y_train)
#     loss = loss_fn(out2, y_train)
    
#     loss.backward()
#     optimizer.step()
#     end_time = time.time()
#     throughput = 100 / (end_time-start_time)
#     # if i % 100 == 0:
#     #     print(f"step : {i} | loss : {loss.item()} | throughput : {throughput}")

# print("linear1 weight grad mean : ", linear1.weight.grad.mean())
# print("linear2 weight grad mean : ", linear2.bias.grad.mean())
# print("linear1 bias grad mean : ", linear1.weight.grad.mean())
# print("linear2 bias grad mean : ", linear2.bias.grad.mean())



# # x_eval = torch.from_numpy(x_eval).float()
# # print("x_eval shape : ", x_eval.shape)

# # with torch.no_grad():
# #     eval_out = linear2(sig1(linear1(x_eval)))
# #     pred = torch.argmax(eval_out, axis=1)
# #     print("pred : ", pred)
# #     print("target : ", y_eval)
    
# #     acc = (pred==y_eval).sum() / len(pred)
# #     print("accuracy : ", acc)
