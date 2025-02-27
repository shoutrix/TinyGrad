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

class AddBackward(BaseBackward):
    def __call__(self, upstream_grad):
        
        if self.sources[0].requires_grad:
            expanded_dims = np.where(np.array(self.sources[0].shape)==1)[0] # for broadcasting different shapes(mostly for the bias)
            if len(expanded_dims)>0:
                self.sources[0].grad += upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[0].grad += upstream_grad
            self.sources[0].grad.no_grad()
        if self.sources[1].requires_grad:
            # print("upstream grad : ", upstream_grad)
            expanded_dims = np.where(np.array(self.sources[1].shape)==1)[0]
            if len(expanded_dims)>0:
                self.sources[1].grad += upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[1].grad += upstream_grad
            self.sources[1].grad.no_grad()

            

class SubBackward(BaseBackward):
    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            expanded_dims = np.where(np.array(self.sources[0].shape)==1)[0] # for broadcasting different shapes
            if len(expanded_dims)>0:
                self.sources[0].grad += upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[0].grad += upstream_grad
            self.sources[0].grad.no_grad()
        if self.sources[1].requires_grad:
            expanded_dims = np.where(np.array(self.sources[1].shape)==1)[0]
            if len(expanded_dims)>0:
                self.sources[1].grad -= upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[1].grad -= upstream_grad
            self.sources[1].grad.no_grad()
            
class MulBackward(BaseBackward):
    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            self.sources[0].grad += upstream_grad * self.sources[1].data
            self.sources[0].grad.no_grad()
        if self.sources[1].requires_grad:
            self.sources[1].grad += upstream_grad * self.sources[0].data
            self.sources[1].grad.no_grad()

class DivBackward(BaseBackward):
    def __init__(self, source, scalar):
        super().__init__(source)
        self.scalar = scalar

    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            self.sources[0].grad += upstream_grad * (1 / self.scalar)
            self.sources[0].grad.no_grad()


class SumBackward(BaseBackward):
    def __init__(self, source, axis=None, keepdims=False):
        super().__init__(source)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            grad_shape = np.ones_like(self.sources[0].shape)
            if self.axis is not None:
                grad_shape = np.array(self.sources[0].shape)
                grad_shape[self.axis] = 1
            self.sources[0].grad += upstream_grad.data.reshape(grad_shape)
            self.sources[0].grad.no_grad()


class MeanBackward(BaseBackward):
    def __init__(self, source, axis=None, keepdims=False):
        super().__init__(source)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            grad_shape = np.ones_like(self.sources[0].shape)
            if self.axis is not None:
                scale = self.sources[0].shape[self.axis]
                grad_shape = np.array(self.sources[0].shape)
                grad_shape[self.axis] = 1
            else:
                scale = np.prod(self.sources[0].shape)
            self.sources[0].grad += (upstream_grad.data.reshape(grad_shape) * (1 / scale))
            self.sources[0].grad.no_grad()


class MatmulBackward(BaseBackward):
    def __call__(self, upstream_grad):
        scale = 1
        if self.sources[0].requires_grad:
            if self.sources[0].name is not None and "Linear_Weight" in self.sources[0].name:
                scale = self.sources[0].shape[0]
            self.sources[0].grad += upstream_grad @ self.sources[1].transpose(1, 0)
            self.sources[0].grad.data /= scale
            self.sources[0].grad.no_grad()
        if self.sources[1].requires_grad:
            if self.sources[1].name is not None and "Linear_Weight" in self.sources[1].name:
                scale = self.sources[1].shape[1]
            self.sources[1].grad += self.sources[0].transpose(1, 0) @ upstream_grad
            self.sources[1].grad.data /= scale
            self.sources[1].grad.no_grad()

class PowerBackward(BaseBackward):
    def __init__(self, source, power):
        super().__init__(source)
        self.power = power

    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            self.sources[0].grad += upstream_grad * (self.power * np.power(self.sources[0].data, self.power - 1))
            self.sources[0].grad.no_grad()

class TransposeBackward(BaseBackward):
    def __init__(self, source, axes):
        super().__init__(source)
        self.axes = axes
        self.inverse_axes = np.argsort(axes)

    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            self.sources[0].grad += np.transpose(upstream_grad, self.inverse_axes)
            self.sources[0].grad.no_grad()


class Tensor:
    def __init__(self, data, requires_grad=False, name=None):
        if isinstance(data, (list, tuple, int, float)):
            data = np.array(data, dtype=np.float32)
        
        self.data = data
        self.shape = data.shape
        self.requires_grad = requires_grad
        self.grad = Tensor(np.zeros_like(data)) if requires_grad else None
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

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        
        self.grad += grad
        def traverse(tensor):
            if tensor.grad is not None:
                mean_ = tensor.grad.data.mean()
            else:
                mean_ = None
            print(tensor.name, tensor.grad, tensor._grad_fn, mean_)
            if tensor._grad_fn:
                tensor._grad_fn(tensor.grad)
                for prev in tensor._prev:
                    traverse(prev)

        traverse(self)
        
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
        # print("sigmoid backward upstream grad : ", upstream_grad.shape)
        # print("sigmoid backward local grad : ", local_grad.shape)
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
        pred = self.sources[0]
        target = self.sources[1]   
        prob = self.sources[2]   
        
        upstream_grad /= pred.shape[0]
        pred_grad = prob
        pred_grad[np.arange(pred.shape[0]), target.data] -= 1
        pred.grad += Tensor(pred_grad)*upstream_grad
        pred.grad.no_grad()


class CrossEntropyLoss:
    @staticmethod
    def __call__(pred, target):
        exp_ = np.exp(pred.data)
        prob = exp_ / exp_.sum(axis=1, keepdims=True)
        
        # print("cross entropy loss : ", target.shape[0], target.data.dtype)
        prob_filtered = prob[np.arange(target.shape[0]), target.data]
        mean_neg_log_likelihood = -np.log(prob_filtered).mean()

        loss = Tensor(mean_neg_log_likelihood, requires_grad=True, name="loss")
        loss._prev = {pred}
        loss._grad_fn = CrossEntropyLossBackward(pred, target, prob)
        return loss



# sinusoid regression



linear1 = Linear(8, 16)
sig1 = Tanh()
linear2 = Linear(16, 2)
loss_fn = CrossEntropyLoss()

x = np.zeros((4, 8))
indices = np.random.choice(x.size, 16, replace=False)
x.flat[indices]=1
y = np.zeros((4))
y_indices = np.any(x[:, :4]==1, axis=1)

y[y_indices] = 1

x_eval = x[::3, :]
y_eval = y[::3]
x_train = np.delete(x, np.arange(0, len(x), 3), axis=0)
y_train = np.delete(y, np.arange(0, len(y), 3))

print(x_train.shape, y_train.shape, x_eval.shape, y_eval.shape)

x_train = Tensor(x_train).float()
y_train = Tensor(y_train).long()
print("input_shape : ", x_train.shape, y_train.shape)

def grad_norm(param):
    print(param.name, param.grad.shape, param.grad.data.max(), param.grad.data.min())
    print("Ufff NAN : ", np.isnan(param.grad.data).any())
    param.grad.data = np.linalg.norm(param.grad.data, ord=2) * 2.0


params = [linear1.weight, linear1.bias, linear2.weight, linear2.bias]
lr = 1e-3


for i in range(5):
    # print(f"\n\nstep : {i}")
    start_time = time.time()
    for param in params:
        param.grad.zero_()
        # print(param.grad)
    out1 = linear1.forward(x_train)
    out1 = sig1(out1)
    out2 = linear2.forward(out1)
    loss = loss_fn(out2, y_train)
        
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
        
    # if i % 100 == 0:
    print(f"\n\nstep : {i} | loss : {loss_data} | throughput : {throughput}")

print("x_eval shape : ", x_eval.shape)

print("linear 1 weight shape : ", linear1.weight.shape)
x_eval = Tensor(x_eval).float()
out1 = linear1.forward(x_eval)
out1 = sig1(out1)
eval_out = linear2.forward(out1)
pred = np.argmax(eval_out.data, axis=1)
print("pred : ", pred)
print("target : ", y_eval)
print(pred.shape, y_eval.shape)

acc = (pred==y_eval).sum() / len(pred)
print("accuracy : ", acc)





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
