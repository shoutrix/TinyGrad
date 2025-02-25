import numpy as np
import matplotlib.pyplot as plt


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
            expanded_dims = np.where(np.array(self.sources[0].shape)==1)[0] # for broadcasting different shapes
            if len(expanded_dims)>0:
                self.sources[0].grad += upstream_grad.mean(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[0].grad += upstream_grad
            self.sources[0].grad.no_grad()
        if self.sources[1].requires_grad:
            expanded_dims = np.where(np.array(self.sources[1].shape)==1)[0]
            if len(expanded_dims)>0:
                self.sources[1].grad += upstream_grad.mean(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[1].grad += upstream_grad
            self.sources[1].grad.no_grad()
            

class SubBackward(BaseBackward):
    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            expanded_dims = np.where(np.array(self.sources[0].shape)==1)[0] # for broadcasting different shapes
            if len(expanded_dims)>0:
                self.sources[0].grad += upstream_grad.mean(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[0].grad += upstream_grad
            self.sources[0].grad.no_grad()
        if self.sources[1].requires_grad:
            expanded_dims = np.where(np.array(self.sources[1].shape)==1)[0]
            if len(expanded_dims)>0:
                self.sources[1].grad -= upstream_grad.mean(axis=tuple(expanded_dims), keepdims=True)
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
                scale = self.sources[0].shape[0]
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

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        
        self.grad += grad
        def traverse(tensor):
            if tensor._grad_fn:
                tensor._grad_fn(tensor.grad)
                for prev in tensor._prev:
                    traverse(prev)

        traverse(self)

    def __repr__(self):
            return f"Tensor(name={self.name}, {self.data}, requires_grad={self.requires_grad}), grad_fn={self._grad_fn}"


class Linear:
    def __init__(self, fan_in, fan_out):
        self.weight = Tensor(np.random.randn(fan_in, fan_out), requires_grad=True, name="Linear_Weight")
        self.bias = Tensor(np.zeros((1, fan_out)), requires_grad=True, name = "Linear_Bias")

    def forward(self, x):
        value = x @ self.weight + self.bias
        return value
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}), grad_fn={self._grad_fn if self._grad_fn else 'None'}"


class SigmoidBackward:
    def __init__(self, value):
        self.value = value
        
    def __repr__(self):
        return "<SigmoidBackward>"

    def __call__(self, upstream_grad):
        local_grad = self.value * (1 - self.value)
        return upstream_grad * local_grad


class Sigmoid:
    def __init__(self):
        self.cache = None        

    def __call__(self, x):
        sigmoid_value = 1 / (1 + np.exp(-x.data))
        out = Tensor(sigmoid_value, requires_grad=x.requires_grad)
        if x.requires_grad:
            out._grad_fn = SigmoidBackward(sigmoid_value)
            out._prev = {x}
        return out



class MSELoss:
    def __init__(self):
        pass

    def __call__(self, pred, target):
        batch_size = pred.shape[0]
        diff_ = (pred - target) ** 2
        loss = diff_.sum() / batch_size
        return loss
            
    

# sinusoid regression

linear1 = Linear(1, 128)
sig1 = Sigmoid()
linear2 = Linear(128, 1)
loss_fn = MSELoss()

x = np.linspace(0, 2 * np.pi, 300)

x_eval = x[::3].reshape(-1, 1)
x_train = np.delete(x, np.arange(0, len(x), 3))
x_train = x_train.reshape(-1, 1)

y_train = np.sin(x_train)

fig = plt.figure(figsize=(20, 5))
plt.scatter(x_train, y_train, color="b", marker="o")
plt.savefig("sin_train_data.png")

x_train = Tensor(x_train)
y_train = Tensor(y_train)
print("input_shape : ", x_train.shape, y_train.shape)

def grad_norm(param):
    print(param.grad.shape, param.grad.data.max(), param.grad.data.min())
    print("Ufff NAN : ", np.isnan(param.grad.data).any())
    param.grad.data = np.linalg.norm(param.grad.data, ord=2) * 2.0


params = [linear1.weight, linear1.bias, linear2.weight, linear2.bias]
lr = 1e-3


# for _ in range(10000):
for param in params:
    param.grad.zero_()
    # print(param.grad)
out1 = linear1.forward(x_train)
out1 = sig1(out1)
out2 = linear2.forward(out1)
loss = loss_fn(out2, y_train)
# print("loss : ", loss)
loss.backward()
for param in params:
    grad_norm(param)
    param.data = param.data - lr * param.grad.data


x_eval = Tensor(x_eval)
print("x_eval shape : ", x_eval.shape)

out1 = linear1.forward(x_eval)
out1 = sig1(out1)
y_eval = linear2.forward(out1)


fig = plt.figure(figsize=(20, 5))
plt.scatter(x_eval.data, y_eval.data, color="b", marker="o")
plt.savefig("sin_eval_data.png")






# import torch
# import torch.nn as nn

# linear1 = nn.Linear(1, 128)
# sig1 = nn.Sigmoid()
# linear2 = nn.Linear(128, 1)
# loss_fn = nn.MSELoss()

# x = np.linspace(0, 2 * np.pi, 300, dtype=np.float32)
# x_eval = x[::3].reshape(-1, 1)
# x_train = np.delete(x, np.arange(0, len(x), 3)).reshape(-1, 1)
# y_train = np.sin(x_train)

# fig = plt.figure(figsize=(20, 5))
# plt.scatter(x_train, y_train, color="b", marker="o")
# plt.savefig("sin_train_data.png")

# x_train = torch.from_numpy(x_train).float()
# y_train = torch.from_numpy(y_train).float()

# print("input_shape : ", x_train.shape, y_train.shape, x_train.dtype, y_train.dtype)

# optimizer = torch.optim.SGD([linear1.weight, linear1.bias, linear2.weight, linear2.bias], lr=1e-3)

# for _ in range(10000):
#     optimizer.zero_grad()
#     out1 = sig1(linear1(x_train))
#     out2 = linear2(out1)
#     loss = loss_fn(out2, y_train)
#     loss.backward()
#     optimizer.step()

# x_eval = torch.from_numpy(x_eval).float()
# print("x_eval shape : ", x_eval.shape)

# with torch.no_grad():
#     y_eval = linear2(sig1(linear1(x_eval)))

# fig = plt.figure(figsize=(20, 5))
# plt.scatter(x_eval.numpy(), y_eval.numpy(), color="b", marker="o")
# plt.savefig("sin_eval_data.png")