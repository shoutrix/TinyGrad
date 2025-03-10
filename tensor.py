import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn

class BaseBackward:
    def __init__(self, *sources):
        self.sources = sources
    
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __call__(self, upstream_grad):
        raise NotImplementedError("Not Implemented")
    
    # def init_grad(self):
    #     for source in self.sources:
    #         if isinstance(source, Tensor) and source.requires_grad:
    #             if source.grad is None:
    #                 source.grad = np.zeros_like(source.data)
    #             else:
    #                 source.grad.fill(0)

class AddBackward(BaseBackward):
    def __call__(self, upstream_grad):
        
        if self.sources[0].requires_grad:
            expanded_dims = np.where(np.array(self.sources[0].shape) == 1)[0]
            if len(expanded_dims) > 0:
                self.sources[0].grad += upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[0].grad += upstream_grad
        
        if self.sources[1].requires_grad:
            expanded_dims = np.where(np.array(self.sources[1].shape) == 1)[0]
            if len(expanded_dims) > 0:
                self.sources[1].grad += upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[1].grad += upstream_grad


class SubBackward(BaseBackward):
    def __call__(self, upstream_grad):
        
        if self.sources[0].requires_grad:
            expanded_dims = np.where(np.array(self.sources[0].shape) == 1)[0]
            if len(expanded_dims) > 0:
                self.sources[0].grad += upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[0].grad += upstream_grad
        
        if self.sources[1].requires_grad:
            expanded_dims = np.where(np.array(self.sources[1].shape) == 1)[0]
            if len(expanded_dims) > 0:
                self.sources[1].grad += -upstream_grad.sum(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[1].grad += -upstream_grad

            
class MulBackward(BaseBackward):
    def __call__(self, upstream_grad):
        
        if self.sources[0].requires_grad:
            expanded_dims = np.where(np.array(self.sources[0].shape) == 1)[0]
            if len(expanded_dims) > 0:
                self.sources[0].grad += (upstream_grad * self.sources[1].data).sum(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[0].grad += (upstream_grad * self.sources[1].data)

        if self.sources[1].requires_grad:
            expanded_dims = np.where(np.array(self.sources[1].shape) == 1)[0]
            if len(expanded_dims) > 0:
                self.sources[1].grad += (upstream_grad * self.sources[0].data).sum(axis=tuple(expanded_dims), keepdims=True)
            else:
                self.sources[1].grad += (upstream_grad * self.sources[0].data)


class DivBackward(BaseBackward):
    def __init__(self, source, scalar):
        super().__init__(source)
        self.scalar = scalar

    def __call__(self, upstream_grad):
        
        if self.sources[0].requires_grad:
            self.sources[0].grad += (upstream_grad * (1 / self.scalar))

class SumBackward(BaseBackward):
    def __init__(self, source, axis=None, keepdims=False):
        super().__init__(source)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, upstream_grad):
        
        if self.sources[0].requires_grad:
            grad_shape = np.ones_like(self.sources[0].data.shape)
            if self.axis is not None:
                grad_shape = np.array(self.sources[0].data.shape)
                grad_shape[self.axis] = 1
            self.sources[0].grad += upstream_grad.reshape(grad_shape)

class MeanBackward(BaseBackward):
    def __init__(self, source, axis=None, keepdims=False):
        super().__init__(source)
        self.axis = axis
        self.keepdims = keepdims

    def __call__(self, upstream_grad):
        
        if self.sources[0].requires_grad:
            grad_shape = np.ones_like(self.sources[0].data.shape)
            if self.axis is not None:
                scale = np.prod([self.sources[0].data.shape[ax] for ax in np.atleast_1d(self.axis)])
                grad_shape = np.array(self.sources[0].data.shape)
                grad_shape[np.atleast_1d(self.axis)] = 1
            else:
                scale = np.prod(self.sources[0].data.shape)
            self.sources[0].grad += (upstream_grad.reshape(grad_shape) * (1 / scale))



class MatmulBackward(BaseBackward):
    def __call__(self, upstream_grad):
        
        
        if self.sources[0].requires_grad:
            self.sources[0].grad += (upstream_grad @ self.sources[1].data.transpose(1, 0))
        
        if self.sources[1].requires_grad:
            self.sources[1].grad += (self.sources[0].data.transpose(1, 0) @ upstream_grad)


class PowerBackward(BaseBackward):
    def __init__(self, source, power):
        super().__init__(source)
        self.power = power

    def __call__(self, upstream_grad):
        
        if self.sources[0].requires_grad:
            grad_value = upstream_grad * (self.power * np.power(self.sources[0].data, self.power - 1))
            self.sources[0].grad += grad_value


class TransposeBackward(BaseBackward):
    def __init__(self, source, axes=None):
        super().__init__(source)
        self.axes = axes
        self.inverse_axes = np.argsort(axes) if axes is not None else None

    def __call__(self, upstream_grad):
        
        if self.sources[0].requires_grad:
            grad_value = np.transpose(upstream_grad.data, self.inverse_axes)
            self.sources[0].grad += grad_value



class Tensor:
    def __init__(self, data, requires_grad=False, name=None):
        if isinstance(data, (list, tuple, int, float)):
            data = np.array(data, dtype=np.float32)
        
        self.data = data
        self.shape = data.shape
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
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
        self.data.fill(0)
        
        
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
                # print(prev.name, prev.shape, prev._grad_fn)
                build_topo_order(prev)
            stack.append(tensor)
                
        build_topo_order(self)
            
        for tensor in reversed(stack):
            # print(tensor.name, tensor.shape, tensor._grad_fn)
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

