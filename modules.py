import numpy as np

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
            self.sources[0].grad += upstream_grad
        if self.sources[1].requires_grad:
            self.sources[1].grad += upstream_grad

class SubBackward(BaseBackward):
    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            self.sources[0].grad += upstream_grad
        if self.sources[1].requires_grad:
            self.sources[1].grad -= upstream_grad

class MulBackward(BaseBackward):
    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            self.sources[0].grad += upstream_grad * self.sources[1].data
        if self.sources[1].requires_grad:
            self.sources[1].grad += upstream_grad * self.sources[0].data

class DivBackward(BaseBackward):
    def __init__(self, source, scalar):
        super().__init__(source)
        self.scalar = scalar

    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            self.sources[0].grad += upstream_grad * (1 / self.scalar)


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
            self.sources[0].grad += upstream_grad.reshape(grad_shape)


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
            self.sources[0].grad += (upstream_grad.reshape(grad_shape) * (1 / scale))


class MatmulBackward(BaseBackward):
    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            self.sources[0].grad += upstream_grad @ self.sources[1].transpose(1, 0)
        if self.sources[1].requires_grad:
            self.sources[1].grad += self.sources[0].transpose(1, 0) @ upstream_grad

class PowerBackward(BaseBackward):
    def __init__(self, source, power):
        super().__init__(source)
        self.power = power

    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            self.sources[0].grad += upstream_grad * (self.power * np.power(self.sources[0].data, self.power - 1))


class TransposeBackward(BaseBackward):
    def __init__(self, source, axes):
        super().__init__(source)
        self.axes = axes
        self.inverse_axes = np.argsort(axes)

    def __call__(self, upstream_grad):
        if self.sources[0].requires_grad:
            self.sources[0].grad += np.transpose(upstream_grad, self.inverse_axes)


class Tensor:
    def __init__(self, data, requires_grad=False, name=None):
        if isinstance(data, (list, tuple)):
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
        assert isinstance(other, Tensor)
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

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        
        self.grad += grad
        def traverse(tensor):
            if tensor._grad_fn:
                print(tensor)
                tensor._grad_fn(tensor.grad)
                for prev in tensor._prev:
                    traverse(prev)

        traverse(self)

    def __repr__(self):
            return f"Tensor(name={self.name}, {self.data}, requires_grad={self.requires_grad}), grad_fn={self._grad_fn}"


class Linear:
    def __init__(self, fan_in, fan_out):
        self.weight = Tensor(np.random.randint(0, 5, (fan_in, fan_out)), requires_grad=True, name="linear weight")
        self.bias = Tensor(np.zeros((1, fan_out)), requires_grad=True, name = "linear bias")

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



# class MSEloss:
#     def __init__(self):
#         pass

#     def __call__(self, pred, target):
#         batch_size = pred.shape
#         diff_ = (pred - target) ** 2
#         loss = diff_.sum() / batch_size
            
    



linear1 = Linear(6, 3)
sig1 = Sigmoid()
linear2 = Linear(3, 2)


x = Tensor(np.random.randint(0, 5, (1, 6)), name="input")

print("input : ", x)
print("linear_1_weight : ", linear1.weight)
print("linear_1_bias : ", linear2.bias)


print("linear_2_weight : ", linear2.weight)
print("linear_2_bias : ", linear2.bias)


out1 = linear1.forward(x)
out1 = sig1(out1)
out2 = linear2.forward(out1)

init_grad = Tensor(np.array([1, 2]), name="initial_gradient")
print("init_grad : ", init_grad)

print("\n\n\n")


out2.backward(init_grad)


print("\n\n\n")
print("linear 2 weight and bias grad : ")
print(linear2.weight.grad)
print(linear2.bias.grad)


print("\n\n\n")
print("linear 1 weight and bias grad : ")
print(linear1.weight.grad)
print(linear1.bias.grad)


print("\n\n\n")
print("linear_1_weight : ", linear1.weight)
print("linear_1_bias : ", linear2.bias)


print("linear_2_weight : ", linear2.weight)
print("linear_2_bias : ", linear2.bias)


