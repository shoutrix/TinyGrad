import numpy as np
from tensor import Tensor, BaseBackward


def add_nonlinearity(name):
    assert name in ["sigmoid", "tanh", "ReLU"]
    print("Adding Non-linearity : ", name)
    if name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    elif name == "ReLU":
        return ReLU()


class init:
    @staticmethod
    def calculate_gain(nonlinearity):
        if nonlinearity == "tanh":
            gain = 5.0 / 3
        elif nonlinearity == "relu":
            gain = np.sqrt(2.0)
        else:
            gain = 1
        return gain
    
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
    def kaiming_uniform_(tensor, mode, nonlinearity):
        print("Initializing weights with kaiming uniform")
        gain = init.calculate_gain(nonlinearity)
        bound = np.sqrt(3 / mode) * gain
        print(f"Initializing weights from a uniform distribution with bound : {-bound}, {bound}")
        tensor.data[:] = np.random.uniform(-bound, bound, tensor.data.shape)

    @staticmethod
    def kaiming_normal_(tensor, mode, nonlinearity):
        print("Initializing weights with kaiming normal")
        gain = init.calculate_gain(nonlinearity)
        bound = gain / np.sqrt(mode)
        print(f"Initializing weights from a normal distribution with mean : 0.0, std : {bound}")
        tensor.data[:] = np.random.normal(loc=0.0, scale=bound, size=tensor.data.shape)

    @staticmethod
    def xavier_uniform_(tensor, nonlinearity):
        print("Initializing weights with xavier uniform")
        gain = init.calculate_gain(nonlinearity)
        fan_in, fan_out = tensor.data.shape
        bound = gain * np.sqrt(6 / (fan_in + fan_out))
        print(f"Initializing weights from a uniform distribution with bound : {-bound}, {bound}")
        tensor.data[:] = np.random.uniform(-bound, bound, tensor.data.shape)

    @staticmethod
    def xavier_normal_(tensor, nonlinearity):
        print("Initializing weights with xavier normal")
        gain = init.calculate_gain(nonlinearity)
        fan_in, fan_out = tensor.data.shape
        bound = gain * np.sqrt(2 / (fan_in + fan_out))
        print(f"Initializing weights from a normal distribution with mean : 0.0, std : {bound}")
        tensor.data[:] = np.random.normal(loc=0.0, scale=bound, size=tensor.data.shape)


class Linear:
    def __init__(self, fan_in, fan_out, weight_init, nonlinearity):
        self.nonlinearity=nonlinearity
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
            init.xavier_uniform_(self.weight, nonlinearity=self.nonlinearity)
        elif self.weight_init == "kaiming":
            init.kaiming_uniform_(self.weight, mode=self.fan_in, nonlinearity=self.nonlinearity)
            
        # Not used
        elif self.weight_init == "He":
            init.He_(self.weight)
        elif self.weight_init == "Xavier_normal":
            init.xavier_normal_(self.weight, nonlinearity=self.nonlinearity)
        elif self.weight_init == "kaiming_normal":
            init.kaiming_normal_(self.weight, mode=self.fan_in, nonlinearity=self.nonlinearity)
        init.zeros_(self.bias)

    def __call__(self, x, training):
        value = x @ self.weight + self.bias
        value.name = "LinearOut"
        return value

    def __repr__(self):
        return f"Linear(fan_in={self.fan_in}, fan_out={self.fan_out})"

class SigmoidBackward(BaseBackward):
    def __call__(self, upstream_grad):
        
        x = self.sources[0]
        value = self.sources[1]
        local_grad = value * (1 - value)
        x.grad += upstream_grad * local_grad

class Sigmoid:
    def __init__(self):
        self.cache = None        

    def __call__(self, x, training):
        sigmoid_value = 1 / (1 + np.exp(-x.data))
        out = Tensor(sigmoid_value, requires_grad=x.requires_grad)
        if x.requires_grad:
            out._grad_fn = SigmoidBackward(x, sigmoid_value)
            out._prev = {x}
            out.name = "SigmoidOut"
        return out

class ReLUBackward(BaseBackward):
    def __call__(self, upstream_grad):
        
        x = self.sources[0]
        local_grad = (x.data > 0).astype(x.data.dtype)
        x.grad += upstream_grad * local_grad


class ReLU:
    def __init__(self):
        self.cache = None
    
    def __call__(self, x, training):
        value = np.maximum(0, x.data)
        out = Tensor(value, requires_grad=x.requires_grad)
        if x.requires_grad:
            out._grad_fn = ReLUBackward(x)
            out._prev = {x}
            out.name = "ReLUOut"
        return out
        


class TanhBackward(BaseBackward):
    def __call__(self, upstream_grad):
        
        x = self.sources[0]
        local_grad = 1 - np.tanh(x.data)**2
        x.grad += upstream_grad * local_grad

class Tanh:
    def __init__(self):
        self.cache = None        

    def __call__(self, x, training):
        # print(type(x))
        tanh_value = np.tanh(x.data)
        out = Tensor(tanh_value, requires_grad=x.requires_grad)
        if x.requires_grad:
            out._grad_fn = TanhBackward(x)
            out._prev = {x}
            out.name = "TanhOut"
        return out

class MSELoss:
    @staticmethod
    def __call__(pred, target):
        one_hot_ = np.zeros_like(pred.data)
        B = one_hot_.shape[0]
        
        one_hot_[np.arange(B), target.data] = 1
        one_hot_ = Tensor(one_hot_)
        
        diff_ = (pred - one_hot_) ** 2
        loss = diff_.sum() / B
        return loss
            
class CrossEntropyLossBackward(BaseBackward):
    def __call__(self, upstream_grad):
        pred = self.sources[0]
        target = self.sources[1]   
        prob = self.sources[2]   
        label_smoothing = self.sources[3]

        n_classes = pred.shape[-1]
        batch_size = pred.shape[0]

        scaled_upstream_grad = upstream_grad / batch_size

        one_hot_target = np.zeros_like(prob)
        one_hot_target[np.arange(batch_size), target.data] = 1.0
        one_hot_target = one_hot_target * (1 - label_smoothing) + label_smoothing / n_classes

        pred_grad = prob - one_hot_target

        pred.grad += pred_grad * scaled_upstream_grad


class CrossEntropyLoss:
    
    def __init__(self, label_smoothing=0.0):
        self.label_smoothing = label_smoothing

    def __call__(self, pred, target):  
        exp_ = np.exp(pred.data - np.max(pred.data, axis=1, keepdims=True))
        prob = exp_ / (exp_.sum(axis=1, keepdims=True) + 1e-8)  

        n_classes = pred.data.shape[-1]
        
        one_hot_target = np.zeros_like(prob)
        one_hot_target[np.arange(target.shape[0]), target.data] = 1.0

        if self.label_smoothing > 0.0:
            one_hot_target = one_hot_target * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        
        loss_value = -np.sum(one_hot_target * np.log(prob + 1e-8), axis=1).mean()
        
        loss = Tensor(loss_value, requires_grad=True, name="loss")
        loss._prev = {pred}
        loss._grad_fn = CrossEntropyLossBackward(pred, target, prob, self.label_smoothing)
        return loss


class DropoutBackward(BaseBackward):
    def __call__(self, upstream_grad):
        x = self.sources[0]
        mask = self.sources[1]
        x.grad += upstream_grad * mask if mask is not None else upstream_grad
    

class Dropout:
    def __init__(self, dropout_p):
        self.dropout_p = dropout_p
    
    def __call__(self, x, train=True):
        x_data = x.data
        if train:
            mask = (np.random.rand(*x_data.shape) > self.dropout_p) / (1 - self.dropout_p)
            value = x_data * mask
            
            out = Tensor(value, name="DropoutOut", requires_grad=True)
            out._prev = {x}
            out._grad_fn = DropoutBackward(x, mask)
            return out
            
        else:
            return x
        

class BatchNormBackward(BaseBackward):
    def __call__(self, upstream_grad):
        
        # print("Batchnorm backward called  !!")
        
        x = self.sources[0]
        x_norm = self.sources[1]
        mean = self.sources[2]
        var = self.sources[3]
        gamma = self.sources[4]
        beta = self.sources[5]
        epsilon = self.sources[6]

        m = upstream_grad.shape[0]

        dL_dbeta = np.sum(upstream_grad, axis=0)
        dL_dgamma = np.sum(upstream_grad * x_norm, axis=0)
        dL_dx_norm = upstream_grad * gamma.data
        dL_dvar = np.sum(dL_dx_norm * (x_norm * -0.5) * (var + epsilon) ** (-1.5), axis=0)
        dL_dmean = np.sum(dL_dx_norm * -1 / np.sqrt(var + epsilon), axis=0) + dL_dvar * np.sum(-2 * x_norm / m, axis=0)
        dL_dx = dL_dx_norm / np.sqrt(var + epsilon) + dL_dvar * 2 * x_norm / m + dL_dmean / m

        gamma.grad += dL_dgamma
        beta.grad += dL_dbeta
        x.grad += dL_dx
        

class BatchNorm:
    def __init__(self, num_features, epsilon=1e-8, momentum=0.1):
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.zeros(num_features)
        
        print("Adding BatchNorm. Yay !!!")

    def __call__(self, x, training=True):
        x_data = x.data
        m = x_data.shape[0]

        if training:
            mean = np.mean(x_data, axis=0)
            var = np.var(x_data, axis=0)

            x_norm = (x_data - mean) / np.sqrt(var + self.epsilon)
            out = self.gamma.data * x_norm + self.beta.data

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            out = Tensor(out)
            out.name = "BatchNormOut"
            out.requires_grad = True
            out._prev = {x, self.gamma, self.beta}
            out._grad_fn = BatchNormBackward(x, x_norm, mean, var, self.gamma, self.beta, self.epsilon)
            return out
            
            
            # mean_ = x.mean(axis=0, keepdims=True)
            # var_ = ((x - mean_)**2).sum(axis=0, keepdims=True) / (x.shape[0] - 1)
            # x_norm = (x - mean_) / (var_ + 1e-10)**0.5

            # out = self.gamma * x_norm + self.beta
            
            # self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean_
            # self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var_
            
            # return out
            
            
        else:
            # print(type(x_data), type(self.running_mean), type(self.running_var), type(self.epsilon))
            x_norm = (x_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma.data * x_norm + self.beta.data
            out = Tensor(out)
            return out


