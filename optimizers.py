import numpy as np
import wandb

class BaseOptimizer:
    def __init__(self):
        pass

class SGD:
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        self.params = params.values()
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.b_cache = {id(param): np.zeros_like(param.data) for param in self.params}

    def step(self):
        for param in self.params:
            if param.grad is not None and param.requires_grad:
                if self.weight_decay != 0:
                    param.grad += self.weight_decay * param.data
                
                if self.momentum > 0:
                    b = self.b_cache[id(param)]
                    b[:] = self.momentum * b + (1 - self.dampening) * param.grad
                    if self.nesterov:
                        param.grad += self.momentum * b
                    else:
                        param.grad = b
                
                param.data -= self.lr * param.grad
                

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0)


class RMSprop:
    def __init__(self, params, lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0):
        self.params = params.values()
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.v_cache = {id(param): np.zeros_like(param.data) for param in self.params}
        self.b_cache = {id(param): np.zeros_like(param.data) for param in self.params}

    def step(self):
        for param in self.params:
            if param.grad is not None and param.requires_grad:
                if self.weight_decay != 0:
                    param.grad += self.weight_decay * param.data
                
                self.v_cache[id(param)] = self.alpha * self.v_cache[id(param)] + (1 - self.alpha) * np.power(param.grad, 2)
                v = self.v_cache[id(param)]

                if self.momentum > 0:
                    self.b_cache[id(param)] = self.momentum * self.b_cache[id(param)] + param.grad / (np.sqrt(v) + self.eps)
                    b = self.b_cache[id(param)]
                    param.data -= self.lr * b
                else:
                    param.data -= self.lr * (param.grad / (np.sqrt(v) + self.eps))

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0)

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = params.values()
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.first_moment = {id(param): np.zeros_like(param.data) for param in self.params}
        self.second_moment = {id(param): np.zeros_like(param.data) for param in self.params}
        self.t = 0

    def step(self):
        self.t += 1

        for param in self.params:
            if param.grad is not None and param.requires_grad:
                if self.weight_decay != 0:
                    param.grad += self.weight_decay * param.data

                self.first_moment[id(param)] = self.beta1 * self.first_moment[id(param)] + (1 - self.beta1) * param.grad
                m = self.first_moment[id(param)]

                self.second_moment[id(param)] = self.beta2 * self.second_moment[id(param)] + (1 - self.beta2) * np.power(param.grad, 2)
                v = self.second_moment[id(param)]

                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)

                param.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))
                
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0)

class NAdam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = params.values()
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.first_moment = {id(param): np.zeros_like(param.data) for param in self.params}
        self.second_moment = {id(param): np.zeros_like(param.data) for param in self.params}
        self.t = 0
        
        
        print("NADAM : ", self.lr, self.eps, self.weight_decay, self.beta1, self.beta2)

    def step(self):
        self.t += 1

        for param in self.params:
            if param.grad is not None and param.requires_grad:
                if self.weight_decay != 0:
                    param.grad += self.weight_decay * param.data

                g_t = param.grad

                self.first_moment[id(param)] = self.beta1 * self.first_moment[id(param)] + (1 - self.beta1) * g_t
                m_t = self.first_moment[id(param)]
                m_hat = (self.beta1 * m_t + (1 - self.beta1) * g_t) / (1 - self.beta1 ** self.t)

                self.second_moment[id(param)] = self.beta2 * self.second_moment[id(param)] + (1 - self.beta2) * np.power(g_t, 2)
                v_t = self.second_moment[id(param)]
                v_hat = v_t / (1 - self.beta2 ** self.t)

                param.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0)