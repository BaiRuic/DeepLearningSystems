"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data

            # Perform weight decay
            grad += self.weight_decay * p.data
            
            # Initialize momentum
            if p not in self.u:
                self.u[p] = ndl.init.zeros(*(p.shape), requires_grad=False)

            # # Update momentum
            self.u[p] = self.momentum * self.u[p] + (1. - self.momentum) * grad
            
            # # Update parameter
            p.data -= self.lr * self.u[p]
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.data
            
            # Perform weight decat
            grad += self.weight_decay * param.data
            
            # Initialize moving averages
            if param not in self.m:
                self.m[param] = ndl.init.zeros(*(param.shape), requires_grad=False)
                self.v[param] = ndl.init.zeros(*(param.shape), requires_grad=False)
            
            # Update moving averages
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * ndl.power_scalar(grad, 2)

            # Bias correction
            m_hat = self.m[param] / (1 - pow(self.beta1, self.t))
            v_hat = self.v[param] / (1 - pow(self.beta2, self.t))
            
            # Update parameter
            param.data -= self.lr * m_hat / (ndl.power_scalar(v_hat, 0.5) + self.eps)
        
        ### END YOUR SOLUTION
