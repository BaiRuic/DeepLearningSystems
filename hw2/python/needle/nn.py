"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from functools import reduce


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, 
                                                     fan_out=out_features, 
                                                     device=device, 
                                                     dtype=dtype,
                                                     requires_grad=True))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(fan_in=out_features,
                                                        fan_out=1,
                                                        device=device,
                                                        dtype=dtype, 
                                                        requires_grad=True).transpose())
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Y = X @ self.weight
        if self.bias:
            Y = Y + ops.broadcast_to(self.bias, Y.shape)
        return Y
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        new_shape = (X.shape[0], reduce((lambda x, y: x * y), X.shape[1:]))
        return ops.reshape(X, (new_shape))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        y_one_hot = init.one_hot(n=logits.shape[1], i=y)
        loss = (ops.summation(log_sum_exp - ops.summation(y_one_hot * logits, axes=(1,)))) / logits.shape[0]
        return loss
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, device=device, dtype=dtype))
        self.running_mean = Tensor(init.zeros(dim, device=device, dtype=dtype))
        self.running_var = Tensor(init.ones(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # 均值
            mean = x.sum(axes=(0,)) / x.shape[0]
            mean_keepdims = mean.broadcast_to(x.shape)
            # 方差
            var = ops.power_scalar(x - mean_keepdims, 2).sum(axes=(0,)) / x.shape[0]
            # 标准差
            std = ops.power_scalar(var+self.eps, 0.5)
            std_keepdims = std.broadcast_to(x.shape)

            # 移动平均
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean 
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var

            y = (x - mean_keepdims) / std_keepdims
            return self.weight.broadcast_to(x.shape) * y + self.bias.broadcast_to(x.shape) 
        else:
            mean_keepdims = self.running_mean.broadcast_to(x.shape)
            std_keepdims = ops.power_scalar(self.running_var+self.eps, 0.5).broadcast_to(x.shape)
            y = (x - mean_keepdims) / std_keepdims
            weight = self.weight.realize_cached_data().broadcast_to(x.shape)
            bias = self.bias.realize_cached_data().broadcast_to(x.shape)
            return weight * y + bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w = Parameter(init.ones(*(1, dim), device=device, dtype=dtype))
        self.b = Parameter(init.zeros(*(1, dim), device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # 计算均值
        mu = ops.summation(x, axes=(1,)) / self.dim
        mu_keepdims = ops.broadcast_to(ops.reshape(mu, (x.shape[0], 1)), x.shape)
        # 计算标准差
        std_2 = ops.summation(ops.power_scalar(x - mu_keepdims, 2), axes=(1,)) / self.dim
        std = ops.power_scalar(std_2+self.eps, 0.5)
        std_keepdims = ops.broadcast_to(ops.reshape(std, (x.shape[0], 1)), x.shape)
        # 做归一化
        y = (x - mu_keepdims) / std_keepdims
        # 平移缩放
        return ops.broadcast_to(self.b, x.shape) + ops.multiply(ops.broadcast_to(self.w, x.shape), y)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(x.shape[0], x.shape[1], p=self.p, dtype="int8", requires_grad=False)
            return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION



