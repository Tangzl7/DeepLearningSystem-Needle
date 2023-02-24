"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


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

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, dtype="float32"))
        self.bias = None
        if bias:
            self.bias = init.kaiming_uniform(out_features, 1, dtype="float32")
            self.bias = Parameter(ops.reshape(self.bias, (1, out_features)))

    def forward(self, X: Tensor) -> Tensor:
        out = ops.matmul(X, self.weight)
        if self.bias is not None:
            out = out + ops.broadcast_to(self.bias, out.shape)
        return out



class Flatten(Module):
    def forward(self, X):
        dim = 1
        for i in X.shape:
            dim = dim * i
        dim = int(dim / X.shape[0])
        return ops.reshape(X, (X.shape[0], dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        batch, feat = logits.shape
        y_one_hot = init.one_hot(feat, y)
        y_hat = ops.summation(logits * y_one_hot, (-1, ))
        loss = ops.summation(ops.logsumexp(logits, (-1, )) - y_hat) / batch
        return loss



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)


    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            shape = list(x.shape)
            shape[0] = 1
            mean = (ops.summation(x, axes=0) / x.shape[0])
            mean_ = ops.broadcast_to(ops.reshape(mean, shape), x.shape)
            var = (ops.summation((x - mean_)**2, axes=0) / x.shape[0])
            var_ = ops.broadcast_to(ops.reshape(var, shape), x.shape)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            norm = (x - mean_) / ((var_ + self.eps) ** 0.5)
        else:
            norm = (x - self.running_mean.broadcast_to(x.shape)) / ((self.running_var + self.eps) ** 0.5).broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))


    def forward(self, x: Tensor) -> Tensor:
        mean = (ops.summation(x, axes=1) / x.shape[-1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var = (ops.summation((x - mean)**2, axes=1) / x.shape[-1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        norm = (x - mean) / ((var + self.eps) ** 0.5)
        return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training == False:
            return x
        mask = init.randb(*x.shape, p=1-self.p)
        return x * mask / (1 - self.p)


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x


