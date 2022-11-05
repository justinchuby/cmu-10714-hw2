"""The module."""
from __future__ import annotations
import math

from typing import Any, Callable

import numpy as np
from needle import ops, init
from needle.autograd import Tensor


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
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


def _child_modules(value: object) -> list[Module]:
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

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list[Module]:
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
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype: str = "float32",
    ):
        """

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to False, the layer will not learn an additive bias.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                in_features,
                out_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        if bias:
            self.bias = Parameter(
                ops.reshape(
                    init.kaiming_uniform(
                        out_features, 1, device=device, dtype=dtype, requires_grad=True
                    ),
                    (1, out_features),
                )
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        r"""Applies a linear transformation to the incoming data: $y = xA^T + b$.

        The input shape is $(N, H_{in})$ where $H_{in}=\text{in_features}$.
        The output shape is $(N, H_{out})$ where $H_{out}=\text{out_features}$.
        """
        ### BEGIN YOUR SOLUTION
        result = X @ self.weight
        if self.bias is None:
            return result
        # return result + ops.broadcast_to(self.bias, (X.shape[0], self.out_features))
        return result + ops.broadcast_to(self.bias, (*X.shape[:-1], self.out_features))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], math.prod(X.shape[1:])))
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
        intermediates = x
        for module in self.modules:
            intermediates = module(intermediates)
        return intermediates
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        classes = logits.shape[1]
        onehot = init.one_hot(classes, y, device=logits.device, dtype=logits.dtype)
        lse = ops.logsumexp(logits, axes=(1,))
        # NOTE: This should be sum, not reduce_mean!
        z_y = ops.summation(logits * onehot, axes=(1,))
        return ops.reduce_mean(lse - z_y)
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, device=None, dtype: str = "float32"
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # NOTE: Do I need gradient?
        self.weight = Parameter(
            init.constant(1, dim, c=1, device=device, dtype=dtype, requires_grad=True)
        )
        self.bias = Parameter(
            init.constant(1, dim, c=0, device=device, dtype=dtype, requires_grad=True)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x is 2D tensor
        expectation = ops.reduce_mean(x, axes=(1,))
        variance = ops.reduce_mean(x**2, axes=(1,)) - expectation**2
        N = x.shape[0]
        weight = ops.broadcast_to(self.weight, (N, self.dim))
        bias = ops.broadcast_to(self.bias, (N, self.dim))
        expectation = ops.broadcast_to(ops.reshape(expectation, (N, 1)), (N, self.dim))
        variance = ops.broadcast_to(ops.reshape(variance, (N, 1)), (N, self.dim))
        return weight * (x - expectation) / ((variance + self.eps) ** (1 / 2)) + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        # Create a zeroed mask
        # NOTE: Remember to divide by (1 - p) to scale the mask
        keep_prob = 1 - self.p
        mask = (
            init.randb(*x.shape, p=keep_prob, device=x.device, dtype=x.dtype)
            / keep_prob
        )
        return x * mask


class Residual(Module):
    """Applies a skip connection given module and input Tensor x, returning module(x) + x."""

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
