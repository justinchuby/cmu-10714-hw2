"""Optimization module"""
from __future__ import annotations

from typing import Sequence

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
    def __init__(
        self,
        params: Sequence[ndl.nn.Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u: dict[ndl.nn.Parameter, ndl.Tensor] = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            # NOTE: Important weight_decay is added to grad first
            if self.weight_decay > 0:
                grad = param.grad.detach() + param.detach() * self.weight_decay
            else:
                grad = param.grad.detach()
            if param in self.u and self.momentum > 0:
                u_t = self.u[param]
                self.u[param] = self.momentum * u_t + (1 - self.momentum) * grad
            else:
                # TODO: What should the initial update be?
                self.u[param] = (1 - self.momentum) * grad

            updated = param.detach() - self.lr * self.u[param]
            param.data = ndl.Tensor(updated, dtype=param.dtype)
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params: Sequence[ndl.nn.Parameter],
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        bias_correction: bool = True,
        weight_decay: float = 0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.t = 0

        self.m: dict[ndl.nn.Parameter, ndl.Tensor] = {}
        self.v: dict[ndl.nn.Parameter, ndl.Tensor] = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # Important: Pay attention to whether or not you are applying bias correction.
        self.t += 1
        for param in self.params:
            if self.weight_decay > 0:
                assert False
                # grad = param.grad.detach() + param.detach() * self.weight_decay
            else:
                grad = param.grad.detach()

            if param in self.m:
                assert self.t > 1
                m_t = self.m[param]
            else:
                assert self.t == 1
                m_t = 0
            if param in self.v:
                assert self.t > 1
                v_t = self.v[param]
            else:
                assert self.t == 1
                v_t = 0

            m_hat = self.m[param] = self.beta1 * m_t + (1 - self.beta1) * grad
            v_hat = self.v[param] = self.beta2 * v_t + (1 - self.beta2) * grad * grad

            if self.bias_correction:
                m_hat = self.m[param] / (1 - self.beta1**self.t)
                v_hat = self.v[param] / (1 - self.beta2**self.t)

            updated = param.detach() - self.lr * m_hat / (
                v_hat ** 0.5 + self.eps
            )

            param.data = ndl.Tensor(updated, dtype=param.dtype)
        ### END YOUR SOLUTION
