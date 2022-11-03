"""Operatpr table."""
# Global operator table.
from numbers import Number
import itertools

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api
from beartype import beartype

from .autograd import NDArray, Op, Tensor, TensorOp, TensorTuple, TensorTupleOp, Value


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a**self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return self.scalar * power_scalar(out_grad, (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return (out_grad / rhs, out_grad * negate(lhs) * power_scalar(rhs, -2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    @beartype
    def __init__(self, axes: tuple[int, int] = (-1, -2)):
        self.axes = axes

    def compute(self, a):
        return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return transpose(out_grad, self.axes)


@beartype
def transpose(a, axes: tuple[int, int] = (-1, -2)):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Reshape into the input shape
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Sum over the broadcasted axes and divide by the product of the new dimensions
        input_shape = node.inputs[0].shape
        reduce_axes = []
        for input_dim, broadcasted_dim, axis in itertools.zip_longest(
            reversed(input_shape),
            reversed(self.shape),
            reversed(range(len(self.shape))),
            fillvalue=-1,
        ):
            if input_dim == broadcasted_dim:
                continue
            assert input_dim == 1 or input_dim == -1
            reduce_axes.append(axis)
        # Need to reshape to maintain dim=1 axes
        return reshape(summation(out_grad, tuple(reduce_axes)), input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        if self.axes == tuple():
            return out_grad
        input_shape = node.inputs[0].shape
        if self.axes is None:
            reduced_axes = set(range(len(input_shape)))
        else:
            reduced_axes = set(self.axes)
        restored_shape = []
        for i, size in enumerate(input_shape):
            # Support negative indexing
            if i in reduced_axes or i - len(input_shape) in reduced_axes:
                restored_shape.append(1)
            else:
                restored_shape.append(size)
        return broadcast_to(reshape(out_grad, tuple(restored_shape)), input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor):
        # NOTE: Since by gradient we mean partial derivative...
        lhs, rhs = node.inputs
        # Determine if anything is broadcasted. If so sum over the broadcasted
        # axes to get the gradient
        lhs_shape = lhs.shape
        rhs_shape = rhs.shape

        if len(lhs_shape) == len(rhs_shape):
            # No broadcasting
            return (out_grad @ transpose(rhs), transpose(lhs) @ out_grad)
        if len(lhs_shape) < len(rhs_shape):
            # Left hand side is broadcasted
            reduce_axes = tuple(range(len(rhs_shape) - len(lhs_shape)))
            return (
                summation(out_grad @ transpose(rhs), axes=reduce_axes),
                transpose(lhs) @ out_grad,
            )

        # Right hand side is broadcasted
        reduce_axes = tuple(range(len(lhs_shape) - len(rhs_shape)))
        return (
            out_grad @ transpose(rhs),
            summation(transpose(lhs) @ out_grad, axes=reduce_axes),
        )


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input_data = node.inputs[0].realize_cached_data()
        relu_data = array_api.maximum(input_data, 0)
        multiplier = relu_data
        multiplier[multiplier > 0] = 1

        # TODO: Still, how to I keep the graph updated?
        return out_grad * Tensor(multiplier)


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes)
        exponentiated = array_api.exp(Z - max_z)
        return array_api.log(array_api.sum(exponentiated, axis=self.axes)) + max_z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
