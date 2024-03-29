"""Operator implementations."""

from numbers import Number
from tkinter.messagebox import NO
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


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
                in_grad.append(init.zeros_like(value))
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
        return a + numpy.float32(self.scalar)

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
        return (a * self.scalar).astype(a.dtype)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        base = node.inputs[0]
        return out_grad * self.scalar * (base ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        dividend, divisor = node.inputs
        return out_grad / divisor, -1 * out_grad * dividend / (divisor ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        new_axes = None

    def compute(self, a):
        new_axes = list(range(0, len(a.shape)))
        if self.axes != None:
            new_axes[self.axes[0]], new_axes[self.axes[1]] = \
                new_axes[self.axes[1]], new_axes[self.axes[0]]
        else:
            new_axes[new_axes[-1]], new_axes[new_axes[-2]] = \
                new_axes[new_axes[-2]], new_axes[new_axes[-1]]
        return a.transpose(new_axes)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        return out_grad.reshape(node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        input = node.inputs[0]
        x_dim = len(input.shape)
        out_grad_dim = len(out_grad.shape)
        expand_dim_num = out_grad_dim - x_dim
        agg_axes = []
        agg_axes.extend([i for i in range(expand_dim_num)])
        agg_axes.extend([i + expand_dim_num for i in range(x_dim) if input.shape[i] == 1])
        agg_axes = tuple(agg_axes)
        return out_grad.sum(agg_axes).reshape(input.shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axis: Optional[tuple] = None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def compute(self, a):
        return a.sum(self.axis, keepdims=self.keepdims)

    def gradient(self, out_grad, node):
        if self.axis != None:
            if self.keepdims == False:
                out_grad.cached_data = array_api.expand_dims(out_grad.cached_data, self.axis) 
            out_grad.cached_data = array_api.broadcast_to(out_grad.cached_data, node.inputs[0].shape)
            return out_grad
        else:
            out_grad.cached_data = array_api.ones_like(node.inputs[0].cached_data) * out_grad.cached_data
            return out_grad


def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        a, b = node.inputs
        if len(a.shape) > len(b.shape):
            b.cached_data = array_api.expand_dims(b.cached_data, (0))
            grad_1, grad_2 = matmul(out_grad, b.transpose()), matmul(a.transpose(), out_grad)
            grad_2 = summation(grad_2, axes=(0, 1))
            return grad_1, grad_2

        if len(b.shape) > len(a.shape):
            a.cached_data = array_api.expand_dims(a.cached_data, (0))
            grad_1, grad_2 = matmul(out_grad, b.transpose()), matmul(a.transpose(), out_grad)
            grad_1 = summation(grad_1, axes=(0, 1))
            return grad_1, grad_2
        return matmul(out_grad, b.transpose()), matmul(a.transpose(), out_grad)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return -1 * out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * array_api.exp(node.inputs[0].cached_data)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.where(a > 0, a, 0)

    def gradient(self, out_grad, node):
        return out_grad * array_api.where(node.inputs[0].cached_data > 0, 1, 0)


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_num = array_api.max(Z, axis=self.axes)
        if self.axes is None:
            return array_api.log(array_api.sum(array_api.exp(Z - max_num), axis=self.axes)) + max_num
        shape = list(Z.shape)
        for i in self.axes:
            shape[i] = 1
        max_num_new = array_api.reshape(max_num, shape)
        max_num_new = array_api.broadcast_to(max_num_new, Z.shape)
        return array_api.log(array_api.sum(array_api.exp(Z - max_num_new), axis=self.axes)) + max_num

    def gradient(self, out_grad, node):
        input = node.inputs[0].cached_data
        max_num = array_api.max(input, axis=self.axes)
        if self.axes is None:
            exp_g = array_api.exp(input - max_num)
            sum_g = array_api.sum(exp_g, axis=self.axes)
            out_grad.cached_data = out_grad.cached_data / sum_g * exp_g
            return out_grad
        shape = list(input.shape)
        for i in self.axes:
            shape[i] = 1
        max_num = array_api.reshape(max_num, shape)
        max_num = array_api.broadcast_to(max_num, input.shape)
        out_grad.cached_data = array_api.reshape(out_grad.cached_data, shape)
        out_grad.cached_data = array_api.broadcast_to(out_grad.cached_data, input.shape)
        exp_g = array_api.exp(input - max_num)
        sum_g = array_api.sum(exp_g, axis=self.axes)
        sum_g = array_api.broadcast_to(array_api.reshape(sum_g, shape), shape=input.shape)
        out_grad.cached_data = out_grad.cached_data / sum_g * exp_g
        return out_grad


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



