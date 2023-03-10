"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


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
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_ = node.inputs[0]
        return out_grad * self.scalar * power_scalar(input_, self.scalar - 1) 
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return divide(out_grad, rhs), negate(divide(multiply(out_grad, lhs), power_scalar(rhs, 2)))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 / self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        """
        # ????????????????????????, self.axes ????????????????????????, ??? axes = [1, 2]??????????????? 1 ??? 2
        # ????????? transpose ??? np.transpose ?????????????????????, np.transpose???axes?????????????????????:
        axes tuple or list of ints, optional
            If specified, it must be a tuple or list which contains a permutation 
            of [0,1,???,N-1] where N is the number of axes of a. The i???th axis of the
            returned array will correspond to the axis numbered axes[i] of the input.
            If not specified, defaults to range(a.ndim)[::-1], which reverses the 
            order of the axes.
        ????????? ??????axes?????????????????? a.shape ???????????????, ??????,axes???????????????????????????return??????????????????;
        ??????,??????????????????????????????,?????????axes=list(range(len(a.shape)))[::-1] 
        """
        _axes = list(range(len(a.shape)))
        if self.axes:
            _axes[self.axes[0]], _axes[self.axes[1]] = _axes[self.axes[1]], _axes[self.axes[0]]
        else:
            # ?????????????????????????????????
            _axes = _axes[:-2] + _axes[-2:][::-1]
        return array_api.transpose(a, _axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape
        return reshape(out_grad, ori_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape

        # ?????????????????????
        axes_ = []
        # 1. ?????? ndim ??? broadcast, ?????? broadcast ?????????dim ???????????? ????????????????????????????????????????????????????????????0???????????????
        axes_ += list(range(len(self.shape)-len(ori_shape)))        
        # 2, ??????????????? broadcast,??????????????????????????????
        for i in range(-1, -len(ori_shape)-1, -1):
            # ????????????????????? broadcast???
            if ori_shape[i] != self.shape[i]:
                assert ori_shape[i] == 1
                axes_.append(i)

        axes_ = tuple(axes_)
        return reshape(summation(out_grad, axes=axes_), ori_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape
        # temp_shape ???????????? ??????sum??????keepdim??????shape, ??????????????????shape?????????
        # ?????? self.axes ?????????????????????????????????(1,),??????keepdim??????shape??? temp_shape = [1] * len(sum?????????dims)
        # ?????? self.axes ?????????????????????keepdim??????shape: ??????shape??? self.axes??????????????? 1
        temp_shape = list(ori_shape)
        if self.axes:
            for i in self.axes:
                temp_shape[i] = 1
        else:
            for i in range(len(temp_shape)):
                temp_shape[i] = 1
        # ????????? out_grad.size ?????????
        temp_node = reshape(out_grad, temp_shape)
        # 2.booadcast: ????????? out_grad.size ?????????
        return broadcast_to(temp_node, ori_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        l_grad = matmul(out_grad, transpose(rhs))
        r_grad = matmul(transpose(lhs), out_grad)

        # ?????? ndim ??????????????????????????? dim ????????????sum?????? ,?????????????????????????????????
        # Tensor???????????? .ndim ??????????????????????????? len(a.shape) ????????? ndim
        l_grad_ndim, r_grad_ndim = len(l_grad.shape), len(r_grad.shape)
        lhs_ndim, rhs_ndim = len(lhs.shape), len(rhs.shape)
        if l_grad_ndim > lhs_ndim:
            axes = tuple(range(l_grad_ndim - lhs_ndim))
            l_grad = summation(l_grad, axes)
        if r_grad_ndim > rhs_ndim:
            axes = tuple(range(r_grad_ndim - rhs_ndim))
            r_grad = summation(r_grad, axes)

        return l_grad, r_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        hs = node.inputs[0]
        return divide(out_grad, hs)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        hs = node.inputs[0]
        return multiply(out_grad, exp(hs))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.mask = (a > 0).astype("int8")
        return array_api.multiply(a, self.mask)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # ?????????????????? mul_saclar
        return mul_scalar(out_grad, self.mask)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

