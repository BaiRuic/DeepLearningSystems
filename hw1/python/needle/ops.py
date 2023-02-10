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
        # 交换给定的两个轴, self.axes 里面最多两个数值, 入 axes = [1, 2]，则交换第 1 和 2
        # 此处的 transpose 和 np.transpose 定义不完全相同, np.transpose对axes参数的描述如下:
        axes tuple or list of ints, optional
            If specified, it must be a tuple or list which contains a permutation 
            of [0,1,…,N-1] where N is the number of axes of a. The i’th axis of the
            returned array will correspond to the axis numbered axes[i] of the input.
            If not specified, defaults to range(a.ndim)[::-1], which reverses the 
            order of the axes.
        即首先 参数axes的长度需要与 a.shape 的长度一致, 其次,axes中的某个数值表示了return值该位置的轴;
        最后,默认是完全交换所有轴,即默认axes=list(range(len(a.shape)))[::-1] 
        """
        _axes = list(range(len(a.shape)))
        if self.axes:
            _axes[self.axes[0]], _axes[self.axes[1]] = _axes[self.axes[1]], _axes[self.axes[0]]
        else:
            # 默认交换的是最后两个轴
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

        # 存储待压缩的轴
        axes_ = []
        # 1. 处理 ndim 的 broadcast, 这里 broadcast 增加的dim 必定是在 较低的轴，所以计算出来增加了几个，然后从0到几就是了
        axes_ += list(range(len(self.shape)-len(ori_shape)))        
        # 2, 处理形状的 broadcast,倒着往回检查不一样的
        for i in range(-1, -len(ori_shape)-1, -1):
            # 不相同，必定是 broadcast了
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
        # temp_shape 用来保存 做了sum但是keepdim住的shape, 因为要利用该shape做广播
        # 如果 self.axes 不存在，则最后的维度为(1,),此时keepdim住的shape： temp_shape = [1] * len(sum之前的dims)
        # 如果 self.axes 存在，那么此时keepdim住的shape: 初始shape中 self.axes对应轴变为 1
        temp_shape = list(ori_shape)
        if self.axes:
            for i in self.axes:
                temp_shape[i] = 1
        else:
            for i in range(len(temp_shape)):
                temp_shape[i] = 1
        # 不涉及 out_grad.size 的改变
        temp_node = reshape(out_grad, temp_shape)
        # 2.booadcast: 涉及到 out_grad.size 的改变
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

        # 如果 ndim 不一样，把多出来的 dim 对应的轴sum起来 ,多出来的轴一定是前面的
        # Tensor没有实现 .ndim 函数，因此只能通过 len(a.shape) 来得到 ndim
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
        # 这里只能使用 mul_saclar
        return mul_scalar(out_grad, self.mask)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

