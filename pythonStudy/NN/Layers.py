import numpy as np


class Layer:
    """
        初始化结构
        self.shape：记录着上个Layer和该Layer所含神经元的个数，具体而言：
            self.shape[0] = 上个Layer所含神经元的个数
            self.shape[1] = 该Layer所含神经元的个数
    """
    def __init__(self, shape):
        self.shape = shape

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    def _activate(self, x, predict=None):
        pass

    # 将激活函数的导函数的定义留给子类定义
    # 需要特别指出的是、这里的参数y其实是
    # 这样设置参数y的原因会马上在后文叙述、这里暂时按下不表
    def derivative(self, y, delta=None):
        return self._derivative(y, delta)

    # 前向传导算法的封装
    def activate(self, x, w, bias):
        return self._activate(x.dot(w) + bias)

    # 反向传播算法的封装，主要是利用上面定义的导函数derivative来完成局部梯度的计算
    # 其中：、、prev_delta；
    def bp(self, y, w, prev_delta):
        return prev_delta.dot(w.T) * self.derivative(y)

    def _derivative(self, y, delta=None):
        pass


class CostLayer(Layer):
    """
        初始化结构
        self._available_cost_functions：记录所有损失函数的字典
        self._available_transform_functions：记录所有特殊变换函数的字典
        self._cost_function、self._cost_function_name：记录损失函数及其名字的两个属性
        self._transform_function 、self._transform：记录特殊变换函数及其名字的两个属性
    """
    def __init__(self, shape, cost_function="MSE", transform=None):
        super(CostLayer, self).__init__(shape)
        self._available_cost_functions = {
            "MSE": CostLayer._mse,
            "SVM": CostLayer._svm,
            "CrossEntropy": CostLayer._cross_entropy
        }
        self._available_transform_functions = {
            "Softmax": CostLayer._softmax,
            "Sigmoid": CostLayer._sigmoid
        }
        self._cost_function_name = cost_function
        self._cost_function = self._available_cost_functions[cost_function]
        if transform is None and cost_function == "CrossEntropy":
            self._transform = "Softmax"
            self._transform_function = CostLayer._softmax
        else:
            self._transform = transform
            self._transform_function = self._available_transform_functions.get(
                transform, None)

    def __str__(self):
        return self._cost_function_name

    def _activate(self, x, predict=None):
        # 如果不使用特殊的变换函数的话、直接返回输入值即可
        if self._transform_function is None:
            return x
        # 否则、调用相应的变换函数以获得结果
        return self._transform_function(x)

    # 由于CostLayer有自己特殊的BP算法，所以这个方法不会被调用、自然也无需定义
    def _derivative(self, y, delta=None):
        pass

    # 计算整合梯度的方法，返回的是负梯度
    def bp_first(self, y, y_pred):
        # 如果是第六节第二、第三种情况则使用-delta（m）=-（v（m）-y）=y-v（m）的计算方法
        if self._cost_function_name == "CrossEntropy" and (
                self._transform == "Softmax" or self._transform == "Sigmoid"):
            return y - y_pred
        # 否则使用普适性公式进行计算
        # -delta（m）=-（损失对v（m）求偏导）或-delta（m）=-（损失对v（m）求偏导）*（激活函数的导数）
        dy = -self._cost_function(y, y_pred)
        if self._transform_function is None:
            return dy
        return dy * self._transform_function(y_pred, diff=True)

    @property
    def calculate(self):
        return lambda y, y_pred: self._cost_function(y, y_pred, False)

    @property
    def cost_function(self):
        return self._cost_function_name

    @cost_function.setter
    def cost_function(self, value):
        self._cost_function_name = value
        self._cost_function = self._available_cost_functions[value]

    def set_cost_function_derivative(self, func, name=None):
        name = "Custom Cost Function" if name is None else name
        self._cost_function_name = name
        self._cost_function = func

    @staticmethod
    def safe_exp(x):
        return np.exp(x - np.max(x, axis=1, keepdims=True))

    @staticmethod
    def _softmax(y, diff=False):
        if diff:
            return y * (1 - y)
        exp_y = CostLayer.safe_exp(y)
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(y, diff=False):
        if diff:
            return y * (1 - y)
        return 1 / (1 + np.exp(-y))

    # Cost Functions

    @staticmethod
    def _mse(y, y_pred, diff=True):
        if diff:
            return -y + y_pred
        return 0.5 * np.average((y - y_pred) ** 2)

    @staticmethod
    def _svm(y, y_pred, diff=True):
        n, y = y_pred.shape[0], np.argmax(y, axis=1)
        correct_class_scores = y_pred[np.arange(n), y]
        margins = np.maximum(0, y_pred - correct_class_scores[..., None] + 1.0)
        margins[np.arange(n), y] = 0
        loss = np.sum(margins) / n
        num_pos = np.sum(margins > 0, axis=1)
        if not diff:
            return loss
        dx = np.zeros_like(y_pred)
        dx[margins > 0] = 1
        dx[np.arange(n), y] -= num_pos
        dx /= n
        return dx

    @staticmethod
    def _cross_entropy(y, y_pred, diff=True):
        if diff:
            return -y / y_pred + (1 - y) / (1 - y_pred)
        # noinspection PyTypeChecker
        return np.average(-y * np.log(np.maximum(y_pred, 1e-12)) - (1 - y) * np.log(np.maximum(1 - y_pred, 1e-12)))


# Activation Layers

class Tanh(Layer):
    def _activate(self, x, predict=None):
        return np.tanh(x)

    def _derivative(self, y, delta=None):
        return 1 - y ** 2


class Sigmoid(Layer):
    def _activate(self, x, predict=None):
        return 1 / (1 + np.exp(-x))

    def _derivative(self, y, delta=None):
        return y * (1 - y)


class ELU(Layer):
    def _activate(self, x, predict=None):
        rs, mask = x.copy(), x < 0
        rs[mask] = np.exp(rs[mask]) - 1
        return rs

    def _derivative(self, y, delta=None):
        _rs, _indices = np.ones(y.shape), y < 0
        _rs[_indices] = y[_indices] + 1
        return _rs


class ReLU(Layer):
    def _activate(self, x, predict=None):
        return np.maximum(0, x)

    def _derivative(self, y, delta=None):
        return y > 0


class Softplus(Layer):
    def _activate(self, x, predict=None):
        return np.log(1 + np.exp(x))

    def _derivative(self, y, delta=None):
        return 1 - 1 / np.exp(y)


class Identical(Layer):
    def _activate(self, x, predict=None):
        return x

    def _derivative(self, y, delta=None):
        return 1