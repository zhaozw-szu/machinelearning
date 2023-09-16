import numpy as np


class Optimizer:
    def __init__(self, lr=0.01, cache=None):
        self.lr = lr
        self._cache = cache

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def feed_variables(self, variables):
        self._cache = [
            np.zeros(var.shape) for var in variables
        ]

    def run(self, i, dw):
        pass

    def update(self):
        pass


class MBGD(Optimizer):
    def run(self, i, dw):
        return self.lr * dw


class Momentum(Optimizer):
    """
           初始化结构（Momentum Update版本）
           self._momentum：记录“惯性”的属性
           self._step：每一步迭代后“惯性”的增量
           self._floor、self._ceiling：“惯性”的最小、最大值
           self._cache：对于Mmentum Update而言、该属性记录的就是“行进速度”
           self._is_nesterov：处理Nesterov Momentum Update的属性
    """
    def __init__(self, lr=0.01, cache=None, epoch=100, floor=0.5, ceiling=0.999):
        Optimizer.__init__(self, lr, cache)
        self._momentum = floor
        self._step = (ceiling - floor) / epoch
        self._floor, self._ceiling = floor, ceiling
        self._is_nesterov = False

    # lr*d*w = lr*dw + pv(t)
    # velocity:v(t-1)
    def run(self, i, dw):
        dw *= self.lr
        velocity = self._cache
        velocity[i] *= self._momentum
        velocity[i] += dw
        # 如果不是Nesterov Momentum Update，直接把v(t)当作更新步伐
        if not self._is_nesterov:
            return velocity[i]
        # 否则，调用公式 pv(t)-lr*dw
        return self._momentum * velocity[i] + dw

    def update(self):
        if self._momentum < self._ceiling:
            self._momentum += self._step


class NAG(Momentum):
    def __init__(self, lr=0.01, cache=None, epoch=100, floor=0.5, ceiling=0.999):
        Momentum.__init__(self, lr, cache, epoch, floor, ceiling)
        self._is_nesterov = True


class RMSProp(Optimizer):
    def __init__(self, lr=0.01, cache=None, decay_rate=0.9, eps=1e-8):
        Optimizer.__init__(self, lr, cache)
        self.decay_rate, self.eps = decay_rate, eps

    def run(self, i, dw):
        self._cache[i] = self._cache[i] * self.decay_rate + (1 - self.decay_rate) * dw ** 2
        return self.lr * dw / (np.sqrt(self._cache[i] + self.eps))


class Adam(Optimizer):
    def __init__(self, lr=0.01, cache=None, beta1=0.9, beta2=0.999, eps=1e-8):
        Optimizer.__init__(self, lr, cache)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def feed_variables(self, variables):
        self._cache = [
            [np.zeros(var.shape) for var in variables],
            [np.zeros(var.shape) for var in variables],
        ]

    def run(self, i, dw):
        self._cache[0][i] = self._cache[0][i] * self.beta1 + (1 - self.beta1) * dw
        self._cache[1][i] = self._cache[1][i] * self.beta2 + (1 - self.beta2) * (dw ** 2)
        return self.lr * self._cache[0][i] / (np.sqrt(self._cache[1][i] + self.eps))


# Factory

class OptFactory:
    available_optimizers = {
        "MBGD": MBGD, "Momentum": Momentum, "NAG": NAG, "RMSProp": RMSProp, "Adam": Adam,
    }

    def get_optimizer_by_name(self, name, variables, lr, epoch=100):
        try:
            optimizer = self.available_optimizers[name](lr)
            if variables is not None:
                optimizer.feed_variables(variables)
            if epoch is not None and isinstance(optimizer, Momentum):
                optimizer.epoch = epoch
            return optimizer
        except KeyError:
            raise NotImplementedError("Undefined Optimizer '{}' found".format(name))
