from Util.Util import DataUtil
from NN.Layers import *
from NN.Networks import *


def main():
    nn = NN()
    epoch = 1000

    x, y = DataUtil.gen_spiral(120, 7, 7, 4)

    nn.add(ReLU((x.shape[1], 24)))
    nn.add(ReLU((24,)))
    nn.add(CostLayer((y.shape[1],), "CrossEntropy"))

    # nn.disable_timing()
    nn.fit(x, y, epoch=epoch, train_rate=0.8, metrics=["acc"])
    nn.evaluate(x, y)
    nn.visualize2d(x, y)
    nn.draw_logs()


def main1():
    nn = NaiveNN()
    epoch = 1000

    x, y = DataUtil.gen_spiral(120, 7, 7, 4)

    nn = NaiveNN()
    nn.add(ReLU((x.shape[1], 24)))
    nn.add(CostLayer((y.shape[1],), "CrossEntropy", transform="Sigmoid"))

    # nn.disable_timing()
    nn.fit(x, y, epoch=epoch, train_rate=0.8, metrics=["acc"])
    nn.evaluate(x, y)
    nn.visualize2d(x, y)
    nn.draw_logs()


if __name__ == '__main__':
    main()
