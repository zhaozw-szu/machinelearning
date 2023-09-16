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
    epoch = 1000
    np.random.seed(5000)

    # x, y = DataUtil.gen_spiral(120, 7, 7, 4)
    x, y = DataUtil.gen_random(120,2)

    nn = NaiveNN()
    nn.add(ReLU((x.shape[1], 24)))
    nn.add(ReLU((24,)))
    nn.add(CostLayer((y.shape[1],), "MSE", transform="Softmax"))

    nn.fit(x, y, epoch=epoch)
    nn.evaluate(x, y)
    nn.visualize2d(x, y)


if __name__ == '__main__':
    main1()
