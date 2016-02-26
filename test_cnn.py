# -*- coding: utf-8 -*-
import numpy
import chainer
import chainer.functions as F
from libdnn import Classifier
import libdnn.visualizer as V
from sklearn.datasets import fetch_mldata

# ネットワーク作って(MLP)
model = chainer.FunctionSet(fh1=F.Linear(28 ** 2, 100), fh2=F.Linear(100, 10))
# 伝搬規則書いて
def forward(self, x, train):
    x = chainer.Variable(x.data.astype(numpy.float32))
    h = F.tanh(self.model.fh1(x))
    y = F.tanh(self.model.fh2(h))
    # tmp = chainer.Variable(numpy.array(y.data, numpy.int32))
    # print tmp.data.shape
    # print tmp.data.dtype
    return y
    # return chainer.Variable(y.data.astype(numpy.float32))
    # return chainer.Variable(y.data, numpy.float32)
    # return numpy.array((y).data, numpy.float32)

# モデル作って
mlp = Classifier(model, gpu=-1) # CUDAします
mlp.set_forward(forward)

# データ作って
mnist = fetch_mldata('MNIST original')
perm = numpy.random.permutation(len(mnist.data))
mnist.data = mnist.data.astype(numpy.float32)
# print mnist.target.dtype
mnist.target = mnist.target.astype(numpy.int32)

train_data, test_data = mnist.data[perm][:60000], mnist.data[perm][60000:]
train_label, test_label = mnist.target[perm][:60000], mnist.data[perm][60000:]

print train_data.dtype
print train_label.dtype


test_label = test_label.astype(numpy.int32)
print test_data.dtype
print test_label.dtype


# 60000(num. of train_data) iter. を1世代として100世代ミニバッチ学習
for epoch in range(100):
    err, acc = mlp.train(train_data, train_label)
    print("%d: err=%f\t acc=%f" % (epoch + 1, err, acc))

# テスト
err, acc = mlp.test(test_data, test_label)
print("on test: %f\t%f" % (err, acc))

# 可視化なんて器用なことも
imager = V.Visualizer(mlp)
plot_filters('fh1', shape=(28, 28))

# 学習済みネットワークも保存できます
mlp.save_param()