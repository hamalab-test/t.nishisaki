# -*- coding: utf-8 -*-
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.

"""
import argparse
import numpy as np
from six.moves import range
# from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

from utility import load_kosode
from utility import makeFolder
# from utility import saveImage
from utility import cnn_saveColorImage
from utility import rbm_saveColorImage
from utility import load_feature
from utility import load_features
from utility import saveFeature
from utility import saveFeatures
from utility import local_contrast_normalization
from utility import load_RGB


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# data_path = 'data/kosode_motif2/test/cnn2_after_train_norm'
data_path = 'data/kosode_motif5/train/cnn2_after_train_norm'
# data_path = 'data/kosode_major_motif/cnn2_after_train_norm'
# data_path = 'data/kosode_motif5/test/group1/cnn2_after_train_norm'
file_num = 80
valid_num = int(file_num * 0.1)
isRGB = True
# cnn_pre_train_lr = 0.1
# rbm_pre_train_lr = 0.1
# cnn_pre_train_epoch = 0
# rbm_pre_train_epoch = 1000
motif_num = 10

batchsize = 10
# n_epoch = 0
n_epoch = 100

rbm_size_list = (3468, 2000, 1000, 500, 300, 200, 100, 50)
# rbm_size_list = (3468, 3000, 2500, 2000, 1500, 1000, 750, 500)
# rbm_size_list = (19200, 10000, 5000, 3000, 1000, 500, 250, 100)

data_set = load_RGB(data_path, file_num * motif_num)
# data_set = load_RGB(data_path, 1177)
# data_set = load_RGB(data_path, 10)
# data_set = load_RGB(data_path, 1200 * 10)
# data_set = load_RGB(data_path, 15*3)
# train_set = np.transpose(data_set[0], (2, 0, 1))

N = (file_num - valid_num) * motif_num

# x_train = data_set
# x_test = data_set
# y_train = data_set
# y_test = data_set
# N_test = y_test.size

# data_list = np.split(data_set, [270, 300, ])
split_list_tmp = [[file_num * i - valid_num, file_num * i] for i in xrange(1, motif_num + 1)]
split_list = np.r_[split_list_tmp[0][0], split_list_tmp[0][1]]

for i in xrange(1, len(split_list_tmp)):
    split = split_list_tmp[i]
    split_list = np.r_[split_list, split[0]]
    split_list = np.r_[split_list, split[1]]

split_list = np.array(split_list)

data_list = np.vsplit(data_set, split_list)
data_list = data_list[:20]
# data_list = np.array(data_list)

x_train = data_list[0]
x_test = data_list[1]
for i in xrange(2, len(data_list) - 1, 2):
    # print x_train.shape, x_test.shape, data_list[i].shape, data_list[i + 1].shape
    x_train = np.r_[x_train, data_list[i]]
    x_test = np.r_[x_test, data_list[i + 1]]
# x_train, x_test = np.split(data_set, split_list)

# print x_train.shape
# print x_test.shape

# print x_train, x_test
x_train = np.array(x_train)
x_test = np.array(x_test)

# y_train, y_test = np.split(data_set, [N])
y_train, y_test = x_train, x_test
N_test = y_test.shape[0]

model = FunctionSet(l1_1=F.Linear(rbm_size_list[0], rbm_size_list[1]),
                    l1_2=F.Linear(rbm_size_list[1], rbm_size_list[0]),
                    l2_1=F.Linear(rbm_size_list[1], rbm_size_list[2]),
                    l2_2=F.Linear(rbm_size_list[2], rbm_size_list[1]),
                    l3_1=F.Linear(rbm_size_list[2], rbm_size_list[3]),
                    l3_2=F.Linear(rbm_size_list[3], rbm_size_list[2]),
                    l4_1=F.Linear(rbm_size_list[3], rbm_size_list[4]),
                    l4_2=F.Linear(rbm_size_list[4], rbm_size_list[3]),
                    l5_1=F.Linear(rbm_size_list[4], rbm_size_list[5]),
                    l5_2=F.Linear(rbm_size_list[5], rbm_size_list[4]),
                    l6_1=F.Linear(rbm_size_list[5], rbm_size_list[6]),
                    l6_2=F.Linear(rbm_size_list[6], rbm_size_list[5]),
                    l7_1=F.Linear(rbm_size_list[6], rbm_size_list[7]),
                    l7_2=F.Linear(rbm_size_list[7], rbm_size_list[6]))
if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()

# model.l1_1.W = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm1_w.txt')
# model.l2_1.W = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm2_w.txt')
# model.l3_1.W = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm3_w.txt')
# model.l4_1.W = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm4_w.txt')
# model.l5_1.W = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm5_w.txt')
# model.l6_1.W = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm6_w.txt')
# model.l7_1.W = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm7_w.txt')

# model.l1_1.b = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm1_b.txt')
# model.l2_1.b = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm2_b.txt')
# model.l3_1.b = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm3_b.txt')
# model.l4_1.b = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm4_b.txt')
# model.l5_1.b = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm5_b.txt')
# model.l6_1.b = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm6_b.txt')
# model.l7_1.b = load_features('data/kosode_motif5/train/rbm_dropout0.0_sigmoid', 'rbm7_b.txt')


# Neural net architecture
def forward(x_data, y_data, layer, train=True):
    x, t = Variable(x_data.astype(np.float32)), Variable(y_data.astype(np.float32))
    # if layer is 1:
    #     h = F.dropout(F.relu(model.l1_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.relu(model.l1_2(h)), train=train, ratio=0.5)
    # elif layer is 2:
    #     h = F.dropout(F.relu(model.l2_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.relu(model.l2_2(h)), train=train, ratio=0.5)
    # elif layer is 3:
    #     h = F.dropout(F.relu(model.l3_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.relu(model.l3_2(h)), train=train, ratio=0.5)
    # elif layer is 4:
    #     h = F.dropout(F.relu(model.l4_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.relu(model.l4_2(h)), train=train, ratio=0.5)
    # elif layer is 5:
    #     h = F.dropout(F.relu(model.l5_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.relu(model.l5_2(h)), train=train, ratio=0.5)
    # elif layer is 6:
    #     h = F.dropout(F.relu(model.l6_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.relu(model.l6_2(h)), train=train, ratio=0.5)
    # elif layer is 7:
    #     h = F.dropout(F.relu(model.l7_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.relu(model.l7_2(h)), train=train, ratio=0.5)

    # if layer is 1:
    #     h = F.dropout(F.sigmoid(model.l1_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.sigmoid(model.l1_2(h)), train=train, ratio=0.5)
    # elif layer is 2:
    #     h = F.dropout(F.sigmoid(model.l2_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.sigmoid(model.l2_2(h)), train=train, ratio=0.5)
    # elif layer is 3:
    #     h = F.dropout(F.sigmoid(model.l3_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.sigmoid(model.l3_2(h)), train=train, ratio=0.5)
    # elif layer is 4:
    #     h = F.dropout(F.sigmoid(model.l4_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.sigmoid(model.l4_2(h)), train=train, ratio=0.5)
    # elif layer is 5:
    #     h = F.dropout(F.sigmoid(model.l5_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.sigmoid(model.l5_2(h)), train=train, ratio=0.5)
    # elif layer is 6:
    #     h = F.dropout(F.sigmoid(model.l6_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.sigmoid(model.l6_2(h)), train=train, ratio=0.5)
    # elif layer is 7:
    #     h = F.dropout(F.sigmoid(model.l7_1(x)), train=train, ratio=0.5)
    #     v = F.dropout(F.sigmoid(model.l7_2(h)), train=train, ratio=0.5)

    if layer is 1:
        h = F.sigmoid(model.l1_1(x))
        v = F.sigmoid(model.l1_2(h))
    elif layer is 2:
        h = F.sigmoid(model.l2_1(x))
        v = F.sigmoid(model.l2_2(h))
    elif layer is 3:
        h = F.sigmoid(model.l3_1(x))
        v = F.sigmoid(model.l3_2(h))
    elif layer is 4:
        h = F.sigmoid(model.l4_1(x))
        v = F.sigmoid(model.l4_2(h))
    elif layer is 5:
        h = F.sigmoid(model.l5_1(x))
        v = F.sigmoid(model.l5_2(h))
    elif layer is 6:
        h = F.sigmoid(model.l6_1(x))
        v = F.sigmoid(model.l6_2(h))
    elif layer is 7:
        h = F.sigmoid(model.l7_1(x))
        v = F.sigmoid(model.l7_2(h))

    # return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    return F.mean_squared_error(v, t)

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

x_train_now = x_train
y_train_now = y_train

makeFolder()

f_loss_train = open('rbm_loss_train.txt', 'w')
f_acc_train = open('rbm_acc_train.txt', 'w')
f_loss_test = open('rbm_loss_test.txt', 'w')
f_acc_test = open('rbm_acc_test.txt', 'w')

# Learning loop
for layer in range(1, 2):
    print('layer', layer)

    if layer is 1:
        x_train_now = x_train
        print x_train_now.shape
        y_train_now = y_train
    elif layer is 2:
        x_train_now = F.relu(model.l1_1(Variable(x_train_now)))
        x_train_now = x_train_now.data
        print x_train_now.shape
        y_train_now = x_train_now
    elif layer is 3:
        x_train_now = F.relu(model.l2_1(Variable(x_train_now)))
        x_train_now = x_train_now.data
        print x_train_now.shape
        y_train_now = x_train_now
    elif layer is 4:
        x_train_now = F.relu(model.l3_1(Variable(x_train_now)))
        x_train_now = x_train_now.data
        print x_train_now.shape
        y_train_now = x_train_now
    elif layer is 5:
        x_train_now = F.relu(model.l4_1(Variable(x_train_now)))
        x_train_now = x_train_now.data
        print x_train_now.shape
        y_train_now = x_train_now
    elif layer is 6:
        x_train_now = F.relu(model.l5_1(Variable(x_train_now)))
        x_train_now = x_train_now.data
        print x_train_now.shape
        y_train_now = x_train_now
    elif layer is 7:
        x_train_now = F.relu(model.l6_1(Variable(x_train_now)))
        x_train_now = x_train_now.data
        print x_train_now.shape
        y_train_now = x_train_now

    for epoch in range(1, n_epoch+1):
        print('epoch:layer', epoch, layer)

        print x_train_now.shape

        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        for i in range(0, N, batchsize):
            x_batch = x_train_now[perm[i:i+batchsize]]
            y_batch = y_train_now[perm[i:i+batchsize]]
            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            optimizer.zero_grads()
            loss = forward(x_batch, y_batch, layer)
            loss.backward()
            optimizer.update()

            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
            # sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        print('train mean loss={}, accuracy={}'.format(
            sum_loss / N, sum_accuracy / N))
        print sum_loss, N

        f_loss_train.write(str(sum_loss / N))
        f_loss_train.write(',')
        f_acc_train.write(str(sum_accuracy / N))
        f_acc_train.write(',')

        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        for i in range(0, N_test, batchsize):
            x_batch = x_test[i:i+batchsize]
            y_batch = y_test[i:i+batchsize]
            if args.gpu >= 0:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            # print i, x_batch.shape

            loss = forward(x_batch, y_batch, layer, train=False)

            sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
            # sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))
        print sum_loss, N_test


        f_loss_test.write(str(sum_loss / N_test))
        f_loss_test.write(',')
        f_acc_test.write(str(sum_accuracy / N_test))
        f_acc_test.write(',')

    f_loss_train.write('\n')
    f_acc_train.write('\n')
    f_loss_test.write('\n')
    f_acc_test.write('\n')

saveFeatures(model.l1_1.W, 'rbm1_w.txt')
saveFeatures(model.l2_1.W, 'rbm2_w.txt')
saveFeatures(model.l3_1.W, 'rbm3_w.txt')
saveFeatures(model.l4_1.W, 'rbm4_w.txt')
saveFeatures(model.l5_1.W, 'rbm5_w.txt')
saveFeatures(model.l6_1.W, 'rbm6_w.txt')
saveFeatures(model.l7_1.W, 'rbm7_w.txt')

saveFeature(model.l1_1.b, 'rbm1_b.txt')
saveFeature(model.l2_1.b, 'rbm2_b.txt')
saveFeature(model.l3_1.b, 'rbm3_b.txt')
saveFeature(model.l4_1.b, 'rbm4_b.txt')
saveFeature(model.l5_1.b, 'rbm5_b.txt')
saveFeature(model.l6_1.b, 'rbm6_b.txt')
saveFeature(model.l7_1.b, 'rbm7_b.txt')

x, t = Variable(data_set), Variable(data_set)
h1 = F.sigmoid(model.l1_1(x))
h2 = F.sigmoid(model.l2_1(h1))
h3 = F.sigmoid(model.l3_1(h2))
h4 = F.sigmoid(model.l4_1(h3))
h5 = F.sigmoid(model.l5_1(h4))
h6 = F.sigmoid(model.l6_1(h5))
y = model.l7_1(h6)

saveFeatures(data_set, 'chainer_features0.txt')
saveFeatures(h1.data, 'chainer_features1.txt')
saveFeatures(h2.data, 'chainer_features2.txt')
saveFeatures(h3.data, 'chainer_features3.txt')
saveFeatures(h4.data, 'chainer_features4.txt')
saveFeatures(h5.data, 'chainer_features5.txt')
saveFeatures(h6.data, 'chainer_features6.txt')
saveFeatures(y.data, 'chainer_features7.txt')
