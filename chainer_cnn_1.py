# -*- coding: utf-8 -*-
from libdnn import Classifier
import chainer
import chainer.functions as F
import numpy as np
from utility import load_RGBimage2cnn
from utility import makeFolder
from utility import saveFeatures
from utility import load_major_motif2cnn
from utility import load_RGBimage
from utility import load_kosode_division_all
from utility import load_kosode_division_group
from utility import load_RGBimage2cnn_listdir
from utility import load_RGBimage2cnn_5pickup


isTrain = True

# data_path = 'data/kosode_all_motif/representations_resize'
# data_path = 'data/kosode_all_motif/dataset_resize'
data_path = 'data/kosode_all_motif/1motif_5pickup/dataset_resize'

data_path2 = 'data/kosode_all_motif/1motif_1pickup/test_representation_images_resize'

# kosode_path = 'data/kosode_division_resize'
kosode_path = 'data/kosode_division_resize/group1/13/dzc_output_files/15'
# data_path = 'data/kosode_motif6/train/google_dataset_resize'
# data_path = 'data/kosode_motif2/test/kosode_smallfield_resize/15/15'
# data_path = 'data/kosode_major_motif_resize'
major_motif_path = 'data/kosode_major_motif_resize'

file_num = 6
file_num2 = 300
kosode_file_num = 1141
# valid_num = int(file_num * 0.1)
isRGB = True
motif_num = 50
batchsize = 25
n_epoch = 50
cnn_param_name = 'cnn.param_12.npy'

major_motif_num = 15

if isTrain:
    # data_set = load_RGBimage2cnn(data_path, motif_num, file_num, isRGB)
    data_set = load_RGBimage2cnn_5pickup(data_path, motif_num, file_num, isRGB)
    # data_set = load_RGBimage(data_path, motif_num, isRGB)

    # train_set = data_set[0]
    # label_set = data_set[1]

    x_train = np.array(data_set[0], np.float32)
    y_train = np.array(data_set[1], np.int32)

    # data_set2 = load_RGBimage2cnn(data_path2, motif_num, file_num2, isRGB)
    # x_test = np.array(data_set2[0], np.float32)
    # y_test = np.array(data_set2[1], np.int32)

    data_set2 = load_RGBimage2cnn_listdir(data_path2, motif_num, file_num2, isRGB)
    x_test = np.array(data_set2[0], np.float32)
    y_test = np.array(data_set2[1], np.int32)

    print x_train.shape, y_train.shape
    print x_test.shape, y_test.shape

    print y_train

    # N = (file_num - valid_num) * motif_num

    # split_list_tmp = [[file_num * i - valid_num, file_num * i] for i in xrange(1, motif_num + 1)]
    # split_list = np.r_[split_list_tmp[0][0], split_list_tmp[0][1]]

    # for i in xrange(1, len(split_list_tmp)):
    #     split = split_list_tmp[i]
    #     split_list = np.r_[split_list, split[0]]
    #     split_list = np.r_[split_list, split[1]]

    # split_list = np.array(split_list)

    # data_list = np.vsplit(train_set, split_list)
    # data_list = np.array(data_list)

    # print data_list.shape

    # x_train = data_list[0]
    # x_test = data_list[1]
    # for i in xrange(2, len(data_list) - 1, 2):
    #     print x_train.shape, x_test.shape, data_list[i].shape, data_list[i + 1].shape
    #     x_train = np.r_[x_train, data_list[i]]
    #     x_test = np.r_[x_test, data_list[i + 1]]
    # # x_train, x_test = np.split(data_set, split_list)
    # print x_train.shape, x_test.shape

    # train_set = np.array(train_set, np.float32)
    # x_train = np.array(x_train, np.float32)
    # x_test = np.array(x_test, np.float32)

    # data_list = np.split(label_set, split_list)
    # data_list = data_list[:motif_num*2]
    # data_list = np.array(data_list)

    # y_train = data_list[0]
    # y_test = data_list[1]
    # for i in xrange(2, len(data_list) - 1, 2):
    #     # print y_train.shape, y_test.shape, data_list[i].shape, data_list[i + 1].shape
    #     y_train = np.r_[y_train, data_list[i]]
    #     y_test = np.r_[y_test, data_list[i + 1]]

    # y_train = np.array(y_train, np.int32)
    # y_test = np.array(y_test, np.int32)
    # N_test = y_test.shape[0]

else:
    # train_set = load_RGBimage(kosode_path, kosode_file_num, isRGB)
    # train_set = load_major_motif2cnn(major_motif_path, ('kiku', 'matsu', 'sakura'), major_motif_num, isRGB)
    train_set = load_RGBimage2cnn_listdir(data_path2, motif_num, file_num2, isRGB)
    x_train = np.array(train_set[0], np.float32)
    print x_train.shape
    n_epoch = 0


# data_set = load_RGBimage(data_path, file_num, isRGB)
# data_set = load_major_motif(data_path, ('kiku', 'matsu', 'sakura'), file_num, isRGB)

# train_set = np.transpose(data_set[0], (0, 2, 1))
# train_set = numpy.transpose(data_set, (2, 0, 1))
# train_set = data_set
# train_set = np.array(train_set, np.float32)

# model = chainer.FunctionSet(
#     conv1=F.Convolution2D(1, 15, 5),
#     bn1=F.BatchNormalization(15),
#     conv2=F.Convolution2D(15, 30, 3, pad=1),
#     bn2=F.BatchNormalization(30),
#     conv3=F.Convolution2D(30, 64, 3, pad=1),
#     fl4=F.Linear(2304, 576),
#     fl5=F.Linear(576, 10)
# )

model = chainer.FunctionSet(
    # conv1=F.Convolution2D(3, 3, 5),
    # bn1=F.BatchNormalization(3),
    # conv2=F.Convolution2D(3, 3, 5),
    # bn2=F.BatchNormalization(3),
    # fl3=F.Linear(867, 800),
    # fl4=F.Linear(800, 10)

    conv1=F.Convolution2D(3, 15, 5),
    bn1=F.BatchNormalization(15),
    conv2=F.Convolution2D(15, 30, 5),
    bn2=F.BatchNormalization(30),
    fl3=F.Linear(8670, 8000),
    fl4=F.Linear(8000, 50 * 5)
)


def forward(self, x, train):
    # h = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
    h = F.max_pooling_2d(F.relu(model.bn1((model.conv1(x)))), 2)
    # h = F.max_pooling_2d(F.relu(model.conv2(h)), 2)
    h = F.max_pooling_2d(F.relu(model.bn2((model.conv2(h)))), 2)
    h = F.dropout(F.relu(model.fl3(h)), train=True)
    y = model.fl4(h)

    return y


def output(self, x, layer):
    # h = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
    h = F.max_pooling_2d(F.relu(model.bn1((model.conv1(x)))), 2)
    if layer == 1:
        return h

    # h = F.max_pooling_2d(F.relu(model.conv2(h)), 2)
    h = F.max_pooling_2d(F.relu(model.bn2((model.conv2(h)))), 2)
    if layer == 2:
        return h

    h = F.dropout(F.relu(model.fl3(h)), train=False)
    if layer == 3:
        return h
    y = model.fl4(h)
    if layer == 4:
        return y

    return None


cnn = Classifier(model, gpu=-1)
cnn.set_forward(forward)
cnn.set_output(output)

if not isTrain:
    cnn.load_param(cnn_param_name)

makeFolder()

# print x_train.shape, x_test.shape
# print y_train.shape, y_test.shape

max_tacc = 0.

for epoch in range(n_epoch):
    print('epoch : %d' % (epoch + 1))
    err, acc = cnn.train(x_train, y_train, batchsize=batchsize)
    print('    err : %f' % (err))
    print('    acc : %f' % (acc))
    # perm = np.random.permutation(len(test_data))
    # terr, tacc = cnn.test(test_data[perm][:100], test_label[perm][:100])
    terr, tacc = cnn.test(x_test, y_test)
    print('    test err : %f' % (terr))
    print('    test acc : %f' % (tacc))
    # if tacc > max_tacc:
    #     max_tacc = tacc
    #     cnn.save_param('cnn.param_' + str(epoch) + '.npy')

    with open('cnn.log', mode='a') as f:
        f.write("%d %f %f %f %f\n" % (epoch + 1, err, acc, terr, tacc))

    if isTrain:
        output_train = output(cnn, chainer.Variable(x_train), 4)
        output_test = output(cnn, chainer.Variable(x_test), 4)
        saveFeatures(output_train.data, 'chainer_features_train_' + str(epoch) + '.txt')
        saveFeatures(output_test.data, 'chainer_features_test_' + str(epoch) + '.txt')

        output_train = np.array(output_train.data)
        output_test = np.array(output_test.data)

        # 訓練データの出力結果がどのラベルか出力
        output_list = []
        for i in xrange(output_train.shape[0]):
            result = np.argmax(output_train[i])
            output_list.append(result)
        print output_list
        print y_train

        output_list = np.array(output_list)

        cnt = 0.
        for y, label_train in zip(output_list / 5, y_train / 5):
            # print y, label_train
            if y == label_train:
                cnt += 1
        train_acc = cnt / len(output_list)
        print train_acc

        # テストデータの出力結果がどのラベルか出力
        output_list = []
        for i in xrange(output_test.shape[0]):
            result = np.argmax(output_test[i])
            output_list.append(result)
        print output_list
        print y_test

        output_list = np.array(output_list)

        cnt = 0.
        for y, label_test in zip(output_list / 5, y_test):
            # print y, label_test
            if y == label_test:
                cnt += 1
        test_acc = cnt / len(output_list)
        print test_acc

        with open('acc.log', mode='a') as f:
            f.write("%d %f %f\n" % (epoch + 1, train_acc, test_acc))

    else:
        y_train = output(cnn, chainer.Variable(x_train), 4)
        saveFeatures(y_train.data, 'chainer_features_train.txt')

# y = output(cnn, chainer.Variable(train_set), 4)
# saveFeatures(y.data, 'chainer_features_y.txt')

cnn.save_param('cnn.param.npy')
