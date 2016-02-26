# -*- coding: utf-8 -*-
import time
import numpy
# import os
# import sys

from cnn import CNN
from utility import load_image, makeFolder, saveImage, saveFeatures, saveW, load_result_image, loadW
from rtrbm import RTRBM
from rbm import RBM


def load_data(file_name):
    f = open(file_name, 'r')

    data_list = []
    for line in f:
        data_each = line.split(',')
        data_each = map(float, data_each)
        data_list.append(data_each)
    return data_list


def load_result_features(result_path, feature_num):
    f = open(result_path, 'r')
    # for line in f.readlines():
    #     for feature in line.split(','):
    #         print feature
    # f.close()

    # feature_list = f.readlines().split(',')
    feature_list = [x.split(',') for x in f.readlines()]
    feature_list = numpy.array(feature_list).astype('float')

    for i in xrange(25):
        feature_list[i] = feature_list[4 * i]

    for i in xrange(12):
        feature_list[25-i-1] = feature_list[i]

    return feature_list[:25]
    # return numpy.r_[feature_list[10:21], list(reversed(feature_list[10:20]))]

if __name__ == '__main__':
    data_path = 'data/kouryu_room/image_gray'

    # file_num = 526
    file_num = 25

    isRGB = False
    # pre_train_lr = 0.1
    pre_train_epoch = 10000
    # node_shape = ((64,48), (58,42), (26,18))
    # node_shape = ((80,52), (74,46), (34,20))
    # filter_shift_list = ((1,1), (2,2))
    input_shape = [64, 48]
    # filter_shape = [7,7]

    data_list = load_image(data_path, file_num, isRGB)

    # 時間計測
    time1 = time.clock()

    # data_list = load_data('feature_14.txt')
    rbm_size_list = (30, 30)

    result_path = 'data/kouryu_room/ImageSpace/feature_RBM30.txt'
    result_data = load_result_features(result_path, 30)

    # W = loadW('result/2015-05-15_02-52-44/rtrbm_timeseries1_W')
    # U = loadW('result/2015-05-15_02-52-44/rtrbm_timeseries1_W')

    print result_data.shape

    makeFolder()

    gradient_check = []

    ####### def __init__(self, W, U, input_v, input_r, data_size, input_size, output_size):
    rtrbm1 = RTRBM(None, None, result_data, None, file_num, rbm_size_list[0], rbm_size_list[1])
    for i in xrange(pre_train_epoch):
        print 'rtrbm1 pre_train:' + str(i)
        rtrbm1.contrast_divergence(i)
    saveW(rtrbm1.getW(), 'rtrbm_timeseries1_W')
    saveW(rtrbm1.getU(), 'rtrbm_timeseries1_U')

    # check
    # rtrbm1 = RTRBM(W, U, result_data, None, file_num, rbm_size_list[0], rbm_size_list[1])

    output_h, output_r = rtrbm1.output_hr()

    saveFeatures(output_h, 'feature_timeseries_h.txt')
    saveFeatures(output_r, 'feature_timeseries_r.txt')

    time2 = time.clock()
    time = time2-time1
    time = int(time)
    time = str(time)
    print 'total_time:' + time
