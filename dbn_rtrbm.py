# -*- coding: utf-8 -*-
import time
# import os
# import sys

from cnn import CNN
from utility import load_image, makeFolder, saveImage, saveFeatures, saveW, load_result_image
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


if __name__ == '__main__':
    data_path = 'data/kouryu_room/image_gray'

    # file_num = 526
    file_num = 100

    isRGB = False
    # pre_train_lr = 0.1
    pre_train_epoch = 1000
    # node_shape = ((64,48), (58,42), (26,18))
    # node_shape = ((80,52), (74,46), (34,20))
    # filter_shift_list = ((1,1), (2,2))
    input_shape = [64, 48]
    # filter_shape = [7,7]

    data_list = load_image(data_path, file_num, isRGB)

    # 時間計測
    time1 = time.clock()

    # data_list = load_data('feature_14.txt')
    rbm_size_list = (468, 234, 117, 58, 30, 14, 7, 7)

    result_path = 'data/kouryu_room/cnn2_after_training'
    result_data = load_result_image(result_path, file_num, isRGB)

    print result_data.shape

    makeFolder()

    gradient_check = []

    ####### def __init__(self, W, U, input_v, input_r, data_size, input_size, output_size):
    rtrbm1 = RTRBM(None, None, result_data, None, file_num, rbm_size_list[0], rbm_size_list[1])
    for i in xrange(pre_train_epoch):
        print 'rtrbm1 pre_train:' + str(i)
        rtrbm1.contrast_divergence(i)
        # rtrbm1.check_params()
    saveW(rtrbm1.getW(), 'rtrbm1_W')
    saveW(rtrbm1.getU(), 'rtrbm1_U')

    rtrbm2 = RTRBM(None, None, rtrbm1.output_hr()[0], None, file_num, rbm_size_list[1], rbm_size_list[2])
    for i in xrange(pre_train_epoch):
        print 'rtrbm2 pre_train:' + str(i)
        rtrbm2.contrast_divergence(i)
    saveW(rtrbm2.getW(), 'rtrbm2_W')
    saveW(rtrbm2.getU(), 'rtrbm2_U')

    rtrbm3 = RTRBM(None, None, rtrbm2.output_hr()[0], None, file_num, rbm_size_list[2], rbm_size_list[3])
    for i in xrange(pre_train_epoch):
        print 'rtrbm3 pre_train:' + str(i)
        rtrbm3.contrast_divergence(i)
    saveW(rtrbm3.getW(), 'rtrbm3_W')
    saveW(rtrbm3.getU(), 'rtrbm3_U')

    rtrbm4 = RTRBM(None, None, rtrbm3.output_hr()[0], None, file_num, rbm_size_list[3], rbm_size_list[4])
    for i in xrange(pre_train_epoch):
        print 'rtrbm4 pre_train:' + str(i)
        rtrbm4.contrast_divergence(i)
    saveW(rtrbm4.getW(), 'rtrbm4_W')
    saveW(rtrbm4.getU(), 'rtrbm4_U')

    output_h, output_r = rtrbm4.output_hr()

    saveFeatures(output_h, 'feature_h.txt')
    saveFeatures(output_r, 'feature_r.txt')

    time2 = time.clock()
    time = time2-time1
    time = int(time)
    time = str(time)
    print 'total_time:' + time
