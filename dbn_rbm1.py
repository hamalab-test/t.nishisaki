# -*- coding: utf-8 -*-
import time
import numpy
import os
import datetime

from cnn import CNN
from utility import load_image, makeFolder, saveImage, saveFeatures, saveW, load_result_image, loadW
from rbm import RBM

if __name__ == '__main__':
    data_path = 'data/4position_rumba/image7000/image_gray'
    file_num = 7000
    isRGB = False
    pre_train_lr = 0.1
    pre_train_epoch = 2000
    # node_shape = ((80,52), (74,46), (35,21))
    node_shape = ((80, 52), (74, 46), (34, 20))
    filter_shift_list = ((1, 1), (2, 2))
    input_shape = [80, 52]
    filter_shape = [7, 7]

    # data_list = load_image(data_path, file_num, isRGB)

    # makeFolder()

    # 時間計測
    # time1 = time.clock()

    # cnn1 = CNN(data_list, filter_shape, filter_shift_list[0], input_shape, node_shape[1], pre_train_lr, pre_train_epoch)

    # output_list = cnn1.output()
    # cnn1.pre_train()

    # result_path = 'data/2014-10-30/cnn1_after_training'
    # result_data = load_result_image(result_path, file_num, isRGB)

    # cnn2 = CNN(result_data, filter_shape, filter_shift_list[1], node_shape[1], node_shape[2], pre_train_lr, pre_train_epoch)

    rbm_size_list = (680, 340, 170, 85, 42, 21, 10, 5)

    result_path = 'data/2014-10-30/cnn2_after_training'
    result_data = load_result_image(result_path, file_num, isRGB)

    result_path = 'data/4position_rumba/image7000/rbm1_train3434'
    result_W = loadW(result_path)

    rbm1 = RBM(result_W, result_data, file_num, rbm_size_list[0], rbm_size_list[1], False)

    ######## RBM2 training ##############

    result_path = 'data/4position_rumba/image7000/rbm2_train2000'
    result_W = loadW(result_path)

    rbm2 = RBM(result_W, rbm1.output(), file_num, rbm_size_list[1], rbm_size_list[2], False)

    ######## RBM3 training ##############
    result_path = 'data/4position_rumba/image7000/rbm3_train2000'
    result_W = loadW(result_path)

    rbm3 = RBM(result_W, rbm2.output(), file_num, rbm_size_list[2], rbm_size_list[3], False)

    ######## RBM4 training ##############
    result_path = 'data/4position_rumba/image7000/rbm4_train2000'
    result_W = loadW(result_path)

    rbm4 = RBM(result_W, rbm3.output(), file_num, rbm_size_list[3], rbm_size_list[4], False)

    ######## RBM5 training ##############
    result_path = 'data/4position_rumba/image7000/rbm5_train2000'
    result_W = loadW(result_path)

    rbm5 = RBM(result_W, rbm4.output(), file_num, rbm_size_list[4], rbm_size_list[5], False)

    ######## RBM6 training ##############
    result_path = 'data/4position_rumba/image7000/rbm6_train2000'
    result_W = loadW(result_path)

    rbm6 = RBM(result_W, rbm5.output(), file_num, rbm_size_list[5], rbm_size_list[6], False)

    ######## RBM7 training ##############
    result_path = 'data/4position_rumba/image7000/rbm7_train2000_5bit'
    result_W = loadW(result_path)

    rbm7 = RBM(result_W, rbm6.output(), file_num, rbm_size_list[6], rbm_size_list[7], False)

    ######## Training #############

    # os.chdir('result')
    # reinput = rbm7.reconstruct_from_input(rbm7.input)
    # reinput = rbm6.reconstruct_from_output(reinput)
    # reinput = rbm5.reconstruct_from_output(reinput)
    # reinput = rbm4.reconstruct_from_output(reinput)
    # reinput = rbm3.reconstruct_from_output(reinput)
    # reinput = rbm2.reconstruct_from_output(reinput)
    # reinput = rbm1.reconstruct_from_output(reinput)
    # saveImage(reinput, node_shape[2], 'rbm7_before_train')
    # os.chdir('../')

    # for i in xrange(pre_train_epoch+1):
    # 	if i % 100 == 0:
    # 		os.chdir('result')
    # 		reinput = rbm7.reconstruct_from_input(rbm7.input)
    # 		reinput = rbm6.reconstruct_from_output(reinput)
    # 		reinput = rbm5.reconstruct_from_output(reinput)
    # 		reinput = rbm4.reconstruct_from_output(reinput)
    # 		reinput = rbm3.reconstruct_from_output(reinput)
    # 		reinput = rbm2.reconstruct_from_output(reinput)
    # 		reinput = rbm1.reconstruct_from_output(reinput)
    # 		saveImage(reinput, node_shape[2], 'rbm7_train' + str(i))
    # 		saveW(rbm7.getW(), 'rbm7_train' + str(i))
    # 		os.chdir('../')
    # 	print 'rbm7 pre_train:' + str(i)
    # 	rbm7.contrast_divergence(i)

    # result_output = rbm7.output()
    # print result_output
    # saveFeatures(result_output)
    # os.chdir('../../')

    ############## 1,0,0  check
    # reinput = rbm7.reconstruct_from_output(( (0,0,0),(1,0,0),(0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1) ))
    reinput = rbm7.reconstruct_from_output(((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)))
    reinput = rbm6.reconstruct_from_output(reinput)
    reinput = rbm5.reconstruct_from_output(reinput)
    reinput = rbm4.reconstruct_from_output(reinput)
    reinput = rbm3.reconstruct_from_output(reinput)
    reinput = rbm2.reconstruct_from_output(reinput)
    reinput = rbm1.reconstruct_from_output(reinput)
    saveImage(reinput, node_shape[2], 'feature_1_0_0_0_0')
    os.chdir('../')

    # time2 = time.clock()
    # time = time2-time1
    # time = int(time)
    # time = str(time)
    # print 'total_time:' + time
