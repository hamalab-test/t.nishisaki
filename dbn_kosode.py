# -*- coding: utf-8 -*-
import time
import numpy
# import csv
import os

from cnn import CNN
from rbm import RBM

from utility import load_kosode
from utility import makeFolder
# from utility import saveImage
from utility import cnn_saveColorImage
from utility import rbm_saveColorImage
from utility import saveFeatures
from utility import saveW
from utility import loadW
from utility import local_contrast_normalization
# from utility import load_RGB
from utility import load_image
from utility import load_motif
from utility import load_major_motif
from utility import load_kosode_division


if __name__ == '__main__':
    data_path = 'data/kosode_motif6/train/google_dataset_resize'
    # data_path = 'data/kosode_major_motif'
    # data_path = 'data/kosode_division_resize'
    file_num = 300
    isRGB = True
    cnn_pre_train_lr = 0.1
    rbm_pre_train_lr = 0.1
    cnn_pre_train_epoch = 100
    rbm_pre_train_epoch = 100
    motif_num = 10

    # node_shape = ((80,52), (74,46), (35,21))
    # node_shape = ((80,52), (74,46), (34,20))
    node_shape = ((80, 80), (74, 74), (34, 34))
    filter_shift_list = ((1, 1), (2, 2))
    input_shape = [80, 80]
    filter_shape = [7, 7]

    data_set = load_kosode(data_path, motif_num, file_num, isRGB)
    # data_set = load_motif(data_path, 10, isRGB)
    # data_set = load_major_motif(data_path, ('kiku', 'matsu', 'sakura'), file_num, isRGB)
    # data_set = load_kosode_division(data_path, 1200, 1, isRGB)
    # print data_set.shape

    train_set = numpy.transpose(data_set[0], (2, 0, 1))
    # train_set = numpy.transpose(data_set, (2, 0, 1))

    # cnn2_output = load_RGB('data/kosode/cnn2_after_train_norm', file_num * motif_num)

    # cnn1_W = loadW('data/kosode_motif2/train/cnn1_after_train')
    # cnn2_W = loadW('data/kosode_motif2/train/cnn2_after_train')

    makeFolder()

    # 時間計測
    time1 = time.clock()

    print '~~~CNN1~~~'

    cnn1 = CNN(train_set, filter_shape, filter_shift_list[0], input_shape, node_shape[1], cnn_pre_train_lr, cnn_pre_train_epoch, isRGB)

    output_list = cnn1.output()
    cnn_saveColorImage(output_list, node_shape[1], 'cnn1_before_train')
    output_list_norm = local_contrast_normalization(output_list)
    cnn_saveColorImage(output_list_norm, node_shape[1], 'cnn1_before_training_norm')

    cnn1.pre_train()
    # cnn1.setW(cnn1_W)

    output_list = cnn1.output()
    cnn_saveColorImage(output_list, node_shape[1], 'cnn1_after_train')
    output_list_norm = local_contrast_normalization(output_list)
    cnn_saveColorImage(output_list_norm, node_shape[1], 'cnn1_after_train_norm')

    print '~~~CNN2~~~'

    cnn2 = CNN(cnn1.output(), filter_shape, filter_shift_list[1], node_shape[1], node_shape[2], cnn_pre_train_lr, cnn_pre_train_epoch, isRGB)
    output_list = cnn2.output()
    cnn_saveColorImage(output_list, node_shape[2], 'cnn2_before_train')
    output_list_norm = local_contrast_normalization(output_list)
    cnn_saveColorImage(output_list_norm, node_shape[2], 'cnn2_before_train_norm')

    cnn2.pre_train()
    # cnn2.setW(cnn2_W)

    output_list = cnn2.output()
    cnn_saveColorImage(output_list, node_shape[2], 'cnn2_after_train')
    output_list_norm = local_contrast_normalization(output_list)
    cnn_saveColorImage(output_list_norm, node_shape[2], 'cnn2_after_train_norm')

    # rbm_size_list = (680, 340, 170, 85, 42, 21, 10, 3)
    rbm_size_list = (3468, 2000, 1000, 500, 300, 200, 100, 50)

    # print '~~~RBM~~~'

    # cnn2_output = numpy.c_[output_list[0], output_list[1], output_list[2]]
    # cnn2_output = local_contrast_normalization(cnn2_output)

    # def __init__(self, W, input, data_size,input_size, output_size, isDropout):
    # rbm1 = RBM(None, cnn2.output(), file_num * motif_num, rbm_size_list[0], rbm_size_list[1], False)
    # rbm1 = RBM(None, cnn2_output, file_num * motif_num, rbm_size_list[0], rbm_size_list[1], False, rbm_pre_train_lr)
    # for i in xrange(rbm_pre_train_epoch):
    #     print 'rbm1 pre_train:' + str(i)
    #     rbm1.contrast_divergence(i)
    #     entropy = rbm1.get_reconstruction_cross_entropy()
    #     f = open('rbm1_entropy.txt', 'a+')
    #     f.write(str(entropy) + '\n')
    #     f.close()
    #     print entropy
    # reinput = rbm1.reconstruct_from_input(rbm1.input)
    # rbm_saveColorImage(reinput, node_shape[2], 'rbm1_after_train')
    # saveW(rbm1.getW(), 'rbm1_after_train')
    # saveFeatures(rbm1.output(), 'rbm1_features.txt')

    # rbm2 = RBM(None, rbm1.output(), file_num * motif_num, rbm_size_list[1], rbm_size_list[2], False, rbm_pre_train_lr)
    # for i in xrange(rbm_pre_train_epoch):
    #     print 'rbm2 pre_train:' + str(i)
    #     rbm2.contrast_divergence(i)
    # reinput = rbm2.reconstruct_from_input(rbm2.input)
    # reinput = rbm1.reconstruct_from_output(reinput)
    # rbm_saveColorImage(reinput, node_shape[2], 'rbm2_after_train')
    # saveW(rbm2.getW(), 'rbm2_after_train')
    # saveFeatures(rbm2.output(), 'rbm2_features.txt')

    # rbm3 = RBM(None, rbm2.output(), file_num * motif_num, rbm_size_list[2], rbm_size_list[3], False, rbm_pre_train_lr)
    # for i in xrange(rbm_pre_train_epoch):
    #     print 'rbm3 pre_train:' + str(i)
    #     rbm3.contrast_divergence(i)
    # reinput = rbm3.reconstruct_from_input(rbm3.input)
    # reinput = rbm2.reconstruct_from_output(reinput)
    # reinput = rbm1.reconstruct_from_output(reinput)
    # rbm_saveColorImage(reinput, node_shape[2], 'rbm3_after_train')
    # saveW(rbm3.getW(),  'rbm3_after_train')
    # saveFeatures(rbm3.output(), 'rbm3_features.txt')

    # rbm4 = RBM(None, rbm3.output(), file_num * motif_num, rbm_size_list[3], rbm_size_list[4], False, rbm_pre_train_lr)
    # for i in xrange(rbm_pre_train_epoch):
    #     print 'rbm4 pre_train:' + str(i)
    #     rbm4.contrast_divergence(i)
    # reinput = rbm4.reconstruct_from_input(rbm4.input)
    # reinput = rbm3.reconstruct_from_output(reinput)
    # reinput = rbm2.reconstruct_from_output(reinput)
    # reinput = rbm1.reconstruct_from_output(reinput)
    # rbm_saveColorImage(reinput, node_shape[2], 'rbm4_after_train')
    # saveW(rbm4.getW(), 'rbm4_after_train')
    # saveFeatures(rbm4.output(), 'rbm4_features.txt')

    # rbm5 = RBM(None, rbm4.output(), file_num * motif_num, rbm_size_list[4], rbm_size_list[5], False, rbm_pre_train_lr)
    # for i in xrange(rbm_pre_train_epoch):
    #     print 'rbm5 pre_train:' + str(i)
    #     rbm5.contrast_divergence(i)
    # reinput = rbm5.reconstruct_from_input(rbm5.input)
    # reinput = rbm4.reconstruct_from_output(reinput)
    # reinput = rbm3.reconstruct_from_output(reinput)
    # reinput = rbm2.reconstruct_from_output(reinput)
    # reinput = rbm1.reconstruct_from_output(reinput)
    # rbm_saveColorImage(reinput, node_shape[2], 'rbm5_after_train')
    # saveW(rbm5.getW(), 'rbm5_after_train')
    # saveFeatures(rbm5.output(), 'rbm5_features.txt')

    # rbm6 = RBM(None, rbm5.output(), file_num * motif_num, rbm_size_list[5], rbm_size_list[6], False, rbm_pre_train_lr)
    # for i in xrange(rbm_pre_train_epoch):
    #     print 'rbm6 pre_train:' + str(i)
    #     rbm6.contrast_divergence(i)
    # reinput = rbm6.reconstruct_from_input(rbm6.input)
    # reinput = rbm5.reconstruct_from_output(reinput)
    # reinput = rbm4.reconstruct_from_output(reinput)
    # reinput = rbm3.reconstruct_from_output(reinput)
    # reinput = rbm2.reconstruct_from_output(reinput)
    # reinput = rbm1.reconstruct_from_output(reinput)
    # rbm_saveColorImage(reinput, node_shape[2], 'rbm6_after_train')
    # saveW(rbm6.getW(), 'rbm6_after_train')
    # saveFeatures(rbm6.output(), 'rbm6_features.txt')

    # rbm7 = RBM(None, rbm6.output(), file_num * motif_num, rbm_size_list[6], rbm_size_list[7], False, rbm_pre_train_lr)
    # for i in xrange(rbm_pre_train_epoch):
    #     print 'rbm7 pre_train:' + str(i)
    #     rbm7.contrast_divergence(i)
    # reinput = rbm7.reconstruct_from_input(rbm7.input)
    # reinput = rbm6.reconstruct_from_output(reinput)
    # reinput = rbm5.reconstruct_from_output(reinput)
    # reinput = rbm4.reconstruct_from_output(reinput)
    # reinput = rbm3.reconstruct_from_output(reinput)
    # reinput = rbm2.reconstruct_from_output(reinput)
    # reinput = rbm1.reconstruct_from_output(reinput)
    # rbm_saveColorImage(reinput, node_shape[2], 'rbm7_after_train')
    # saveW(rbm7.getW(), 'rbm7_after_train')

    # result_output = rbm7.output()
    # print result_output

    # saveFeatures(result_output, 'rbm7_features.txt')

    # os.chdir('../../')

    # time2 = time.clock()
    # time = time2-time1
    # time = int(time)
    # time = str(time)
    # print 'total_time:' + time
