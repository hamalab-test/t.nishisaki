# -*- coding: utf-8 -*-
import time
# import numpy
# import csv
import os

from cnn import CNN
from utility import load_image, makeFolder, saveImage, saveFeatures, saveW
from rbm import RBM

if __name__ == '__main__':
    data_path = 'data/4position_rumba/image_gray'
    file_num = 3
    isRGB = False
    pre_train_lr = 0.1
    pre_train_epoch = 10
    # node_shape = ((80,52), (74,46), (35,21))
    node_shape = ((80, 52), (74, 46), (34, 20))
    filter_shift_list = ((1, 1), (2, 2))
    input_shape = [80, 52]
    filter_shape = [7, 7]

    data_list = load_image(data_path, file_num, isRGB)

    makeFolder()

    # 時間計測
    time1 = time.clock()

    cnn1 = CNN(data_list, filter_shape, filter_shift_list[0], input_shape, node_shape[1], pre_train_lr, pre_train_epoch)

    output_list = cnn1.output()
    saveImage(output_list, node_shape[1], 'cnn1_before_training')

    cnn1.pre_train()
    output_list = cnn1.output()
    saveImage(output_list, node_shape[1], 'cnn1_after_training')

    # for i in xrange(pre_train_epoch):
    # 	cnn1.pre_train()
    # 	output_list = cnn1.output()
    # 	saveImage(output_list, (74,46))

    cnn2 = CNN(cnn1.output(), filter_shape, filter_shift_list[1], node_shape[1], node_shape[2], pre_train_lr, pre_train_epoch)
    output_list = cnn2.output()
    saveImage(output_list, node_shape[2], 'cnn2_before_train')

    cnn2.pre_train()
    output_list = cnn2.output()
    saveImage(output_list, node_shape[2], 'cnn2_after_train')

    rbm_size_list = (680, 340, 170, 85, 42, 21, 10, 3)

    # def __init__(self, W, input, data_size,input_size, output_size, isDropout):
    rbm1 = RBM(None, cnn2.output(), file_num, rbm_size_list[0], rbm_size_list[1], False)
    for i in xrange(pre_train_epoch):
        print 'rbm1 pre_train:' + str(i)
        rbm1.contrast_divergence()
    reinput = rbm1.reconstruct_from_input(rbm1.input)
    saveImage(reinput, node_shape[2], 'rbm1_after_train')
    saveW(rbm1.getW(), 'rbm1_after_train')

    rbm2 = RBM(None, rbm1.output(), file_num, rbm_size_list[1], rbm_size_list[2], False)
    for i in xrange(pre_train_epoch):
        print 'rbm2 pre_train:' + str(i)
        rbm2.contrast_divergence()
    reinput = rbm2.reconstruct_from_input(rbm2.input)
    reinput = rbm1.reconstruct_from_output(reinput)
    saveImage(reinput, node_shape[2], 'rbm2_after_train')
    saveW(rbm2.getW(), 'rbm2_after_train')

    rbm3 = RBM(None, rbm2.output(), file_num, rbm_size_list[2], rbm_size_list[3], False)
    for i in xrange(pre_train_epoch):
        print 'rbm3 pre_train:' + str(i)
        rbm3.contrast_divergence()
    reinput = rbm3.reconstruct_from_input(rbm3.input)
    reinput = rbm2.reconstruct_from_output(reinput)
    reinput = rbm1.reconstruct_from_output(reinput)
    saveImage(reinput, node_shape[2], 'rbm3_after_train')
    saveW(rbm3.getW(),  'rbm3_after_train')

    rbm4 = RBM(None, rbm3.output(), file_num, rbm_size_list[3], rbm_size_list[4], False)
    for i in xrange(pre_train_epoch):
        print 'rbm4 pre_train:' + str(i)
        rbm4.contrast_divergence()
    reinput = rbm4.reconstruct_from_input(rbm4.input)
    reinput = rbm3.reconstruct_from_output(reinput)
    reinput = rbm2.reconstruct_from_output(reinput)
    reinput = rbm1.reconstruct_from_output(reinput)
    saveImage(reinput, node_shape[2], 'rbm4_after_train')
    saveW(rbm4.getW(), 'rbm4_after_train')

    rbm5 = RBM(None, rbm4.output(), file_num, rbm_size_list[4], rbm_size_list[5], False)
    for i in xrange(pre_train_epoch):
        print 'rbm5 pre_train:' + str(i)
        rbm5.contrast_divergence()
    reinput = rbm5.reconstruct_from_input(rbm5.input)
    reinput = rbm4.reconstruct_from_output(reinput)
    reinput = rbm3.reconstruct_from_output(reinput)
    reinput = rbm2.reconstruct_from_output(reinput)
    reinput = rbm1.reconstruct_from_output(reinput)
    saveImage(reinput, node_shape[2], 'rbm5_after_train')
    saveW(rbm5.getW(), 'rbm5_after_train')

    rbm6 = RBM(None, rbm5.output(), file_num, rbm_size_list[5], rbm_size_list[6], False)
    for i in xrange(pre_train_epoch):
        print 'rbm6 pre_train:' + str(i)
        rbm6.contrast_divergence()
    reinput = rbm6.reconstruct_from_input(rbm6.input)
    reinput = rbm5.reconstruct_from_output(reinput)
    reinput = rbm4.reconstruct_from_output(reinput)
    reinput = rbm3.reconstruct_from_output(reinput)
    reinput = rbm2.reconstruct_from_output(reinput)
    reinput = rbm1.reconstruct_from_output(reinput)
    saveImage(reinput, node_shape[2], 'rbm6_after_train')
    saveW(rbm6.getW(), 'rbm6_after_train')

    rbm7 = RBM(None, rbm6.output(), file_num, rbm_size_list[6], rbm_size_list[7], False)
    for i in xrange(pre_train_epoch):
        print 'rbm7 pre_train:' + str(i)
        rbm7.contrast_divergence()
    reinput = rbm7.reconstruct_from_input(rbm7.input)
    reinput = rbm6.reconstruct_from_output(reinput)
    reinput = rbm5.reconstruct_from_output(reinput)
    reinput = rbm4.reconstruct_from_output(reinput)
    reinput = rbm3.reconstruct_from_output(reinput)
    reinput = rbm2.reconstruct_from_output(reinput)
    reinput = rbm1.reconstruct_from_output(reinput)
    saveImage(reinput, node_shape[2], 'rbm7_after_train')
    saveW(rbm7.getW(), 'rbm7_after_train')

    result_output = rbm7.output()
    print result_output

    saveFeatures(result_output)

    os.chdir('../../')

    # f = open('data.csv', 'ab') #ファイルが無ければ作る、の'a'を指定します
    # csvWriter = csv.writer(f)
    # csvWriter.writerow(result_output)          #1行書き込み
    # f.close()

    time2 = time.clock()
    time = time2-time1
    time = int(time)
    time = str(time)
    print 'total_time:' + time
