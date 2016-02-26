# -*- coding: utf-8 -*-

import numpy
from logistic_regression import LogisticRegression
from utility import load_features
from utility import saveW
from utility import saveFeatures
from utility import makeFolder


if __name__ == '__main__':
    data_path = 'data/kosode_motif5/judge_dropout0.0_sigmoid'
    file_name = 'motif_chainer_features1.txt'
    feature_list = load_features(data_path, file_name)
    # fine_tune_epoch = 10000
    fine_tune_epoch = 0
    fine_tune_lr = 0.1

    print feature_list.shape

    file_num = 80
    motif_num = 10
    data_size = feature_list.shape[0]

    input_size = feature_list.shape[1]
    output_size = motif_num

    W = load_features(data_path, 'W_1.txt')

    makeFolder()

    # label = numpy.zeros((data_size, output_size))

    # for i in xrange(data_size):
    #     index = i / file_num
    #     label[i][index] = 1

    # LR = LogisticRegression(feature_list, label, input_size, output_size, data_size, fine_tune_lr)
    LR = LogisticRegression(feature_list, None, input_size, output_size, data_size, fine_tune_lr)
    LR.W = W
    for i in xrange(fine_tune_epoch):
        print 'epoch: ' + str(i)
        LR.fine_tune()
    # output_list = LR.predict(feature_list)
    # output_list = LR.predict_direct(feature_list)
    output_list = LR.predict_sigmoid(feature_list)

    saveW(LR.getW(), 'LR_after_train')
    saveFeatures(output_list, 'LR_judge.txt')
    # saveFeatures(label, 'label.txt')
