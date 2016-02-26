# -*- coding: utf-8 -*-
import time
import numpy
import os
import datetime


from cnn import CNN
from utility import load_image, makeFolder, saveImage, saveFeatures, saveW, load_result_image, loadW
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

    # makeFolder()

    # 時間計測
    # time1 = time.clock()

    cnn1 = CNN(data_list, filter_shape, filter_shift_list[0], input_shape, node_shape[1], pre_train_lr, pre_train_epoch)

    # output_list = cnn1.output()
    # cnn1.pre_train()

    result_path = 'data/2014-10-30/cnn1_after_training'
    result_data = load_result_image(result_path, file_num, isRGB)

    cnn2 = CNN(result_data, filter_shape, filter_shift_list[1], node_shape[1], node_shape[2], pre_train_lr, pre_train_epoch)

    rbm_size_list = (680, 340, 170, 85, 42, 21, 10, 3)

    result_path = 'data/2014-10-30/rbm1_after_train'
    result_data = load_result_image(result_path, file_num, isRGB)
    result_W = loadW(result_path)

    rbm1 = RBM(result_W, result_data, file_num, rbm_size_list[0], rbm_size_list[1], False)

    result_path = 'data/2014-10-30/rbm2_after_train'
    result_data = load_result_image(result_path, file_num, isRGB)
    result_W = loadW(result_path)

    rbm2 = RBM(result_W, result_data, file_num, rbm_size_list[1], rbm_size_list[2], False)

    result_path = 'data/2014-10-30/rbm3_after_train'
    result_data = load_result_image(result_path, file_num, isRGB)
    result_W = loadW(result_path)

    rbm3 = RBM(result_W, result_data, file_num, rbm_size_list[2], rbm_size_list[3], False)

    result_path = 'data/2014-10-30/rbm4_after_train'
    result_data = load_result_image(result_path, file_num, isRGB)
    result_W = loadW(result_path)

    rbm4 = RBM(result_W, result_data, file_num, rbm_size_list[3], rbm_size_list[4], False)

    result_path = 'data/2014-10-30/rbm5_after_train'
    result_data = load_result_image(result_path, file_num, isRGB)
    result_W = loadW(result_path)

    rbm5 = RBM(result_W, result_data, file_num, rbm_size_list[4], rbm_size_list[5], False)

    result_path = 'data/2014-10-30/rbm6_after_train'
    result_data = load_result_image(result_path, file_num, isRGB)
    result_W = loadW(result_path)

    rbm6 = RBM(result_W, result_data, file_num, rbm_size_list[5], rbm_size_list[6], False)

    result_path = 'data/2014-10-30/rbm7_after_train'
    result_data = load_result_image(result_path, file_num, isRGB)
    result_W = loadW(result_path)

    rbm7 = RBM(result_W, result_data, file_num, rbm_size_list[6], rbm_size_list[7], False)

    # Fine-tuning!!!
    finetuning_epoch = 30
    finetuning_lr = 0.1
    for epoch in xrange(finetuning_epoch):
        if epoch % 10 == 0:
            print 'finetuning epoch: ' + str(epoch)

            # ここで途中段階の特徴，重みを出力
            os.chdir('result')
            # d = datetime.datetime.today()
            # dir_name = d.strftime('%Y-%m-%d_%H-%M-%S')
            # print dir_name

            # if not os.path.exists(dir_name):
            # 	os.mkdir(dir_name)
            # os.chdir(dir_name)

            dir_name = 'finetune_epoch' + str(epoch)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            os.chdir(dir_name)

            # Feature
            output = rbm1.output_from_input(rbm1.input)
            output = rbm2.output_from_input(output)
            output = rbm3.output_from_input(output)
            output = rbm4.output_from_input(output)
            output = rbm5.output_from_input(output)
            output = rbm6.output_from_input(output)
            output = rbm7.output_from_input(output)
            saveFeatures(output)

            # RBM1 Save
            reinput = rbm1.reconstruct_from_input(rbm1.input)
            saveImage(reinput, node_shape[2], 'rbm1_after_train')
            saveW(rbm1.getW(), 'rbm1_after_train')

            # RBM2 Save
            output = rbm1.output_from_input(rbm1.input)
            output = rbm2.output_from_input(output)
            reinput = rbm2.reconstruct_from_output(output)
            reinput = rbm1.reconstruct_from_output(reinput)
            saveImage(reinput, node_shape[2], 'rbm2_after_train')
            saveW(rbm2.getW(), 'rbm2_after_train')

            # RBM3 Save
            output = rbm1.output_from_input(rbm1.input)
            output = rbm2.output_from_input(output)
            output = rbm3.output_from_input(output)
            reinput = rbm3.reconstruct_from_output(output)
            reinput = rbm2.reconstruct_from_output(reinput)
            reinput = rbm1.reconstruct_from_output(reinput)
            saveImage(reinput, node_shape[2], 'rbm3_after_train')
            saveW(rbm3.getW(), 'rbm3_after_train')

            # RBM4 Save
            output = rbm1.output_from_input(rbm1.input)
            output = rbm2.output_from_input(output)
            output = rbm3.output_from_input(output)
            output = rbm4.output_from_input(output)
            reinput = rbm4.reconstruct_from_output(output)
            reinput = rbm3.reconstruct_from_output(reinput)
            reinput = rbm2.reconstruct_from_output(reinput)
            reinput = rbm1.reconstruct_from_output(reinput)
            saveImage(reinput, node_shape[2], 'rbm4_after_train')
            saveW(rbm4.getW(), 'rbm4_after_train')

            # RBM5 Save
            output = rbm1.output_from_input(rbm1.input)
            output = rbm2.output_from_input(output)
            output = rbm3.output_from_input(output)
            output = rbm4.output_from_input(output)
            output = rbm5.output_from_input(output)
            reinput = rbm5.reconstruct_from_output(output)
            reinput = rbm4.reconstruct_from_output(reinput)
            reinput = rbm3.reconstruct_from_output(reinput)
            reinput = rbm2.reconstruct_from_output(reinput)
            reinput = rbm1.reconstruct_from_output(reinput)
            saveImage(reinput, node_shape[2], 'rbm5_after_train')
            saveW(rbm5.getW(), 'rbm5_after_train')

            # RBM6 Save
            output = rbm1.output_from_input(rbm1.input)
            output = rbm2.output_from_input(output)
            output = rbm3.output_from_input(output)
            output = rbm4.output_from_input(output)
            output = rbm5.output_from_input(output)
            output = rbm6.output_from_input(output)
            reinput = rbm6.reconstruct_from_output(output)
            reinput = rbm5.reconstruct_from_output(reinput)
            reinput = rbm4.reconstruct_from_output(reinput)
            reinput = rbm3.reconstruct_from_output(reinput)
            reinput = rbm2.reconstruct_from_output(reinput)
            reinput = rbm1.reconstruct_from_output(reinput)
            saveImage(reinput, node_shape[2], 'rbm6_after_train')
            saveW(rbm6.getW(), 'rbm6_after_train')

            # RBM7 Save
            output = rbm1.output_from_input(rbm1.input)
            output = rbm2.output_from_input(output)
            output = rbm3.output_from_input(output)
            output = rbm4.output_from_input(output)
            output = rbm5.output_from_input(output)
            output = rbm6.output_from_input(output)
            output = rbm7.output_from_input(output)
            reinput = rbm7.reconstruct_from_output(output)
            reinput = rbm6.reconstruct_from_output(reinput)
            reinput = rbm5.reconstruct_from_output(reinput)
            reinput = rbm4.reconstruct_from_output(reinput)
            reinput = rbm3.reconstruct_from_output(reinput)
            reinput = rbm2.reconstruct_from_output(reinput)
            reinput = rbm1.reconstruct_from_output(reinput)
            saveImage(reinput, node_shape[2], 'rbm7_after_train')
            saveW(rbm7.getW(), 'rbm7_after_train')

            os.chdir('../../')

        output = rbm1.output()
        output = rbm2.output_from_input(output)
        output = rbm3.output_from_input(output)
        output = rbm4.output_from_input(output)
        output = rbm5.output_from_input(output)
        output = rbm6.output_from_input(output)
        output = rbm7.output_from_input(output)

        input = rbm7.reconstruct_from_output(output)
        input = rbm6.reconstruct_from_output(input)
        input = rbm5.reconstruct_from_output(input)
        input = rbm4.reconstruct_from_output(input)
        input = rbm3.reconstruct_from_output(input)
        input = rbm2.reconstruct_from_output(input)
        input = rbm1.reconstruct_from_output(input)

        for i in xrange(input.shape[0]):
            first_input = rbm1.input[i]
            last_input = input[i]

            delta = [x-y for(x, y) in zip(first_input, last_input)]
            delta = numpy.array(delta)

            # RBM1 finetune
            W = rbm1.W
            for j in xrange(W.shape[0]):
                W[j] = W[j] + finetuning_lr * delta
            rbm1.W = W
            delta = rbm1.output_from_input(delta)

            # RBM2 finetune
            W = rbm2.W
            for j in xrange(W.shape[0]):
                W[j] = W[j] + finetuning_lr * delta
            rbm2.W = W
            delta = rbm2.output_from_input(delta)

            # RBM3 finetune
            W = rbm3.W
            for j in xrange(W.shape[0]):
                W[j] = W[j] + finetuning_lr * delta
            rbm3.W = W
            delta = rbm3.output_from_input(delta)

            # RBM4 finetune
            W = rbm4.W
            for j in xrange(W.shape[0]):
                W[j] = W[j] + finetuning_lr * delta
            rbm4.W = W
            delta = rbm4.output_from_input(delta)

            # RBM5 finetune
            W = rbm5.W
            for j in xrange(W.shape[0]):
                W[j] = W[j] + finetuning_lr * delta
            rbm5.W = W
            delta = rbm5.output_from_input(delta)

            # RBM6 finetune
            W = rbm6.W
            for j in xrange(W.shape[0]):
                W[j] = W[j] + finetuning_lr * delta
            rbm6.W = W
            delta = rbm6.output_from_input(delta)

            # RBM7 finetune
            W = rbm7.W
            for j in xrange(W.shape[0]):
                W[j] = W[j] + finetuning_lr * delta
            rbm7.W = W
            delta = rbm7.output_from_input(delta)

    # result_output = rbm7.output()
    # print result_output

    # saveFeatures(result_output)

    # os.chdir('../../')


    # time2 = time.clock()
    # time = time2-time1
    # time = int(time)
    # time = str(time)
    # print 'total_time:' + time
