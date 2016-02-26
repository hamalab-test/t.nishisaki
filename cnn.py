# -*- coding: utf-8 -*-
import numpy

from utility import sigmoid

numpy_rng = numpy.random.RandomState(1234)


class CNN():
    def __init__(self, input, filter_shape, filter_shift, prev_shape, post_shape, lr, epoch, isRGB):
        self.input = input
        self.filter_shape = filter_shape
        self.filter_shift = filter_shift
        self.prev_shape = prev_shape
        self.post_shape = post_shape
        self.lr = lr
        self.epoch = epoch
        self.isRGB = isRGB
        # if isRGB:
        #     self.WR = numpy.array(numpy.random.uniform(low=-1.0, high=1.0,     size=(filter_shape[0], filter_shape[1])))
        #     self.WG = numpy.array(numpy.random.uniform(low=-1.0, high=1.0,     size=(filter_shape[0], filter_shape[1])))
        #     self.WB = numpy.array(numpy.random.uniform(low=-1.0, high=1.0,     size=(filter_shape[0], filter_shape[1])))
        # else:
        #     self.W = numpy.array(numpy.random.uniform(low=-1.0, high=1.0,     size=(filter_shape[0], filter_shape[1])))
        self.W = numpy.array(numpy.random.uniform(low=-1.0, high=1.0, size=(filter_shape[0], filter_shape[1])))
        self.bias = 0.0

    def pre_train(self):
        for ep in xrange(self.epoch):
            print 'pretrain epoch:' + str(ep+1)

            loss = 0.0

            if self.isRGB:
                for i_rgb in xrange(3):
                    data_input = self.input[i_rgb]

                    for i in xrange(data_input.shape[0]):
                        input_vector = data_input[i]

                        # 入力データの１次元ベクトルを２次元に直す
                        input = []
                        for j in xrange(self.prev_shape[1]):
                            # j 0~51
                            input.append(input_vector[j*self.prev_shape[0]:     (j+1)*self.prev_shape[0]])

                        input = numpy.array(input)

                        for y in xrange(self.post_shape[1]):
                            for x in xrange(self.post_shape[0]):
                                # print input.shape
                                # print x,y
                                # print x+self.filter_shape[0], y+self.filter_shape[1]

                                input_dot = input[y*self.filter_shift[0]:y*self.filter_shift[0]+self.filter_shape[1],
                                                  x*self.filter_shift[1]:x*self.filter_shift[1]+self.filter_shape[0]]

                                now_W = self.W
                                # if i_rgb == 0:
                                #     now_W = self.WR
                                # if i_rgb == 1:
                                #     now_W = self.WG
                                # if i_rgb == 2:
                                #     now_W = self.WB

                                output = [a*b for (a, b) in zip(input_dot, now_W)]
                                output = numpy.array(output)
                                output = output.sum() + self.bias
                                output_possible = sigmoid(output)

                                # 0,1の２値にする必要があるのかは不明
                                # output_state = numpy_rng.binomial(n=1, p=output_possible)
                                # print output_state

                                visible = output_possible * now_W
                                visible_possible = sigmoid(visible)

                                hidden_output = [a*b for (a, b) in zip(visible_possible, now_W)]
                                hidden_output = numpy.array(hidden_output)
                                hidden_output = hidden_output.sum() + self.bias
                                hidden_possible = sigmoid(hidden_output)

                                dw = numpy.zeros(self.filter_shape)
                                # print x,y
                                # print input_dot.shape
                                # print visible_possible.shape

                                dw = input_dot*output_possible - visible_possible*hidden_possible

                                loss += numpy.average(dw * dw)

                                # if self.isRGB:
                                #     if i_rgb == 0:
                                #         self.WR += self.lr * dw
                                #     elif i_rgb == 1:
                                #         self.WG += self.lr * dw
                                #     elif i_rgb == 2:
                                #         self.WB += self.lr * dw
                                # else:
                                #     self.W += self.lr * dw
                                self.W += self.lr * dw

                print loss

            else:
                for i in xrange(self.input.shape[0]):
                    input_vector = self.input[i]

                    # 入力データの１次元ベクトルを２次元に直す
                    input = []
                    for j in xrange(self.prev_shape[1]):
                        # j 0~51
                        input.append(input_vector[j*self.prev_shape[0]:     (j+1)*self.prev_shape[0]])

                    input = numpy.array(input)

                    for y in xrange(self.post_shape[1]):
                        for x in xrange(self.post_shape[0]):
                            # print input.shape
                            # print x,y
                            # print x+self.filter_shape[0], y+self.filter_shape[1]

                            input_dot = input[y*self.filter_shift[0]:y*self.filter_shift[0]+self.filter_shape[1],
                                              x*self.filter_shift[1]:x*self.filter_shift[1]+self.filter_shape[0]]

                            output = [a*b for (a, b) in zip(input_dot, self.W)]
                            output = numpy.array(output)
                            output = output.sum() + self.bias
                            output_possible = sigmoid(output)

                            # 0,1の２値にする必要があるのかは不明
                            # output_state = numpy_rng.binomial(n=1, p=output_possible)
                            # print output_state

                            visible = output_possible * self.W
                            visible_possible = sigmoid(visible)

                            print 'check!!'
                            print visible_possible.shape
                            print self.W

                            hidden_output = [a*b for (a, b) in zip(visible_possible, self.W)]
                            hidden_output = numpy.array(hidden_output)
                            hidden_output = hidden_output.sum() + self.bias
                            hidden_possible = sigmoid(hidden_output)

                            dw = numpy.zeros(self.filter_shape)
                            # print x,y
                            # print input_dot.shape
                            # print visible_possible.shape

                            dw = input_dot*output_possible - visible_possible*hidden_possible
                            self.W += self.lr * dw
                            # print numpy.var(dw)

    def output(self):
        output_rgblist = []
        # print self.input.shape

        if self.isRGB:
            for i_rgb in xrange(3):
                output_list = []
                data_input = self.input[i_rgb]

                for i in xrange(data_input.shape[0]):
                    if i % 100 == 0:
                        print 'output image:' + str(i), data_input.shape[0]
                    output_row = []

                    input_vector = data_input[i]

                    # print 'cnn output check!!'
                    # print input_vector.shape

                    # 入力データの１次元ベクトルを２次元に直す
                    input = []
                    for j in xrange(self.prev_shape[1]):
                        # j 0~51
                        input.append(input_vector[j*self.prev_shape[0]: (j+1)*self.prev_shape[0]])

                    input = numpy.array(input)

                    for y in xrange(self.post_shape[1]):
                        for x in xrange(self.post_shape[0]):
                            # x 0~73    74
                            # y 0~51    46
                            # input [52,80]

                            input_dot = input[y*self.filter_shift[0]:y*self.filter_shift[0]+self.filter_shape[1],
                                              x*self.filter_shift[1]:x*self.filter_shift[1]+self.filter_shape[0]]

                            now_W = self.W
                            # if i_rgb == 0:
                            #     now_W = self.WR
                            # if i_rgb == 1:
                            #     now_W = self.WG
                            # if i_rgb == 2:
                            #     now_W = self.WB

                            output = [a*b for (a, b) in zip(input_dot, now_W)]
                            output = numpy.array(output)
                            output = output.sum() + self.bias
                            output_possible = sigmoid(output)

                            output_row.append(output_possible)

                    output_list.append(output_row)

                output_rgblist.append(output_list)
                # return numpy.array(output_list)
            return numpy.array(output_rgblist)

        else:
            for i in xrange(self.input.shape[0]):
                print 'output image:' + str(i)
                output_row = []

                input_vector = self.input[i]

                # 入力データの１次元ベクトルを２次元に直す
                input = []
                for j in xrange(self.prev_shape[1]):
                    # j 0~51
                    input.append(input_vector[j*self.prev_shape[0]: (j+1)*self.prev_shape[0]])

                input = numpy.array(input)

                for y in xrange(self.post_shape[1]):
                    for x in xrange(self.post_shape[0]):
                        # x 0~73	74
                        # y 0~51	46
                        # input [52,80]

                        # print input.shape
                        # print x,y
                        # print self.post_shape

                        input_dot = input[y*self.filter_shift[0]:y*self.filter_shift[0]+self.filter_shape[1],
                                          x*self.filter_shift[1]:x*self.filter_shift[1]+self.filter_shape[0]]

                        output = [a*b for (a, b) in zip(input_dot, self.W)]
                        output = numpy.array(output)
                        output = output.sum() + self.bias
                        output_possible = sigmoid(output)

                        output_row.append(output_possible)

                # [Height, Width] を [Width, Height]に変換する？？？不要？
                # output_mat = []
                # for x in xrange(self.post_shape[0]):
                # 	# output_mat_vec = []
                # 	for y in xrange(self.post_shape[1]):
                # 		# print x,y
                # 		index = y*self.post_shape[1] + x
                # 		# print index
                # 		output_mat.append(output_row[index])

                output_list.append(output_row)
                # output_list.append(output_mat)

            return numpy.array(output_list)

    # デバッグ用関数 シグモイドを通さず出力値を見る
    def output_raw(self):
        output_list = []
        # output = numpy.array(output)
        # print output
        # output = numpy.array([])
        for i in xrange(self.input.shape[0]):
            # print i
            output_row = []
            # output_row = numpy.array([])

            input_vector = self.input[i]

            # 入力データの１次元ベクトルを２次元に直す
            # input [Width, Height] になっているのに注意
            input = []
            print self.prev_shape
            for j in xrange(self.prev_shape[1]):
                # j 0~51
                input.append(input_vector[j*self.prev_shape[0]: (j+1)*self.prev_shape[0]])

            input = numpy.array(input)
            print input

            for x in xrange(self.post_shape[0]):
                for y in xrange(self.post_shape[1]):
                    # print x,y

                    input_dot = input[x:x+self.filter_shape[0], y:y+self.filter_shape[1]]

                    output = [a*b for (a, b) in zip(input_dot, self.W)]
                    output = numpy.array(output)
                    output = output.sum() + self.bias
                    # output_possible = sigmoid(output)
                    # print output_possible

                    output_row.append(output)
            output_list.append(output_row)
        return output_list

    def setW(self, W):
        self.W = W
