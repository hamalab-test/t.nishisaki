# -*- coding: utf-8 -*-

# 入力データは時系列画像を想定。input = [画像のインデックス，画素w*h]の２次元配列。
# 各時刻ごとの画像に対して誤差を計算し，全画像で計算が完了するまでを１ステップとする。
# ある時刻Tの学習をするのにt=0,…,Tまでの結果を利用するため，計算量が膨大になってしまう

import numpy

from utility import sigmoid

numpy_rng = numpy.random.RandomState(1234)

numpy.seterr(all='raise', under='ignore')


class RTRBM:
    def __init__(self, W, U, input_v, input_r, data_size, input_size, output_size):
        if W is None:
            W = numpy.array(numpy_rng.uniform(
                                                # low=-1.0, high=1.0,\
                                                low=-4 * numpy.sqrt(6. / (output_size + input_size)),
                                                high=4 * numpy.sqrt(6. / (output_size + input_size)),
                                                size=(output_size, input_size)))

        if U is None:
            U = numpy.array(numpy_rng.uniform(
                                                low=-0.0, high=0.0, \
                                                # low=-4 * numpy.sqrt(6. / (output_size + input_size)),
                                                # high=4 * numpy.sqrt(6. / (output_size + input_size)),
                                                size=(output_size, output_size)))

        self.W = W
        self.U = U
        self.input_v = input_v
        self.input_r = input_r
        self.data_size = data_size
        self.input_size = input_size
        self.output_size = output_size

        self.vbias = numpy.zeros(input_size)
        self.hbias = numpy.zeros(output_size)
        self.lr = 0.01

    def contrast_divergence(self, epoch):
        v_list = self.input_v
        h_list = []
        r_list = []
        d_list = []

        v_iteration_list = []
        h_iteration_list = []

        # t=0,,,Tまで隠れ層の出力hとリカレントの出力rを計算する
        for i in xrange(self.data_size):
            input = v_list[i]

            r = 0
            h = 0
            if i == 0:
                r = numpy.dot(input, self.W.T) + self.hbias
                h = numpy.dot(input, self.W.T) + self.hbias
            else:
                r = numpy.dot(input, self.W.T) + self.hbias + numpy.dot(r_list[i-1], self.U.T)
                h = numpy.dot(input, self.W.T) + self.hbias + numpy.dot(r_list[i-1], self.U.T)

            r = sigmoid(r)
            # h = numpy.dot(input, self.W.T) + self.hbias
            h = sigmoid(h)

            h_list.append(h)
            r_list.append(r)

            # CD iterationはとりあえず１で試す
            v_iteration = numpy.dot(h, self.W) + self.vbias
            v_iteration = sigmoid(v_iteration)

            h_iteration = 0
            if i == 0:
                h_iteration = numpy.dot(v_iteration, self.W.T) + self.hbias
            else:
                h_iteration = numpy.dot(v_iteration, self.W.T) + self.hbias + numpy.dot(r_list[i-1], self.U.T)
            h_iteration = sigmoid(h_iteration)

            v_iteration_list.append(v_iteration)
            h_iteration_list.append(h_iteration)

        # print numpy.array(v_iteration_list).shape
        # print numpy.array(h_iteration_list).shape

        v_list = numpy.array(v_list)
        h_list = numpy.array(h_list)
        r_list = numpy.array(r_list)
        d_list = numpy.array(d_list)
        v_iteration_list = numpy.array(v_iteration_list)
        h_iteration_list = numpy.array(h_iteration_list)

        d_reverse_list = []

        for i in reversed(xrange(self.data_size)):
            h_diff = h_list[i] - h_iteration_list[i]

            d = 0
            if i == self.data_size - 1:
                d = numpy.dot(self.U, h_diff)
            else:
                d = numpy.dot(self.U, d_reverse_list[-1] * r_list[i] * (1 - r_list[i]) + h_diff)

            d_reverse_list.append(d)

        d_reverse_list = numpy.array(d_reverse_list)

        d_list = []
        for i in xrange(len(d_reverse_list)):
            d_list.append(d_reverse_list[len(d_reverse_list) - i - 1])
        d_list = numpy.array(d_list)

        # calculate W H
        delta_H_W = [numpy.dot(h[numpy.newaxis, :].T, v[numpy.newaxis, :]) \
                   - numpy.dot(ha[numpy.newaxis, :].T, va[numpy.newaxis, :]) \
                   for v,h,va,ha \
                   in zip(v_list, h_list, v_iteration_list, h_iteration_list)]

        delta_H_W = numpy.array(delta_H_W)
        delta_H_W = numpy.average(delta_H_W, axis=0)

        # calculate W Q2
        # delta_Q2_W = [numpy.dot(d*r*(1-r)[numpy.newaxis, :].T, v[numpy.newaxis, :]) \
        #              for d,r,v \
        #              in zip(d_list, r_list, v_list)]

        _delta_Q2_W = [d*r*(1-r) \
                     for d,r,v \
                     in zip(d_list, r_list, v_list)]

        _delta_Q2_W = numpy.array(_delta_Q2_W)

        delta_Q2_W = [numpy.dot(dr[numpy.newaxis, :].T, v[numpy.newaxis, :]) \
                      for dr,v \
                      in zip(_delta_Q2_W, v_list)]

        delta_Q2_W = numpy.average(delta_Q2_W, axis=0)

        # calculate W delta
        # delta_W = delta_H_W + delta_Q2_W
        delta_W = delta_H_W

        # calculate U delta
        _delta_Q2_U = _delta_Q2_W + (h_list - h_iteration_list)
        delta_Q2_U = numpy.dot(_delta_Q2_U.T, r_list)
        # delta_Q2_U = numpy.average(delta_Q2_U, axis=0)
        # 上の式，Uは(output_size, output_size)なので実行できても意味が通ってるか確認すべき

        # !!! check d_list
        # print numpy.max(d_list)
        # print numpy.min(d_list)
        # print numpy.average(d_list)

        delta_U = delta_Q2_U

        # calculate vbias
        delta_vbias = v_list - v_iteration_list
        delta_vbias = numpy.average(delta_vbias, axis=0)

        # calculate hbias
        delta_hbias = h_list - h_iteration_list + _delta_Q2_W
        delta_hbias = numpy.average(delta_hbias, axis=0)

        ####################
        # gradient check
        ####################

        f = open('gradient_check_U_10000.txt', 'a+')
        f.write(str(numpy.sum(numpy.fabs(delta_U))) + '\n')
        print numpy.sum(numpy.fabs(delta_U))
        f.close()

        ####################
        # parameter update
        ####################

        # print 'delta_U'
        # print delta_U
        # print 'U'
        # print self.U

        self.W += self.lr * delta_W
        self.U += self.lr * delta_U - 0.01 * self.U
        # self.U += self.lr * delta_U
        # self.vbias += self.lr * delta_vbias
        # self.hbias += self.lr * delta_hbias

    def getW(self):
        return self.W

    def getU(self):
        return self.U

    def output_hr(self):
        h_list = []
        r_list = []

        for i in xrange(self.data_size):
        # for i in xrange(4):
        #     input = []
        #     if i is 0:
        #         input = self.input_v[0]
        #     elif i is 1:
        #         input = self.input_v[19]
        #     elif i is 2:
        #         input = self.input_v[39]
        #     elif i is 3:
        #         input = self.input_v[19]

            input = self.input_v[i]

            # r = 0
            # if i == 0:
            #     r = numpy.dot(input, self.W.T) + self.hbias
            # else:
            #     r = numpy.dot(input, self.W.T) + self.hbias + numpy.dot(r_list[i-1], self.U.T)

            if i == 0:
                r = numpy.dot(input, self.W.T) + self.hbias
                h = numpy.dot(input, self.W.T) + self.hbias

                r = (r - numpy.min(r)) / numpy.max(r - numpy.min(r)) * 6 - 3
                h = (h - numpy.min(h)) / numpy.max(h - numpy.min(h)) * 6 - 3

                # print numpy.max(h)
                # print numpy.min(h)
            else:
                # print 'feature_check!!!!'
                # print 'r_list[i-1]'
                # print r_list[i-1]
                # print 'self.U.T'
                # print self.U.T

                # print 'r_list[i-1]'
                # print r_list[i-1]
                # print 'self.U.T'
                # print self.U.T

                tmp_v = numpy.dot(input, self.W.T)
                tmp_r = numpy.dot(r_list[i-1], self.U.T)

                tmp_v = (tmp_v - numpy.min(tmp_v)) / numpy.max(tmp_v - numpy.min(tmp_v)) * 6 - 3
                tmp_r = (tmp_r - numpy.min(tmp_r)) / numpy.max(tmp_r - numpy.min(tmp_r)) * 6 - 3

                h = tmp_v + tmp_r
                r = tmp_v + tmp_r

                # h = tmp_v
                # r = tmp_v

                f = open('check_vrh.txt', 'a+')
                # check_v = numpy.dot(input, self.W.T)
                # check_r = numpy.dot(r_list[i-1], self.U.T)
                # check_h = numpy.dot(input, self.W.T) + self.hbias + numpy.dot(r_list[i-1], self.U.T)
                str_v = ''
                str_r = ''
                str_h = ''
                for i_check in xrange(30):
                    str_v += str(tmp_v[i_check]) + ','
                    str_r += str(tmp_r[i_check]) + ','
                    str_h += str(h[i_check]) + ','
                f.write(str_v + '\n')
                f.write(str_r + '\n')
                f.write(str_h + '\n')
                f.write('\n')
                f.close()

                print numpy.max(h)
                print numpy.min(h)

                # r = numpy.dot(input, self.W.T) + self.hbias + numpy.dot(r_list[i-1], self.U.T)
                # h = numpy.dot(input, self.W.T) + self.hbias + numpy.dot(r_list[i-1], self.U.T)

            r = sigmoid(r)
            h = sigmoid(h)

            h_list.append(h)
            r_list.append(r)

        h_list = numpy.array(h_list)
        r_list = numpy.array(r_list)

        # print 'output_hr check'
        # print numpy.max(self.input_v)
        # print numpy.min(self.input_v)
        # print numpy.average(self.input_v)

        # print numpy.max(self.W)
        # print numpy.min(self.W)
        # print numpy.average(self.W)

        # print numpy.max(h_list)
        # print numpy.min(h_list)
        # print numpy.average(h_list)

        return h_list, r_list

    def check_params(self):
        print 'check_params'
        print '~W~'
        print 'max:' + str(numpy.max(self.W))
        print 'min:' + str(numpy.min(self.W))
        print 'ave:' + str(numpy.average(self.W))
        print '~U~'
        print 'max:' + str(numpy.max(self.U))
        print 'min:' + str(numpy.min(self.U))
        print 'ave:' + str(numpy.average(self.U))
        print ''
