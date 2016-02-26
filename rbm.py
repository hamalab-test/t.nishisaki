# -*- coding: utf-8 -*-
import numpy
import os

from utility import sigmoid
# from utility import softmax

numpy_rng = numpy.random.RandomState(1234)


class RBM:
    def __init__(self, W, input, data_size, input_size, output_size, isRGB, lr):
        self.input_size = input_size
        self.output_size = output_size
        self.data_size = data_size

        if W is None:
            W = numpy.array(numpy.random.uniform(
                                                low=-0.0, high=0.0,\
                                                # low=-4 * numpy.sqrt(6. / (output_size + input_size)),
                                                # high=4 * numpy.sqrt(6. / (output_size + input_size)),
                                                size=(output_size, input_size))
            )
        self.W = W
        self.input = input
        self.vbias = numpy.zeros(input_size)
        self.hbias = numpy.zeros(output_size)
        self.learning_rate = lr
        self.isRGB = isRGB

    def showInfo(self):
        print '~ showInfo() ~'

        print 'max'
        print numpy.max(self.W)

        print 'min'
        print numpy.min(self.W)

        print 'ave'
        print numpy.mean(self.W)

    def getW(self):
        return self.W

    def reconstruct_from_input(self, input):
        output = numpy.dot(input, self.W.T) + self.hbias
        hidden_possible = sigmoid(output)
        input_after = numpy.dot(hidden_possible, self.W) + self.vbias
        input_possible = sigmoid(input_after)
        assert input.shape == input_possible.shape
        return input_possible

    def reconstruct_from_output(self, output):
        input = numpy.dot(output, self.W) + self.vbias
        input_possible = sigmoid(input)
        # assert self.input.shape == input_possible.shape
        return input_possible

    def contrast_divergence(self, epoch):
        dw = numpy.zeros((self.output_size, self.input_size))
        dvb = numpy.zeros(self.input_size)
        dhb = numpy.zeros(self.output_size)

        output = numpy.dot(self.input, self.W.T) + self.hbias
        hidden_possible = sigmoid(output)
        # hidden_state = numpy_rng.binomial(n=1, p=hidden_possible)

        dw += self.learning_rate * numpy.dot(hidden_possible.T, self.input)
        dvb += self.learning_rate * numpy.mean(self.input, axis=0)
        dhb += self.learning_rate * numpy.mean(hidden_possible, axis=0)

        visible_output = numpy.dot(hidden_possible, self.W) + self.vbias
        visible_possible = sigmoid(visible_output)
        # visible_state = numpy_rng.binomial(n=1, p=visible_possible)

        hidden_output = numpy.dot(visible_possible, self.W.T) + self.hbias
        hidden_possible_after = sigmoid(hidden_output)

        dw -= self.learning_rate * numpy.dot(hidden_possible_after.T, visible_possible)
        dvb -= self.learning_rate * numpy.mean(visible_possible, axis=0)
        dhb -= self.learning_rate * numpy.mean(hidden_possible_after, axis=0)

        ####################
        # parameter update
        ####################

        self.W += dw / self.data_size
        # self.hbias += dhb / self.data_size
        # self.vbias += dvb / self.data_size

    def contrast_divergence_binomial(self, epoch):
        dw = numpy.zeros((self.output_size, self.input_size))
        # dvb = numpy.zeros(self.input_size)
        # dhb = numpy.zeros(self.output_size)

        output = numpy.dot(self.input, self.W.T) + self.hbias
        hidden_possible = sigmoid(output)
        hidden_state = numpy_rng.binomial(n=1, p=hidden_possible)

        dw += self.learning_rate * numpy.dot(hidden_state.T, self.input)
        # dvb += self.learning_rate * numpy.mean(self.input, axis=0)
        # dhb += self.learning_rate * numpy.mean(hidden_possible, axis=0)

        visible_output = numpy.dot(hidden_state, self.W) + self.vbias
        visible_possible = sigmoid(visible_output)
        visible_state = numpy_rng.binomial(n=1, p=visible_possible)

        hidden_output = numpy.dot(visible_state, self.W.T) + self.hbias
        hidden_possible_after = sigmoid(hidden_output)

        dw -= self.learning_rate * numpy.dot(hidden_possible_after.T, visible_state)
        # dvb -= self.learning_rate * numpy.mean(visible_state, axis=0)
        # dhb -= self.learning_rate * numpy.mean(hidden_possible_after, axis=0)

        ####################
        # parameter update
        ####################
        self.W += dw / self.data_size

    def contrast_divergence_eachdata(self, epoch):
        # print 'input.shape[0] : '+ str(self.input.shape[0])
        total_delta = numpy.zeros((self.output_size, self.input_size))

        for i in xrange(self.input.shape[0]):
            train_input = self.input[i]

            dw = numpy.zeros((self.output_size, self.input_size))
            dvb = numpy.zeros(self.input_size)
            dhb = numpy.zeros(self.output_size)

            output = numpy.dot(train_input, self.W.T) + self.hbias
            hidden_possible = sigmoid(output)
            # hidden_state = numpy_rng.binomial(n=1, p=hidden_possible)

            delta_W = []
            for j in xrange(hidden_possible.shape[0]):
                # j: 0~340
                delta_W_elem = train_input * hidden_possible[j]
                delta_W.append(delta_W_elem)
            delta_W = numpy.array(delta_W)

            dw += delta_W
            dvb += numpy.mean(train_input, axis=0)
            dhb += numpy.mean(hidden_possible, axis=0)

            visible_output = numpy.dot(hidden_possible, self.W) + self.vbias
            visible_possible = sigmoid(visible_output)

            hidden_output = numpy.dot(visible_possible, self.W.T) + self.hbias
            hidden_possible_after = sigmoid(hidden_output)

            delta_W = []
            for j in xrange(hidden_possible_after.shape[0]):
                # j: 0~340
                delta_W_elem = visible_possible * hidden_possible_after[j]
                delta_W.append(delta_W_elem)
            delta_W = numpy.array(delta_W)

            dw -= delta_W
            dvb -= numpy.mean(visible_possible, axis=0)
            dhb -= numpy.mean(hidden_possible_after, axis=0)

            # data_sizeで割るのは結局必要なのか？
            # self.W += self.learning_rate * dw

            total_delta += dw

            error = numpy.sum(numpy.abs(dw))
            print error

            os.chdir('result/rbm1_train' + str(epoch))
            f = open('error.txt', 'a')
            f.write(str(error) + ',')
            f.close()
            os.chdir('../../')

        # total_delta: -700~700 / 7000
        self.W += self.learning_rate * total_delta / self.data_size

        os.chdir('result/rbm1_train' + str(epoch))
        f = open('error.txt', 'a')
        f.write('\n')
        f.close()
        os.chdir('../../')

    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = numpy.dot(self.input, self.W.T) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy =  - numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v) +
            (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1))
        
        return cross_entropy

    def output(self):
        output = numpy.dot(self.input, self.W.T) + self.hbias
        hidden_possible = sigmoid(output)
        return hidden_possible

    def output_from_input(self, input):
        output = numpy.dot(input, self.W.T) + self.hbias
        hidden_possible = sigmoid(output)
        return hidden_possible
