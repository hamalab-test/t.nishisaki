# -*- coding: utf-8 -*-

import numpy
import cPickle
from utility import softmax
from utility import sigmoid

numpy_rng = numpy.random.RandomState(1234)


class LogisticRegression:
    def __init__(self, input, label, input_size, output_size, data_size, lr):
        self.input = input
        self.label = label
        self.input_size = input_size
        self.output_size = output_size
        self.data_size = data_size
        self.W = numpy.array(numpy.random.uniform(low=-1.0, high=1.0, size=(output_size, input_size)))
        self.b = numpy.zeros(output_size)
        self.lr = lr

    def fine_tune(self):
        # print self.input.shape
        # print self.W.shape

        output = numpy.dot(self.input, self.W.T) + self.b
        hidden_possible = softmax(output)
        d_y = self.label - hidden_possible

        print '     loss : ' + str(numpy.sum(d_y ** 2))

        self.W += self.lr * numpy.dot(d_y.T, self.input) / self.data_size
        # self.W += (self.learning_rate * numpy.dot(d_y.T, self.input) - self.learning_rate * 0.1 * self.W) / 10000

        self.b += self.lr * numpy.mean(d_y, axis=0)

        # print 'check fine_tune'
        # print self.input.shape
        # print self.W.shape
        # print self.learning_rate
        # print self.b.shape

        # print numpy.max(numpy.dot(d_y.T, self.input))
        # print numpy.max(self.W)

    def predict(self, input):
        output = numpy.dot(input, self.W.T) + self.b
        hidden_possible = softmax(output)
        return hidden_possible

    def predict_direct(self, input):
        output = numpy.dot(input, self.W.T) + self.b
        # hidden_possible = softmax(output)
        # return hidden_possible
        return output

    def predict_sigmoid(self, input):
        output = numpy.dot(input, self.W.T) + self.b
        hidden_possible = sigmoid(output)
        return hidden_possible

    def inverse_predict(self):
        print 'Logistic Regression : self.W'
        print self.W
        print numpy.max(self.W)
        print numpy.min(self.W)
        print numpy.mean(self.W)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

        inverse_input = numpy.dot((1,0,0,0,0,0,0,0,0,0), self.W)
        # print inverse_input.shape
        # print inverse_input
        inverse_input_possible = softmax(inverse_input)
        inverse_input_sample = numpy_rng.binomial(n=1, p=inverse_input_possible)
        print inverse_input_sample

        f = open('inverse_log.txt', 'w')
        cPickle.dump(inverse_input_sample, f)
        f.close()

        return inverse_input_sample

    def getW(self):
        return self.W
