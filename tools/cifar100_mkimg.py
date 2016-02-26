import numpy
import cPickle
import os
from PIL import Image
import more_itertools


data_path = '../data/cifar-100-python/train'

f = open(data_path)
dict = cPickle.load(f)
f.close()

data = dict['data']
fine_labels = dict['fine_labels']

fine_labels = numpy.array(fine_labels)
idx = numpy.where(fine_labels < 50)

target = fine_labels[idx]
x_train = data[idx]

os.chdir('../data/cifar-100-python/img')

for i in xrange(50):
	os.mkdir('category' + str(i+1))

count_list = numpy.zeros(50, dtype=numpy.int)

for (x,y) in zip(x_train, target):
	os.chdir('category' + str(y+1))

	print x.shape
	x = more_itertools.chunked(x, 1024)
	rgb = list(x)
	rgb = numpy.array(rgb)

	print rgb.shape

	rgb = [more_itertools.chunked(a, 32) for a in rgb]
	rgb2 = [list(rgb) for rgb in rgb]
	rgb2 = numpy.array(rgb2)

	print rgb2.shape

	rgb2 = numpy.transpose(rgb2, (1, 2, 0))

	img = Image.fromarray(numpy.uint8(rgb2))

	count_list[y] += 1

	img.save('image' + str(count_list[y]) + '.jpg')




	os.chdir('../')

