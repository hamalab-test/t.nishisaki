# -*- coding: utf-8 -*-
import Image, os, numpy

data_path = '../data/kosode/google_gray'
data_size = 80

os.chdir(data_path)


for i in xrange(12):
    print 'label:' + str(i+1)
    os.chdir('motif' + str(i+1))

    for num in xrange(data_size):
        print '    image:' + str(num+1)

        file_name = 'image' + str(num+1) + '.jpg'
        image1 = Image.open(file_name)
        size = image1.size

        image2 = Image.new('L', size)
        data = image1.getdata()

        for j in xrange(len(data)):
            r, g, b = data[j]
            image2.putpixel((j % size[0], j / size[0]), r * 0.2126 + g * 0.7152 + b * 0.0722)

        image2.save(file_name)

    os.chdir('../')
