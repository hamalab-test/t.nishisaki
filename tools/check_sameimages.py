# -*- coding: utf-8 -*-
import os
import numpy
from PIL import Image
from PIL import ImageOps

# 画像の短い辺に合わせてアスペクト比を維持しつつリサイズし，余った領域を両端からトリミングする。

data_path = '../data/kosode_all_motif/1motif_5pickup/dataset_resize'
# resize_shape = (80, 80)
# data_size = 300
file_num = 60
motif_num = 50

os.chdir(data_path)

hist_list = []


def compare_hist(hist1, hist2):
    total = 0.
    for (data1, data2) in zip(hist1, hist2):
        data1 = float(data1)
        data2 = float(data2)
        total += numpy.sqrt((data1 - data2) ** 2)
    return total


for i_motif in xrange(motif_num):
    print 'motif:' + str(i_motif + 1)
    os.chdir('motif' + str(i_motif + 1))

    for i_pickup in xrange(5):
        folder_name = 'pickup' + str(i_pickup + 1)
        os.chdir(folder_name)

        for i_file in xrange(file_num):
            file_name = 'image' + str(i_file + 1) + '.jpg'
            img = Image.open(file_name)
            img = ImageOps.grayscale(img)
            img_hist = img.histogram()
            img_hist = numpy.array(img_hist)

            hist_list.append(img_hist)

        os.chdir('../')

    os.chdir('../')

hist_list = numpy.array(hist_list)
print hist_list.shape

for i in xrange(motif_num * 5):
    for j in xrange(i+1, motif_num * 5):
        result = compare_hist(hist_list[i], hist_list[j])
        if result == 100.:
            print result







