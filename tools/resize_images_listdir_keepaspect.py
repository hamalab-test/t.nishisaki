# -*- coding: utf-8 -*-
import os
from PIL import Image

# 画像の短い辺に合わせてアスペクト比を維持しつつリサイズし，余った領域を両端からトリミングする。

data_path = '../data/caltech101/dataset_resize/test'
resize_shape = (80, 80)
# data_size = 300
motif_num = 50

os.chdir(data_path)

for i in xrange(motif_num):
    print 'label:' + str(i+1)
    os.chdir(str(i+1))

    # for pickup_num in xrange(5):
    #     print pickup_num
    #     folder_name = 'pickup' + str(pickup_num+1)
    #     os.chdir(folder_name)

    files = os.listdir(os.getcwd())
    for file_name in files:
        if file_name.split('.')[-1] == 'jpg':
            print i+1, file_name
            image = Image.open(file_name)
            size = image.size
            if size[0] > size[1]:
                aspect_rate = (float)(resize_shape[1]) / size[1]
                image2 = image.resize(((int)(size[0]*aspect_rate), resize_shape[1]))
                x = (int)((image2.size[0] - resize_shape[0]) / 2)
                y = 0
                print ((int)(size[0]*aspect_rate), resize_shape[1])
                print (x, y, x + resize_shape[0], y + resize_shape[1])
                image2.crop((x, y, x + resize_shape[0], y + resize_shape[1])).save(file_name)

            else:
                aspect_rate = (float)(resize_shape[0]) / size[0]
                image2 = image.resize((resize_shape[0], (int)(size[1]*aspect_rate)))
                x = 0
                y = (int)((image2.size[1] - resize_shape[1]) / 2)
                print (resize_shape[0], (int)(size[1]*aspect_rate))
                print (x, y, x + resize_shape[0], y + resize_shape[1])
                image2.crop((x, y, x + resize_shape[0], y + resize_shape[1])).save(file_name)

            # image = Image.open(file_name).resize(resize_shape).save(file_name)
            # image.thumbnail(resize_shape, Image.ANTIALIAS)
            # image.save(file_name)
    os.chdir('../')

    # os.chdir('../')
