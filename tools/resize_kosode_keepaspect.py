# -*- coding: utf-8 -*-
import os
from PIL import Image

# 画像の短い辺に合わせてアスペクト比を維持しつつリサイズし，余った領域を両端からトリミングする。

data_path = '../data/kosode_division_resize'
resize_shape = (80, 80)
# data_size = 1200
# motif_num = 50

os.chdir(data_path)

for i in xrange(1, 12):
    folder_name = 'group' + str(i)
    os.chdir(folder_name)

    for j in xrange(10):
        if i == 11 and j >= 3:
            break
        folder_name = str(i*10 + j+1)
        print folder_name
        os.chdir(folder_name)
        os.chdir('dzc_output_files/15')

        jpg_count = 0
        for x_file in xrange(50):
            for y_file in xrange(50):
                file_name = str(x_file) + '_' + str(y_file) + '.jpg'
                if os.path.isfile(file_name):
                    # print file_name
                    image = Image.open(file_name)
                    size = image.size
                    if size[0] < resize_shape[0] or size[1] < resize_shape[1]:
                        os.remove(file_name)
                        continue

                    jpg_count += 1
                    save_name = 'image' + str(jpg_count) + '.jpg'
                    if size[0] > size[1]:
                        aspect_rate = (float)(resize_shape[1]) / size[1]
                        image2 = image.resize(((int)(size[0]*aspect_rate), resize_shape[1]))
                        x = (int)((image2.size[0] - resize_shape[0]) / 2)
                        y = 0
                        # print ((int)(size[0]*aspect_rate), resize_shape[1])
                        # print (x, y, x + resize_shape[0], y + resize_shape[1])
                        image2.crop((x, y, x + resize_shape[0], y + resize_shape[1])).save(save_name)

                    else:
                        aspect_rate = (float)(resize_shape[0]) / size[0]
                        image2 = image.resize((resize_shape[0], (int)(size[1]*aspect_rate)))
                        x = 0
                        y = (int)((image2.size[1] - resize_shape[1]) / 2)
                        # print (resize_shape[0], (int)(size[1]*aspect_rate))
                        # print (x, y, x + resize_shape[0], y + resize_shape[1])
                        image2.crop((x, y, x + resize_shape[0], y + resize_shape[1])).save(save_name)

                    os.remove(file_name)

        os.chdir('../../../')
    os.chdir('../')
