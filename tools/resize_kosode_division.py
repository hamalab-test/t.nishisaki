import os
from PIL import Image

from os.path import join, relpath
from glob import glob

data_path = '../data/kosode_division_resize'
# resize_shape = (80, 80)
resize_shape = (80, 80)
data_size = 28

os.chdir(data_path)

for i in xrange(10):
    folder_name = 'group' + str(i+1)
    os.chdir(folder_name)
    print folder_name

    for j in xrange(10):
        folder_name = str((i+1)*10 + j+1)
        print folder_name
        os.chdir(folder_name)
        os.chdir('dzc_output_files/15')

        path = ''
        files = [relpath(x, path) for x in glob(join(path, '*'))]
        for k, file_name in enumerate(files):
            image = Image.open(file_name).resize(resize_shape).save('image' + str(k+1) + '.jpg')

        os.chdir('../../../')

    os.chdir('../')

# for num in xrange(data_size):
#     print 'image:' + str(num+1)
#     file_name = 'image' + str(num+1) + '.jpg'
#     image = Image.open(file_name).resize(resize_shape).save(file_name)

# os.chdir('../')
