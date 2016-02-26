import os
from PIL import Image

data_path = '../data/kosode_all_motif/representations_resize'
resize_shape = (80, 80)
# resize_shape = (227, 227)
data_size = 53

os.chdir(data_path)

for num in xrange(data_size):
    print 'image:' + str(num+1)
    file_name = 'image' + str(num+1) + '.jpg'
    image = Image.open(file_name).resize(resize_shape).save(file_name)

os.chdir('../')
