import os
from PIL import Image

data_path = '../data/kosode_motif2/test/kosode_smallfield_resize'
resize_shape = (80, 80)
data_sizes = (2, 6, 20, 80, 320, 1209)

os.chdir(data_path)

for i in xrange(50, 63):
    print('i', i)
    os.chdir(str(i))
    for j in xrange(10, 16):
        print('j', j)
        os.chdir(str(j))
        for num in xrange(data_sizes[j - 10]):
            print 'image:' + str(num+1)
            file_name = 'image' + str(num+1) + '.jpg'
            image = Image.open(file_name).resize(resize_shape).save(file_name)
        os.chdir('../')
    os.chdir('../')
os.chdir('../')
