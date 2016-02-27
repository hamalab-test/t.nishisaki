from PIL import Image
import os
import numpy

data_dir = 'rbm1_after_train'
image_size = (34, 20)
W_shape = (340, 680)
file_size = 340

# file_count = (10,10)

os.chdir(data_dir)

# total_image = Image.new('L', (image_size[0]*file_count[0], image_size[1]*file_count[1]))

f = open('W.txt')
lines = f.readlines()
f.close()

os.chdir('../')

new_dir = data_dir + '_W'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
os.chdir(new_dir)


for i, line in enumerate(lines):
    print i

    W_elem = []

    values = line.split(',')
    for value in values:
        W_elem.append(float(value))
    # W.append(W_elem)

    W_max = numpy.max(W_elem)
    W_min = numpy.min(W_elem)

    W_elem = [pix-W_min for pix in W_elem]
    W_elem = [pix/W_max for pix in W_elem]
    W_elem = [pix*255 for pix in W_elem]

    assert numpy.max(W_elem) > 1.0 and numpy.min(W_min) < -1.0

    image = Image.new('L', image_size)
    for j in xrange(image_size[0] * image_size[1]):
        # print j / data_shape[0], j % data_shape[0]
        image.putpixel((j % image_size[0], j / image_size[0]), W_elem[j])
    image.save('visual' + str(i) + '.jpg')


# W = numpy.array(W)

# total_image.save(data_dir + '.jpg')
