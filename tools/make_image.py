import cPickle
from PIL import Image

# file_name = 'inverse_log.txt'
# save_name = 'inverse_image.jpg'
# image_size = (10,50)

file_name = 'inverse_input_log.txt'
save_name = 'inverse_input_image.jpg'
image_size = (28,28)

f = open(file_name)
l = cPickle.load(f)

for i in xrange(len(l)):
	l[i] *= 255
	# l[i] *= 100

image = Image.new('L', image_size)

for i in xrange(len(l)):
	image.putpixel((i%image_size[0], i/image_size[1]), l[i])

image.save(save_name)
