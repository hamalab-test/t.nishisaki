from PIL import Image
from PIL import ImageStat
import os

data_dir = 'local_4'

os.chdir(data_dir)

for i in xrange(3600):
	file_name = 'image_' + str(i+1) + '.jpg'
	image = Image.open(file_name)
	stat = ImageStat.Stat(image)
	size = image.size
	# print stat.sum

	# print image.getdata()

	# regulation
	max = 0
	min = 255
	for x in xrange(size[0]):
		for y in xrange(size[1]):
			pix = image.getpixel((x,y))
			if pix > max:
				max = pix
			if pix < min:
				min = pix


	if max - min == 0:
		continue

	for x in xrange(size[0]):
		for y in xrange(size[1]):
			pix = image.getpixel((x,y))
			image.putpixel((x,y), 1.0 * (pix - min)/(max - min)*255 )
	# image.show()
	image.save(file_name)
