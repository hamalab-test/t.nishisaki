import os
import numpy
from PIL import Image

# dir_name = 'local_M1_5'
dir_name = 'rekihaku_300_edges'
dir_name_new = '0or1_1'
	
os.chdir(dir_name)

file_size = 10000

for i in xrange(file_size):
	print i
	image = Image.open('image_' + str(i+1) + '.jpg').convert('L')
	imgArray = numpy.asarray(image)
	maxcol, maxrow = image.size
	imgArray.flags.writeable = True
	for x in xrange(maxrow):
		for y in xrange(maxcol):
			if(imgArray[x,y] > 50):
				imgArray[x,y] = 255
			else:
				imgArray[x,y] = 0
	image_new = Image.fromarray(numpy.uint8(imgArray))

	if not os.path.exists(dir_name_new):
		os.mkdir(dir_name_new)
		
	os.chdir(dir_name_new)
	image_new.save('image_' + str(i+1) + '.jpg')
	os.chdir('../')

