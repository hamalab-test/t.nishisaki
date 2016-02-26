from PIL import Image
from PIL import ImageFilter
import os

data_dir = 'rekihaku_300_gray_not_contrast_2'

os.chdir(data_dir)

for i in xrange(100):
	file_name = 'image_' + str(i+1) + '.jpg'

	image = Image.open(file_name)
	image2 = image.filter(ImageFilter.FIND_EDGES)
	image2.save(file_name)