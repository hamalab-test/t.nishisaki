import Image, ImageFilter, os

os.chdir('rekihaku_300_gray_not_contrast_2')

for i in xrange(100):
	file_name = 'image_' + str(i+1) + '.jpg'
	img = Image.open(file_name)
	img2 = img.filter(ImageFilter.MedianFilter(5))
	img2.save(file_name)