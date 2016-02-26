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

	print image.getdata()

	count = 0
	white_sum = 0
	for x in xrange(size[0]):
		for y in xrange(size[1]):
			# print image.getpixel((x,y))
			if image.getpixel((x,y)) > 230:
				# image.putpixel((x,y), stat.mean)
				count += 1
				white_sum += image.getpixel((x,y))

	print count

	sum = stat.sum[0] - 255 * count
	sum = stat.sum[0] - white_sum
	kosode_count = size[0] * size[1] - count
	
	print kosode_count
	if kosode_count is 0:
		average = 0
	else:
		average  = sum / kosode_count
	# print average
	assert 0 <= average <= 255

	for x in xrange(size[0]):
		for y in xrange(size[1]):
			if image.getpixel((x,y)) > 230:
				image.putpixel((x,y), average)
		
	# regulation
	# max = 0
	# min = 255
	# for x in xrange(size[0]):
	# 	for y in xrange(size[1]):
	# 		pix = image.getpixel((x,y))
	# 		if pix > max:
	# 			max = pix
	# 		if pix < min:
	# 			min = pix



	# for x in xrange(size[0]):
	# 	for y in xrange(size[1]):
	# 		pix = image.getpixel((x,y))
	# 		image.putpixel((x,y), 1.0 * (pix - min)/(max - min)*255 )
	# image.show()
	image.save(file_name)
