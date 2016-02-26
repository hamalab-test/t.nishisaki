import os
from PIL import Image

# image = Image.open('H-35-1-copy.jpg')
# data = image.getdata()

# dir_name = 'rekihaku_300_gray_contrast_local'
dir_name = 'rekihaku_300_edges'

# if not os.path.exists(dir_name):
# 	os.mkdir(dir_name)
	
os.chdir(dir_name)

# os.mkdir('local')

for i in xrange(100):
	image = Image.open('image_' + str(i+1) + '.jpg')
	data = image.getdata()

	image_origin_size = image.size
	image_after_size = [30,30]
	divide_size = [10,10]

	for j in xrange(divide_size[0]*divide_size[1]):

		image_new = Image.new('L', (image_after_size[0],image_after_size[1]))
		# image_new = Image.new('RGB', (100,100))

		for x in xrange(image_after_size[0]):
			for y in xrange(image_after_size[0]):
				# r,g,b = data[x+100*(i%10)+1000*y+100*1000*(i/10)]
				# image_new.putpixel((x,y), r*0.2126 + g*0.7152 + b*0.0722)

				image_new.putpixel((x,y), data[x
					+image_after_size[0]*(j%divide_size[0])
					+image_origin_size[0]*y
					+image_after_size[0]*image_origin_size[0]*(j/divide_size[0])])
			
		if not os.path.exists('local_M1_3'):
			os.mkdir('local_M1_3')
		
		os.chdir('local_M1_3')
		image_new.save('image_' + str(i*divide_size[0]*divide_size[1]+j+1) + '.jpg')
		os.chdir('../')
