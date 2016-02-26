import os
from PIL import Image


image = Image.open('H-35-1-copy.jpg')

data = image.getdata()

dir_name = 'rekihaku_local_100_gray'

os.mkdir(dir_name)
os.chdir(dir_name)

for i in xrange(100):
	image_new = Image.new('L', (100,100))	
	# image_new = Image.new('RGB', (100,100))
	for x in xrange(100):
		for y in xrange(100):
			r,g,b = data[x+100*(i%10)+1000*y+100*1000*(i/10)]
			image_new.putpixel((x,y), r*0.2126 + g*0.7152 + b*0.0722)

			# image_new.putpixel((x,y), data[x+100*(i%10)+1000*y+100*1000*(i/10)])
			
	image_new.save('image_' + str(i+1) + '.jpg')
