import os
from PIL import Image

data_path = '../data/kosode_motif2/train/motif'
file_name = 'image1.png'
resize_shape = (80,80)


os.chdir(data_path)


image = Image.open(file_name).resize(resize_shape).save(file_name)
# size = image.size
# image_new = Image.new('RGB', (1000,1000))
