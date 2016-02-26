from PIL import Image
import numpy
from utility import load_RGB


data_path = 'data/kosode_motif2/train/cnn2_after_train_norm_13567'
file_num = 80
motif_num = 5

data_set = load_RGB(data_path, file_num * motif_num)

print data_set.shape
print data_set[0]


