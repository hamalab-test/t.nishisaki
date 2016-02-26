from PIL import Image
import os
import math

motif_num = 1

# data_dir = '../data/kosode_motif2/test/kosode_smallfield_origin/sample/judge_result_sigmoid_epoch200/rejudge/' + str(motif_num)
# data_dir = '../data/kosode_motif1_re/train/cnn2_after_train_norm/R'
# data_dir = '../data/kosode_major_motif_resize/kiku'
# data_dir = '../data/kosode_division_resize/judge_group1_result_sigmoid_epoch200/' + str(motif_num)
# data_dir = '../data/kosode_motif6/train/google_dataset_resize/motif' + str(motif_num)

# data_dir = '../data/kosode_division_resize/judge_result/' + str(motif_num)
# data_dir = '../data/kosode_division_resize/judge_result_pickup/'
data_dir = '../data/caltech101/dataset_resize/google/39'

os.chdir(data_dir)

image_size = (80, 80)

# for i_motif in xrange(41, 42):
    # os.chdir(str(i_motif))

files = os.listdir(os.getcwd())
temp = int(math.sqrt(len(files))) + 1
file_count = (temp, temp)

total_image = Image.new('RGB', (image_size[0]*file_count[0], image_size[1]*file_count[1]))

for i_file, file_name in enumerate(files):
    this_image = Image.open(file_name)

    total_x = i_file % file_count[0]
    total_y = i_file / file_count[0]

    print i_file

    for x in xrange(this_image.size[0]):
        for y in xrange(this_image.size[1]):
            px = this_image.getpixel((x, y))
            total_image.putpixel((image_size[0] * total_x + x, image_size[1] * total_y + y), px)

    # os.chdir('../')

# total_image.save('total_visualization_' + rbm_layer + '.jpg')
# total_image.save('total_visualization_' + str(i_motif) + '.jpg')
total_image.save('total_visualization_flamingo.jpg')
# total_image.save('total_visualization_kiku15.jpg')
