from PIL import Image
import os
import math

# data_date = '2014-04-10_16-27-06'
# data_date = '2014-05-13_09-42-02'
# rbm_layer = 'rbm_layer_0'
# data_dir = 'visualization/' + data_date + '/' + rbm_layer

# data_dir = '../data/kosode_motif2/test/kosode_smallfield_origin/sample/judge_result_sigmoid_epoch200/rejudge/' + str(motif_num)
# data_dir = '../data/kosode_motif1_re/train/cnn2_after_train_norm/R'
data_dir = '../data/kosode_all_motif/1motif_5pickup/dataset_resize'
# data_dir = '../data/kosode_division_resize/judge_result/' + str(motif_num)
# data_dir = '../data/kosode_motif6/train/google_dataset_resize/motif' + str(motif_num)

# image_size = (34, 34)
# image_size = (514, 514)
# image_size = (227, 227)
image_size = (80, 80)

# file_count = (3, 3)
file_count = (10, 10)

file_size = file_count[0] * file_count[1]
# file_size = 9
# file_count_elm = int(math.sqrt(file_size)) + 1
# file_count = (file_count_elm, file_count_elm)

os.chdir(data_dir)

# motif_num = 1
# pickup_num = 1

for i_motif in xrange(50):
    os.chdir('motif' + str(i_motif + 1))

    for i_pickup in xrange(5):
        os.chdir('pickup' + str(i_pickup + 1))

        total_image = Image.new('RGB', (image_size[0]*file_count[0], image_size[1]*file_count[1]))

        for i in xrange(file_size):
            # file_name = 'visual' + str(file_size * (motif_num - 1) + i) + '.jpg'
            file_name = 'image' + str(i+1) + '.jpg'
            this_image = Image.open(file_name)

            total_x = i % file_count[0]
            total_y = i / file_count[0]

            print i

            for x in xrange(this_image.size[0]):
                for y in xrange(this_image.size[1]):
                    px = this_image.getpixel((x, y))
                    total_image.putpixel((image_size[0] * total_x + x, image_size[1] * total_y + y), px)

        os.chdir('../')
        # total_image.save('total_visualization_' + rbm_layer + '.jpg')
        total_image.save('total_visualization_' + str(i_motif + 1) + '_' + str(i_pickup + 1) + '.jpg')
        # total_image.save('total_visualization_kiku15.jpg')

    os.chdir('../')
