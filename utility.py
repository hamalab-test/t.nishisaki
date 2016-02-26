# -*- coding: utf-8 -*-
import numpy
import os
import datetime
from PIL import Image


def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2


def load_image(data_path, file_num, isRGB):
    os.chdir(data_path)

    pixel_list = []
    for i in xrange(file_num):
        image = Image.open('image' + str(i+1) + '.png')
        image_data = image.getdata()
        pixel_list_element = []

        if not isRGB:
            for px in image_data:
                pixel_list_element += px
            pixel_list.append(pixel_list_element)
        else:
            for px in image_data:
                pixel_list_element.append(px)
            pixel_list.append(pixel_list_element)

    pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return pixel_list


def load_RGBimage(data_path, file_num, isRGB):
    os.chdir(data_path)

    data_set = []

    pixel_list = []
    for i in xrange(file_num):
        image = Image.open('image' + str(i+1) + '.jpg')
        image_data = image.getdata()
        pixel_list_element = []

        if not isRGB:
            for px in image_data:
                pixel_list_element += px
            pixel_list.append(pixel_list_element)
        else:
            for px in image_data:
                pixel_list_element.append(px)
            pixel_list.append(pixel_list_element)

    pixel_list = numpy.split(pixel_list, [i * image.size[0] for i in xrange(1, image.size[0] + 1)], axis=1)[:image.size[0]]
    pixel_list = numpy.transpose(pixel_list, (1, 3, 0, 2))

    pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    data_set.append(pixel_list)

    label_list = []
    for m_i in xrange(file_num):
        # label_list.extend([m_i] * file_num)
        label_list.extend([m_i])

    label_list = numpy.array(label_list)
    # print label_list.shape

    data_set.append(label_list)

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    # return pixel_list
    return data_set


def load_motif(data_path, file_num, isRGB):
    os.chdir(data_path)

    pixel_list = []
    for i in xrange(file_num):
        image = Image.open('motif' + str(i+1) + '.png')
        image_data = image.getdata()
        pixel_list_element = []

        if not isRGB:
            for px in image_data:
                pixel_list_element += px
            pixel_list.append(pixel_list_element)
        else:
            for px in image_data:
                pixel_list_element.append(px[0:3])
            pixel_list.append(pixel_list_element)

    pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return pixel_list


def load_kosode(data_path, motif_num, file_num, isRGB):
    os.chdir(data_path)

    data_set = []
    pixel_list = []
    for m_i in xrange(motif_num):
        os.chdir('motif' + str(m_i + 1))

        for i in xrange(file_num):
            image = Image.open('image' + str(i+1) + '.jpg')
            image_data = image.getdata()
            pixel_list_element = []

            if not isRGB:
                for px in image_data:
                    pixel_list_element += px
                pixel_list.append(pixel_list_element)
            else:
                for px in image_data:
                    pixel_list_element.append(px)
                pixel_list.append(pixel_list_element)

        os.chdir('../')

    pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    data_set.append(pixel_list)

    label_list = []
    for m_i in xrange(motif_num):
        label_list.extend([m_i + 1] * file_num)

    label_list = numpy.array(label_list)
    # print label_list.shape

    data_set.append(label_list)

    # print data_set

    # data_set = numpy.array(data_set)
    # print data_set.shape

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return data_set


def load_RGBimage2cnn(data_path, motif_num, file_num, isRGB):
    os.chdir(data_path)

    data_set = []
    pixel_list = []
    for m_i in xrange(4, motif_num+4):
        print 'motif' + str(m_i)
        os.chdir('motif' + str(m_i))

        for i in xrange(file_num):
            image = Image.open('image' + str(i+1) + '.jpg')
            image_data = image.getdata()
            pixel_list_element = []

            for px in image_data:
                pixel_list_element.append(px)
            pixel_list.append(pixel_list_element)

            # print numpy.array(pixel_list).shape

        os.chdir('../')

    # print numpy.array(pixel_list).shape
    # print [i * 80 for i in xrange(1, 80 + 1)]
    pixel_list = numpy.split(pixel_list, [i * image.size[0] for i in xrange(1, image.size[0] + 1)], axis=1)[:image.size[0]]
    pixel_list = numpy.transpose(pixel_list, (1, 3, 0, 2))

    # pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    data_set.append(pixel_list)

    label_list = []
    for m_i in xrange(motif_num):
        label_list.extend([m_i] * file_num)

    label_list = numpy.array(label_list)
    # print label_list.shape

    data_set.append(label_list)

    # print data_set

    # data_set = numpy.array(data_set)
    # print data_set.shape

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return data_set


def load_RGBimage2cnn_5pickup(data_path, motif_num, file_num, isRGB):
    os.chdir(data_path)

    data_set = []
    pixel_list = []
    for m_i in xrange(1, motif_num+1):
        print 'motif' + str(m_i)
        os.chdir('motif' + str(m_i))

        for i in xrange(5):
            os.chdir('pickup' + str(i+1))

            for i in xrange(file_num):
                image = Image.open('image' + str(i+1) + '.jpg')
                image_data = image.getdata()
                pixel_list_element = []

                for px in image_data:
                    pixel_list_element.append(px)
                pixel_list.append(pixel_list_element)

                # print numpy.array(pixel_list).shape

            os.chdir('../')
        os.chdir('../')

    # print numpy.array(pixel_list).shape
    # print [i * 80 for i in xrange(1, 80 + 1)]
    pixel_list = numpy.split(pixel_list, [i * image.size[0] for i in xrange(1, image.size[0] + 1)], axis=1)[:image.size[0]]
    pixel_list = numpy.transpose(pixel_list, (1, 3, 0, 2))

    # pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    data_set.append(pixel_list)

    label_list = []
    for m_i in xrange(motif_num * 5):
        label_list.extend([m_i] * file_num)

    label_list = numpy.array(label_list)
    # print label_list.shape

    data_set.append(label_list)

    # print data_set

    # data_set = numpy.array(data_set)
    # print data_set.shape

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return data_set


def load_RGBimage2cnn_listdir(data_path, motif_num, file_num, isRGB):
    os.chdir(data_path)

    data_set = []
    pixel_list = []
    label_list = []
    count_list = []
    for m_i in xrange(1, motif_num+1):
        print 'motif' + str(m_i)
        os.chdir('motif' + str(m_i))

        files = os.listdir(os.getcwd())
        count = 0
        for file_name in files:
            if file_name.split('.')[-1] == 'jpg':
                count += 1
                image = Image.open(file_name)
                image_data = image.getdata()
                pixel_list_element = []

                for px in image_data:
                    pixel_list_element.append(px)
                pixel_list.append(pixel_list_element)

        count_list.append(count)
        os.chdir('../')

    # print numpy.array(pixel_list).shape
    # print [i * 80 for i in xrange(1, 80 + 1)]
    pixel_list = numpy.split(pixel_list, [i * image.size[0] for i in xrange(1, image.size[0] + 1)], axis=1)[:image.size[0]]
    pixel_list = numpy.transpose(pixel_list, (1, 3, 0, 2))

    # pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    data_set.append(pixel_list)

    for m_i in xrange(motif_num):
        label_list.extend([m_i] * count_list[m_i])

    label_list = numpy.array(label_list)
    print label_list
    print label_list.shape

    data_set.append(label_list)

    # print data_set

    # data_set = numpy.array(data_set)
    # print data_set.shape

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return data_set


def load_kosode_division(data_path, file_num, group_num, isRGB):
    os.chdir(data_path)

    pixel_list = []

    folder_name = 'group' + str(group_num)
    os.chdir(folder_name)
    print folder_name

    for j in xrange(10):
        folder_name = str((group_num)*10 + j+1)
        print folder_name
        os.chdir(folder_name)
        os.chdir('dzc_output_files/15')

        for k in xrange(file_num):
            image = Image.open('image' + str(k+1) + '.jpg')
            image_data = image.getdata()
            pixel_list_element = []

            if not isRGB:
                for px in image_data:
                    pixel_list_element += px
                pixel_list.append(pixel_list_element)
            else:
                for px in image_data:
                    pixel_list_element.append(px)
                pixel_list.append(pixel_list_element)

        os.chdir('../../../')
    os.chdir('../')

    pixel_list = numpy.split(pixel_list, [i * image.size[0] for i in xrange(1, image.size[0] + 1)], axis=1)[:image.size[0]]
    pixel_list = numpy.transpose(pixel_list, (1, 3, 0, 2))

    pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return pixel_list


def load_kosode_division_all(data_path, file_num, isRGB):
    os.chdir(data_path)

    pixel_list = []

    for i in xrange(1, 12):
        folder_name = 'group' + str(i)
        os.chdir(folder_name)

        for j in xrange(10):
            if i == 11 and j >= 4:
                break
            folder_name = str(i*10 + j+1)
            print folder_name
            os.chdir(folder_name)
            os.chdir('dzc_output_files/15')

            for image_num in xrange(file_num):
                image = Image.open('image' + str(image_num+1) + '.jpg')
                image_data = image.getdata()
                pixel_list_element = []

                if not isRGB:
                    for px in image_data:
                        pixel_list_element += px
                    pixel_list.append(pixel_list_element)
                else:
                    for px in image_data:
                        pixel_list_element.append(px)
                    pixel_list.append(pixel_list_element)

            os.chdir('../../../')
        os.chdir('../')

    pixel_list = numpy.split(pixel_list, [i * image.size[0] for i in xrange(1, image.size[0] + 1)], axis=1)[:image.size[0]]
    pixel_list = numpy.transpose(pixel_list, (1, 3, 0, 2))

    pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return pixel_list


def load_kosode_division_group(data_path, file_num, group_num, isRGB):
    os.chdir(data_path)

    pixel_list = []
    folder_name = 'group' + str(group_num)
    os.chdir(folder_name)

    for j in xrange(10):
        if group_num == 11 and j >= 4:
            break
        folder_name = str(group_num*10 + j+1)
        print folder_name
        os.chdir(folder_name)
        os.chdir('dzc_output_files/15')

        for image_num in xrange(file_num):
            image = Image.open('image' + str(image_num+1) + '.jpg')
            image_data = image.getdata()
            pixel_list_element = []

            if not isRGB:
                for px in image_data:
                    pixel_list_element += px
                pixel_list.append(pixel_list_element)
            else:
                for px in image_data:
                    pixel_list_element.append(px)
                pixel_list.append(pixel_list_element)

        os.chdir('../../../')
    os.chdir('../')

    pixel_list = numpy.split(pixel_list, [i * image.size[0] for i in xrange(1, image.size[0] + 1)], axis=1)[:image.size[0]]
    pixel_list = numpy.transpose(pixel_list, (1, 3, 0, 2))

    pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return pixel_list


def load_major_motif2cnn(data_path, motif_list, file_num, isRGB):
    os.chdir(data_path)

    pixel_list = []
    for m_i in xrange(len(motif_list)):
        os.chdir(motif_list[m_i])

        for i in xrange(file_num):
            image = Image.open('image' + str(i+1) + '.jpg')
            image_data = image.getdata()
            pixel_list_element = []

            if not isRGB:
                for px in image_data:
                    pixel_list_element += px
                pixel_list.append(pixel_list_element)
            else:
                for px in image_data:
                    pixel_list_element.append(px)
                pixel_list.append(pixel_list_element)

        os.chdir('../')

    pixel_list = numpy.split(pixel_list, [i * image.size[0] for i in xrange(1, image.size[0] + 1)], axis=1)[:image.size[0]]
    pixel_list = numpy.transpose(pixel_list, (1, 3, 0, 2))

    print numpy.array(pixel_list).shape

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    pixel_list = numpy.array(pixel_list)
    pixel_list = pixel_list.astype(float)
    pixel_list /= 255

    return pixel_list


def makeFolder():
    os.chdir('result')
    d = datetime.datetime.today()
    dir_name = d.strftime('%Y-%m-%d_%H-%M-%S')
    print dir_name

    assert not os.path.exists(dir_name)
    os.mkdir(dir_name)
    os.chdir(dir_name)


def saveImage(data_list, data_shape, dir_name):
    # os.chdir('result')
    # d = datetime.datetime.today()
    # dir_name = d.strftime('%Y-%m-%d_%H-%M-%S')

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    # assert not os.path.exists(dir_name)
    # os.mkdir(dir_name)
    # os.chdir(dir_name)

    # print numpy.array(data_list).shape

    # for i in xrange(numpy.array(data_list).shape[0]):
    for i in xrange(numpy.array(data_list).shape[0]):
        data = data_list[i]
        data = [pix*255 for pix in data]
        image = Image.new('L', data_shape)
        for j in xrange(data_shape[0] * data_shape[1]):
            # print j / data_shape[0], j % data_shape[0]
            image.putpixel((j % data_shape[0], j / data_shape[0]), data[j])
        image.save('visual' + str(i) + '.jpg')
        # image.show()

    os.chdir('../')
    # os.chdir('../../')


def cnn_saveColorImage(data_list, data_shape, dir_name):
    # os.chdir('result')
    # d = datetime.datetime.today()
    # dir_name = d.strftime('%Y-%m-%d_%H-%M-%S')

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    # assert not os.path.exists(dir_name)
    # os.mkdir(dir_name)
    # os.chdir(dir_name)

    for i_rgb in xrange(3):
        dir_name_rgb = ['R', 'G', 'B']
        if not os.path.exists(dir_name_rgb[i_rgb]):
            os.mkdir(dir_name_rgb[i_rgb])
        os.chdir(dir_name_rgb[i_rgb])

        data_rgb = data_list[i_rgb]

        # for i in xrange(numpy.array(data_list).shape[0]):
        for i in xrange(numpy.array(data_rgb).shape[0]):
            data = data_rgb[i]
            data = [pix*255 for pix in data]
            image = Image.new('L', data_shape)
            for j in xrange(data_shape[0] * data_shape[1]):
                # print j / data_shape[0], j % data_shape[0]
                image.putpixel((j % data_shape[0], j / data_shape[0]), data[j])
            image.save('visual' + str(i) + '.jpg')
            # image.show()

        os.chdir('../')
        # os.chdir('../../')
    os.chdir('../')


def rbm_saveColorImage(data_list, data_shape, dir_name):
    # os.chdir('result')
    # d = datetime.datetime.today()
    # dir_name = d.strftime('%Y-%m-%d_%H-%M-%S')

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    for i_rgb in xrange(3):
        dir_name_rgb = ['R', 'G', 'B']
        if not os.path.exists(dir_name_rgb[i_rgb]):
            os.mkdir(dir_name_rgb[i_rgb])
        os.chdir(dir_name_rgb[i_rgb])

        data_rgb = data_list[i_rgb]

        # for i in xrange(numpy.array(data_list).shape[0]):
        for i in xrange(numpy.array(data_rgb).shape[0]):
            data = data_rgb[i]
            data = [pix*255 for pix in data]
            image = Image.new('L', data_shape)
            for j in xrange(data_shape[0] * data_shape[1]):
                # print j / data_shape[0], j % data_shape[0]
                image.putpixel((j % data_shape[0], j / data_shape[0]), data[j])
            image.save('visual' + str(i) + '.jpg')
            # image.show()

        os.chdir('../')
        # os.chdir('../../')
    os.chdir('../')


def saveW(W, dir_name):
    # assert not os.path.exists(dir_name)
    # os.mkdir(dir_name)
    # os.chdir(dir_name)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    f = open('W.txt', 'w')
    for i in xrange(W.shape[0]):
        data_str = ''
        for j in xrange(W.shape[1]):
            if j == 0:
                data_str += str(W[i][j])
            else:
                data_str += ',' + str(W[i][j])
        data_str += '\n'
        f.writelines(data_str)
    f.close()

    os.chdir('../')


def saveFeature(output, file_name):
    # assert not os.path.exists(dir_name)
    # os.mkdir(dir_name)
    # os.chdir(dir_name)

    f = open(file_name, 'w')
    data_str = ''
    for i in xrange(len(output)):
        if i == 0:
            data_str += str(output[i])
        else:
            data_str += ',' + str(output[i])
    data_str += '\n'
    f.writelines(data_str)
    f.close()


def saveFeatures(output, file_name):
    # assert not os.path.exists(dir_name)
    # os.mkdir(dir_name)
    # os.chdir(dir_name)

    f = open(file_name, 'w')
    for i in xrange(output.shape[0]):
        data_str = ''
        for j in xrange(output.shape[1]):
            if j == 0:
                data_str += str(output[i][j])
            else:
                data_str += ',' + str(output[i][j])
        data_str += '\n'
        f.writelines(data_str)
    f.close()

    # os.chdir('../')


# 画像表示用
def debug_showImage(data_list, data_shape):
    # for i in xrange(numpy.array(data_list).shape[0]):
    for i in xrange(10):
        data = data_list[i]
        data = [i*255 for i in data]
        image = Image.new('L', data_shape)
        for j in xrange(data_shape[0] * data_shape[1]):
            # print j / data_shape[0], j % data_shape[0]
            image.putpixel((j % data_shape[0], j / data_shape[0]), data[j])
        # image.save('test_w_value_'+ str(i) + '.jpg')
        image.show()


def load_result_image(result_path, file_num, isRGB):
    os.chdir(result_path)

    data_list = []
    for i in xrange(file_num):
        image = Image.open('visual' + str(i) + '.jpg')
        image_data = image.getdata()
        data_list_element = []

        if isRGB:
            for px in image_data:
                data_list_element += px
            data_list.append(data_list_element)
        else:
            for px in image_data:
                data_list_element.append(px)
            data_list.append(data_list_element)

    data_list = numpy.array(data_list)

    # print numpy.max(data_list)
    # print numpy.min(data_list)

    # print numpy.mean(data_list,axis=1)
    data_list = data_list.astype(float)
    data_list /= 255

    # print numpy.max(data_list)
    # print numpy.min(data_list)

    dir_count = len(result_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return data_list


def loadW(dir_name):
    # if not os.path.exists(dir_name):
    # 	os.mkdir(dir_name)
    os.chdir(dir_name)

    f = open('W.txt')
    lines = f.readlines()
    f.close()

    W = []
    for line in lines:
        # print line
        W_elem = []

        values = line.split(',')
        for value in values:
            W_elem.append(float(value))
        W.append(W_elem)

    dir_count = len(dir_name.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    W = numpy.array(W)

    return W


def load_feature(data_path, file_name):
    os.chdir(data_path)

    f = open(file_name)
    line = f.readlines()
    f.close()

    feature = []

    values = line[0].split(',')
    for value in values:
        feature.append(float(value))

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return numpy.array(feature)


def load_features(data_path, file_name):
    os.chdir(data_path)

    f = open(file_name)
    lines = f.readlines()
    f.close()

    feature_list = []
    for line in lines:
        # print line
        feature = []

        values = line.split(',')
        for value in values:
            feature.append(float(value))
        feature_list.append(feature)

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    return numpy.array(feature_list)


def load_RGB(data_path, file_num):
    os.chdir(data_path)

    rgb_list = ['R', 'G', 'B']
    data_list = []

    for i in xrange(file_num):

        data_list_element = []

        for j in xrange(3):
            os.chdir(rgb_list[j])

            image = Image.open('visual' + str(i) + '.jpg')
            image_data = image.getdata()

            for px in image_data:
                data_list_element.append(px)

            os.chdir('../')
        data_list.append(data_list_element)

    data_list = numpy.array(data_list)
    data_list = data_list.astype(float)
    data_list /= 255

    dir_count = len(data_path.split('/'))

    return_dir = ''
    for i in xrange(dir_count):
        return_dir += '../'

    os.chdir(return_dir)

    # print data_list.shape

    return data_list


def local_contrast_normalization(data):
    max = numpy.max(data)
    min = numpy.min(data)

    data = (data - min) / (max - min)

    return data
