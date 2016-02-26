import os
import shutil

judge_path = 'judge/motif50_acc0.35'
# data_path = 'data/kosode_motif2/test/kosode_smallfield_origin/sample'
data_path = 'data/kosode_division_resize'
folder_name = 'judge_result_pickup'

# group_num = 1
# kosode_num = 19
file_num = 1141

os.chdir('../')

for i_kosode in xrange(11, 114):
    group_num = (i_kosode - 1) / 10

    os.chdir(judge_path)
    judge_name = 'chainer_features_' + str(i_kosode) + '.txt'
    f = open(judge_name, 'r')
    lines = f.readlines()
    f.close()
    os.chdir('../../')

    result_list = []
    for i, line in enumerate(lines):
        print i, line
        line = line.split(',')

        max = 0.0
        max_i = -1
        for j, txt in enumerate(line):
            if float(txt) > max:
                max = float(txt)
                max_i = j

        print max, max_i
        result_list.append(max_i)

    print result_list

    os.chdir(data_path)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    os.chdir(folder_name)

    for i_mkdir in xrange(50):
        if not os.path.exists(str(i_mkdir + 1)):
            os.mkdir(str(i_mkdir + 1))

    # os.mkdir('1')
    # os.mkdir('2')
    # os.mkdir('3')
    # os.mkdir('4')
    # os.mkdir('5')
    # os.mkdir('6')
    # os.mkdir('7')
    # os.mkdir('8')
    # os.mkdir('9')
    # os.mkdir('10')

    for i, num in enumerate(result_list):
        if num != -1:
            # if i / file_num == i_kosode - group_num * 10 - 1:
            file_name = 'image' + str(i % file_num + 1) + '.jpg'
            shutil.copy('../group' + str(group_num) + '/' + str(i_kosode) + '/dzc_output_files/15/' + file_name, str(num+1) + '/' + str(i_kosode) + '_' + str(i) + '.jpg')

    os.chdir('../../../')
