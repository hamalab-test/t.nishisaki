import os
import shutil

judge_path = '../data/kosode_motif5/judge'
judge_name = 'test_LR_judge.txt'
data_path = 'data/kosode_motif2/test/kosode_smallfield_origin/sample'
folder_name = 'judge_result_sigmoid_epoch200'

os.chdir(judge_path)
f = open(judge_name, 'r')
lines = f.readlines()
f.close()

result_list = []
for i, line in enumerate(lines):
    # print i
    line = line.split(',')

    max = 0.0
    max_i = -1
    for j, txt in enumerate(line):
        if float(txt) > max:
            max = float(txt)
            max_i = j

    # print max, max_i
    result_list.append(max_i)

print result_list

os.chdir('../../../')

os.chdir(data_path)
os.mkdir(folder_name)
os.chdir(folder_name)

os.mkdir('1')
os.mkdir('2')
os.mkdir('3')
os.mkdir('4')
os.mkdir('5')
os.mkdir('6')
os.mkdir('7')
os.mkdir('8')
os.mkdir('9')
os.mkdir('10')

for i, num in enumerate(result_list):
    file_name = 'image' + str(i+1) + '.jpg'
    shutil.copy('../' + file_name, str(num+1))
