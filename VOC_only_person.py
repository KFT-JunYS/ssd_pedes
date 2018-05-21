import sys
import os

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random

data_path   = '/home/JunYS/data/VOCdevkit/'
types       = ['2007', '2012']
classes     = ['person']

for class_ in classes:
    for type_ in types:
        fd_total            = open(data_path + 'VOC' + type_ + '/ImageSets/Main/' + class_ + '_only_trainvaltest.txt', 'wt')
        fd_trainval         = open(data_path + 'VOC' + type_ + '/ImageSets/Main/' + class_ + '_trainval.txt', 'rt')
        fd_trainval_list    = fd_trainval.read().split('\n')

        for data in fd_trainval_list[:-1]:
            s_data = data.split(' ', 1)

            if s_data[1] == '-1':
                continue

            fd_total.write(s_data[0] + '\n')

        if type_ == '2007':
            fd_test         = open(data_path + 'VOC' + type_ + '/ImageSets/Main/' + class_ + '_test.txt', 'rt')
            fd_test_list    = fd_test.read().split('\n')

            for data in fd_test_list[:-1]:
                s_data = data.split(' ', 1)

                if s_data[1] == '-1':
                    continue

                fd_total.write(s_data[0] + '\n')


for class_ in classes:
    for type_ in types:
        fd      = open(data_path + 'VOC' + type_ + '/ImageSets/Main/' + class_ + '_only_trainvaltest.txt', 'rt')

        for line in fd.read().split('\n')[:-1]:
            doc     = ET.parse(data_path + 'VOC' + type_ + '/Annotations/' + line + '.xml')
            root    = doc.getroot()

            # print(ET.dump(root))
            # print("################################")

            for obj in root.findall('object'):
                name = obj.find('name').text

                if name != 'person':
                    # 해당 태그를 삭제한다
                    root.remove(obj)

            # print(ET.dump(root))
            outfile = os.path.join(data_path, 'VOC' + type_ + '/Annotations', line + ".xml")
            # print(outfile)

            print("Generating annotation xml file of picture: ", line)
            ET.ElementTree(root).write(outfile)


for class_ in classes:
    for type_ in types:
        train_fd    = open(data_path + 'VOC' + type_ + '/ImageSets/Main/train.txt', 'wt')
        val_fd      = open(data_path + 'VOC' + type_ + '/ImageSets/Main/val.txt', 'wt')
        test_fd     = open(data_path + 'VOC' + type_ + '/ImageSets/Main/test.txt', 'wt')
        fd          = open(data_path + 'VOC' + type_ + '/ImageSets/Main/' + class_ + '_only_trainvaltest.txt', 'rt')

        fd_list = fd.read().split('\n')

        for data in fd_list[:-1]:

            seed = random.random()

            if seed > 0.4:
                seed = random.random()
                if seed > 0.4:
                    train_fd.write(data + '\n')
                else:
                    val_fd.write(data + '\n')
            else:
                test_fd.write(data + '\n')