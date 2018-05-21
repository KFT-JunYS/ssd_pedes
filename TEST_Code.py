# from itertools import product as product
# from math import sqrt as sqrt
#
# image_size      = 300
# feature_maps    = [38, 19, 10, 5, 3, 1]
# steps           = [8, 16, 32, 64, 100, 300]
# min_sizes       = [30, 60, 111, 162, 213, 264]
# max_sizes       = [60, 111, 162, 213, 264, 315]
# aspect_ratios   = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
#
# mean = []
# for k, f in enumerate(feature_maps):
#     # print(f)
#     for i, j in product(range(f), repeat=2):
#         f_k = image_size / steps[k]
#         # unit center x,y
#         cx = (j + 0.5) / f_k
#         cy = (i + 0.5) / f_k
#
#         # aspect_ratio: 1
#         # rel size: min_size
#         s_k = min_sizes[k] / image_size
#         mean += [cx, cy, s_k, s_k]
#
#         # aspect_ratio: 1
#         # rel size: sqrt(s_k * s_(k+1))
#         s_k_prime = sqrt(s_k * (max_sizes[k] / image_size))
#         mean += [cx, cy, s_k_prime, s_k_prime]
#
#         # rest of aspect ratios
#         for ar in aspect_ratios[k]:
#             mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
#             mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
#
#####################################################################################################################
#
''''''
''' 본 코드를 돌릴 시에 "annotation_cache" 폴더를 반드시 지울 것 '''

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import random
#
# data_path   = '/home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages'
# img_list    = os.listdir(data_path)
#
# train_fd    = open('/home/JunYS/data/VOCdevkit/VOC2014_instance/ImageSets/Main/train.txt', 'wt')
# val_fd      = open('/home/JunYS/data/VOCdevkit/VOC2014_instance/ImageSets/Main/val.txt', 'wt')
# test_fd     = open('/home/JunYS/data/VOCdevkit/VOC2014_instance/ImageSets/Main/test.txt', 'wt')
#
# TRAIN       = ['set00', 'set01', 'set02', 'set03', 'set04', 'set05']
#
# ''' 보통 칼텍 데이터는 0~5번을  train, 나머지 데이터를 test 로 사용한다. '''
# for i in range(len(img_list)):
#
#     # if img_list[i].split('_')[0] in TRAIN:
#     #     print(img_list[i], ' - ', 'Train')
#     # else:
#     #     print(img_list[i], ' - ', 'Val')
#
#     seed = random.random()
#
#     if img_list[i].split('_')[0] in TRAIN:
#         if seed > 0.005:
#             # train_fd.write(img_list[i][:-4] + ' -1\n')
#             train_fd.write(img_list[i][:-4] + '\n')
#         else:
#             val_fd.write(img_list[i][:-4] + '\n')
#     else:
#         # val_fd.write(img_list[i][:-4] + ' -1\n')
#         if i % 30 == 0:
#             test_fd.write(img_list[i][:-4] + '\n')
#
# ''' 전체 이미지에서 train/val 데이터를 비율에 맞춰 나누기 위함 '''
# # for i in range(len(img_list)):
# #     seed = random.random()
# #
# #     if seed > 0.1:
# #         # train_fd.write(img_list[i][:-4] + ' -1\n')
# #         train_fd.write(img_list[i][:-4] + '\n')
# #     else:
# #         # val_fd.write(img_list[i][:-4] + ' -1\n')
# #         val_fd.write(img_list[i][:-4] + '\n')

#####################################################################################################################
#
''' caltechToVoc_10x.py 로 데이터 만든 이후 Train 및 기타 코드들을 동작하게 하기 위해 데이터 위치 조정 '''
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from lxml import etree, objectify
# #
# def visualize_bbox(xml_file, img_file):
#     import cv2
#     tree = etree.parse(xml_file)
#     # load image
#     image = cv2.imread(img_file)
#     # get bbox
#     for bbox in tree.xpath('//bndbox'):
#         coord = []
#         for corner in bbox.getchildren():
#             coord.append(int(float(corner.text)))
#         # draw rectangle
#         # coord = [int(x) for x in coord]
#         image = cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
#     # visualize image
#     cv2.imshow("test", image)
#     cv2.waitKey(0)
#
#
# if not os.path.exists('/home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages'):
#     os.mkdir('/home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
#
# if not os.path.exists('/home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations'):
#     os.mkdir('/home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
#
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set00/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set01/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set02/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set03/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set04/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set05/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set06/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set07/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set08/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set09/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/set10/frame/* /home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages')
#
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set00/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set01/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set02/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set03/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set04/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set05/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set06/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set07/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set08/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set09/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
# os.system('cp -r /home/JunYS/data/Caltech_pedestrian/VOC_type/annotations/set10/bbox/* /home/JunYS/data/VOCdevkit/VOC2014_instance/Annotations')
#
# # 데이터 정상적으로 뽑힌게 맞는지 확인
# data_path   = '/home/JunYS/data/VOCKftec/Annotations_TEST'
# img_list    = os.listdir(data_path)
#
# for i in range(len(img_list)):
#     # print(img_list[i].split('.')[0])
#
#     xml_file = "/home/JunYS/data/VOCKftec/Annotations_TEST/" + img_list[i].split('.')[0] + ".xml"
#     img_file = "/home/JunYS/data/VOCKftec/JPEGImages/" + img_list[i].split('.')[0] + ".png"
#
#     print(xml_file)
#     print(img_file)
#
#     visualize_bbox(xml_file, img_file)

#####################################################################################################################

# VOC_CLASSES = (  # always index 0
#     'person', 'BBBB')
#
# print(VOC_CLASSES[0])
# print(VOC_CLASSES[1])
# print(len(VOC_CLASSES))
#
# VOC_CLASSES1 = (  # always index 0
#     'person',)
#
# print(VOC_CLASSES1[0])
# print(len(VOC_CLASSES1))

#####################################################################################################################

''' Caltech 보행자로 테스트한 결과물 (SSD, ACF 간의 차이 비교) '''

# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import csv
# import math
#
# PR = False  # PR curve : true / Miss rate : False
#
# # with open('../SSD_caltech/ssd300_120000/test/person_test_pr.pkl', 'rb') as f:
# with open('../SSD_caltech/ssd300_120000/val/person_val_pr.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# recall      = data['rec']
# precision   = data['prec']
# precision.sort()
# precision = precision[::-1]
#
# print('pkl - ', np.array(precision).shape)
#
# # [recall, precision]
# fd = open('AcfCaltech_recall_precision.csv', 'r', encoding='utf-8')
# rdr = csv.reader(fd)
#
# for i, line in enumerate(rdr):
#     # for pr curve
#     # if i == 0:
#     #     recall_     = line
#     # else:
#     #     precision_  = line
#
#     # for miss rate curve
#     if i == 0:
#         recall_     = np.array(line).astype(float)
#     else:
#         precision_  = np.array(line).astype(float)
#         precision_.sort()
#         precision_ = precision_[::-1]
#
#
# fig, ax = plt.subplots()
#
# if PR:
#     # for pr curve
#     ax.plot(recall, precision, 'k--', color='blue', label='SSD')
#     ax.plot(recall_, precision_, 'k-', color='red', label='ACF')
# else:
#
#     # for miss rate curve
#     ax.loglog(1-precision, 1-recall, 'k--', color='blue', label='SSD')
#     ax.loglog(1-precision_, 1-recall_, 'k-', color='red', label='ACF')
#
#     # miss rate 계산 하는 곳
#     init_value = -2
#     mr_value = []
#     for fppi, mr in zip(1-precision, 1-recall):
#         if fppi >= pow(10, init_value):
#             mr = math.log10(max(mr, 1e-6))
#             print(mr)
#             mr_value.append(mr)
#
#             if init_value == 0:
#                 break
#             init_value += 0.25
#
#     print( math.exp(np.mean(mr_value)) )
#
#
#
#
# plt.legend(loc='upper right')
#
# if PR:
#     # for pr curve
#     ax.set_xlabel("recall")
#     ax.set_ylabel("precision")
# else:
#     # for miss rate curve
#     ax.set_xlabel("fppi")
#     ax.set_ylabel("miss rate")
#
# if PR:
#     plt.savefig('fig_caltech_Pr.jpg')
# else:
#     plt.savefig('fig_caltech_missrate.jpg')
#
# plt.show()
#
# print("finish!")
# fd.close()

#####################################################################################################################

''' Caltech data set 의 평균 RGB 값 계산 - 미완성 '''
# import os
# import cv2 as cv
# import numpy as np
# import torch
#
# ''' 샘플코드
# data = np.array(torch.rand(4, 4, 3))
# print(data)
#
# print('###')
# print(np.mean(np.mean(data, axis=1), axis=0))
# '''
#
# # data_path = '/home/JunYS/data/VOCdevkit/VOC2014_instance/JPEGImages'    # for Caltech
# data_path_2007 = '/home/JunYS/data/VOCdevkit/VOC2007/JPEGImages'        # for VOC
# data_path_2012 = '/home/JunYS/data/VOCdevkit/VOC2007/JPEGImages'        # for VOC
#
# img_list    = os.listdir(data_path_2007)
# list        = []
#
# for img in img_list:
#     image       = cv.imread(data_path_2007 + '/' + img)
#     img_np      = np.array(image)
#
#     rgb_mean    = np.mean(np.mean(img_np, axis=1), axis=0)
#     # print(rgb_mean.shape)
#
#     list.append(rgb_mean)
#
# list_np = np.array(list)
# list_np.mean(axis=1)
# print(list_np.mean(axis=1))

#####################################################################################################################

''' 
clamp 는 입력 값이 min 을 넘지 않을 경우 min 으로 고정, max 를 넘을 경우 max 로 고정하게 해주는 구나...
    ex) torch.clamp(input=3.6, min=2, max=4) = 3.6
        torch.clamp(input=1.6, min=2, max=4) = 2
        torch.clamp(input=4.5, min=2, max=4) = 4 
'''
# import torch
#
# data = torch.rand(4, 4)
# data[0] = 3.6
#
# print(data)
# print(torch.clamp(data, 2, 3))

''' 특정열 없애버리고 싶을 때 사용 '''

# import numpy as np
#
# A = [
#     [1, 2, 3, 4, 5],
#     [6, 7, 8, 9, 0]
# ]
#
# B = [
#         [[1, 2, 3, 4, 5]],
#         [[6, 7, 8, 9, 0]],
# ]
#
# print('A = ', A, 'shape = ', np.array(A).shape)
# print('B = ', B, 'shape = ', np.array(B).shape)
#
# C = np.array(B).squeeze(axis=1)
# print('C = ', C, 'shape = ', C.shape)

# import math
# print(math.isinf(1))

#####################################################################################################################

# from numpy import random
#
# for i in range(100):
#     print(random.uniform(1, 4))

#####################################################################################################################

''' XML 특정 태그 삭제 방법 '''

# import sys
# if sys.version_info[0] == 2:
#     import xml.etree.cElementTree as ET
# else:
#     import xml.etree.ElementTree as ET
#
# doc     = ET.parse('/home/JunYS/data/VOCdevkit/VOC2007/Annotations/000001.xml')
# root    = doc.getroot()
#
# print(ET.dump(root))
# print("################################")
#
# for obj in root.findall('object'):
#     name = obj.find('name').text
#
#     if name != 'person':
#         # 해당 태그를 삭제한다
#         root.remove(obj)
#
# print(ET.dump(root))

#####################################################################################################################

''' 특정 weight 만 업데이트 진행하기 위한 테스트 '''
# TEST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# NET  = [1, 2, 3, 12, 13, 14, 15]
#
# ignored_params = list(map(id, TEST))
# print(ignored_params)
#
# base_params = filter(lambda p: id(p) not in ignored_params, NET)
#
# for data in base_params:
#     print(data)

#####################################################################################################################

''' 21개 클래스 중에 자신이 원하는 특정 클래스의 weight 값만 읽어오기 위한 테스트 '''
# import torch
# import enum
# import numpy as np
# from collections import OrderedDict
#
# data = torch.load('/home/JunYS/PycharmProjects/SSD_caltech/weights/ssd300_mAP_77.43_v2.pth',
#                   map_location=lambda storage, loc: storage)
#
# order_dict_data = OrderedDict()
#
# class TYPE(enum.Enum):
#     BACKGROUND  = 0
#     AEROPLANE   = 1
#     BICYCLE     = 2
#     BIRD        = 3
#     BOAT        = 4
#     BOTTLE      = 5
#     BUS         = 6
#     CAR         = 7
#     CAT         = 8
#     CHAIR       = 9
#     COW         = 10
#     DININGTABLE = 11
#     DOG         = 12
#     HORSE       = 13
#     MOTORBIKE   = 14
#     PERSON      = 15
#     POTTEDPLANT = 16
#     SHEEP       = 17
#     SOFA        = 18
#     TRAIN       = 19
#     TVMONITOR   = 20
#
# TYPE_NUMBER = len(TYPE)
#
# for key, value in data.items():
#
#     list = []
#
#     if key[:4] == 'conf':
#         if np.array(value).ndim == 4:
#             back = value[TYPE.BACKGROUND.value::TYPE_NUMBER, :, :, :]
#             pers = value[TYPE.PERSON.value::TYPE_NUMBER, :, :, :]
#             for i in range(len(back)):
#                 list.append(back[i])
#                 list.append(pers[i])
#
#             list = torch.stack(list, 0)
#
#         else:
#             back = value[TYPE.BACKGROUND.value::TYPE_NUMBER]
#             pers = value[TYPE.PERSON.value::TYPE_NUMBER]
#             for i in range(len(back)):
#                 list.append(back[i])
#                 list.append(pers[i])
#
#         order_dict_data[key] = torch.FloatTensor(list)
#     else:
#         order_dict_data[key] = value
#
# torch.save(order_dict_data, 'TEST.pth')

#####################################################################################################################

''' Scale-aware Fast R-CNN for pedestrian Detection 논문 내의 한 수식 테스트 '''
# import math
#
# h = [2, 2, 2, 10, 10, 10]
# avg_h = sum(h) / len(h)
#
# A = 1 / math.exp(-(h[0]-avg_h))
# B = 1 / math.exp(-(h[5]-avg_h))
# print(A)
# print(B)

#####################################################################################################################

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import random
#
# data_path   = '/home/JunYS/data/VOCKftec/Annotations_ori'
# xml_list    = os.listdir(data_path)
#
# train_fd    = open('/home/JunYS/data/VOCKftec/ImageSets/Main/train.txt', 'wt')
#
# for i in range(len(xml_list)):
#     train_fd.write(xml_list[i][:-4] + '\n')

#####################################################################################################################

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import random
#
# data_path   = '/home/JunYS/data/VOCKftec/Annotations_TEST'
# xml_list    = os.listdir(data_path)
#
# val_fd      = open('/home/JunYS/data/VOCKftec/ImageSets/Main/val.txt', 'wt')
# test_fd     = open('/home/JunYS/data/VOCKftec/ImageSets/Main/test.txt', 'wt')
#
# for i in range(len(xml_list)):
#     seed = random.random()
#
#     if seed <= 0.02:
#         val_fd.write(xml_list[i][:-4] + '\n')
#     else:
#         test_fd.write(xml_list[i][:-4] + '\n')

#####################################################################################################################

''' float tensor '''
# import torch
# A = torch.cuda.FloatTensor([[0, 0]])
#
# print(A, A.shape)

#####################################################################################################################



