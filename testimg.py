from __future__ import print_function
import sys
import os
import argparse
import torch

import numpy as np
import cv2
from torch.autograd import Variable

from data import VOC_CLASSES, VOC

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_2014_17260.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--num_classes', default=2, type=int,
                    help='number of classified object')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--img_file', default="/home/JunYS/set04_V010_I00420.png", help='Name of Image File')
parser.add_argument('--keep_aspect', action='store_true', help='Keep the aspect ratio') #type=bool -> flag
args = parser.parse_args()

print(args.img_file)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def visualize_bbox(image, detections):
    # get bbox
    scale = torch.Tensor(image.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = VOC_CLASSES[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            color = (i*10, 0, 255-i*10)
            image = cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), color, 2)
            image = cv2.putText(image, display_txt, (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 1, color)
            j+=1

    # visualize image
    cv2.imshow("test", image)
    cv2.waitKey(0)
    
def test_img():
    # load net
    from ssd import build_ssd
    size = VOC['image_size']
    size_y = VOC['min_dim']
    # num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, args.num_classes) # initialize SSD
    net.load_weights(args.trained_model)
    print('Finished loading model!')

    data_path   = '/home/JunYS/data/VOCKftec/Annotations'
    xml_list    = os.listdir(data_path)


    for i in range(len(xml_list)):
        img_file = '/home/JunYS/data/VOCKftec/JPEGImages/' + xml_list[i][:-4] + '.png'

        # load data
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        # image0 = cv2.imread(args.img_file, cv2.IMREAD_COLOR)
        # msize = min(image0.shape[:2])
        # image = image0[:msize,:msize,:]
        x = cv2.resize(image, (size, size_y)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)

        xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = net(xx)
        detections = y.data

        visualize_bbox(image, detections)

if __name__ == '__main__':
    test_img()
