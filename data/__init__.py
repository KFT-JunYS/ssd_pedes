from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .config import *
import cv2
import numpy as np
from data import VOC, v2_512


def base_transform(image, size, mean):
    # x = cv2.resize(image, size).astype(np.float32)
    x = cv2.resize(image, (VOC['image_size'], VOC['min_dim'])).astype(np.float32)
    # print(np.array(x).shape)

    # x = cv2.resize(image, (VOC['min_dim'][1], VOC['min_dim'][0])).astype(np.float32)

    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)

    x -= mean
    x = x.astype(np.float32)

    # cv2.imshow('TEST', x * 255)
    # cv2.waitKey(0)

    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
