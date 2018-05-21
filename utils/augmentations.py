import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import matplotlib.pyplot as plt
from data import v2, v1, VOC, v2_512


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        # image = cv2.resize(image, self.size)
        image = cv2.resize(image, (VOC['image_size'], VOC['min_dim']))
        # image = cv2.resize(image, (VOC['min_dim'][1], VOC['min_dim'][0]))

        # # for Debug
        # COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        #
        # for i in range(len(boxes)):
        #     # pt = [boxes[i][0] * 533, boxes[i][1] * 300, boxes[i][2] * 533, boxes[i][3] * 300]
        #     # pt = np.array(pt)
        #     pt = boxes[i] * 300
        #
        #     cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), COLORS[i % 3], 2)
        #
        # cv2.imshow('image', image / 255)
        # cv2.waitKey(0)

        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2): # 0, 1 둘중에 하나 나옴 numpy 의 random 이용
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in point form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    # 이미지 전체 영역에서 0.1 ~ 0.9 사이의 영역을 랜덤 크롭하고,  GT 박스 영역을 크롭한 이미지에 맞춰줌
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:

            # # for debugging for expand function
            # tmp = (boxes[:, 3] - boxes[:, 1]) * 300 / height
            # print([i for i in tmp if i < 30])

            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                # w = random.uniform(0.3 * width, width)
                # h = random.uniform(0.3 * height, height)
                #
                # # aspect ratio constraint b/t .5 & 2
                # if h / w < 0.5 or h / w > 2:
                #     continue

                # 보행자의 경우 가로 세로 비율이 거의 일정하므로 원본 이미지 비율인 1.33 : 1 을 고정으로 맞춘다. # for Kftec --> 1:1
                w = h = random.uniform(0.3 * height, height)
                # w = h * 16 / 9

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                # 찾아야 할 객체가 없으면 버림
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 2)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)

        ######################################################
        # print(boxes)
        tmp = (boxes[:, 3] - boxes[:, 1]) * 300 / expand_image.shape[0]
        # print(tmp)

        # 작은 이미지는 38 X 38 피쳐맵에서 걸러지는데 한 개 그리드 셀에서 검출할 수 있는 최소 크기는 30
        small = [i for i in tmp if i < 30]

        if len(small) != 0:
            return image, boxes, labels
        ######################################################

        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'), # hue (색상), saturation (탁함정도) 을 변경하기 위해 RGB --> HSV (Hue, Saturation, brightness Value) 로 변경
            RandomSaturation(), # 그림판에서 색 편집 > 위, 아래 (채도, 탁함정도) 로 약간의 변경
            RandomHue(),        # 그림판에서 색 편집 > 좌, 우 (색조) 로 약간의 변경
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]

        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise() # 색의 다양성을 위한 RGB 채널 변경 (우리는 필요X)

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):  # 0, 1 둘중에 하나 나옴 numpy 의 random 이용
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])

        im, boxes, labels = distort(im, boxes, labels)

        # return self.rand_light_noise(im, boxes, labels)
        return im, boxes, labels

'''
데이터 셋의 이미지를 
 - 확대하거나 
 - 좌우 반전시키거나
 - 사람의 영역이 0.1 ~ 0.9 까지만 포함시키게 한다거나

하는 방법을 이미지를 읽어올때 마다 적용시킨다. 
이렇게 할 경우 원본 이미지와 GT 가 들어갈 수 도 있고, 원본 이미지의 변형 형태와 변형된 GT 가 들어갈 수도 있다.

실제 데이터는 그냥 두고,
학습 과정에서 이와 같은 과정을 거치게 된다.

이러한 경우 원본 이미지 대비 여러가지 데이터들이 생성되기 때문에 데이터 셋의 전체 표현력이
올라가 최종 결과가 좋아지는 효과가 있다.
'''
class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        img_, boxes_, labels_ = self.augment(img, boxes, labels)

        return img_, boxes_, labels_

""" 변경점
1) expand 제거 (300 X 300 크기 내에 아주 작은 이미지가 들어감. 이럴 경우 사람이 너무 작아 판별이 어려워짐)
    - VOC 의 데이터 셋에는 큰 사람만 존재하기 때문에 작은 사람 검출을 위해 조정한 것으로 보임
    
2) 종횡비 1.33 : 1 을 고정
    - VOC 는 다양한 비율의 동,식물, 물체 등을 구분하기 위해 랜덤의 종횡비를 사용하였지만 사람의 경우 
      거의 일정한 비율로 존재하기 때문임

3) RandomLightingNoise 제거 --> 색의 다양성을 위한 RGB 채널 변경 (우리는 필요X)
    - 초록색 후미 등 과 같은 이상한 색이 나타남.. 예를 들어 앵무새의 경우 다양한 색을 가지기 때문에 이러한 코드가 
      들어간 듯 하나 Caltech 의 경우 이러한 색 변환은 학습에 방해가 됨
"""