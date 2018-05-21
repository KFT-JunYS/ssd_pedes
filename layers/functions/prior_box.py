from __future__ import division
import torch
from math import sqrt as sqrt
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.min_dim = cfg['min_dim']

        #################
        self.image_size = cfg['image_size']
        self.begins = [4, 8, 8, 16, 80, 144]  # inserted by scjung
        #################

        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        # if self.version == 'v2':

        #################
        # a_r = self.min_dim[0] / self.min_dim[1]  # added by scjung: 300/400
        a_r = self.min_dim / self.image_size  # added by scjung: 300/400
        #################

        if self.version == 'VOC':
            for k, f in enumerate(self.feature_maps):

                # for i, j in product(range(f), repeat=2):
                for i, j in product(range(f[0]), range(f[1])):  # modified by scjung
                    # f_k = self.image_size / self.steps[k]
                    # # unit center x,y
                    # cx = (j + 0.5) / f_k
                    # cy = (i + 0.5) / f_k
                    #
                    # # aspect_ratio: 1
                    # # rel size: min_size
                    # s_k = self.min_sizes[k]/self.image_size
                    # mean += [cx, cy, s_k, s_k]
                    #
                    # # aspect_ratio: 1
                    # # rel size: sqrt(s_k * s_(k+1))
                    # s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    # mean += [cx, cy, s_k_prime, s_k_prime]
                    #
                    # # rest of aspect ratios
                    # for ar in self.aspect_ratios[k]:
                    #     # mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]    # width
                    #     mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]    # height
                    #     # mean += [cx, cy, s_k_prime*sqrt(ar), s_k_prime/sqrt(ar)]    # width
                    #     mean += [cx, cy, s_k_prime/sqrt(ar), s_k_prime*sqrt(ar)]    # height
                    #
                    # f_k = self.min_dim / self.steps[k][0]
                    # f_k_x = self.image_size / self.steps[k][1]  # added by scjung

###################################################

                    f_k = self.min_dim / self.steps[k]
                    f_k_x = self.image_size / self.steps[k] # added by scjung

                    # unit center x,y
                    cx = (j + 0.5) / f_k_x  # modified by scjung
                    cy = (i + 0.5) / f_k

                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k = self.min_sizes[k] / self.min_dim
                    s_k_x = self.min_sizes[k] / self.image_size  # added by scjung
                    mean += [cx, cy, s_k_x, s_k]

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.min_dim))
                    s_k_x_prime = sqrt(s_k_x * (self.max_sizes[k] / self.image_size))  # added by scjung
                    mean += [cx, cy, s_k_x_prime, s_k_prime]  # modified by scjung

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k_x * sqrt(ar), s_k / sqrt(ar)]  # modified by scjung
                        mean += [cx, cy, s_k_x / sqrt(ar), s_k * sqrt(ar)]  # modified by scjung

###################################################

                    # # f_k_x = self.min_dim[1] / self.steps[k] # added by scjung
                    # f_k = self.min_dim / self.steps[k]  # added by scjung
                    # b_x = self.begins[k] / self.steps[k]
                    # # unit center x,y
                    # cx = (j + b_x) / f_k * a_r  # modified by scjung
                    # cy = (i + b_x) / f_k
                    #
                    # # aspect_ratio: 1
                    # # rel size: min_size
                    # s_k = self.min_sizes[k] / self.min_dim
                    # s_k_x = s_k  # * a_r
                    # mean += [cx, cy, s_k_x, s_k]
                    #
                    # # aspect_ratio: 1
                    # # rel size: sqrt(s_k * s_(k+1))
                    # # s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    # s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.min_dim))
                    # # s_k_prime = s_k * 1.25
                    #
                    # s_k_x_prime = s_k_prime  # * a_r
                    # mean += [cx, cy, s_k_x_prime, s_k_prime]
                    #
                    # # rest of aspect ratios
                    # for ar in self.aspect_ratios[k]:
                    #     # mean += [cx, cy, s_k_x * sqrt(ar), s_k / sqrt(ar)]
                    #     mean += [cx, cy, s_k_x / sqrt(ar), s_k * sqrt(ar)]

        else:
            # original version generation of prior (default) boxes
            for i, k in enumerate(self.feature_maps):
                print(self.image_size, k)
                step_x = step_y = self.image_size/k
                for h, w in product(range(k), repeat=2):
                    c_x = ((w+0.5) * step_x)
                    c_y = ((h+0.5) * step_y)
                    c_w = c_h = self.min_sizes[i] / 2
                    s_k = self.image_size  # 300
                    # aspect_ratio: 1,
                    # size: min_size
                    mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                             (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    if self.max_sizes[i] > 0:
                        # aspect_ratio: 1
                        # size: sqrt(min_size * max_size)/2
                        c_w = c_h = sqrt(self.min_sizes[i] *
                                         self.max_sizes[i])/2
                        mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                 (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    # rest of prior boxes
                    for ar in self.aspect_ratios[i]:
                        if not (abs(ar-1) < 1e-6):
                            c_w = self.min_sizes[i] * sqrt(ar)/2
                            c_h = self.min_sizes[i] / sqrt(ar)/2
                            mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                     (c_x+c_w)/s_k, (c_y+c_h)/s_k]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
