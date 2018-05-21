# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
ddir = os.path.join(home,"data/")               # for KFTEC
# ddir = os.path.join(home,"data/VOCdevkit/")

# note: if you used our download scripts, this should be right
VOCroot = ddir # path to VOCdevkit root dir

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4


#SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
v2 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 111, 162, 213, 264],

    'max_sizes' : [60, 111, 162, 213, 264, 315],

    'aspect_ratios' : [[2, 3],     # 2:1 1:2
                       [2, 3, 4, 5],  # 2:1 1:2 3:1 1:3
                       [2, 3, 4, 5],  # 2:1 1:2 3:1 1:3
                       [2, 3, 4, 5],  # 2:1 1:2 3:1 1:3
                       [2, 3],     # 2:1 1:2
                       [2, 3]],    # 2:1 1:2

    # 'aspect_ratios': [[2, 3],  # 2:1 1:2 3:1 1:3
    #                   [2, 3],  # 2:1 1:2 3:1 1:3
    #                   [2, 3],  # 2:1 1:2 3:1 1:3
    #                   [2, 3],  # 2:1 1:2 3:1 1:3
    #                   [2, 3],  # 2:1 1:2
    #                   [2]],    # 2:1 1:2

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'v2',
}

########################
VOC_SC = {#400x300
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'min_dim': [300, 400],  # yx h-w
    'feature_maps': [[38, 50], [19, 25], [10, 13], [5, 7], [3, 5], [1, 3]], #400x300 yx

    # 'min_dim': [300, 533], #yx h-w
    # 'feature_maps': [[38, 67], [19, 33], [10, 17], [5, 9], [3, 7], [1, 5]], #533

    'steps': [8, 16, 32, 64, 64, 64],
    'begins' : [4, 8, 8, 16, 80, 144],
    'min_sizes': [25, 40, 59, 94, 161, 228],
    'max_sizes': [40, 59, 94, 161, 228, 322],
    'aspect_ratios' : [[2, 3],     # 2:1 1:2
               [2, 3, 4, 5],  # 2:1 1:2 3:1 1:3
               [2, 3, 4, 5],  # 2:1 1:2 3:1 1:3
               [2, 3, 4, 5],  # 2:1 1:2 3:1 1:3
               [2, 3],     # 2:1 1:2
               [2, 3]],    # 2:1 1:2
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
########################

VOC = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,

    # 'feature_maps': [[90, 160], [45, 80], [23, 40], [12, 20], [10, 18], [8, 16]], #1280 X 720
    # 'image_size': 1280,
    # 'feature_maps': [[38, 67], [19, 33], [10, 17], [5, 9], [3, 7], [1, 5]], #533
    # 'image_size': 533,
    'feature_maps': [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]], #300
    'image_size': 300,

    'min_dim': 300,

    # 'steps': [8, 16, 32, 64, 64, 64],         # 400
    # 'min_sizes': [25, 40, 59, 94, 161, 228],  # 400
    # 'max_sizes': [40, 59, 94, 161, 228, 322], # 400
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],

    'aspect_ratios' : [[2],     # 2:1 1:2
                       [2, 3],  # 2:1 1:2 3:1 1:3
                       [2, 3],  # 2:1 1:2 3:1 1:3
                       [2, 3],  # 2:1 1:2 3:1 1:3
                       [2],     # 2:1 1:2
                       [2]],    # 2:1 1:2

    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

#SSD512 CONFIGS
v2_512 = {

    # 'feature_maps': [[64, 114], [32, 57], [16, 29], [8, 15], [4, 8], [2, 4], [1, 3]],
    # 'image_size': 910,
    'feature_maps' : [[64, 64], [32, 32], [16, 16], [8, 8], [4, 4], [2, 2], [1, 1]],
    'image_size': 512,

    'min_dim' : 512,

    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [36, 77, 154, 230, 307, 384, 460],
    'max_sizes': [77, 154, 230, 307, 384, 460, 537],


    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'v2_512',
}


# use average pooling layer as last layer before multibox layers
v1 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim' : 300,

    'steps' : [8, 16, 32, 64, 100, 300],

    'min_sizes' : [30, 60, 114, 168, 222, 276],

    'max_sizes' : [-1, 114, 168, 222, 276, 330],

    # 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'aspect_ratios' : [[1,1,2,1/2],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],
                        [1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3],[1,1,2,1/2,3,1/3]],

    'variance' : [0.1, 0.2],

    'clip' : True,

    'name' : 'v1',
}
