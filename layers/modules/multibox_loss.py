# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, log_sum_exp
import math


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos         # negative 와 positive 를 3: 1 비율로 하기 위함
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, priors = predictions
        '''
        loc_data  = batch_size(8) 만큼의 각각 이미지에 대한 피쳐맵에 대한 그리드 셀 (8732) 별로 
                    (x1, y1, x2, y2) 4개의 위치 값을 예측
        conf_data = batch_size(8) 만큼의 각각 이미지에 대한 피쳐맵에 대한 그리드 셀 (8732) 별로
                    2 개의 클래스에 대한 confidence 값을 예측
        priors    = default box 
        '''
        # print(loc_data)     # [torch.cuda.FloatTensor of size 8x8732x4 (GPU 0)]
        # print(conf_data)    # [torch.cuda.FloatTensor of size 8x8732x2 (GPU 0)]
        # print(priors)       # [torch.cuda.FloatTensor of size 26196x4 (GPU 0)]

        num = loc_data.size(0);
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        # conf_t = torch.FloatTensor(num, num_priors)
        '''
        아래 크기의 loc_target 과 conf_target 변수를 선언하고 match 함수를 통해 해당 값을 채우게 됨
        '''
        # print(loc_t)  # [torch.cuda.FloatTensor of size 8x8732x4 (GPU 0)]
        # print(conf_t) # [torch.LongTensor of size 8x8732]

        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data

            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

            '''
            그럼 이 시점에  loc_target 과 conf_target 변수 내에 
            8732개의 박스 별 객체 위치(loc)와 객체 있는지 여부(conf) 값들이 있겠지?
            '''

            ################################################
            # 이미지 중에 사람 객체가 없을 경우 (x1, y1, x2, y2, name) 을 0으로 세팅했기 때문에
            # x2 값이 제로 일경우 해당 이미지의 conf 값을 제로로 변경해줌
            # if 0 in targets[idx][0, 3].data:
            #     conf_t[idx][:-1]    = 0
            #     loc_t[idx][:-1]     = 0.
            ################################################

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        '''
        8732개의 박스 중에 0보다 큰값을 갖는 (객체가 있는) 곳에 True(1) 마스킹을 하네?
        그럼 pos 에는 8732 box 에 객체가 있는 곳에만 1 로 세팅이 되어 있을거야.
        '''
        pos = conf_t > 0
        # print(pos)          # [torch.cuda.ByteTensor of size 8x8732 (GPU 0)]
        # print(pos.dim())    # 2

        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]

        '''
        print(pos.unsqueeze(pos.dim())) [8X8732 --> 8x8732x1] [1, 2, 3]
        가로로 있던 값을 [[1],
                       [2],
                       [3]] 세로로 늘여놨네?
        그리고  8 X8732X4를 8 X8732X4 로 expand_as 를 했고..
        [[1],         [[1, 1, 1, 1],
         [2],    -->   [2, 2, 2, 2],
         [3]]          [3, 3, 3, 3]] 이런식으로..
        '''
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        '''
        loc_data(8x8732x4) 에서 객체가 있는 위치에서의 loc 예측 값을 가져오고, 
        아래는 예측 값이 18개가 있을 때
        '''
        loc_p = loc_data[pos_idx].view(-1, 4)
        # print(loc_p)  # [torch.cuda.FloatTensor of size 18x4 (GPU 0)]

        '''
        동일하게 target 에서 값을 추출해서 L1_loss 를 구해줌
        '''
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        '''
        batch_conf.gather(1, conf_t.view(-1, 1) 는 이렇게 값이 있을 때,
         8.0319  0.5783
         8.7334  0.8604
        [torch.cuda.FloatTensor of size 2x2 (GPU 0)]
        
        아래나온 숫자 인덱스에 맞게 값을 추출
        Variable containing:
        0
        0
        [torch.cuda.LongTensor of size 2x1 (GPU 0)]
        
        만일 0, 1 이 있었다면 8.0319 0.8604 두 값이 선택됨
        Variable containing:
          8.0319
          8.7334
        [torch.cuda.FloatTensor of size 2x1 (GPU 0)]
        '''
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # print(batch_conf)  # [torch.cuda.FloatTensor of size 69856x2 (GPU 0)]

        # print(batch_conf[0:2])
        # print(batch_conf.gather(1, conf_t.view(-1, 1)))
        # print(conf_t.view(-1, 1))

        '''
        앞 수식에서 클래스를 분류하고, 뒷 수식에서 실제 해당하는 각 셀이 갖는 클래스 값을 빼서 loss_c 를 구하고 있음
        '''
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # print(loss_c)

        # Hard Negative Mining
        '''
        하드 네가티브 마이닝을 위해 일단 객체가 존재하는 영역을 0으로 마스킹하고,
        loss_c.view(num, -1) 를 통해 batch_size X 8732 로 다시 쪼갬
        Variable containing:
         0.0000  0.0005  0.0007  ...   0.0003  0.0000  0.0002
         0.0000  0.0004  0.0007  ...   0.0000  0.0000  0.0003
         0.0000  0.0008  0.0007  ...   0.0000  0.0000  0.0000
                  ...             ⋱             ...          
         0.0000  0.0004  0.0006  ...   0.0000  0.0000  0.0000
         0.0000  0.0003  0.0004  ...   0.0000  0.0000  0.0000
         0.0000  0.0011  0.0016  ...   0.0000  0.0000  0.0003
        [torch.cuda.FloatTensor of size 8x8732 (GPU 0)]
        '''
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        # print(loss_c)  # [torch.cuda.FloatTensor of size 8x8732 (GPU 0)]

        '''
        이렇게 되면, 8장의 사진 중에 사람은 아니지만 사람으로 오검출 된 것들을 알 수가 있고, 
        이를 정렬하여 가장 사람과 유사한 오검출을 Negative 로 설정 
        
         8308  7395  7157  ...   8385  8443  8445
         6701  2483  6815  ...   8729  8730  8731
         3015  2851  3003  ...   8729  8730  8731
               ...          ⋱          ...       
         8341  8281  8655  ...   7022  8283  8343
         8639  8284  8638  ...   8654  8710  8726
         8248  8651  8249  ...   8670  8710  8717
        [torch.cuda.LongTensor of size 8x8732 (GPU 0)]
        '''
        _, loss_idx = loss_c.sort(1, descending=True)  # 로스를 내림차순 정렬 후에 인덱스를 추려냄
        # print(loss_idx, loss_c.type)

        '''
        이미 인덱스가 잘 정리되었을 건데 왜 또 sort 를 하지?
        아래 쪽에서 positive  3개 일때 9개의 negative 를 뽑는데 이럴 경우 0~ 9번 박스에 가장 큰 값이 오게 하기 위함.
        위에 8308 번이 가장 큰값인데. 정렬을 한번 더해서 인덱스를 뽑으면, 8308 번은 0번 인덱스를 갖게됨.
         8709  2464  1675  ...   8628  8514  2095
         8700   948  1124  ...   8730  8731  8596
         8714  2574  2433  ...   8730  8685  8731
               ...          ⋱          ...       
         2518  4286  3457  ...   8720  8589  8630
         8688  3059  2112  ...   8730  8731  8632
         8691  2541   781  ...   7792  8536  7680
        [torch.cuda.LongTensor of size 8x8732 (GPU 0)]
        '''
        _, idx_rank = loss_idx.sort(1)
        # print(idx_rank)

        '''
        각 이미지 별 positive 개수를 만들어냄
            9
            9
           15
           15
           18
            4
           14
            8
        [torch.cuda.LongTensor of size 8x1 (GPU 0)]
        '''
        num_pos = pos.long().sum(1, keepdim=True)
        # print(num_pos)

        '''
            3
           45
           39
           18
           27
           39
           54
            9
        [torch.cuda.LongTensor of size 8x1 (GPU 0)]
        '''
        # print(self.negpos_ratio*num_pos)
        # print(pos.size(), pos.size(1)-1)  # torch.Size([8, 8732]) 8731

        '''
        각 이미지 별 positive 개수가 전체 박스 수를 넘을 수 없기 때문에 clamp 를 이용해서 max 값을 넘지 않도록 해줌
           36
           18
           78
           63
           33
           15
           12
           45
        [torch.cuda.LongTensor of size 8x1 (GPU 0)]
        '''
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # print(num_neg)

        '''
           36    36    36  ...     36    36    36
            9     9     9  ...      9     9     9
            9     9     9  ...      9     9     9
               ...          ⋱          ...       
           21    21    21  ...     21    21    21
            3     3     3  ...      3     3     3
           15    15    15  ...     15    15    15
        [torch.cuda.LongTensor of size 8x8732 (GPU 0)]
        '''
        # print(num_neg.expand_as(loss_idx))

        '''
        A = [1, 2, 3, 4, 5], B = [3, 3, 3, 3, 3]
        A < B  를 하면 
        [1, 1, 0, 0, 0] 이 나온다.
        '''

        '''
        위 쪽에서 0~negative 개수 까지 가장 큰 스코어를 갖는 녀석을 모아놨기 때문에 
        num_neg 보다 작은 것들만 1로 세팅하면 negative 인덱스를 뽑을 수 있음
        '''
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # neg = loss_idx < num_neg.expand_as(loss_idx)
        # print(neg)


        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # print(pos)
        # print(pos.unsqueeze(2))
        # print(pos_idx)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
