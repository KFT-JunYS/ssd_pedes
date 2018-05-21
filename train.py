import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, VOC, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES, BaseTransform
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time
import eval
import math

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='VOC', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
# parser.add_argument('--resume', default='/home/JunYS/PycharmProjects/SSD_caltech/weights/ssd300_2014_4315.pth', type=str, help='Resume from checkpoint')
parser.add_argument('--resume', default='/home/JunYS/PycharmProjects/SSD_caltech/TEST.pth', type=str, help='Resume from checkpoint')
# parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=1000000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')

# 학습도중 Inf 혹은 Nan 으로 발산하게 될 경우 learning rate 을 줄여준다.
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

# weight decay 가 큰 값일 수록 가중치에 큰 패널티가 부과된다. 보통의 경우 W 가 너무 커지는 것을 방지하기 위해 사용.
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.5, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=True, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
args = parser.parse_args()

iter_continue = 0   # 이어서 돌릴때, weight 값도 이어서 저장할 수 있도록 0 부터 시작하지 않도록
                    # step_index 도 바꿔주자. 12만번 돌린 상태라면 해당 값은 6!

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# cfg = (v1, v2)[args.version == 'v2']
cfg = 'VOC'
# VOC -->v2_512 (cfg, prior_box.py, ssd.py, augmentations.py, BaseTransform func)
# ssd --> ssd512 (train.py, eval.py)
# ssd_dim = (300, 300) --> 512

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
train_sets = [('Kftec', 'train')]     # for KFTEC
# train_sets = [('2014_instance', 'train')]
ssd_dim = 300  # only support 300 now
means = (104, 117, 123)  # only support voc now
num_classes = len(VOC_CLASSES) + 1

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_ssd('train', ssd_dim, num_classes)
# net = ssd_net

# VGG 16 의 feature 는 잘 학습되었다고 가정하고, 해당 파라미터들은 학습시키지 않는다.
ignored_params = list(map(id, ssd_net.vgg.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, ssd_net.parameters())

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net = net.cuda()

def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

for name in net.children():
    print(name)

# # 아래 처럼 하더라도 vgg 파라미터 갱신되는 이슈가 있음
# for param in net.module.vgg.parameters():
#     param.require_grad = False
#
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=args.momentum, weight_decay=args.weight_decay)

# # vgg 파라미터 제외한 업데이트
# optimizer = optim.SGD(base_params, lr=args.lr,
#                       momentum=args.momentum, weight_decay=args.weight_decay)

# vgg 파라미터 포함한 업데이트
optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)


def train():
    net.train()
    print('Loading Dataset...')

    # dataset     = VOCDetection(args.voc_root, train_sets, SSDAugmentation(ssd_dim, means), AnnotationTransform())
    dataset = VOCDetection(args.voc_root, [('Kftec', 'train')], SSDAugmentation(ssd_dim, means), AnnotationTransform())
    dataset_val = VOCDetection(args.voc_root, [('Kftec', 'val')], BaseTransform(ssd_dim, means), AnnotationTransform(keep_difficult=True))

    print(len(dataset), len(dataset_val))

    epoch           = 1
    epoch_size      = len(dataset) // args.batch_size
    epoch_size_val  = len(dataset_val) // args.batch_size

    print('Training SSD on', dataset.name)
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros(1).cpu(),
            # Y=torch.zeros((1, 2)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD train-val Loss',
                # legend=['Loss_train', 'Loss_val']
                legend=['Loss_train']
            )
        )

        lot_ap = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros(1).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='mAP',
                title='Current SSD val-set mAp',
                legend=['ap_val']
            )
        )

    batch_iterator      = None
    batch_iterator_val  = None

    data_loader     = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=False)


    data_loader_val = data.DataLoader(dataset_val, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=False)

    print(data_loader)
    print(data_loader_val)
    print(epoch_size, epoch_size_val)

    mAP = []


    for iteration in range(args.start_iter, args.iterations):
        iteration += iter_continue

        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator      = iter(data_loader)

        if (not batch_iterator_val) or (iteration % epoch_size_val == 0):
            batch_iterator_val  = iter(data_loader_val)

        if iteration % epoch_size == 0 and iteration > 0:
            epoch += 1


        # 통상적으로 20 epochs 당 0.1 정도로 줄임 or 5에폭당 0.5 정도 줄임
        # 하지만, SSD 기본 코드가 lr decay 를 [80000, 100000, 120000] 에 대해 0.1 씩 세번만 진행한것으로 보아
        # 일률적으로 적용하는 것보다는 lr 고정 후 추이를 지켜보고 적용하는 것이 좋을 것으로 판단
        # augmentation 을 통해 데이터가 96배 정도 뻥튀기 되었다고 가정
        # if iteration % (epoch_size * 5 * 96) == 0 and iteration > 0:
        #     step_index += 1
        #     cur_lr = adjust_learning_rate(optimizer, args.gamma, step_index)
        #     print('##################### cur_lr = ', cur_lr)

        # load train data
        images, targets         = next(batch_iterator)
        images_val, targets_val = next(batch_iterator_val)

        if args.cuda:
            images      = Variable(images.cuda())
            images_val  = Variable(images_val.cuda())
            targets     = [Variable(anno.cuda(), volatile=True) for anno in targets]
            targets_val = [Variable(anno.cuda(), volatile=True) for anno in targets_val]
        else:
            images      = Variable(images)
            images_val  = Variable(images_val)
            targets     = [Variable(anno, volatile=True) for anno in targets]
            targets_val = [Variable(anno, volatile=True) for anno in targets_val]

        # forward
        t0      = time.time()
        out     = net(images)
        out_val = net(images_val)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c

        loss.backward()
        optimizer.step()

        t1 = time.time()

        if iteration % 10 == 0:
        # if iteration >= 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' - ' + repr(epoch) + 'epoch'
                  ' || Loss: %.4f || loc: %.4f, || conf: %.4f' % (loss.data[0], loss_l.data[0], loss_c.data[0]), end=' ')

            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())

            # # 디버그를 위해 vgg weight 갱신되는지 확인하기 위함
            # for index__, datadata in enumerate(ssd_net.vgg.parameters()):
            #     print(datadata[0])
            #     break

        # if iteration % epoch_size == 0:
        #     loss_l_val, loss_c_val = criterion(out_val, targets_val)
        #     loss_val = loss_l_val + loss_c_val
        #
        #     print(' || Loss_val: %.4f || loc_val: %.4f, || conf_val: %.4f' % (loss_val.data[0], loss_l_val.data[0], loss_c_val.data[0]), end=' ')

        if args.visdom and math.isinf(loss) == False: # and math.isinf(loss_val) == False:
            viz.line(
                X=torch.ones((1, 1)).cpu() * iteration,
                # X=torch.ones((1, 2)).cpu() * iteration,
                # Y=torch.Tensor([loss_l.data[0] + loss_c.data[0], loss_l_val.data[0] + loss_c_val.data[0]]).unsqueeze(0).cpu(),
                Y=torch.Tensor([loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )

        if iteration % epoch_size == 0:
            print('Saving state, iter:', iteration)
            file_path = 'weights/ssd300_2014_' + repr(iteration) + '.pth'
            torch.save(ssd_net.state_dict(), file_path)

            net_val = build_ssd('test', ssd_dim, num_classes)  # initialize SSD
            net_val.load_state_dict(torch.load(file_path))
            net_val.eval()

            ap_val = eval.test_net('eval/', net_val, args.cuda, dataset_val,
                          BaseTransform(ssd_dim, means), 5, ssd_dim, thresh=0.01)

            mAP.append(ap_val)

            if np.mean(mAP) > ap_val:
                step_index += 1
                cur_lr = adjust_learning_rate(optimizer, args.gamma, step_index)
                print('##################### cur_lr = ', cur_lr)

            viz.line(
                X=torch.ones((1, 1)).cpu() * iteration,
                Y=torch.Tensor([ap_val]).unsqueeze(0).cpu(),
                win=lot_ap,
                update='append'
            )

    torch.save(ssd_net.state_dict(), args.save_folder + '' + args.version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    train()



'''
[to do list]
1) 큰 데이터와 작은 데이터에 대해서는 1:2 비율까지만 사용하고 있는데 이를 1:4 까지 사용하도록 변경하고,
   2:1 3:1 비율은 불필요하므로 제거
   
2) VOC + Caltech 1class classification
  - expand + rand hue + rand ar : VOC person only (20 --> 1 class 로 변경 시 기존 결과가 유지되는지?)
  - expand + rand hue + rand ar : VOC person only + added caltech data set 
    (caltech 데이터 추가 시 기존 VOC 결과 감소 없이 caltech 을 잘 측정하는지?)
    
3) VOC + Caltech 20class classification
  - VOC 베이스에 단순히 person 데이터만 추가하는 효과
'''