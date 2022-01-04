import os
from collections import OrderedDict
import argparse
import time
import datetime
import math
import logging
from pathlib import Path

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from models.retinaface import RetinaFace


parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset',
                    default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25',
                    help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3,
                    type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None,
                    help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./result/',
                    help='Location to save checkpoint models')

args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123)  # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = os.path.join(args.save_folder, args.network)

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# config log =============================================================
log_path = f'{save_folder}/train.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# tensorboard config
tb_writer = SummaryWriter(Path(save_folder) / 'log')

# ========================================================================

net = RetinaFace(cfg=cfg)
logger.debug("Printing net...")
logger.debug(net)


if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = False


optimizer = optim.SGD(net.parameters(), lr=initial_lr,
                      momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

# load resume weight =====================================================
if args.resume_net is not None:
    logger.debug('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    for key in state_dict.keys():
        if key == 'model':
            net.load_state_dict(state_dict[key])
        if key == 'optimizer':
            optimizer.load_state_dict(state_dict[key])
        # if key == 'scaler':
            # scaler.load_state_dict(state_dict[key])
        # if key == 'scheduler':
        #     scheduler.load_state_dict(state_dict[key])
# ========================================================================


def train():
    net.train()

    logger.debug('Loading Dataset...')
    train_dataset = WiderFaceDetection(
        training_dataset, preproc(img_dim, rgb_mean))
    train_dataloader = data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)

    epoch_begin = 0 + args.resume_epoch
    epoch_size = math.ceil(len(train_dataset) / batch_size)

    start_iter = args.resume_epoch * epoch_size if args.resume_epoch > 0 else 0
    max_iter = max_epoch * epoch_size

    iter_num = start_iter
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    # train ==================================================================================================
    for epoch in range(epoch_begin, max_epoch):
        for images, targets in train_dataloader:
            t0 = time.time()

            # 根据迭代数调整学习率
            iter_num += 1
            if iter_num in stepvalues:
                step_index += 1
            lr = adjust_learning_rate(
                optimizer, gamma, epoch, step_index, iter_num, epoch_size)

            # load train data
            images, targets = images.cuda(), [anno.cuda() for anno in targets]

            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()

            t1 = time.time()
            # log
            batch_time = t1 - t0
            eta = int(batch_time * (max_iter - iter_num))

            info = {
                'Epoch': f'{epoch}/{max_epoch}',
                'step': f'{(iter_num % epoch_size) + 1}/{epoch_size}',
                'Iter': f'{iter_num}/{max_iter}',
                'loss': {
                    'Loc': float(f'{loss_l.item():.4f}'),
                    'Cla': float(f'{loss_c.item():.4f}'),
                    'Landm': float(f'{loss_landm.item():.4f}'),
                },
                'LR': float(f'{lr:.8f}'),
                'Batchtime': float(f'{batch_time:.4f}'),
                'ETA': str(datetime.timedelta(seconds=eta)),
            }
            logger.debug('train: ' + str(info))

            tb_writer.add_scalar(
                'loc_loss', info['loss']['Loc'], iter_num, walltime=None)
            tb_writer.add_scalar(
                'cls_loss', info['loss']['Cla'], iter_num, walltime=None)
            tb_writer.add_scalar(
                'landm_loss', info['loss']['Landm'], iter_num, walltime=None)

            if (epoch + 1) % 10 == 0 or ((epoch + 1) % 5 == 0 and (epoch + 1) > cfg['decay1']):
                # eval =====================================================================

                # ==========================================================================
                save_path = os.path.join(
                    save_folder, cfg['name'] + '_epoch_' + str(epoch + 1) + '.pth')
                save_state(save_path, model=net,
                           optimizer=optimizer, scaler=None)

    save_path = os.path.join(save_folder, cfg['name'] + '_Final.pth')
    save_state(save_path, model=net, optimizer=optimizer, scaler=None)

    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')

    # ======================================================================================================


def save_state(save_path, model=None, optimizer=None, scheduler=None, scaler=None):
    if isinstance(save_path, str):
        save_path = Path(save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(exist_ok=True, parents=True)

    status = OrderedDict(model=model.state_dict() if model is not None else None,
                         optimizer=optimizer.state_dict() if optimizer is not None else None,
                         scheduler=scheduler.state_dict() if scheduler is not None else None,
                         scaler=scaler.state_dict() if scaler is not None else None)

    torch.save(status, save_path)


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
