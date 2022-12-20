import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from wsol.regression import regression_timm
from dataloaders.LCHPInitDataset import LCHPInitDataset, collate_fn
from utils import log_msg, AverageMeter, ProgressMeter, TopKLocAccuracyMetric, TopKAccuracyMetric, ModelCheckpoint, compute_maxiou_onevsmulti_batch

from torch.cuda.amp import autocast, GradScaler



parser = argparse.ArgumentParser(description='PyTorch WSOL Training')

parser.add_argument('--cls-backbone', metavar='N', default='inception_v3')
parser.add_argument('--feature-index', default=-1, type=int)
parser.add_argument('--reg-backbone', metavar='N', default='inception_v3')
parser.add_argument('--norm', metavar='N', default='')

parser.add_argument('--dataset', default='CUB', help='dataset name in [CUB, TINY, IMAGENET]')
parser.add_argument('--dataset-path', default='/root/skb2/dataset', help='where to save')
parser.add_argument('--save-dir', default=None, help='where to save')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warm-up-epochs', default=5, type=int, metavar='N',
                    help='number of warm up epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--momentum-encoder', default=0.9, type=float, metavar='V', 
                    help='the momentum rate of encoder')
parser.add_argument('--image-size', default=224, type=int, metavar='V', 
                    help='the solution of raw image')



parser.add_argument('--val-batch-size', default=0, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--reg-lr', '--reg-learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate of reg net')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--reg-scheduler', default='const', type=str, metavar='LR',
                    help='scheduler to decay reg lr')
parser.add_argument('--decay-freq', default=-1, type=int, metavar='V', 
                    help='frequncy for decay lr')
parser.add_argument('--decay-gamma', default=0.9, type=float, metavar='V', 
                    help='gamma for decay lr')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://202.120.36.216:30501', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--amp', dest='amp', action='store_true',
                    help='if to train with torch amp')
parser.add_argument('--custwd', dest='custwd', action='store_true',
                    help='Use custom weight decay. It improves the accuracy and makes quantization easier.')
parser.add_argument('--tag', default='test', type=str,
                    help='the tag for identifying the log and model files. Just a string.')

best_locgt = 0

def sgd_optimizer(model, lr, momentum, weight_decay, use_custwd):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_locgt
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    # net_cls = getCamModel(num_classes=num_classes, ckpt=None)
    net_reg = regression_timm(pretrained=True, backbone=args.reg_backbone, feature_index=-1)

    is_main = not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and torch.distributed.get_rank() == 0)

    if is_main:
        if args.save_dir is None: 
            args.save_dir =  './WSOL_TIMM/%s_norm(%s)_%s_bs%d_regLR%s_m%s_%d/'%(args.tag, args.norm, args.dataset, args.batch_size, str(args.reg_lr), str(args.momentum_encoder), int(time.time()))
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        log_file =  os.path.join(args.save_dir, 'train.log') 
        log_msg('epochs {}, reg_lr {}, weight_decay {}'.format(args.epochs, args.reg_lr, args.weight_decay), log_file)
        log_msg('--------args----------', log_file)
        for k in sorted(list(vars(args).keys())):
            log_msg('%s:\t\t%s' % (k, vars(args)[k]), log_file)
        log_msg('--------args----------\n', log_file)


    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.



        if args.gpu is not None:
            # DDP模式，一个进程仅使用一张GPU的情况。
            # 两种情况条件下均可触发：（仅仅需要考虑第一种）
            # 1） 指定multiprocessing-distributed时（mp.spawn）
                # arg.gpu为单node的进程index序号，启动参数的args.batch_size为单node总batch大小。
            # 2） 同时指定arg.gpu、arg.world_size及arg.rank时（手动多进程）
                # 相当于手动启动每个进程，其中arg.world_size为全局进程数，arg.rank为全局进程序号，arg.gpu为进程使用的可见gpu编号。 
            torch.cuda.set_device(args.gpu)
            net_reg.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            net_reg = torch.nn.parallel.DistributedDataParallel(net_reg, device_ids=[args.gpu],find_unused_parameters=True)


        else:
            # DDP模式，一个进程同时使用多个GPU的情况。
            # 触发条件，需同时满足以下条件：
            # 1) 不指定multiprocessing-distributed
            # 2) 不指定args.gpu
            # 3) 指定arg.world_size及arg.rank，其中arg.world_size为全局节点数，arg.rank为全局节点序号
            net_reg.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            net_reg = torch.nn.parallel.DistributedDataParallel(net_reg,find_unused_parameters=True)
        


    elif args.gpu is not None:
        # 单机单卡模式，仅仅使用args.gpu指定的单显卡
        # 触发条件，同时满足：
        # 1） 不指定arg.world_size、arg.rank及multiprocessing-distributed
        # 2） 指定args.gpu
        torch.cuda.set_device(args.gpu)
        net_reg = net_reg.cuda(args.gpu)

    else:
        # 单机多卡DP模式，使用单机上的DP多卡并行
        # 触发条件，同时满足：
            # 1） 不指定arg.world_size、arg.rank及multiprocessing-distributed
            # 2） 不指定args.gpu
        # DataParallel will divide and allocate batch_size to all available GPUs
        net_reg = torch.nn.DataParallel(net_reg).cuda()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    print(args.dataset)

    train_dataset, validate_dataset = LCHPInitDataset(dataset=args.dataset, phase='train', resize=(args.image_size, args.image_size), dataset_path=args.dataset_path), \
                                    LCHPInitDataset(dataset=args.dataset, phase='val', resize=(args.image_size, args.image_size), dataset_path=args.dataset_path)
    dataloader_collate_fn = collate_fn
    print(train_dataset.num_classes)
    print(validate_dataset.num_classes)


    train_loader, val_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=dataloader_collate_fn), \
                                torch.utils.data.DataLoader(validate_dataset, batch_size=args.val_batch_size if args.val_batch_size !=0 else args.batch_size, shuffle=False, collate_fn=dataloader_collate_fn,
                                        num_workers=args.workers, pin_memory=True)


    # define loss function (criterion) and optimizer
    criterion_mse = nn.MSELoss().cuda(args.gpu)


    optimizer_reg = sgd_optimizer(net_reg, args.reg_lr, args.momentum, args.weight_decay, args.custwd)


    reg_lr_scheduler = setScheduler(optimizer_reg, train_loader, args.reg_scheduler, args)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_locgt = checkpoint['best_locgt']
            # if args.gpu is not None:
            #     # best_locgt may be from a checkpoint from a different GPU
            #     best_locgt = best_locgt.to(args.gpu)
            net_reg.load_state_dict(checkpoint['net_reg_state_dict'])

            optimizer_reg.load_state_dict(checkpoint['optimizer_reg'])

            reg_lr_scheduler.load_state_dict(checkpoint['reg_scheduler'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True



    if args.evaluate:
        validate(val_loader, net_reg, args)
        return

    

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        start_time = time.time()
        epoch_reg_loss = train(train_loader, net_reg, criterion_mse, optimizer_reg, epoch, args, reg_lr_scheduler, is_main=is_main, log_file=log_file)
        train_msg = 'Epoch: {:03d}, Reg Loss: ({:.4f}), Time {:3.2f}'.format(
                    epoch, epoch_reg_loss, time.time() - start_time)


        if is_main:
            log_msg(train_msg, log_file)

            # evaluate on validation set
            start_time = time.time()
            mean_iou_logits, epoch_loc_acc_logits = validate(val_loader, net_reg, args)
            val_msg = 'Epoch: {:03d},  \
                Mean Logit IoU: {:.2f}, \
                Logit Loc Acc: ({:.2f}, {:.2f}, {:.2f}), Time {:3.2f}'.format(
                epoch,
                mean_iou_logits * 100, 
                epoch_loc_acc_logits[0][0],  epoch_loc_acc_logits[0][1], epoch_loc_acc_logits[1],
                time.time() - start_time)
            
            log_msg(val_msg, log_file)

            # remember best acc@1 and save checkpoint
            is_best = epoch_loc_acc_logits[1] > best_locgt
            best_locgt = max(epoch_loc_acc_logits[1], best_locgt)

            save_checkpoint({
                'epoch': epoch + 1,
                'reg_backbone': args.reg_backbone,
                'net_reg_state_dict': net_reg.state_dict(),
                'best_locgt': best_locgt,
                'optimizer_reg' : optimizer_reg.state_dict(),
                'reg_scheduler': reg_lr_scheduler.state_dict(),
            }, is_best, filename = os.path.join(args.save_dir,'{}.pth.tar'.format(args.reg_backbone)),
                best_filename=os.path.join(args.save_dir,'{}_best.pth.tar'.format(args.reg_backbone)))


def train(train_loader, net_reg, criterion_mse, optimizer_reg, epoch, args, reg_lr_scheduler, is_main, log_file):


    # loss and metric
    batch_time = AverageMeter(name='BT')
    data_time = AverageMeter(name='DT')

    dreg_loss_container = AverageMeter(name='dreg_loss')


    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, dreg_loss_container],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    net_reg.train()

    scaler = GradScaler()



    end = time.time()
    for i, (images, target, bboxes) in enumerate(train_loader):
        # measure data loading time
        data_time(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        bboxes = bboxes.squeeze().cuda(args.gpu, non_blocking=True)


        net_reg.zero_grad()

        optimizer_reg.zero_grad()


        if args.amp:
            with autocast():

                reg_logits = net_reg(images) 
                reg_loss = criterion_mse(reg_logits, bboxes)

            # compute gradient and do SGD step. (In torc amp way)
            scaler.scale(reg_loss).backward()
            scaler.step(optimizer_reg)
            scaler.update()
        else:
            # ori reg loss 

            reg_logits = net_reg(images) 
            reg_loss = criterion_mse(reg_logits, bboxes)
            # compute gradient and do SGD step. (In torc amp way)
            reg_loss.backward()

            optimizer_reg.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_reg_loss = dreg_loss_container(reg_loss.item())


        # measure elapsed time
        batch_time(time.time() - end)
        end = time.time()

        if is_main and i == 0:
            message = 'reg_lr:{}'.format(reg_lr_scheduler.get_last_lr()[0])
            log_msg(message, log_file)

        if is_main and i != 0 and  i % args.print_freq == 0:
            progress.display(i)

        reg_lr_scheduler.step()
                
    return epoch_reg_loss





def validate(val_loader, net_reg, args):
    # loss and metric
    batch_time = AverageMeter(name='BT')
    data_time = AverageMeter(name='DT')

    loc_metric_logits = TopKLocAccuracyMetric(topk=(1, 5), name='loc_logit_acc')
    iou_logits_container = AverageMeter(name='iou_logits')

    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, iou_logits_container, loc_metric_logits],
        prefix='Test: ')



    # switch to evaluate mode
    net_reg.eval()

    end = time.time()
    with torch.no_grad():
        for i, (X, y, locs) in enumerate(val_loader):
            # obtain data
            data_time(time.time() - end)

            batch_size = X.size()[0]

            if args.gpu is not None:
                X = X.cuda(args.gpu, non_blocking=True)
                y = y.cuda(args.gpu, non_blocking=True).long()
                locs = locs.cuda(args.gpu, non_blocking=True)

            regression_logits = net_reg(X)  


            iou_logits = compute_maxiou_onevsmulti_batch(regression_logits,locs)
            mean_iou_logits = iou_logits_container(iou_logits.sum(0),batch_size)

            gt_one_hot  = F.one_hot(y, num_classes=val_loader.dataset.num_classes)
            epoch_loc_acc_logits = loc_metric_logits(gt_one_hot, y, iou_logits.cuda(args.gpu, non_blocking=True))

            # measure elapsed time
            batch_time(time.time() - end)
            end = time.time()

            if i !=0 and i % args.print_freq == 0:
                progress.display(i)
        progress.display(len(val_loader))

    return mean_iou_logits, epoch_loc_acc_logits


def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def setScheduler(optimizer, dataloader, scheduler, args):
    import math
    if scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.epochs * len(dataloader) // args.decay_freq, gamma=args.decay_gamma)
    elif scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs * len(dataloader))
    elif scheduler == 'const':
        scheduler = StepLR(optimizer, step_size=1, gamma=1)
    elif scheduler == 'warmup_cos':
        # warm_up_with_cosine_lr
        T_max = args.epochs * len(dataloader)
        warmup_t = args.warm_up_epochs * len(dataloader)
        warm_up_with_cosine_lr = lambda t: t / warmup_t if t <= warmup_t else 0.5 * ( math.cos((t - warmup_t) / (T_max - warmup_t) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    elif scheduler == 'warmup_const':
        warmup_t = args.warm_up_epochs * len(dataloader)
        warm_up_with_const_lr = lambda t: t / warmup_t if t <= warmup_t else 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_const_lr)
    else:
        raise ValueError
    
    return scheduler


if __name__ == '__main__':
    main()
