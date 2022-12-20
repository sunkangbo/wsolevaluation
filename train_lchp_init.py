"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import os
import cv2
import pickle
import random
import torch
import torch.nn as nn
import torch.optim

from configs.config_lchp import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer, normalize_scoremap, PseudoBoxComputer
from util import string_contains_any, t2n
from evaluation import compute_bboxes_from_scoremaps


import wsol
import wsol.method


from wsol.regression import regression_timm

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch = [] \
            if split == 'val' else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['cls_loss', 'reg_loss', 'classification', 'localization', 'localizationtop1']
    _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "TINY": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def __init__(self):
        self.args = get_configs()
        set_random_seed(self.args.seed)
        print(self.args)
        self.performance_meters = self._set_performance_meters()
        self.reporter = self.args.reporter

        self.cls_model_main = self._set_cls_model()
        self.cls_model_ema = self._set_cls_model()
        self.reg_model = self._set_reg_model()

        self.load_ckpt_from_path(checkpoint_path=self.args.ckpt, mode=self.args.ckpt_mode)


        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()
        self.mse_loss = nn.MSELoss().cuda()
        self.cls_optimizer = self._set_cls_optimizer()
        self.reg_optimizer = self._set_reg_optimizer()

        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class)

    def _set_performance_meters(self):
        self._EVAL_METRICS += ['{}_localization_IOU_{}'.format(model_name, threshold)
                               for threshold in self.args.iou_threshold_list
                               for model_name in ['main', 'ema']]
        self._EVAL_METRICS += ['{}_localizationtop1_IOU_{}'.format(model_name, threshold)
                               for threshold in self.args.iou_threshold_list
                               for model_name in ['main', 'ema']]
        self._EVAL_METRICS += ['{}_localization'.format(model_name)
                               for model_name in ['main', 'ema']]
        self._EVAL_METRICS += ['{}_localizationtop1'.format(model_name)
                               for model_name in ['main', 'ema']]


        self._EVAL_METRICS += ['regnet_{}_localization_IOU_{}'.format(model_name, threshold)
                               for threshold in self.args.iou_threshold_list
                               for model_name in ['main', 'ema']]
        self._EVAL_METRICS += ['regnet_{}_localizationtop1_IOU_{}'.format(model_name, threshold)
                               for threshold in self.args.iou_threshold_list
                               for model_name in ['main', 'ema']]
        self._EVAL_METRICS += ['regnet_{}_localization'.format(model_name)
                               for model_name in ['main', 'ema']]
        self._EVAL_METRICS += ['regnet_{}_localizationtop1'.format(model_name)
                               for model_name in ['main', 'ema']]


        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if 'loss' in metric else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_cls_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        model = wsol.__dict__[self.args.architecture](
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            pretrained_path=self.args.pretrained_path,
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            acol_drop_threshold=self.args.acol_threshold)
        model = model.cuda()
        # print(model)
        return model
    
    def _load_cls_ckpt(self):
        if self.args.cls_ckpt_path is not None:
            print("Loading checkpoint from {}".format(self.args.cls_ckpt_path))
            checkpoint = torch.load(self.args.cls_ckpt_path)
            self.cls_model_main.load_state_dict(checkpoint['model_state_dict'])
            self.cls_model_ema.load_state_dict(checkpoint['model_state_dict'])
            self.cls_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def _set_reg_model(self):
        model = regression_timm(pretrained=True, backbone=self.args.reg_backbone, feature_index=-1)
        model = model.cuda()
        return model

    def _set_cls_optimizer(self):
        param_features = []
        param_classifiers = []

        def param_features_substring_list(architecture):
            for key in self._FEATURE_PARAM_LAYER_PATTERNS:
                if architecture.startswith(key):
                    return self._FEATURE_PARAM_LAYER_PATTERNS[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        for name, parameter in self.cls_model_main.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    param_features.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_classifiers.append(parameter)
            else:
                if self.args.architecture in ('vgg16', 'inception_v3'):
                    param_classifiers.append(parameter)
                elif self.args.architecture == 'resnet50':
                    param_features.append(parameter)

        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': self.args.cls_lr},
            {'params': param_classifiers,
             'lr': self.args.cls_lr * self.args.cls_lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        return optimizer

    def _set_reg_optimizer(self):
        optimizer = torch.optim.SGD(self.reg_model.parameters(), lr=self.args.reg_lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        return optimizer

    @torch.no_grad()
    def _momentum_update_encoder(self):
        """
        Momentum update of the encoder
        """
        for param_1, param_2 in zip(self.cls_model_main.parameters(), self.cls_model_ema.parameters()):
            param_2.data = param_2.data * self.args.ema_m + param_1.data * (1. - self.args.ema_m)

    @torch.no_grad()
    def _pseudo_bboxes_generator(self, images, target, model_name='main'):
        cls_model = self.cls_model_main if model_name == 'main' else self.cls_model_ema
        image_size = images.shape[2:]
        H,W = image_size
        cams = cls_model(images, target, return_cam_logits=False, return_cam=True)
        cams = t2n(cams)

        pseudo_bboxes = []
        for cam in cams:
            cam_resized = cv2.resize(cam, image_size,
                                    interpolation=cv2.INTER_CUBIC)
            cam_normalized = normalize_scoremap(cam_resized)
            bboxes, _ = compute_bboxes_from_scoremaps(scoremap=cam_normalized,
                    scoremap_threshold_list=[random.uniform(self.args.tau_lower, self.args.tau_upper)], multi_contour_eval=False)
            bbox = bboxes[0][0]
            revised_bbox = [1.0 * bbox[0]/float(W-1), 1.0 * bbox[1]/float(H-1), 1.0 * bbox[2]/float(W-1), 1.0 * bbox[3]/float(H-1)]
            pseudo_bboxes.append(revised_bbox)
        return torch.tensor(pseudo_bboxes).float()
                    

    def _wsol_cls_training(self, images, target):
        if (self.args.wsol_method == 'cutmix' and
                self.args.cutmix_prob > np.random.rand(1) and
                self.args.cutmix_beta > 0):
            images, target_a, target_b, lam = wsol.method.cutmix(
                images, target, self.args.cutmix_beta)
            output_dict = self.cls_model_main(images)
            logits = output_dict['logits']
            loss = (self.cross_entropy_loss(logits, target_a) * lam +
                    self.cross_entropy_loss(logits, target_b) * (1. - lam))
            return logits, loss

        if self.args.wsol_method == 'has':
            images = wsol.method.has(images, self.args.has_grid_size,
                                     self.args.has_drop_rate)

        output_dict = self.cls_model_main(images, target)
        logits = output_dict['logits']

        if self.args.wsol_method in ('acol', 'spg'):
            loss = wsol.method.__dict__[self.args.wsol_method].get_loss(
                output_dict, target, spg_thresholds=self.args.spg_thresholds)
        else:
            loss = self.cross_entropy_loss(logits, target)

        return logits, loss
    
    def _wsol_reg_training(self, images, target):

        # print(target)
        logits = self.reg_model(images)
        loss = self.mse_loss(logits, target)

        return logits, loss

    def train(self, split):
        if self.args.freeze_cls:
            self._train_reg(split)
        else:
            self._train_all(split)

    def _train_all(self, split):

        self.cls_model_main.train()
        self.reg_model.train()
        loader = self.loaders[split]

        total_cls_loss = 0.0
        total_reg_loss = 0.0

        num_correct = 0
        num_images = 0

        for batch_idx, (images, target, _) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()
            if batch_idx % int(len(loader) / 10) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            # reg training
            pseudo_bboxes = self._pseudo_bboxes_generator(images, target, model_name='main')

            cls_logits, cls_loss = self._wsol_cls_training(images, target)
            _, reg_loss = self._wsol_reg_training(images, pseudo_bboxes.detach().cuda())

            self.cls_optimizer.zero_grad()
            self.reg_optimizer.zero_grad()


            cls_loss.backward()
            reg_loss.backward()

            self.cls_optimizer.step()
            self.reg_optimizer.step()

            # ema
            self._momentum_update_encoder()



            num_images += images.size(0)
            total_reg_loss += reg_loss.item() * images.size(0)
            reg_loss_average = total_reg_loss / float(num_images)
            self.performance_meters[split]['reg_loss'].update(reg_loss_average)

            # cal cls acc
            pred = cls_logits.argmax(dim=1)
            num_correct += (pred == target).sum().item()

            # cal total loss
            total_cls_loss += cls_loss.item() * images.size(0)

            classification_acc = num_correct / float(num_images) * 100
            cls_loss_average = total_cls_loss / float(num_images)

            self.performance_meters[split]['classification'].update(
                classification_acc)
            self.performance_meters[split]['cls_loss'].update(cls_loss_average)


        return dict(classification_acc=classification_acc, cls_loss=cls_loss_average, reg_loss=reg_loss_average)


    def _train_reg(self, split):
        self.cls_model_main.eval()
        self.reg_model.train()
        loader = self.loaders[split]

        total_cls_loss = 0.0
        total_reg_loss = 0.0

        num_correct = 0
        num_images = 0

        for batch_idx, (images, target, _) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()
            num_images += images.size(0)
            if batch_idx % int(len(loader) / 10) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            # reg training
            pseudo_bboxes = self._pseudo_bboxes_generator(images, target, model_name='main')


            with torch.no_grad():
                cls_logits, cls_loss = self._wsol_cls_training(images, target)


            _, reg_loss = self._wsol_reg_training(images, pseudo_bboxes.detach().cuda())
            self.reg_optimizer.zero_grad()
            reg_loss.backward()
            self.reg_optimizer.step()

            total_cls_loss += cls_loss.item() * images.size(0)
            total_reg_loss += reg_loss.item() * images.size(0)


            cls_loss_average = total_cls_loss / float(num_images)
            reg_loss_average = total_reg_loss / float(num_images)

            
            # cal cls acc
            pred = cls_logits.argmax(dim=1)
            num_correct += (pred == target).sum().item()
            classification_acc = num_correct / float(num_images) * 100

            self.performance_meters[split]['classification'].update(
                classification_acc)
            self.performance_meters[split]['cls_loss'].update(cls_loss_average)
            self.performance_meters[split]['reg_loss'].update(reg_loss_average)

            # print("classification_acc: ", classification_acc, "cls_loss_average: ", cls_loss_average, "reg_loss_average: ", reg_loss_average)

        return dict(classification_acc=classification_acc, cls_loss=cls_loss_average, reg_loss=reg_loss_average)



    def print_performances(self):
        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    print("Split {}, metric {}, current value: {}".format(
                        split, metric, current_performance))
                    if split != 'test':
                        print("Split {}, metric {}, best value: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_value))
                        print("Split {}, metric {}, best epoch: {}".format(
                            split, metric,
                            self.performance_meters[split][metric].best_epoch))
        if  hasattr(self, 'gt_optimal_tau'):
            print('last epoch gt_optimal_tau {}'.format(self.gt_optimal_tau))
        if  hasattr(self, 'top1_optimal_tau'):
            print('last epoch top1_optimal_tau {}'.format(self.top1_optimal_tau))

    def save_performances(self):
        log_path = os.path.join(self.args.log_folder, 'performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.performance_meters, f)

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, image_ids) in enumerate(loader):
            images = images.cuda()
            targets = targets.cuda()
            output_dict = self.cls_model_main(images)
            pred = output_dict['logits'].argmax(dim=1)

            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100
        return classification_acc

    def evaluate(self, epoch, split, model_name='main'):
        print("Evaluate epoch {}, split {}".format(epoch, split))
        if model_name == 'main':
            model = self.cls_model_main
        else:
            model = self.cls_model_ema
        model.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['classification'].update(accuracy)

        cam_computer = CAMComputer(
            model=model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            log_folder=self.args.log_folder,
        )
        cam_performance_gt, cam_optimal_gt_tau = cam_computer.compute_and_evaluate_cams_gt()
        cam_performance_top1, cam_optimal_top1_tau, = cam_computer.compute_and_evaluate_cams_top1()

        # cam_performance_gt, cam_performance_top1 = cam_performance

        if self.args.multi_iou_eval or self.args.dataset_name == 'OpenImages':
            loc_score_gt = np.average(cam_performance_gt)
            loc_score_top1 = np.average(cam_performance_top1)
            gt_optimal_tau = np.average(cam_optimal_gt_tau)
            top1_optimal_tau = np.average(cam_optimal_top1_tau)

        else:
            loc_score_gt = cam_performance_gt[self.args.iou_threshold_list.index(50)]
            loc_score_top1 = cam_performance_top1[self.args.iou_threshold_list.index(50)]
            gt_optimal_tau = cam_optimal_gt_tau[self.args.iou_threshold_list.index(50)]
            top1_optimal_tau = cam_optimal_top1_tau[self.args.iou_threshold_list.index(50)]




        self.performance_meters[split]['{}_localization'.format(model_name)].update(loc_score_gt)
        self.performance_meters[split]['{}_localizationtop1'.format(model_name)].update(loc_score_top1)

        

        if self.args.dataset_name in ('CUB', 'ILSVRC', 'TINY'):
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    '{}_localization_IOU_{}'.format(model_name, IOU_THRESHOLD)].update(
                    cam_performance_gt[idx])
                self.performance_meters[split][
                    '{}_localizationtop1_IOU_{}'.format(model_name, IOU_THRESHOLD)].update(
                    cam_performance_top1[idx])
        
        self.gt_optimal_tau = gt_optimal_tau
        self.top1_optimal_tau = top1_optimal_tau

    def evaluate_lchp(self, epoch, split, model_name='main'):
        print("Evaluate RegNet in epoch {}, split {}".format(epoch, split))
        self.reg_model.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['classification'].update(accuracy)

        cls_model = self.cls_model_main if model_name=='main' else self.cls_model_ema

        pseudo_computer = PseudoBoxComputer(
            cls_model=cls_model,
            reg_model=self.reg_model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            log_folder=self.args.log_folder,
        )

        
        _, loc_top1_accs, loc_gt_accs = pseudo_computer.compute_accs()
        # cam_performance_gt, cam_performance_top1 = cam_performance

        if self.args.multi_iou_eval or self.args.dataset_name == 'OpenImages':
            loc_gt_acc = np.average(loc_gt_accs)
            loc_top1_acc = np.average(loc_top1_accs)

        else:
            loc_gt_acc = loc_gt_accs[self.args.iou_threshold_list.index(50)]
            loc_top1_acc = loc_top1_accs[self.args.iou_threshold_list.index(50)]




        self.performance_meters[split]['regnet_{}_localization'.format(model_name)].update(loc_gt_acc)
        self.performance_meters[split]['regnet_{}_localizationtop1'.format(model_name)].update(loc_top1_acc)

        

        if self.args.dataset_name in ('CUB', 'ILSVRC', 'TINY'):
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    'regnet_{}_localization_IOU_{}'.format(model_name, IOU_THRESHOLD)].update(
                    loc_gt_accs[idx])
                self.performance_meters[split][
                    'regnet_{}_localizationtop1_IOU_{}'.format(model_name, IOU_THRESHOLD)].update(
                    loc_top1_accs[idx])
          

    def _torch_save_model(self, filename, epoch):
        torch.save({'architecture': self.args.architecture,
                    'epoch': epoch,
                    'cls_main_state_dict': self.cls_model_main.state_dict(),
                    'cls_ema_state_dict': self.cls_model_ema.state_dict(),
                    'reg_state_dict': self.reg_model.state_dict(),
                    'cls_optimizer': self.cls_optimizer.state_dict(),
                    'reg_optimizer': self.reg_optimizer.state_dict(),
                    },
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch, split):
        if (self.performance_meters[split][self._BEST_CRITERION_METRIC]
                .best_epoch) == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('best'), epoch)
        if self.args.epochs == epoch:
            self._torch_save_model(
                self._CHECKPOINT_NAME_TEMPLATE.format('last'), epoch)

    def report_train(self, train_performance, epoch, split='train'):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        reporter_instance.add(
            key='{split}/classification'.format(split=split),
            val=train_performance['classification_acc'])
        reporter_instance.add(
            key='{split}/cls_loss'.format(split=split),
            val=train_performance['cls_loss'])
        reporter_instance.add(
            key='{split}/reg_loss'.format(split=split),
            val=train_performance['reg_loss'])
        reporter_instance.write()

    def report(self, epoch, split):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        for metric in self._EVAL_METRICS:
            reporter_instance.add(
                key='{split}/{metric}'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].current_value)
            reporter_instance.add(
                key='{split}/{metric}_best'
                    .format(split=split, metric=metric),
                val=self.performance_meters[split][metric].best_value)
        reporter_instance.write()

    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch % self.args.cls_lr_decay_frequency == 0:
            for param_group in self.cls_optimizer.param_groups:
                param_group['lr'] *= 0.1

    def load_checkpoint(self, checkpoint_type):
        if checkpoint_type not in ('best', 'last'):
            raise ValueError("checkpoint_type must be either best or last.")
        checkpoint_path = os.path.join(
            self.args.log_folder,
            self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if 'cls_main_state_dict' in checkpoint:
                self.cls_model_main.load_state_dict(checkpoint['cls_main_state_dict'], strict=True)
            else:
                print("Warnning! No cls_main_state_dict in checkpoint.")
            if 'cls_ema_state_dict' in checkpoint:
                self.cls_model_ema.load_state_dict(checkpoint['cls_ema_state_dict'], strict=True)
            else:
                print("Warnning! No cls_ema_state_dict in checkpoint.")
            if 'reg_state_dict' in checkpoint:
                self.reg_model.load_state_dict(checkpoint['reg_state_dict'], strict=True)
            else:
                print("Warnning! No reg_state_dict in checkpoint.")
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))

    def load_ckpt_from_path(self, checkpoint_path, mode=['cls', 'reg']):
        if checkpoint_path is None:
            print("Warnning! No pretraiend checkpoint assigned.")
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if 'cls_main_state_dict' in checkpoint and 'cls' in mode:
                self.cls_model_main.load_state_dict(checkpoint['cls_main_state_dict'], strict=True)
            else:
                print("Warnning! No cls_main_state_dict in checkpoint or cls not in mode")
            if 'cls_ema_state_dict' in checkpoint and 'cls' in mode:
                self.cls_model_ema.load_state_dict(checkpoint['cls_ema_state_dict'], strict=True)
            else:
                print("Warnning! No cls_ema_state_dict in checkpoint or cls not in mode")
            if 'reg_state_dict' in checkpoint and 'reg' in mode:
                self.reg_model.load_state_dict(checkpoint['reg_state_dict'], strict=True)
            else:
                print("Warnning! No reg_state_dict in checkpoint or reg not in mode.")
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))


def main():
    trainer = Trainer()

    print("===========================================================")
    print("Start epoch 0 ...")
    # trainer.evaluate(epoch=0, split='val', model_name='main')
    trainer.evaluate_lchp(epoch=0, split='val', model_name='main')
    trainer.print_performances()
    trainer.report(epoch=0, split='val')
    trainer.save_checkpoint(epoch=0, split='val')
    print("Epoch 0 done.")

    for epoch in range(trainer.args.epochs):
        print("===========================================================")
        print("Start epoch {} ...".format(epoch + 1))
        trainer.adjust_learning_rate(epoch + 1)
        train_performance = trainer.train(split='train')
        trainer.report_train(train_performance, epoch + 1, split='train')
        trainer.evaluate(epoch + 1, split='val', model_name='main')
        trainer.evaluate_lchp(epoch + 1, split='val', model_name='main')
        trainer.print_performances()
        trainer.report(epoch + 1, split='val')
        trainer.save_checkpoint(epoch + 1, split='val')
        print("Epoch {} done.".format(epoch + 1))

    print("===========================================================")
    print("Final epoch evaluation on test set ...")

    trainer.load_checkpoint(checkpoint_type=trainer.args.eval_checkpoint_type)
    trainer.evaluate(trainer.args.epochs, split='test', model_name='main')
    trainer.evaluate_lchp(trainer.args.epochs, split='test', model_name='ema')
    trainer.print_performances()
    trainer.report(trainer.args.epochs, split='test')
    trainer.save_performances()


if __name__ == '__main__':
    main()
