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
import pickle
import random
import torch
import torch.nn as nn
import torch.optim

import cv2


from evaluation import compute_bboxes_from_scoremaps
from util import t2n

from config import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer, normalize_scoremap
from util import string_contains_any
import wsol
import wsol.method


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


class Eval(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _EVAL_METRICS = ['loss', 'classification', 'localization', 'localizationtop1']
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
        self.model = self._set_model()
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            val_transform=True)

    def _set_performance_meters(self):
        self._EVAL_METRICS += ['localization_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]
        
        self._EVAL_METRICS += ['localizationtop1_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                         higher_is_better=False
                                         if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self):
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
        print(model)
        return model


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

        if self.gt_optimal_tau:
            print('gt_optimal_tau {}'.format(self.gt_optimal_tau))
        if self.top1_optimal_tau:
            print('top1_optimal_tau {}'.format(self.top1_optimal_tau))



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
            output_dict = self.model(images)
            pred = output_dict['logits'].argmax(dim=1)

            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100
        return classification_acc

    def evaluate(self, split):
        self.model.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.performance_meters[split]['classification'].update(accuracy)

        cam_computer = CAMComputer(
            model=self.model,
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
        cam_performance_top1, cam_optimal_top1_tau = cam_computer.compute_and_evaluate_cams_top1()



        loc_score_gt = cam_performance_gt[self.args.iou_threshold_list.index(50)]
        loc_score_top1 = cam_performance_top1[self.args.iou_threshold_list.index(50)]
        gt_optimal_tau = cam_optimal_gt_tau[self.args.iou_threshold_list.index(50)]
        top1_optimal_tau = cam_optimal_top1_tau[self.args.iou_threshold_list.index(50)]




        self.performance_meters[split]['localization'].update(loc_score_gt)
        self.performance_meters[split]['localizationtop1'].update(loc_score_top1)

        

        if self.args.dataset_name in ('CUB', 'ILSVRC', 'TINY'):
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    'localization_IOU_{}'.format(IOU_THRESHOLD)].update(
                    cam_performance_gt[idx])
                self.performance_meters[split][
                    'localizationtop1_IOU_{}'.format(IOU_THRESHOLD)].update(
                    cam_performance_top1[idx])
        
        self.gt_optimal_tau = gt_optimal_tau
        self.top1_optimal_tau = top1_optimal_tau
        
  
    def inference(self, split):
        self.model.eval()
        # optimal_tau = self.gt_optimal_tau
        optimal_tau = 0.138

        print("Computing and evaluating cams.")
        save_file = os.path.join(self.args.log_folder, 'pseudo_bboxes.txt')
        if not os.path.exists(self.args.log_folder):
            os.makedirs(self.args.log_folder)
        lines = []
        with open(save_file, 'w') as f:
            for images, targets, image_ids in self.loaders[split]:
                image_size = images.shape[2:]
                images = images.cuda()
                cams = self.model(images, targets, return_cam_logits=False, return_cam=True)
                cams = t2n(cams)

                for cam,image_id in zip(cams, image_ids):
                    cam_resized = cv2.resize(cam, image_size,
                                            interpolation=cv2.INTER_CUBIC)
                    cam_normalized = normalize_scoremap(cam_resized)
                    bboxes, _ = compute_bboxes_from_scoremaps(scoremap=cam_normalized,scoremap_threshold_list=[optimal_tau], multi_contour_eval=False)
                    lines.append('{},{}\n'.format(image_id, ','.join(str(int(item/(self.args.crop_size-1)*63)) for item in bboxes[0][0]))) 
            f.writelines(lines)
        return 

   

    def load_checkpoint(self,):
        checkpoint_path = self.args.ckpt
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))


def main():
    evaler = Eval()

    print("===========================================================")
    print("Start eval ...")
    evaler.load_checkpoint()
    # evaler.evaluate(split='test')
    # evaler.print_performances()
    evaler.inference(split='train')
    # evaler.save_performances()


if __name__ == '__main__':
    main()
