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

import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd

from evaluation import BoxEvaluator, PseudoBoxEvaluator
from evaluation import MaskEvaluator
from evaluation import configure_metadata
from evaluation import compute_bboxes_from_scoremaps
from util import t2n

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator_gt = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator,
                          "TINY": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)
        self.evaluator_top1 = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator,
                          "TINY": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)

    def compute_and_evaluate_cams_gt(self):
        print("Computing and evaluating cams.")

        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]
            images = images.cuda()
            cams = self.model(images, targets, return_cam=True)
            cams = t2n(cams)

            for cam, image_id in zip(cams, image_ids):
                cam_resized = cv2.resize(cam, image_size,
                                        interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                if self.split in ('val', 'test'):
                    cam_path = ospj(self.log_folder, 'scoremaps_gt', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                self.evaluator_gt.accumulate(cam_normalized, image_id)

        return self.evaluator_gt.compute()
    
    def compute_and_evaluate_cams_top1(self):
        print("Computing and evaluating cams.")

        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]
            images = images.cuda()
            cams, logits = self.model(images, labels=None, return_cam_logits=True)
            cams = t2n(cams)
            logits = t2n(logits)

            for cam, image_id, logit, target in zip(cams, image_ids, logits, targets):
                cam_resized = cv2.resize(cam, image_size,
                                        interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                if self.split in ('val', 'test'):
                    cam_path = ospj(self.log_folder, 'scoremaps_top1', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                self.evaluator_top1.accumulate(cam_normalized, image_id, target, logit, top1=True)

        return self.evaluator_top1.compute()

    

    def inference_bboxes(self, optimal_tau):
        print("Computing and evaluating cams.")
        save_file = ospj(self.log_folder, 'pseudo_bboxes.txt')
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        lines = []
        
        with open(save_file, 'w') as f:
            for images, targets, image_ids in self.loader:
                image_size = images.shape[2:]
                images = images.cuda()
                cams = self.model(images, targets, return_cam_logits=False, return_cam=True)
                cams = t2n(cams)

                for cam,image_id in zip(cams, image_ids):
                    cam_resized = cv2.resize(cam, image_size,
                                            interpolation=cv2.INTER_CUBIC)
                    cam_normalized = normalize_scoremap(cam_resized)
                    bboxes, _ = compute_bboxes_from_scoremaps(scoremap=cam_normalized,scoremap_threshold_list=[optimal_tau], multi_contour_eval=False)
                    lines.append('{},{}\n'.format(image_id, ','.join(bboxes[0])))
            f.writelines(lines)
        return 
    


class PseudoBoxComputer(object):
    def __init__(self, cls_model, reg_model, loader, metadata_root,
                 iou_threshold_list, dataset_name, split, log_folder=None):
        self.cls_model = cls_model
        self.reg_model = reg_model

        self.cls_model.eval()
        self.reg_model.eval()

        self.loader = loader
        self.split = split
        self.log_folder = log_folder

        metadata = configure_metadata(metadata_root)

        self.evaluator = PseudoBoxEvaluator(metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          iou_threshold_list=iou_threshold_list,
                                          cam_threshold_list=None,
                                          mask_root=None,
                                          multi_contour_eval=None)


    def compute_accs(self):
        print("Computing {cls acc} and {top1/gt loc acc} from cls model and reg model.")

        for images, targets, image_ids in self.loader:
            images = images.cuda()

            output_dict = self.cls_model(images)
            
            cls_logits = output_dict['logits']
            reg_logits = self.reg_model(images)


            cls_logits = t2n(cls_logits)
            reg_logits = t2n(reg_logits)

            for cls_logit, reg_logit, target, image_id in zip(cls_logits, reg_logits, targets, image_ids):
                self.evaluator.accumulate(cls_logit, reg_logit, target, image_id)

        return self.evaluator.compute()
        

    def inference_bboxes(self, optimal_tau):
        print("Computing and evaluating cams.")
        save_file = ospj(self.log_folder, 'pseudo_bboxes.txt')
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        lines = []
        
        with open(save_file, 'w') as f:
            for images, targets, image_ids in self.loader:
                image_size = images.shape[2:]
                images = images.cuda()
                cams = self.model(images, targets, return_cam_logits=False, return_cam=True)
                cams = t2n(cams)

                for cam,image_id in zip(cams, image_ids):
                    cam_resized = cv2.resize(cam, image_size,
                                            interpolation=cv2.INTER_CUBIC)
                    cam_normalized = normalize_scoremap(cam_resized)
                    bboxes, _ = compute_bboxes_from_scoremaps(scoremap=cam_normalized,scoremap_threshold_list=[optimal_tau], multi_contour_eval=False)
                    lines.append('{},{}\n'.format(image_id, ','.join(bboxes[0])))
            f.writelines(lines)
        return 
   