import sys
import os
import torch.utils.data as data
import cv2
import math
import numpy as np
import random
import torch
import datetime
from datasets.gaic_transforms import croptransform, reverse
from util.box_ops import box_iou
from torchvision.ops import nms

import warnings
MOS_MEAN = 2.95
MOS_STD = 0.8
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def prepare_original_data(image):



    orig_size = torch.as_tensor(list(image.shape[:2]))

    target = {'orig_size': orig_size}
              # 'orig_good_boxes': good_bbox}

    return image, target



class custom(data.Dataset):

    def __init__(self, args, set, imgpath=None, transform=None):


        self._imgpath = imgpath
        self.set = set
        self.transform = transform


    def __getitem__(self, idx):

        image = cv2.imread(self._imgpath[idx])

        image, target = prepare_original_data(image)

        if self.transform is not None:
            image, target = self.transform(image, target)
        target['image_id'] = torch.as_tensor(int(self._imgpath[idx].split('/')[-1].split('.')[0]))

        return image, target

    def __len__(self):
        return len(self._imgpath)



def generate_bboxes(image):

    bins = 12.0
    h = image.shape[0]
    w = image.shape[1]
    step_h = h / bins
    step_w = w / bins
    annotations = list()
    for x1 in range(0,4):
        for y1 in range(0,4):
            for x2 in range(8,12):
                for y2 in range(8,12):
                    if (x2-x1)*(y2-y1)>0.4999*bins*bins and (y2-y1)*step_w/(x2-x1)/step_h>0.5 and (y2-y1)*step_w/(x2-x1)/step_h<2.0:
                        annotations.append([float(step_h*(0.5+x1)),float(step_w*(0.5+y1)),float(step_h*(0.5+x2)),float(step_w*(0.5+y2))])

    return annotations

def generate_bboxes_16_9(image):

    h = image.shape[0]
    w = image.shape[1]
    h_step = 9
    w_step = 16
    annotations = list()
    for i in range(14,30):
        out_h = h_step*i
        out_w = w_step*i
        if out_h < h and out_w < w and out_h*out_w>0.4*h*w:
            for w_start in range(0,w-out_w,w_step):
                for h_start in range(0,h-out_h,h_step):
                    annotations.append([float(h_start), float(w_start), float(h_start+out_h-1), float(w_start+out_w-1)])
    return annotations

def generate_bboxes_4_3(image):

    h = image.shape[0]
    w = image.shape[1]
    h_step = 12
    w_step = 16
    annotations = list()
    for i in range(14, 30):
        out_h = h_step*i
        out_w = w_step*i
        if out_h < h and out_w < w and out_h*out_w>0.4*h*w:
            for w_start in range(0, w-out_w,w_step):
                for h_start in range(0, h-out_h,h_step):
                    annotations.append([float(h_start),float(w_start),float(h_start+out_h-1),float(w_start+out_w-1)])
    return annotations

def generate_bboxes_1_1(image):

    h = image.shape[0]
    w = image.shape[1]
    h_step = 12
    w_step = 12
    annotations = list()
    for i in range(14,30):
        out_h = h_step*i
        out_w = w_step*i
        if out_h < h and out_w < w and out_h*out_w>0.4*h*w:
            for w_start in range(0,w-out_w,w_step):
                for h_start in range(0,h-out_h,h_step):
                    annotations.append([float(h_start),float(w_start),float(h_start+out_h-1),float(w_start+out_w-1)])
    return annotations



def build_dataset(args):

    imgdir = os.path.join(args.dataset_root)

    imglist = os.listdir(imgdir)
    imgpath = []
    for image in imglist:
        imgpath.append(os.path.join(imgdir, image))


    transform = croptransform(set='val', imgsize_test=args.image_size_test)

    dataset = custom(args=args, set='val', imgpath=imgpath, transform=transform)

    return dataset

