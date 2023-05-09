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


def prepare_original_data(image, annofile, thresh, good_num=-1, nms_thresh=0.8):
    # image: image read from cv2, therefore, it is numpy array
    # annofile: coordinates and the corresponding scores loaded from files
    # thresh: score thresh to define good crops
    # the output is not processed since augmentation needs original format
    # the output image is BGR and unit8 (0-255) numpy of original size
    # the output target score range is 1-5
    # the output target boxes is [xmin, ymin, xmax, ymax] of original size
    # if there is no crop whose score is higher than the threshold
    # we will use the crops with the highest score
    if thresh * good_num > 0:
        raise Exception('thresh and good_num cannot be both positive or both negative')

    bbox = list()
    score = list()

    good_score = list()
    good_bbox = list()

    for annotation in annofile:

        annotation_split = annotation.split()
        current_score = float(annotation_split[4])
        xmin = float(annotation_split[1])
        ymin = float(annotation_split[0])
        xmax = float(annotation_split[3])
        ymax = float(annotation_split[2])

        if current_score != -2:

            bbox.append([xmin, ymin, xmax, ymax])
            score.append(current_score)

            if thresh > 0:
                if current_score >= thresh:
                    good_bbox.append([xmin, ymin, xmax, ymax])
                    good_score.append(current_score)

    bbox = torch.as_tensor(bbox)
    # iou_mat = box_iou(bbox, bbox)[0]
    # torch.set_printoptions(edgeitems=100)
    # print(iou_mat)
    # exit()
    score = torch.as_tensor(score)

    if good_num > 0:
        good_score, indices = torch.sort(score, descending=True)
        good_score = good_score[:good_num]
        good_bbox = bbox[indices[:good_num]]
    else:
        good_bbox = torch.as_tensor(good_bbox)
        good_score = torch.as_tensor(good_score)

    if good_bbox.shape[0] == 0:
        best_score = torch.max(score)
        max_index = (score == best_score).nonzero()
        best_bbox = bbox[max_index]
        bestnum = max_index.shape[0]
        best_bbox = best_bbox.view(bestnum, -1).contiguous()
        best_score = best_score.unsqueeze(0).repeat(bestnum)
        good_bbox = best_bbox
        good_score = best_score

    if nms_thresh > 0:
        # there may be some very close crops with high scores
        # we only keep the one with the highest score
        id_remained = nms(good_bbox, good_score, iou_threshold=nms_thresh)
        good_bbox = good_bbox[id_remained]
        good_score = good_score[id_remained]


    orig_size = torch.as_tensor(list(image.shape[:2]))
    label = torch.zeros_like(good_score).type(torch.LongTensor)

    target = {'boxes': bbox, 'good_boxes': good_bbox,
              'scores': score, 'good_scores': good_score,
              'orig_size': orig_size, 'labels':label,}
              # 'orig_good_boxes': good_bbox}

    return image, target



class GAICD(data.Dataset):

    def __init__(self, args, set, imgpath=None, annopath=None, transform=None):


        self._imgpath = imgpath
        self._annopath = annopath
        self.set = set
        self.good_thresh = args.good_thresh
        self.transform = transform
        self.good_num = args.good_num
        self.nms_thresh = args.nms_thresh


    def __getitem__(self, idx):

        image = cv2.imread(self._imgpath[idx])
        with open(self._annopath[idx], 'r') as fid:
            annotations_txt = fid.readlines()

        image, target = prepare_original_data(image, annotations_txt, self.good_thresh, self.good_num, self.nms_thresh)
        target['image_id'] = torch.as_tensor(int(self._imgpath[idx].split('/')[-1].split('.')[0]))

        if self.transform is not None:
            image, target = self.transform(image, target)


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



def build_gaic(image_set, args):

    imgdir = os.path.join(args.dataset_root, 'images/' + image_set)
    annodir = os.path.join(args.dataset_root, 'annotations/' + image_set)
    imglist = os.listdir(imgdir)
    imgpath = []
    annopath = []
    for image in imglist:
        imgpath.append(os.path.join(imgdir, image))
        annopath.append(os.path.join(annodir, image[:-3] + 'txt'))

    transform = croptransform(set=image_set, imgsize_test=args.image_size_test)

    dataset = GAICD(args=args, set=image_set, imgpath=imgpath,
                              annopath=annopath, transform=transform)

    return dataset

