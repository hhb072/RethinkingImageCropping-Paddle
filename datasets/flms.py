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
from scipy.io import loadmat

import warnings
MOS_MEAN = 2.95
MOS_STD = 0.8
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def prepare_original_data(image, annos):
    # image: image read from cv2, therefore, it is numpy array
    # annofile: 10 good boxes, may have invalid box

    good_bbox = list()

    for annotation in annos:

        xmin = float(annotation[1])
        ymin = float(annotation[0])
        xmax = float(annotation[3])
        ymax = float(annotation[2])

        if xmin != -1:
            good_bbox.append([xmin, ymin, xmax, ymax])

    good_bbox = torch.tensor(good_bbox)

    orig_size = torch.as_tensor(list(image.shape[:2]))

    target = {'good_boxes': good_bbox,
              'orig_size': orig_size}

    return image, target



class flms(data.Dataset):

    def __init__(self, args, imgpath=None, annopath=None, anchorfile=None, transform=None):


        self._imgpath = imgpath
        self._annopath = annopath
        self.achor_normalized = self.anchor_process(anchorfile)

        self.transform = transform
        self.good_num = args.good_num
        self.nms_thresh = args.nms_thresh

    def anchor_process(self, anchorfile):

        anchor_normalized = []
        for acs in anchorfile:
            anchor = acs[:-1].split(' ')
            anchor = [float(ac) for ac in anchor]
            anchor_normalized.append(anchor)

        return anchor_normalized

    def __getitem__(self, idx):

        image = cv2.imread(self._imgpath[idx])
        annos = self._annopath[idx]

        image, target = prepare_original_data(image, annos)
        target['image_id'] = torch.tensor(idx)

        if self.transform is not None:
            image, target = self.transform(image, target)

        # image = reverse(image)
        # print(image)
        # cv2.imwrite('/data1/gengyun.jia/ConditionalDETR_crop/1.jpg', image)
        # exit()

        return image, target

    def __len__(self):
        return len(self._imgpath)



def build_flms(image_set, args):

    path = os.path.join(args.flms_dataset_root, '500_image_dataset.mat')
    ann = loadmat(path)
    anchorpath = os.path.join(args.flms_dataset_root, 'pdefined_anchor.txt')
    with open(anchorpath, "r") as f:
        anchorfile = f.readlines()

    imgpath = []
    annopath = []

    for idx in range(500):
        name = ann['img_gt'][idx][0][0].tolist()[0]
        imgpath.append(os.path.join(args.flms_dataset_root, 'image', name))
        annopath.append(ann['img_gt'][idx][0][1].tolist())

    transform = croptransform(set='val', imgsize_test=args.image_size_test)

    dataset = flms(args=args, imgpath=imgpath,
                   annopath=annopath, anchorfile=anchorfile, transform=transform)

    return dataset, imgpath

