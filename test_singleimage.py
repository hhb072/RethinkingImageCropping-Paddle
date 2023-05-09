# ------------------------------------------------------------------------
# Conditional DETR for Image Cropping
# ------------------------------------------------------------------------
# Modified from ConditionalDETR (https://github.com/Atten4Vis/ConditionalDETR)
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import cv2
import datasets
import util.misc as utils
from datasets.custom_dataset import build_dataset
from engine import evaluate_single
from models import build_model
from collections import OrderedDict
import torch.distributed as dist


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_gamma', default=2, type=float)
    parser.add_argument('--soft_iou_thresh', default=0.85, type=float,
                        help='only iou with a gt crop is bigger than this, it can use the score of the gt')
    parser.add_argument('--soft_bound', default=0.5, type=float,
                        help='for redudant queries, their soft label should not bigger than this at the good dimension,'
                             'set this to 0 means do not use soft label')
    parser.add_argument('--use_valid_smooth', default=0, type=int)

    # dataset parameters
    parser.add_argument('--dataset_root', default='/data1/GAIC_journal/')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--image_size_test', default=600, type=int)
    parser.add_argument('--good_thresh', default=4.0, type=float)
    parser.add_argument('--good_num', default=-1, type=int)
    parser.add_argument('--nms_thresh', default=0.8, type=float)
    parser.add_argument('--topk_num', default=10, type=int)

    parser.add_argument('--crop_savepath', default='/data1/ConditionalDETR_ImageCropping/results/single/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/data1/ConditionalDETR_ImageCropping/cocopretrained/ConditionalDETR_r50dc5_epoch50.pth',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))

    args.distributed = False
    # args.test = True

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    dataset = build_dataset(args=args)

    sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader_test = DataLoader(dataset, args.batch_size, sampler=sampler,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        pretrained_dict = checkpoint['model']
        try:
            pretrained_dict = OrderedDict({k: v for k, v in pretrained_dict.items()})
            model_without_ddp.load_state_dict(pretrained_dict)
        except RuntimeError:
            if args.num_queries != 300:
                print('both cls head and query embedding are randomly initialized')
                pretrained_dict = OrderedDict({k: v for k, v in pretrained_dict.items() if ('class_embed' not in k)
                                               and ('query_embed' not in k)})
            else:
                print('cls head is randomly initialized')
                pretrained_dict = OrderedDict({k: v for k, v in pretrained_dict.items() if ('class_embed' not in k)})
            model_dict = model_without_ddp.state_dict()
            model_dict.update(pretrained_dict)
            model_without_ddp.load_state_dict(OrderedDict(model_dict))


    results = evaluate_single(model, postprocessors, data_loader_test, device)


    for image_id, resdict in results.items():

        pred_box_sorted = resdict['boxes']
        pred_score_sorted = resdict['scores']

        img = cv2.imread(os.path.join(args.dataset_root, str(image_id) + '.jpg'))

        annotation_savepath = os.path.join(args.crop_savepath,
                                           str(image_id) + '_' + 'annotation' + '.jpg')
        cropped_path = os.path.join(args.crop_savepath, str(image_id))
        if not os.path.exists(cropped_path):
            os.mkdir(cropped_path)

        displaynum = 10

        for idx, (score, box) in enumerate(zip(pred_score_sorted[:displaynum], pred_box_sorted[:displaynum])):

            xmin = max(int(box[0].item()), 0)
            ymin = max(int(box[1].item()), 0)
            xmax = min(int(box[2].item()), img.shape[1] - 1)
            ymax = min(int(box[3].item()), img.shape[0] - 1)

            cropped_img = img[ymin: ymax + 1, xmin: xmax + 1, :]

            crop_savefile = os.path.join(cropped_path, str(image_id) + '_' + str(idx) + '.jpg')
            try:
                cv2.imwrite(crop_savefile, cropped_img)
            except:
                print('empty image under bbox : {}'.format([xmin, ymin, xmax, ymax]))


            if idx < 5:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
                cv2.putText(img, str(score.item()), (xmin, ymin - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
        try:
            cv2.imwrite(annotation_savepath, img)
        except:
            print('maybe the img is empty, check the path: {}'.format(os.path.join(args.dataset_root, str(image_id) + '.jpg')))





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
