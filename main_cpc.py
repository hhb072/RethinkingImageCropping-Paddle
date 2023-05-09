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

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets.cpc import build_cpc
from datasets.flms import build_flms
from engine_cpc import evaluate, train_one_epoch, evaluate_flms
from models.conditional_detr_cpc import build
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
    parser.add_argument('--start_ema', default=45, type=int)
    parser.add_argument('--force_ema', default=0, type=int)
    parser.add_argument('--moving_average_decay', default=0.5, type=float)
    parser.add_argument('--ema_initialzed', default=0, type=int)

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
    parser.add_argument('--num_queries', default=24, type=int,
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
                        help='only when iou with a gt crop is bigger than this, it can use the score of the gt')
    parser.add_argument('--soft_bound', default=0.0, type=float,
                        help='for redundant queries, their soft label should not bigger than this at the good dimension,'
                             'set this to 0 means do not use soft label')
    parser.add_argument('--use_valid_smooth', default=0, type=int)

    # dataset parameters
    parser.add_argument('--dataset_root', default='/data1/CPCDataset_spit/')
    parser.add_argument('--flms_dataset_root', default='/data1/FLMS/')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--image_size_test', default=600, type=int)
    parser.add_argument('--good_thresh', default=2.0, type=float)
    parser.add_argument('--good_num', default=-1, type=int)
    parser.add_argument('--nms_thresh', default=0.8, type=float)
    parser.add_argument('--topk_num', default=10, type=int)

    parser.add_argument('--output_dir', default='/data1/ConditionalDETR_ImageCropping/results/cpc_results/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/data1/ConditionalDETR_ImageCropping/ConditionalDETR_r50dc5_epoch50.pth',
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
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors, postprocessors_flms = build(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model_without_ddp.conditionalDETR.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.conditionalDETR.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.conditionalDETR.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train, pathlist_train = build_cpc(image_set='train', args=args)
    dataset_val, pathlist_val = build_cpc(image_set='val', args=args)
    dataset_test, _ = build_cpc(image_set='val', args=args)
    dataset_flms, pathlist_flms = build_flms(image_set='val', args=args)


    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        sampler_flms = torch.utils.data.SequentialSampler(dataset_flms)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        sampler_flms = torch.utils.data.SequentialSampler(dataset_flms)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_flms = DataLoader(dataset_flms, args.batch_size, sampler=sampler_flms,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.conditionalDETR.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        pretrained_dict = checkpoint['model']
        try:
            pretrained_dict = OrderedDict({k: v for k, v in pretrained_dict.items()})
            model_without_ddp.conditionalDETR.load_state_dict(pretrained_dict)
        except RuntimeError:
            if args.num_queries != 300:
                print('both cls head and query embedding are randomly initialized')
                pretrained_dict = OrderedDict({k: v for k, v in pretrained_dict.items() if ('class_embed' not in k)
                                               and ('query_embed' not in k)})
            else:
                print('cls head is randomly initialized')
                pretrained_dict = OrderedDict({k: v for k, v in pretrained_dict.items() if ('class_embed' not in k)})
            model_dict = model_without_ddp.conditionalDETR.state_dict()
            model_dict.update(pretrained_dict)
            model_without_ddp.conditionalDETR.load_state_dict(OrderedDict(model_dict))

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # if args.test:
    #     test_stats, evaluator = evaluate(model, criterion, postprocessors,
    #                                      data_loader_test, 'test', pathlist_val, device, args.output_dir)
    #     final_results_dict_fixed, _ = evaluator.summary(use_score_thresh=False, use_predefined_thresh=False)
    #     final_results_dict_threshed, diff_dict = evaluator.summary(use_score_thresh=True, use_predefined_thresh=True)
    #     evaluator.display_and_save_metrics(final_results_dict=final_results_dict_fixed,
    #                                        mode='topk', epoch=-1)
    #     evaluator.display_and_save_metrics(final_results_dict=final_results_dict_threshed,
    #                                        mode='thresh', epoch=-1)
    #     if args.output_dir:
    #         # evaluator.save_cropping_results(dataset_path=args.dataset_root)
    #         utils.save_on_master(evaluator, output_dir / "test_evaluator.pth")
    #     return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.conditionalDETR.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, evaluator = evaluate(model.module.conditionalDETR, criterion, postprocessors,
                                         data_loader_val, args.num_queries, 'val', device, args.output_dir, epoch)

        evaluator_flms = evaluate_flms(model.module.conditionalDETR, criterion, postprocessors_flms,
                                         data_loader_flms, 'val', pathlist_flms,
                                         device, args.output_dir, epoch)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if evaluator is not None:
            evaluator.evaluate()

            if utils.is_main_process():
                (output_dir / 'val').mkdir(exist_ok=True)
                filenames = ['latest.pth']
                if epoch % 50 == 0:
                    filenames.append(f'{epoch:03}.pth')
                for name in filenames:
                    torch.save(evaluator,
                               output_dir / "val" / name)
                evaluator.display_and_save_apmetrics(iou_thresh=0.90, epoch=epoch)
                evaluator.display_and_save_apmetrics(iou_thresh=0.85, epoch=epoch)
                evaluator.display_and_save_accmetrics(iou_thresh=0.90, epoch=epoch)
                evaluator.display_and_save_accmetrics(iou_thresh=0.85, epoch=epoch)
        dist.barrier()

        if evaluator_flms is not None:
            final_iou_max_dict, final_iou_mean_dict = evaluator_flms.summary()
            if utils.is_main_process():
                (output_dir / 'val').mkdir(exist_ok=True)
                filenames = ['latest_flms.pth']
                if epoch % 50 == 0:
                    filenames.append(f'{epoch:03}.pth')
                for name in filenames:
                    torch.save(evaluator,
                               output_dir / "val" / name)
                evaluator_flms.display_and_save_metrics(final_iou_max_dict, final_iou_mean_dict, epoch=epoch)

        if utils.is_dist_avail_and_initialized():
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(1)
    main(args)
