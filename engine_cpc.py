# ------------------------------------------------------------------------
# Conditional DETR for Image Cropping
# ------------------------------------------------------------------------
# Modified from ConditionalDETR (https://github.com/Atten4Vis/ConditionalDETR)
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import numpy as np
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.cpc_eval import cpc_evaluator
from datasets.flms_eval import flms_evaluator
from datasets.gaic_transforms import reverse

def train_one_epoch(model, criterion, data_loader, optimizer,
                    device, epoch, args):
    model.module.conditionalDETR.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    iterid = 0

    if (epoch >= args.start_ema or args.force_ema) and not args.ema_initialzed:
        model.module.reset_moving_average()
        model.module.netT = model.module._get_target_network()
        args.ema_initialzed = 1
        model.module.netT.eval()
        if epoch > args.start_ema:
            print('currently we do not support resume from checkpoints whose ema has started')
            raise Exception('repeated loading of teacher')

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        iterid += 1

        outputs = model.module.conditionalDETR(samples)

        for datadict in targets:
            datadict['pseudo_label'] = None

        if epoch >= args.start_ema or args.force_ema:

            with torch.no_grad():
                model.module.netT.eval()
                teacher_output = model.module.produce_pseudo_label(samples)
                pseudo_label = teacher_output['pred_logits'].sigmoid().detach()
                for idx, datadict in enumerate(targets):
                    datadict['pseudo_label'] = pseudo_label[idx]

        loss_dict = criterion(outputs, targets, epoch)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.module.conditionalDETR.parameters(), args.clip_max_norm)
        optimizer.step()
        if (epoch >= args.start_ema) and args.enable_ema:
            model.update_moving_average()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, num_queries, set, device, output_dir, epoch=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    evaluator = cpc_evaluator(output_dir, device=device, set=set, num_queries=num_queries)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        for datadict in targets:
            datadict['pseudo_label'] = None

        loss_dict = criterion(outputs, targets, epoch)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results_origin, results_oriorder = postprocessors['bbox'](outputs, orig_target_sizes)
        results = results_origin

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if evaluator is not None:
            evaluator.update(batchresults=res, batchgt=targets)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if evaluator is not None:
        evaluator.evaluate()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats, evaluator


@torch.no_grad()
def evaluate_flms(model, criterion, postprocessors, data_loader, set, pathlist,
             device, output_dir, epoch=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    cls_threshes = [0]

    evaluator = flms_evaluator(output_dir, iou_thresh=0.85, pathlist=pathlist, set=set, cls_threshes=cls_threshes)

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        for datadict in targets:
            datadict['pseudo_label'] = None

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results_origin,_ = postprocessors['bbox'](outputs, orig_target_sizes)
        results = results_origin

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if evaluator is not None:
            evaluator.update(batchresults=res, batchgt=targets)

    if evaluator is not None:
        evaluator.evaluate()


    return evaluator