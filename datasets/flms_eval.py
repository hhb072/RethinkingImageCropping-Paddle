import os
import contextlib
import copy
import numpy as np
import cv2
import csv
import torch
import torchvision.ops as ops
import torch.distributed as dist
from torch.distributed import ReduceOp
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import util.box_ops as box_ops
from scipy.optimize import linear_sum_assignment
from util.misc import all_gather, is_dist_avail_and_initialized, get_world_size, get_rank, is_main_process
from collections import Counter

# this can calculate the acc_kn

class flms_evaluator(object):
    def __init__(self, save_path, iou_thresh, pathlist, set='val', cls_threshes=None, default_cls_thresh=0):

        self.cls_threshes = cls_threshes
        self.iou_thresh = iou_thresh
        self.default_cls_thresh = default_cls_thresh
        self.save_path = save_path
        self.crop_save_root = os.path.join(self.save_path, set, 'cropped_images_flms')
        if not os.path.exists(self.crop_save_root):
            os.makedirs(self.crop_save_root)
        self.set = set
        self.score_mean = 2.95
        self.score_std = 0.8
        self.pathlist = pathlist

        self.results = {}
        self.gt = []

        self.cls_score_all = torch.as_tensor([])
        self.cls_results = {}
        self.cls_results_id = {}
        self.cls_results_score = {}
        self.pr_good_num = {}
        self.cropscale = {}
        self.maxquality = {}
        self.meanquality = {}
        self.iou_mean_threshes = {}
        self.iou_max_nums = {}
        self.iou_mean_nums = {}
        self.boxes_ioufiltered = {}
        self.precision = {}
        self.recall = {}
        self.f_score = {}
        self.gt_good_num = 0

    def update(self, batchresults, batchgt):
        batchgt = self.post_process_gt(batchgt)
        self.results.update(batchresults)
        self.gt.extend(batchgt)

        device = list(batchresults.values())[0]['scores'].device

        self.cls_score_all = self.cls_score_all.type_as(self.cls_score_all).to(device)

        for idx, (imgid, imgdata) in enumerate(batchresults.items()):
            self.cls_score_all = torch.cat((self.cls_score_all, imgdata['scores']), 0)
            # self.cls_score_all.extend(imgdata['scores'].cpu().tolist())
            if batchgt[idx]['image_id'] != imgid:
                raise Exception('results do not match gt')


    def post_process_gt(self, batchgt):
        processed_batchgt = []
        for datadict in batchgt:
            # datadict['scores'] = datadict['scores'] * self.score_std + self.score_mean
            # datadict['good_scores'] = datadict['good_scores'] * self.score_std + self.score_mean
            orig_size = datadict['orig_size']
            h, w = orig_size[0], orig_size[1]
            norm_factor = torch.tensor([w, h, w, h], dtype=torch.float32).to(datadict['good_boxes'].device)
            # minscale = torch.as_tensor([0.35, 0.35, 0.55, 0.55]).to(datadict['boxes'].device)
            # maxscale = torch.as_tensor([0.65, 0.65, 0.95, 0.95]).to(datadict['boxes'].device)
            minscale = torch.as_tensor([0., 0., 0., 0.]).to(datadict['good_boxes'].device)
            maxscale = torch.as_tensor([1., 1., 1., 1.]).to(datadict['good_boxes'].device)
            datadict['good_boxes'] = datadict['good_boxes'] * (maxscale - minscale) + minscale
            datadict['good_boxes'] = box_ops.box_cxcywh_to_xyxy(datadict['good_boxes'])
            datadict['good_boxes'] = datadict['good_boxes'] * norm_factor
            processed_batchgt.append(datadict)
            self.gt_good_num += datadict['good_boxes'].shape[0]

        return processed_batchgt


    def gather_obj_from_processes(self, obj):
        if not is_dist_avail_and_initialized():
            return obj
        dist.barrier()
        world_size = get_world_size()
        gather_obj_list = [None for _ in range(world_size)]

        dist.all_gather_object(gather_obj_list, obj)

        if type(obj) == dict:
            gather_obj = {}
            for d in gather_obj_list:
                gather_obj.update(d)
        elif type(obj) == list:
            gather_obj = []
            for d in gather_obj_list:
                gather_obj.append(d)
        elif type(obj) == torch.Tensor:
            gather_obj = torch.cat(gather_obj_list, 0).type_as(obj)
        else:
            raise Exception('the objects to be gathered need to be lists or dictionaries')

        return gather_obj

    def gather_final_results(self, results_dict):

        if not is_dist_avail_and_initialized():
            return results_dict

        gathered_results_dict = {}
        for key, fres in results_dict.items():
            if type(fres) != torch.Tensor:
                fres = torch.as_tensor(fres)
            dist.all_reduce(fres, op=ReduceOp.SUM)
            gathered_results_dict.update({key:fres})

        return gathered_results_dict


    def set_cls_thresholds(self):
        if self.cls_threshes is not None:
            for thresh in self.cls_threshes:
                self.pr_good_num[thresh] = 0
            return

        self.cls_score_all = self.gather_obj_from_processes(self.cls_score_all)

        if type(self.cls_score_all) == list:
            self.cls_score_all = torch.as_tensor(self.cls_score_all)
        self.threshes = torch.unique(self.cls_score_all).sort(descending=False)[0]

        if torch.as_tensor([self.default_cls_thresh]).type_as(self.threshes) not in self.threshes:
            # add default thresh in the thresh list
            self.threshes = torch.cat((self.threshes, torch.as_tensor([self.default_cls_thresh]).type_as(self.threshes)),
                                      0).sort(descending=False)[0]
        self.cls_threshes = self.threshes.tolist()

        for thresh in self.cls_threshes:
            self.pr_good_num[thresh] = 0
        return

    def boxes_cls_thresholds(self):
        # reults: {image_id: {scores:torch.Tensor([0.7, 0.2]), labels: torch.Tensor([0,0]), boxes:torch.Tensor([[x,y,x,y],[]])}}
        # cls_results:{image_id:{thresh1: bboxes, thresh2:bboxes,...},...}
        # cls_results_id: {image_id:{thresh1: bbox_id, thresh2:bbox_id,...},}
        # pr_good_num: {thresh1: box_num, thresh2: box_num,....}
        initscale = torch.zeros(len(self.cls_threshes)).tolist()
        self.cropscale = dict(zip(self.cls_threshes, initscale))

        for idx, (image_id, value) in enumerate(self.results.items()):
            # key: image_id
            scores_all = value['scores']
            score_max, max_id = torch.max(scores_all, 0)
            if max_id.shape == torch.Size([]):
                max_id = max_id.unsqueeze(0)
            threshes = torch.as_tensor(self.cls_threshes).type_as(scores_all).to(scores_all.device)
            binary_indices = (scores_all >= threshes.unsqueeze(1))
            boxes_all = value['boxes']
            cls_results_current = {}
            cls_results_id_current = {}
            cls_results_score_current = {}
            img_h, img_w = self.gt[idx]['orig_size'][0], self.gt[idx]['orig_size'][0]
            assert self.gt[idx]['image_id'] == image_id
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(scores_all.device)


            for idx, thresh in enumerate(self.cls_threshes):
                indexes = torch.where(binary_indices[idx, :] == True)[0]
                if indexes.shape[0] == 0:
                    indexes = max_id
                cls_results_current[thresh] = boxes_all[indexes]
                cls_results_id_current[thresh] = indexes
                cls_results_score_current[thresh] = scores_all[indexes]

                normalized_bbox = box_ops.box_xyxy_to_cxcywh(boxes_all[indexes] / scale_fct)
                self.cropscale[thresh] += torch.sum(normalized_bbox[:, 2] * normalized_bbox[:, 3]) / \
                                          normalized_bbox.shape[0]
                self.pr_good_num[thresh] += torch.tensor(cls_results_current[thresh].shape[0]).to(scores_all.device)

            self.cls_results[image_id] = cls_results_current
            self.cls_results_id[image_id] = cls_results_id_current
            self.cls_results_score[image_id] = cls_results_score_current

        self.pr_good_num = self.gather_final_results(self.pr_good_num)

        self.cropscale= self.gather_final_results(self.cropscale)

        self.gt_good_num = self.gather_final_results({'gt': torch.tensor(self.gt_good_num).to(scores_all.device)})


    def iou_pr_gt(self):
        #
        # pr_gt_ioudict : {image_id: N*M tensor}, N is the number of queries, M is the number of gt good boxes

        pr_gt_ioudict = {}


        for idx, (imgid, boxes_data) in enumerate(self.results.items()):


            if imgid != self.gt[idx]['image_id']:
                raise Exception('the gt does not match the prediction ')

            gt_good_boxes = self.gt[idx]['good_boxes']

            pr_boxes = boxes_data['boxes'].type_as(gt_good_boxes)

            iou_mat = ops.box_iou(pr_boxes, gt_good_boxes)

            pr_gt_ioudict[imgid] = iou_mat


        return pr_gt_ioudict

    def iou_for_threshes(self):
        # results:  {image_id: [{scores:torch.Tensor(0.7), labels: torch.Tensor(1), boxes:torch.Tensor([x,y,x,y])}, {...}, {...}]}
        # cls_results_id: {image_id:{thresh1: query_id, thresh2:query_id,...},}
        # gt: [{'boxes':[[]], 'good_boxes':[[]], 'score':[], 'good_score':[]},{}]
        # iou_mean_threshes : {image_id:{thresh1:iou, thresh2:iou,...},}
        # boxes_ioufiltered : {image_id:{thresh1:boxes, thresh2:boxes,...},}

        pr_gt_ioudict = self.iou_pr_gt() # {image_id: N*M tensor} N is the number of queries, M is the number of gt good boxes
        device = self.gt[0]['good_boxes'].device

        for idx, (imgid, thresh_dict) in enumerate(self.cls_results_id.items()):

            if imgid != self.gt[idx]['image_id']:
                raise Exception('the gt does not mathch the prediction ')

            pr_gt_ioumat_eachimg = pr_gt_ioudict[imgid]

            if torch.sum(torch.isnan(pr_gt_ioumat_eachimg)) > 0:
                raise Exception('Found NaN in the iou matrix')

            maxiou_for_each_pr, maxiou_indices_gt = torch.max(pr_gt_ioumat_eachimg, dim=1)

            iou_eachimg = {}

            for thresh, pr_good_boxes_id in thresh_dict.items():
                pr_good_num = pr_good_boxes_id.shape[0]
                maxiou_clsthresh_current = maxiou_for_each_pr[pr_good_boxes_id].view(pr_good_num).contiguous()

                if pr_good_num > 0:
                    iou_eachimg[thresh] = torch.mean(maxiou_clsthresh_current)
                else:
                    iou_eachimg[thresh] = torch.as_tensor(0).type_as(maxiou_clsthresh_current).to(device)

            self.iou_mean_threshes[imgid] = iou_eachimg


    def iou_for_nums(self):

        pr_gt_ioudict = self.iou_pr_gt() # {image_id: N*M tensor} N is the number of queries, M is the number of gt good boxes
        device = self.gt[0]['good_boxes'].device

        for idx, (imgid, value) in enumerate(self.results.items()):
            # key: image_id
            boxes_all = value['boxes']
            pr_gt_ioumat_eachimg = pr_gt_ioudict[imgid]
            iou_eachimg_mean = {}
            iou_eachimg_max = {}

            for topk_id in range(1, boxes_all.shape[0] + 1):
                current_topbox = boxes_all[:topk_id]
                current_ioumat = pr_gt_ioumat_eachimg[:topk_id]
                maxiou_for_each_pr, maxiou_indices_gt = torch.max(current_ioumat, dim=1)
                iou_eachimg_max[topk_id] = torch.max(maxiou_for_each_pr)
                iou_eachimg_mean[topk_id] = torch.mean(maxiou_for_each_pr)

            self.iou_mean_nums[imgid] = iou_eachimg_mean
            self.iou_max_nums[imgid] = iou_eachimg_max





    def evaluate(self):
            # the used metrics:
            # (1) cls_results: {image_id:{thresh1: bboxes, thresh2:bboxes,...},...}
            # (2) iou_mean_threshes: {image_id:{thresh1:iou, thresh2:iou,...},...}
            #     for each predicted good box, we compute iou with all the gt good boxes,
            #     and select the best one as the iou for this box, then we average
            #     all the iou for boxes in each thresh.
            # (3) boxes_ioufiltered: {image_id:{thresh1:boxes, thresh2:boxes,...},...}
            #     We further filter the cls_results using iou thresh: in each
            #     thresh, only the boxes whose max iou is bigger than a threshold are maintained
            # (4) precision: {image_id:{thresh1:pr, thresh2:pr,...},...}
            #     only the boxes satisfying two conditions are regarded as correct predictions:
            #     1: cls score is higher than the cls thresh.
            #     2: max iou with good gt is higher than the iou thresh
            #     we calculate the ratio: number of correct prediction / number of gt good box
            #     this is a strict metric
            # (5) recall: {image_id:{thresh1:recall, thresh2:recall,...},...}
            #     each correct prediction (defined in precision) match a gt good box
            #     we obtain how many unique gt good box are matched
            #     and calculate the ratio of it to the total number of gt good box
            #     this reflect the diversity the predictions
            # (6) f_score: 2 * precision * recall / (precision + recall +eps):
            #     Harmonic mean of precision and recall
            # Note that in this version of evaluator, we do not compute the scores for predictions
            # this is because we can not ensure that each prediction can match a gt box

            # self.gather_necessary_data()
            self.set_cls_thresholds()
            self.boxes_cls_thresholds()
            self.iou_for_nums()


    def summary(self):

        device = list(self.results.values())[0]['boxes'].device
        boxnum = list(self.results.values())[0]['boxes'].shape[0]
        self.iou_mean_final = torch.tensor(0.0).to(device)
        self.evalnum = 0

        filter_thresh = 0

        final_iou_mean_dict = {}
        final_iou_max_dict = {}
        for idx in range(1, boxnum + 1):
            final_iou_mean_dict[idx] = 0
            final_iou_max_dict[idx] = 0

        for imgid, ioudict_mean in self.iou_mean_nums.items():
            self.evalnum += 1
            for idx in range(1, boxnum + 1):

                final_iou_mean_dict[idx] += ioudict_mean[idx]
                final_iou_max_dict[idx] += self.iou_max_nums[imgid][idx]

        self.evalnum = torch.tensor(self.evalnum).to(self.iou_mean_final.device)
        self.evalnum = self.gather_final_results({'evalnum':self.evalnum})
        final_iou_mean_dict = self.gather_final_results(final_iou_mean_dict)
        final_iou_max_dict = self.gather_final_results(final_iou_max_dict)

        for num, iou_mean in final_iou_mean_dict.items():
            final_iou_mean_dict[num] = iou_mean / self.evalnum['evalnum']
            final_iou_max_dict[num] = final_iou_max_dict[num] / self.evalnum['evalnum']

        return final_iou_max_dict, final_iou_mean_dict


    def display_and_save_metrics(self, final_iou_max_dict, final_iou_mean_dict, epoch=None):
        # mode: 'topk' or 'thresh'
        # assert mode == 'topk' or mode == 'thresh'

        print_menu = 'Epoch {:5s}: [ IoU_mean_1={:.4f}]'

        # if not is_dist_avail_and_initialized() or is_main_process():
        sepoch = 'test' if epoch == None else str(epoch)
        print(print_menu.format(sepoch, final_iou_max_dict[1].item()))
        if self.save_path is not None:
            # save final results in a csv file
            logpath = os.path.join(self.save_path, self.set)
            if not os.path.exists(logpath):
                os.makedirs(logpath)

            logfile_max = os.path.join(logpath,  'result_flms_max' + '.csv')
            logger_content_max = [x.item() for x in final_iou_max_dict.values()] + [sepoch]
            if not os.path.exists(logfile_max):
                headers = ['IoU_max_' + str(x) for x in final_iou_max_dict.keys()] + ['Epoch']
                with open(logfile_max, 'a') as f:
                    fcsv = csv.writer(f)
                    fcsv.writerow(headers)
            with open(logfile_max, 'a') as f:
                fcsv = csv.writer(f)
                fcsv.writerow(logger_content_max)

            logfile_mean = os.path.join(logpath,  'result_flms_mean' + '.csv')
            logger_content_mean = [x.item() for x in final_iou_mean_dict.values()] + [sepoch]
            if not os.path.exists(logfile_max):
                headers = ['IoU_mean_' + str(x) for x in final_iou_mean_dict.keys()] + ['Epoch']
                with open(logfile_mean, 'a') as f:
                    fcsv = csv.writer(f)
                    fcsv.writerow(headers)
            with open(logfile_mean, 'a') as f:
                fcsv = csv.writer(f)
                fcsv.writerow(logger_content_mean)

    def save_cropping_results(self, dataset_path, epoch, only_bbox=True):

        # if (not is_dist_avail_and_initialized()) or is_main_process():

        crop_savepath = os.path.join(self.crop_save_root, str(epoch))
        if not os.path.exists(crop_savepath) and is_main_process():
            # if mkdir in all processes, there may be error that one process has created the path
            # and other processes just begin to create the path
            os.mkdir(crop_savepath)

        if is_dist_avail_and_initialized():
            dist.barrier()


        for image_id, boxdict in self.cls_results.items():

            imgname = self.pathlist[image_id].split('/')[-1].split('.')[0]

            pr_good_bbox = boxdict[self.default_cls_thresh]

            pr_cls_score_sorted, sort_id = self.cls_results_score[image_id][self.default_cls_thresh].sort(descending=True)

            img = cv2.imread(self.pathlist[image_id])

            annotation_savepath = os.path.join(crop_savepath,
                                               imgname + '_' + 'annotation' + '.jpg')

            for idx, score in zip(sort_id, pr_cls_score_sorted):
                bbox = pr_good_bbox[idx]
                xmin = max(int(bbox[0].item()), 0)
                ymin = max(int(bbox[1].item()), 0)
                xmax = min(int(bbox[2].item()), img.shape[1] - 1)
                ymax = min(int(bbox[3].item()), img.shape[0] - 1)

                if not only_bbox:
                    cropped_img = img[ymin: ymax + 1, xmin: xmax + 1, :]
                    crop_savefile = os.path.join(crop_savepath,
                                            imgname + '_' + str(idx) + '_' + '.jpg')
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
                print('maybe the img is empty, check the path: {}'.format(os.path.join(dataset_path, 'images', self.set, str(image_id) + '.jpg')))






















if __name__ == '__main__':

    scores = torch.as_tensor([0.3, 0.6, 0.9, 0.3, 0.7])
    labels = torch.as_tensor([0.0, 1.0, 1.0, 0.0, 1.0])
    boxes = torch.as_tensor([[0, 0, 10, 10], # good 4
                             [20, 20, 35, 45], # good 3
                             [5, 5, 15, 20], # good 1
                             [5, 5, 25, 15], # good 5
                             [15, 10, 30, 30]]) # bad 2
    res = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    results = {torch.as_tensor(1):res, torch.as_tensor(2):res}

    bbox = torch.as_tensor([[0, 0, 10, 10],
                             [20, 20, 35, 45],
                             [5, 5, 15, 20],
                            [8, 5, 18, 22],
                             [5, 5, 25, 15],
                            [7, 15, 25, 30],
                             [15, 10, 30, 30]])

    good_bbox = torch.as_tensor([[0, 0, 10, 10],
                             [20, 20, 35, 45],
                             [5, 5, 15, 20],
                            [8, 5, 18, 22],
                             [5, 5, 25, 15]])

    score = torch.as_tensor([4, 4.5, 4, 4.1, 5, 3, 2])
    good_score = torch.as_tensor([4, 4.5, 4, 4.1, 5])

    gt = [{'boxes':bbox, 'good_boxes':good_bbox, 'scores':score, 'good_scores':good_score, 'image_id':torch.as_tensor(1)},
          {'boxes':bbox, 'good_boxes':good_bbox, 'scores':score, 'good_scores':good_score, 'image_id':torch.as_tensor(2)}]
    #
    # eval = cpc_evaluator(save_path='', iou_thresh=0.7)
    #
    # eval.update(batchresults=results, batchgt=gt)
    #
    # final_results_dict = eval.evaluate()
    #















