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
import util.box_ops as box_ops
from util.misc import all_gather, is_dist_avail_and_initialized, get_world_size, get_rank, is_main_process
from collections import Counter
import pickle
import time
import scipy.io as scio


class gaic_evaluator(object):
    def __init__(self, save_path, device, set='val', num_queries=90):

        self.save_path = save_path
        self.crop_save_root = os.path.join(self.save_path, set, 'cropped_images')
        if not os.path.exists(self.crop_save_root):
            os.makedirs(self.crop_save_root)
        self.set = set
        self.device = device

        self.iou_threshes = torch.tensor(list(np.arange(0.85, 0.951, 0.01)))
        # self.iou_threshes = torch.tensor([0.9])
        self.maxDets = torch.tensor([5, 10, 20, 40, 60, 90])
        self.maxDets = self.maxDets[torch.where(self.maxDets <= num_queries)]
        if num_queries not in self.maxDets:
            self.maxDets = torch.cat((self.maxDets, torch.tensor([num_queries])), 0)

        # self.maxDets = torch.tensor([90])
        self.recall_threshes = torch.from_numpy(np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True))
        T = self.iou_threshes.shape[0]
        self.gtm_list = []
        self.dtm_list = []
        self.score_list = []
        self.imageid_list = []
        self.acc5 = torch.zeros(T).to(device)
        self.acc10 = torch.zeros(T).to(device)
        self.image_num = torch.tensor([0]).to(device)
        self.center_x = []
        self.center_y = []
        self.area = []
        self.score_oriorder_list = []

    def update(self, batchresults, batchgt):
        batchgt = self.post_process_gt(batchgt)

        for image_order, (image_id, pred_dict) in enumerate(batchresults.items()):
            assert image_id == batchgt[image_order]['image_id']
            pred_boxes_current = pred_dict['boxes']
            gt_boxes_current = batchgt[image_order]['good_boxes']
            gt_scores_current_sorted, gt_order = torch.sort(batchgt[image_order]['good_scores'], descending=True)
            gt_boxes_current = gt_boxes_current[gt_order]
            pred_scores_current = pred_dict['scores']
            iou_mat_current = ops.box_iou(pred_boxes_current, gt_boxes_current)

            top5, top10 = self.get_acc(ioumat=iou_mat_current)
            self.acc5 += top5
            self.acc10 += top10

            self.image_num += 1

            gtm, dtm = self.get_matchid_eachimage(pred_boxes_current, gt_boxes_current, iou_mat_current)
            self.gtm_list.append(gtm)
            self.dtm_list.append(dtm)
            self.score_list.append(pred_scores_current)
            self.imageid_list.append(image_id)


    def update_oriorder(self, batchresults):

        for image_order, (image_id, pred_dict) in enumerate(batchresults.items()):

            pred_boxes_current = pred_dict['boxes']
            pred_scores_current = pred_dict['scores']


            self.center_x.append(pred_boxes_current[:, 0])
            self.center_y.append(pred_boxes_current[:, 1])
            self.area.append(pred_boxes_current[:, 2] * pred_boxes_current[:, 3])

            self.score_oriorder_list.append(pred_scores_current)

    def statistic_anchors(self):
        self.center_x = torch.stack(self.center_x, 0) # 500 * 90
        self.center_y = torch.stack(self.center_y, 0)
        self.area = torch.stack(self.area, 0)
        self.score_oriorder_list = torch.stack(self.score_oriorder_list, 0)
        mean_center_x = torch.mean(self.center_x, 0).squeeze()
        mean_center_y = torch.mean(self.center_y, 0).squeeze()
        mean_area = torch.mean(self.area, 0).squeeze()
        mean_score = torch.mean(self.score_oriorder_list, 0).squeeze()

        var_center_x = torch.var(self.center_x, 0, unbiased=True).squeeze()
        var_center_y = torch.var(self.center_y, 0, unbiased=True).squeeze()
        var_area = torch.var(self.area, 0, unbiased=True).squeeze()
        var_score = torch.var(self.score_oriorder_list, 0, unbiased=True).squeeze()

        std_center_x = torch.std(self.center_x, 0, unbiased=True).squeeze()
        std_center_y = torch.std(self.center_y, 0, unbiased=True).squeeze()
        std_area = torch.std(self.area, 0, unbiased=True).squeeze()
        std_score = torch.std(self.score_oriorder_list, 0, unbiased=True).squeeze()




    def post_process_gt(self, batchgt):
        processed_batchgt = []
        for datadict in batchgt:
            # datadict['scores'] = datadict['scores'] * self.score_std + self.score_mean
            # datadict['good_scores'] = datadict['good_scores'] * self.score_std + self.score_mean
            orig_size = datadict['orig_size']
            h, w = orig_size[0], orig_size[1]
            norm_factor = torch.tensor([w, h, w, h], dtype=torch.float32).to(datadict['boxes'].device)
            # minscale = torch.as_tensor([0.35, 0.35, 0.55, 0.55]).to(datadict['boxes'].device)
            # maxscale = torch.as_tensor([0.65, 0.65, 0.95, 0.95]).to(datadict['boxes'].device)
            minscale = torch.as_tensor([0., 0., 0., 0.]).to(datadict['boxes'].device)
            maxscale = torch.as_tensor([1., 1., 1., 1.]).to(datadict['boxes'].device)
            datadict['boxes'] = datadict['boxes'] * (maxscale - minscale) + minscale
            datadict['good_boxes'] = datadict['good_boxes'] * (maxscale - minscale) + minscale
            datadict['boxes'] = box_ops.box_cxcywh_to_xyxy(datadict['boxes'])
            datadict['good_boxes'] = box_ops.box_cxcywh_to_xyxy(datadict['good_boxes'])
            datadict['boxes'] = datadict['boxes'] * norm_factor
            datadict['good_boxes'] = datadict['good_boxes'] * norm_factor
            processed_batchgt.append(datadict)

        return processed_batchgt


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


    def get_acc(self, ioumat):

        ioum_top5 = ioumat[:5, 0]
        ioum_top10 = ioumat[:10, 0]
        T = self.iou_threshes.shape[0]
        top5 = torch.zeros(T).to(self.device)
        top10 = torch.zeros(T).to(self.device)
        for t, iouthresh in enumerate(self.iou_threshes):
            if torch.sum(ioum_top5 >= iouthresh) > 0:
                top5[t] = 1
            if torch.sum(ioum_top10 >= iouthresh) > 0:
                top10[t] = 1

        return top5, top10


    def get_matchid_eachimage(self, dt, gt, ioumat):

        device = gt.device
        T = len(self.iou_threshes)
        G = gt.shape[0]
        D = dt.shape[0]
        gtm = -torch.ones((T, G)).to(device)
        dtm = -torch.ones((T, D)).to(device)


        for tind, iouth in enumerate(self.iou_threshes):
            for dind, pred_boxes in enumerate(dt):
                m = -1
                for gind, gt_boxes in enumerate(gt):

                    if gtm[tind, gind] >= 0:
                        continue
                    if ioumat[dind, gind] < iouth:
                        continue

                    iou = ioumat[dind, gind]
                    m = gind

                if m == -1:
                    continue

                dtm[tind, dind] = m
                gtm[tind, m] = dind

        return gtm, dtm


    def get_precision_recall_mat(self):

        # all_dtm = all_gather(self.dtm_list)
        # all_gtm = all_gather(self.gtm_list)
        # all_scores = all_gather(self.score_list)
        # torch.cuda.empty_cache()
        #
        # merged_dtm = []
        # for p in all_dtm:
        #     merged_dtm.extend(p)
        #
        # merged_gtm = []
        # for p in all_gtm:
        #     merged_gtm.extend(p)
        #
        # merged_scores = []
        # for p in all_scores:
        #     merged_scores.extend(p)
        merged_dtm = self.dtm_list
        merged_gtm = self.gtm_list
        merged_scores = self.score_list


        self.iou_threshes = self.iou_threshes.to(self.device)
        self.recall_threshes = self.recall_threshes.to(self.device)
        self.maxDets = self.maxDets.to(self.device)

        T = self.iou_threshes.shape[0]
        R = self.recall_threshes.shape[0]
        M = self.maxDets.shape[0]
        precision = -torch.ones((T, R, M)).to(self.device)
        recall = -torch.ones((T, M)).to(self.device)
        scores = -torch.ones((T, R, M)).to(self.device)

        gtm_full = torch.cat([s.to(self.device) for s in merged_gtm], dim=1)
        gtnum = gtm_full.shape[1]


        for m, maxDet in enumerate(self.maxDets):

            pred_scores_full = torch.cat([s[0: maxDet].to(self.device) for s in merged_scores], dim=0)
            inds = torch.argsort(-pred_scores_full)
            dtScoresSorted = pred_scores_full[inds]
            dtm_full = torch.cat([d[:, 0: maxDet].to(self.device) for d in merged_dtm], dim=1)[:, inds]

            tps = (dtm_full >= 0).type_as(dtm_full)
            fps = (dtm_full < 0).type_as(dtm_full)

            tp_cumsum = torch.cumsum(tps, dim=1)
            fp_cumsum = torch.cumsum(fps, dim=1)

            for t, (tp, fp) in enumerate(zip(tp_cumsum, fp_cumsum)):


                pred_num = tp.shape[-1]
                rc = tp / gtnum
                pr = tp / (fp + tp + np.spacing(1).item())
                q = torch.zeros((R,)).to(self.device)
                ss = torch.zeros((R,)).to(self.device)

                recall[t, m] = rc[-1]

                for i in range(pred_num - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                inds = torch.searchsorted(rc, self.recall_threshes, right=False)
                try:
                    for ri, pi in enumerate(inds):
                        q[ri] = pr[pi]
                        ss[ri] = dtScoresSorted[pi]
                except:
                    pass
                precision[t, :, m] = q
                scores[t, :, m] = ss

        self.precision = precision
        self.recall = recall
        self.scores = scores
        self.mAP = torch.mean(precision, dim=1)


    def evaluate(self):

        self.get_precision_recall_mat()
        self.acc_all = torch.cat([self.acc5, self.acc10, self.image_num], 0)
        #
        # if is_dist_avail_and_initialized():
        #     dist.all_reduce(self.acc_all, op=ReduceOp.SUM)
        #     torch.cuda.empty_cache()


    def display_and_save_apmetrics(self, iou_thresh=None, epoch=None):

        if iou_thresh is None:
            iou_thresh = 0.9

        iou_ind = torch.argmin(torch.abs(self.iou_threshes - iou_thresh))

        mAP_target = self.mAP[iou_ind, :]

        print_menu = 'Epoch {:5s}: [| mAP_5={:.4f} | mAP_10={:.4f} | mAP_20={:.4f} | mAP_40={:.4f} | mAP_60={:.4f} |' \
                     ' mAP_90={:.4f} | IoUThresh={:.4f} |]'

        # if not is_dist_avail_and_initialized() or is_main_process():
        sepoch = 'test' if epoch == None else str(epoch)
        print(print_menu.format(sepoch, mAP_target[0].item(),
                                mAP_target[1].item(),
                                mAP_target[2].item(),
                                mAP_target[3].item(),
                                mAP_target[4].item(),
                                mAP_target[5].item(),
                                iou_thresh))

        if self.save_path is not None:
            # save final results in a csv file
            logpath = os.path.join(self.save_path, self.set)
            if not os.path.exists(logpath):
                os.makedirs(logpath)
            logfile = os.path.join(logpath,  'result_AP_' + str(iou_ind.item()) + '.csv')
            logger_content = [mAP_target[0].item(),
                              mAP_target[1].item(),
                              mAP_target[2].item(),
                              mAP_target[3].item(),
                              mAP_target[4].item(),
                              mAP_target[5].item(),
                              iou_thresh,
                              epoch]
            if not os.path.exists(logfile):
                headers = ['mAP_5', 'mAP_10', 'mAP_20', 'mAP_40', 'mAP_60', 'mAP_90', 'IoUThresh', 'Epoch']
                with open(logfile, 'a') as f:
                    fcsv = csv.writer(f)
                    fcsv.writerow(headers)
            with open(logfile, 'a') as f:
                fcsv = csv.writer(f)
                fcsv.writerow(logger_content)


    # def display_and_save_apmetrics(self, iou_thresh=None, epoch=None):
    #
    #     if iou_thresh is None:
    #         iou_thresh = 0.9
    #
    #     iou_ind = torch.argmin(torch.abs(self.iou_threshes - iou_thresh))
    #
    #     mAP_target = self.mAP[iou_ind, :]
    #
    #     print_menu = 'Epoch {:5s}: [| mAP_5={:.4f} | mAP_10={:.4f} | mAP_20={:.4f} | mAP_40={:.4f} |' \
    #                  ' | IoUThresh={:.4f} |]'
    #
    #     # if not is_dist_avail_and_initialized() or is_main_process():
    #     sepoch = 'test' if epoch == None else str(epoch)
    #     print(print_menu.format(sepoch, mAP_target[0].item(),
    #                             mAP_target[1].item(),
    #                             mAP_target[2].item(),
    #                             mAP_target[3].item(),
    #                             iou_thresh))
    #
    #     if self.save_path is not None:
    #         # save final results in a csv file
    #         logpath = os.path.join(self.save_path, self.set)
    #         if not os.path.exists(logpath):
    #             os.makedirs(logpath)
    #         logfile = os.path.join(logpath,  'result_AP_' + str(iou_ind.item()) + '.csv')
    #         logger_content = [mAP_target[0].item(),
    #                           mAP_target[1].item(),
    #                           mAP_target[2].item(),
    #                           mAP_target[3].item(),
    #                           iou_thresh,
    #                           epoch]
    #         if not os.path.exists(logfile):
    #             headers = ['mAP_5', 'mAP_10', 'mAP_20', 'mAP_40', 'IoUThresh', 'Epoch']
    #             with open(logfile, 'a') as f:
    #                 fcsv = csv.writer(f)
    #                 fcsv.writerow(headers)
    #         with open(logfile, 'a') as f:
    #             fcsv = csv.writer(f)
    #             fcsv.writerow(logger_content)


    def display_and_save_accmetrics(self, iou_thresh=None, epoch=None):

        if iou_thresh is None:
            iou_thresh = 0.9

        iou_ind = torch.argmin(torch.abs(self.iou_threshes - iou_thresh))
        T = self.iou_threshes.shape[0]

        acc5_final = self.acc_all[:T] / self.acc_all[-1]
        acc10_final = self.acc_all[T: 2 * T] / self.acc_all[-1]
        acc5_target = acc5_final[iou_ind]
        acc10_target = acc10_final[iou_ind]

        print_menu = 'Epoch {:5s}: [| acc5={:.4f} | acc10={:.4f} | IoUThresh={:.4f} |]'

        # if not is_dist_avail_and_initialized() or is_main_process():
        sepoch = 'test' if epoch == None else str(epoch)
        print(print_menu.format(sepoch, acc5_target.item(), acc10_target.item(), iou_thresh))

        if self.save_path is not None:
            # save final results in a csv file
            logpath = os.path.join(self.save_path, self.set)
            if not os.path.exists(logpath):
                os.makedirs(logpath)
            logfile = os.path.join(logpath,  'result_acc_' + str(iou_ind.item()) + '.csv')
            logger_content = [acc5_target.item(), acc10_target.item(), iou_thresh, epoch]
            if not os.path.exists(logfile):
                headers = ['acc5', 'acc10', 'IoUThresh', 'Epoch']
                with open(logfile, 'a') as f:
                    fcsv = csv.writer(f)
                    fcsv.writerow(headers)
            with open(logfile, 'a') as f:
                fcsv = csv.writer(f)
                fcsv.writerow(logger_content)






















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
















