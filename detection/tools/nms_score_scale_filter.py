# -*- coding: utf-8 -*-
'''
Time:DATE{TIME}
Author: Wangzhihui
File:single_test.py
Date:
Email:jillianwang@didiglobal.com
Company: Computer Vision at DiDi Map
Function: Please enter the function of script
'''
import os
import sys
import numpy as np
import json
import copy


def rect_intersection(rect1, rect2):
    rect = [max(rect1[0], rect2[0]),
            max(rect1[1], rect2[1]),
            min(rect1[2], rect2[2]),
            min(rect1[3], rect2[3])]
    rect[2] = max(rect[2], rect[0])
    rect[3] = max(rect[3], rect[1])
    return rect

def rect_area(rect):
    return float(max(0.0, (rect[2] - rect[0]) * (rect[3] - rect[1])))

def calc_iou(rect1, rect2):
    i = rect_intersection(rect1, rect2)
    area_i = rect_area(i)
    area1 = rect_area(rect1)
    area2 = rect_area(rect2)
    return area_i / (area1 + area2 - area_i)


def cal_max_iou_score(box_ind, bboxes):
    max_iou = 0
    ind = -1
    box = bboxes[box_ind]
    box = [box[0], box[1],box[2], box[3]]
    iou_list = []
    score_list = []
    for i, bbox_all in enumerate(bboxes):
        bbox = [bbox_all[0], bbox_all[1], bbox_all[2], bbox_all[3]]
        iou = calc_iou(box, bbox)
        iou_list.append(iou)
        score_list.append(bbox_all[5])
    iou_list[box_ind] = 0.
    score_list[box_ind] = 0.

    iou_array = np.array(iou_list)
    score_array = np.array(score_list)
    return iou_array, score_array

def score_scale_filter(bboxes, cfg, img):
    new_bbs = []
    for bbs in bboxes:
        bb = bbs[:4]
        label = bbs[4]
        score = bbs[5]
       
        if cfg.improve_recall:
            det_score = cfg.det_score  - 0.1
        else:
            det_score = cfg.det_score
        
        #det_score = cfg.det_score
        # score filter
        if score > det_score:
            # scale filter
            if (((bb[2] - bb[0]) * (bb[3] - bbs[1])) / (img.shape[0] * img.shape[1])) < 0.6:
                # car aspect ratio filter
                if label != 'car' or (label == 'car' and ((bb[3] - bb[1]) / (bbs[2] - bbs[0]) < 1.5)):
                    new_bbs.append(bbs)
    return new_bbs


def nms(bboxes, cfg):

    final_new_bbs = copy.deepcopy(bboxes)
    dele_list = []
    for ind in range(len(bboxes)):
        if ind in dele_list:
            continue

        iou_array, score_array = cal_max_iou_score(ind, bboxes)

        # nms for all score bbs
        ov_thr_inds = np.where(iou_array > cfg.nms_thr)[0].tolist()
        ov_thr_scores = score_array[iou_array > cfg.nms_thr].tolist()

        if ov_thr_inds:  # 有overlap>cfg.Detection.nms_thr的bbs
            # compare score
            ov_thr_inds.append(ind)
            ov_thr_scores.append(bboxes[ind][-1])

            for ov_ind in ov_thr_inds:
                if ov_ind != ov_thr_inds[ov_thr_scores.index(max(ov_thr_scores))] and (ov_ind not in dele_list):
                    final_new_bbs.pop(final_new_bbs.index(bboxes[ov_ind]))
                    dele_list.append(ov_ind)
        # nms for low score bbs
        if cfg.improve_recall:
            if (ind not in dele_list) and (
                    bboxes[ind][-1] > (cfg.det_score - 0.1) and bboxes[ind][-1] < cfg.det_score):  # 若bboxes[ind]得分大于0.2,小于0.3
                sc_thr_scores = score_array[iou_array > 0.].tolist()  # 与bb有overlap的bbs
                if sc_thr_scores and (max(sc_thr_scores) > cfg.det_score):
                    final_new_bbs.pop(final_new_bbs.index(bboxes[ind]))
                    dele_list.append(ind)
    return final_new_bbs

def detection_results_filter(cfg, bboxes, img):
    # scale and car aspect ratio filter
    bboxes = score_scale_filter(bboxes, cfg, img)
    # nms between diff classes
    bboxes = nms(bboxes, cfg)
    return bboxes

