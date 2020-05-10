# -*- coding: utf-8 -*-
'''
Time:DATE{TIME}
Author: Wangzhihui
File:nms_bbs.py
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

def cal_max_iou(box_ind, bboxes):
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
        score_list.append(bbox_all[4])
    iou_list[box_ind] = 0.
    score_list[box_ind] = 0.

    iou_array = np.array(iou_list)
    score_array = np.array(score_list)

    return iou_array, score_array


def nms_bbs(bboxes, score_thr, cfg):
    ## nms between diff classes
    final_new_bbs = copy.deepcopy(bboxes)
    dele_list = []
    for ind in range(len(bboxes)):
        if ind in dele_list:
            continue

        # iou_r, iou_ind, score_array, score_inds = cal_max_iou(ind, bboxes,img)
        iou_array, score_array = cal_max_iou(ind, bboxes)

        # nms for all score bbs
        ov_thr_inds = np.where(iou_array > cfg.nms_thr)[0].tolist()
        ov_thr_scores = score_array[iou_array > cfg.nms_thr].tolist()

        if ov_thr_inds:
            # compare score
            ov_thr_inds.append(ind)
            ov_thr_scores.append(bboxes[ind][-2])

            for ov_ind in ov_thr_inds:
                if ov_ind != ov_thr_inds[ov_thr_scores.index(max(ov_thr_scores))] and (ov_ind not in dele_list):
                    final_new_bbs.pop(final_new_bbs.index(bboxes[ov_ind]))
                    dele_list.append(ov_ind)

        # nms for low score bbs
        if cfg.improve_recall:
            if (ind not in dele_list) and (bboxes[ind][-2] > (score_thr - 0.1) and bboxes[ind][-2] < score_thr):
                sc_thr_inds = np.where(iou_array > 0.)[0].tolist()

                sc_thr_scores = score_array[iou_array > 0.].tolist() 
                if max(sc_thr_scores) > cfg.det_score:
                    final_new_bbs.pop(final_new_bbs.index(bboxes[ind]))
                    dele_list.append(ind)
    return final_new_bbs

