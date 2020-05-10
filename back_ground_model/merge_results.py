# -*- coding: utf-8 -*-
'''
Time:DATE{TIME}
Author: Wangzhihui
File:merge_results.py
Date:
Email:jillianwang@didiglobal.com
Company: Computer Vision at DiDi Map
Function: Please enter the function of script
'''
import os
import sys
import numpy as np
import json


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

def cal_max_iou_ind(box, bboxes):
    max_iou = 0
    ind = -1
    box = [box[0], box[1],box[2], box[3]]
    iou_list = []
    score_list = []
    for i, bbox_all in enumerate(bboxes):
        bbox = [bbox_all[0], bbox_all[1], bbox_all[2], bbox_all[3]]
        # if bbox == box:
        #     iou_list.append(0)
        #     continue
        iou = calc_iou(box, bbox)
        iou_list.append(iou)
        score_list.append(bbox_all[4])

    max_iou = max(iou_list)
    ind = iou_list.index(max_iou)
    return max_iou, ind


def merge_results(dets, bgs):
    
    final_bbs = []
    final_bgs = []
    if dets:
        if not bgs:
            final_bbs.extend(dets)
        else:
            final_bbs.extend(dets)
            
            final_bgs = []
            for bb in bgs:
                # 计算与检测结果的IOU
                max_iou, ind = cal_max_iou_ind(bb, dets)
                if max_iou <= 0.0:
                    final_bgs.append(bb)
    else:
        if bgs:
            final_bgs.extend(bgs)
    final_bbs.extend(final_bgs)
    return final_bbs
