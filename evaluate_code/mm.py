# coding=utf-8

"""
Time:2020-04-05 13:16
Author: XingTengfei
File:mm.py
Email:xingtengfei@didiglobal.com
Company: Computer Vision at DiDi Map
"""

import os
import sys
import json
sys.path.insert(0, './evaluate_code')
import cal_line_seg_angle
import math

def load_one_traj(ori_list, pixel_interval):
    """
    point_list 转换成 轨迹段
    注意：输入的轨迹点是list格式，后面会转换成tuple
    :param ori_list:
    :return:
    """
    point_list = []
    # pre_point = ori_list[0]
    # pre_point = (int(pre_point[0]), int(pre_point[1]))
    # point_list.append(pre_point)
    for i in range(0, len(ori_list)-1):
        point_i = ori_list[i]
        point_j = ori_list[i+1]
        x_dis = abs(point_j[0] - point_i[0])
        y_dis = abs(point_j[1] - point_i[1])
        max_dis = max(x_dis, y_dis)
        point = (int(point_i[0]), int(point_i[1]), max_dis)
        if i > 1:
            if point[0] == pre_point[0] and point[1] == pre_point[1]:
                continue
        pre_point = point
        point_list.append(pre_point)
    point_list.append((int(ori_list[-1][0]), int(ori_list[-1][1]), 0))
    seg_list = []
    dis_all = 0
    for i in range(0, len(point_list)-1):
        if i == 0:
           pre_point = point_list[i][0:-1]
           dis_all += point_list[i][-1]
           continue
        if dis_all < pixel_interval:
            dis_all += point_list[i][-1]
            continue
        seg_list.append([pre_point, point_list[i][0:-1]])
        pre_point = point_list[i][0:-1]
        dis_all = point_list[i][-1]
    seg_list.append([pre_point, point_list[-1][0:-1]])
    return seg_list

def load_one_traj_frame(ori_list, fps_interval):
    """
    point_list 转换成 轨迹段
    注意：输入的轨迹点是list格式，后面会转换成tuple
    :param ori_list:
    :return:
    """
    # fps_interval = 10
    point_list = []
    pre_point = ori_list[0]
    pre_point = (int(pre_point[0]), int(pre_point[1]))
    point_list.append(pre_point)
    if len(ori_list) <= 10:
        end_point = (int(ori_list[-1][0]), int(ori_list[-1][1]))
        point_list.append(end_point)
    else:
        for i in range(0, len(ori_list), fps_interval):
            point = ori_list[i]
            point = (int(point[0]), int(point[1]))

            if point[0] == pre_point[0] and point[1] == pre_point[1]:
                continue
            pre_point = point
            point_list.append(pre_point)
    seg_list = []
    for i in range(1, len(point_list)):
        seg_list.append([point_list[i-1], point_list[i]])
    return seg_list

def load_traj_gt(traj_gt_file_name):
    """
    读入建模好的轨迹，输出字典格式
    :return:
    """
    # dir_path = '/Users/didi/Documents/AICity/debug_0405/trajectory'
    # file_name = 'cam_5.json'

    json_data = json.load(open(traj_gt_file_name))

    dic_trajes = {}
    for cam_name in json_data:
        for traj_name in json_data[cam_name]:
            ori_list = json_data[cam_name][traj_name]
            point_list = []
            for point in ori_list:
                point_list.append((int(point[0]), int(point[1])))
            dic_trajes[traj_name] = point_list

    # list of segment
    dic_traj_segs = {}
    for traj_name in dic_trajes:
        trajes = dic_trajes[traj_name]
        seg_list = []
        for i in range(1, len(trajes)):
            seg_list.append([trajes[i-1], trajes[i]])
        dic_traj_segs[traj_name] = seg_list

    # list of segment and weights
    for traj_name in dic_trajes:
        traj_segs = dic_traj_segs[traj_name]
        # print(traj_name)
        angles = [0]
        angle_sum = 0
        for i in range(1, len(traj_segs)):
            angle = cal_line_seg_angle.line_seg_angle(traj_segs[i-1], traj_segs[i])
            angles.append(angle)
            angle_sum += angle
        for i in range(len(traj_segs)):
            traj_segs[i].append(angles[i]/angle_sum)
    return dic_traj_segs

def cal_seg_length(seg):
    return cal_point_dist(seg[0], seg[1])

def cal_point_dist(point1, point2):
    dist = (point2[0] -point1[0])**2 + (point2[1] -point1[1])**2
    return math.sqrt(dist)

def cal_seg_traj_dist(seg, traj_segs_gt):
    seg_length = cal_seg_length(seg)+0.0001
    dist = float("inf")
    angle_diff = 0

    for index, seg_gt in enumerate(traj_segs_gt):
        temp_dist = cal_point_dist(seg[0], seg_gt[0]) + cal_point_dist(seg[1], seg_gt[1])
        temp_dist /= (2*seg_length)
        if dist > temp_dist:
            dist = temp_dist
            angle_diff = cal_line_seg_angle.line_seg_angle(seg, seg_gt)
            seg_nearest = index
    return dist, angle_diff, seg_nearest

def cal_traj_dist(traj_segs, traj_segs_gt):
    dist_total = 0
    angle_diff_total = 0
    seg_nearest_movement = 0
    for index, seg in enumerate(traj_segs):
        dist, angle_diff, seg_nearest = cal_seg_traj_dist(seg, traj_segs_gt)
        dist_total += dist
        angle_diff_total += angle_diff
        if index == len(traj_segs) - 1:
           seg_nearest_movement = seg_nearest
    dist_total /= len(traj_segs)
    angle_diff_total /= len(traj_segs)
    return dist_total, angle_diff_total, seg_nearest_movement

def mm(traj_segs, dic_traj_segs_gt):
    mm_result = {}
    for traj_name_gt in dic_traj_segs_gt:
        traj_segs_gt = dic_traj_segs_gt[traj_name_gt]

        dist, angle_diff, last_near_seg = cal_traj_dist(traj_segs, traj_segs_gt)
        left_dis = 1 - float(last_near_seg)/len(dic_traj_segs_gt[traj_name_gt])
        mm_result[traj_name_gt] = (dist, angle_diff, left_dis)
    return mm_result

def mm_nearest(traj_segs, dic_traj_segs_gt, movement_id):
    min_dis = 9999.0
    min_angle = 9999.0
    left_life = 0
    for traj_name_gt in dic_traj_segs_gt:
        if traj_name_gt.split("_")[0] != movement_id:
            continue
        traj_segs_gt = dic_traj_segs_gt[traj_name_gt]
        dist, angle_diff, last_near_seg = cal_traj_dist(traj_segs, traj_segs_gt)
        if dist < min_dis and angle_diff<min_angle:
            min_dis = dist
            min_angle = angle_diff
            left_life = 1 - float(last_near_seg)/len(dic_traj_segs_gt[traj_name_gt])
    return left_life

if __name__ == '__main__':
    json_infor = json.load(open('/Users/didi/Code/AICity2020/Track1/count_ids/compare_results/new_0407/cam_3.json'))
    for id in json_infor:
        bboxes = json_infor[id]['boxes']
        gt = load_one_traj(bboxes, 50)
        pred = load_one_traj(bboxes, 50)
