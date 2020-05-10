# coding: utf-8
# Copyright (c) 2020-present, DiDi, Inc.
# All rights reserved.
# 
#   File:count_num_use_in_out.py
#   Author: BaiBing
#   Email:baibing@didiglobal.com
#   Group: Computer Vision at DiDi Map
#   Created:2020-03-17
#   Description: {input this file description}

import os
import sys
import json
import time
import numpy as np
from collections import Counter
from .mm import *
import copy
import time
import math

# id_list = []
cam_name_to_id = {
    'cam_1':'1',
    'cam_1_dawn':'2',
    'cam_1_rain':'3',
    'cam_2':'4',
    'cam_2_rain':'5',
    'cam_3':'6',
    'cam_3_rain':'7',
    'cam_4':'8',
    'cam_4_dawn':'9',
    'cam_4_rain':'10',
    'cam_5':'11',
    'cam_5_dawn':'12',
    'cam_5_rain':'13',
    'cam_6':'14',
    'cam_6_snow':'15',
    'cam_7':'16',
    'cam_7_dawn':'17',
    'cam_7_rain':'18',
    'cam_8':'19',
    'cam_9':'20',
    'cam_10':'21',
    'cam_11':'22',
    'cam_12':'23',
    'cam_13':'24',
    'cam_14':'25',
    'cam_15':'26',
    'cam_16':'27',
    'cam_17':'28',
    'cam_18':'29',
    'cam_19':'30',
    'cam_20':'31',
}

fz_cam_name = {
    'cam_1_dawn':'cam_1',
    'cam_1_rain':'cam_1',
    'cam_2_rain':'cam_2',
    'cam_3_rain':'cam_3',
    'cam_4_dawn':'cam_4',
    'cam_4_rain':'cam_4',
    'cam_5_dawn':'cam_5',
    'cam_5_rain':'cam_5',
    'cam_6_snow':'cam_6',
    'cam_7_dawn':'cam_7',
    'cam_7_rain':'cam_7',
}

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def get_label(id, label_list, bboxes_truck, movement_id, indent, cam_name):
   
    cnt = Counter(label_list)
    label = cnt.most_common(1)[0][0]
 
    max_size_thr_cam  = ['cam_4', 'cam_4_dawn', 'cam_4_rain', 'cam_6', 'cam_6_snow', 'cam_10', 'cam_11', 'cam_12', 'cam_15']
    h_thr_cam = ['cam_3']

    if cam_name in max_size_thr_cam:
        # calculate analysis data
        w_list = []
        h_list = []
        side_list = []
        for i, bb in enumerate(bboxes_truck):
            w = bb[2] - bb[0]
            h = bb[3] - bb[1]
            w_list.append(w)
            h_list.append(h)
            side_list.append(math.sqrt(w*h))
        max_len = max(max(w_list), max(h_list))
        max_w = max(w_list)
        max_h = max(h_list)
        # start judgement
        if movement_id == '5':
            if max_len < 260 and label == 2:
                if indent == 1:
                    label = 1
                else:
                    label = 0
        elif movement_id != '5':
            if max_len < 400 and label == 2:
                if indent == 1:
                    label = 1
                else:
                    label = 0
            if cam_name == 'cam_4':
                if movement_id == '11' and label==2:
                    if max(side_list)<300:
                        label = 1
                elif movement_id == '10' and label==2:
                    if max(side_list)<450:
                        label =  1
            if cam_name == 'cam_4_dawn':
                if movement_id == '11' and label==2:
                    if max(side_list)<400:
                        label = 1
                elif movement_id == '10' and label==2:
                    if max(side_list)<450:
                        label = 1
            if cam_name == 'cam_5' and label == 2:
                if movement_id == '5' and max_h < 90:
                    label = 1
                if movement_id == '7' and max_h > 100:
                    label = 1
                if movement_id == '9' and max_h < 210:
                    label = 1
                if movement_id == '10' and max_w/max_h < 2.0:
                    label = 1
            if cam_name == 'cam_7' and label == 2:
                if movement_id == '2' and max_w <= 80:
                    label = 1

    elif cam_name in h_thr_cam:
        new_label_list = []
        for i, bb in enumerate(bboxes_truck):
            h = bb[3] - bb[1]        
            if h<55 and label_list[i] == 2:
                new_label_list.append(1)
            else:
                new_label_list.append(label_list[i])
        label_list = new_label_list
        cnt = Counter(label_list)
        label = cnt.most_common(1)[0][0]
    return label
    
    
def clean_tracks_first(all_ids, not_care_infor):
    if_else_indent = not_care_infor[-1]
    not_care = not_care_infor[0:-1]
    new_res = dict()
    delete_list = []
    #清除非目标轨迹的统计结果
    if if_else_indent == 1:
        #清除非目标轨迹的统计结果
        for id in all_ids:
            id_infor = all_ids[id]
            for not_care_movement in not_care:
                move_test = str(int(float(not_care_movement)))
                if move_test in id_infor['in'] or move_test in id_infor['out']:
                    delete_list.append(id)
                    continue
            new_res[id] = all_ids[id]
    else:
        for id in all_ids:
            id_infor = all_ids[id]
            indent = 0
            for not_care_movement in not_care:
                move_test = str(int(float(not_care_movement)))
                if move_test in id_infor['in'] and move_test in id_infor['out']:
                    indent = 1
                    delete_list.append(id)
                    continue
            if indent == 0:
                new_res[id] = all_ids[id]
    return new_res

def cal_all_ids_in_out(track_res, roi_config, not_care_infor):
    all_ids = dict()
    id_list = []
    for track_line in track_res:
        frame, id, x, y, x1, y1, _, _, _, _, label = track_line.strip().split(",")
        if id not in id_list:
            id_list.append(id)
        if label == 'truck':
            label = 2
        else:
            label = 1
        cal_box = [float(x), float(y), float(x1), float(y1)]
        save_box_infor = [(float(x1) + float(x)) / 2.0, (float(y1) + float(y)) / 2.0, int(float(frame)), label]
        if id not in all_ids:
            all_ids[id] = dict()
            all_ids[id]['in'] = dict()
            all_ids[id]['out'] = dict()
            all_ids[id]['boxes'] = []
            all_ids[id]['boxes_truck'] = []
            all_ids[id]['labels'] = []
        all_ids[id]['boxes'].append(save_box_infor)
        all_ids[id]['boxes_truck'].append(cal_box)
        all_ids[id]['labels'].append(label)
        for roi in roi_config:
            iou = calc_iou(cal_box, roi[0:-1])
            if iou > 0:
                if 'in' in roi[-1]:
                    m_roi_id = roi[-1].replace("in","")
                    if m_roi_id not in all_ids[id]['in']:
                        all_ids[id]['in'][m_roi_id] = dict()
                        all_ids[id]['in'][m_roi_id]['frame'] = []
                        all_ids[id]['in'][m_roi_id]['label'] = []
                    all_ids[id]['in'][m_roi_id]['frame'].append(frame)
                    all_ids[id]['in'][m_roi_id]['label'].append(label)
                else:
                    m_roi_id = roi[-1].replace("out","")
                    if m_roi_id not in all_ids[id]['out']:
                        all_ids[id]['out'][m_roi_id] = dict()
                        all_ids[id]['out'][m_roi_id]['frame'] = []
                        all_ids[id]['out'][m_roi_id]['label'] = []
                    all_ids[id]['out'][m_roi_id]['frame'].append(frame)
                    all_ids[id]['out'][m_roi_id]['label'].append(label)
    if not_care_infor != [0, 0]:
        all_ids = clean_tracks_first(all_ids, not_care_infor)
    return all_ids

def decision_true_false(config_rois, bboxes, in_id):
    yes_or_no = False
    for infor in config_rois:
        if '{}in'.format(in_id) not in infor[-1]:
            continue
        panduan_roi = infor[0:-1]
    x, y, x1, y1 = panduan_roi
    for index, box in enumerate(bboxes):
        if index > 10:
           continue
        x_b, y_b = box[0:2]
        if x < x_b < x1 and y < y_b < y1:
            yes_or_no = True
            break
    return yes_or_no

def find_movement_frame_label(id, in_ids, out_ids, labels, config_rois, bboxes, bbox_truck, cam_name):
    movement = []
    the_first_in = 9999999999
    the_first_in_id = 0
    the_last_out = 0
    the_last_out_id = 0
    for m in out_ids:
        if m in in_ids:
            movement.append(m)
    if len(movement) == 0:
        return 0, 0, 0
    for index, m in enumerate(movement):
        in_id = in_ids[m]['frame']
        out_id = out_ids[m]['frame']
        if index == 0:
            the_first_in = float(in_id[0])
            the_first_in_id = m
            the_last_out = float(out_id[-1])
            the_last_out_id = m
        else:
            if float(in_id[0]) <= the_first_in:
                #需要同时满足，入轨迹帧数最小，出轨迹帧数最大
                before_out_id = out_ids[str(int(the_first_in_id))]['frame'][-1]
                this_out_id = out_id[-1]
                if float(this_out_id) >= float(before_out_id):
                    the_first_in = float(in_id[0])
                    the_first_in_id = m
            if float(out_id[-1]) >= the_last_out:
               before_in_id = in_ids[str(int(the_last_out_id))]['frame'][0]
               this_in_id = in_id[0]
               if float(this_in_id) <= float(before_in_id):
                   the_last_out = float(out_id[-1])
                   the_last_out_id = m
    if the_first_in_id == the_last_out_id and the_first_in < the_last_out:
        if_point_in_rois = decision_true_false(config_rois, bboxes, the_first_in_id)
        if if_point_in_rois:
            out_frame = the_last_out
            out_movement = the_last_out_id
            out_label = get_label(id, labels, bbox_truck, out_movement, 1, cam_name)
            if out_label == 0:
                return 0, 0, 0
            return out_movement, out_frame, out_label
        else:
            return 0, 0, 0
    else:
        return 0, 0, 0


def count_nums_new(all_ids_in_out, config_rois, cam_name):
    movement_counts = []
    every_movement_counts_track_life = dict()
    dis_count_ids = dict()
    dis_count_ids['in'] = []
    dis_count_ids['out'] = []
    dis_count_ids['in_and_out'] = []
    dis_count_ids['no_in_and_out'] = []
    for id in all_ids_in_out:
        #轨迹不完整，判断为轨迹断裂，则不进行判断
        if len(all_ids_in_out[id]['out']) == 0:
            if len(all_ids_in_out[id]['in']) == 0:
                dis_count_ids['no_in_and_out'].append(id)
                continue
            else:
                dis_count_ids['in'].append(id)
                continue

        if len(all_ids_in_out[id]['in']) == 0:
            if len(all_ids_in_out[id]['out']) == 0:
                dis_count_ids['no_in_and_out'].append(id)
                continue
            else:
                dis_count_ids['out'].append(id)
                continue
        in_ids = all_ids_in_out[id]['in']
        out_ids = all_ids_in_out[id]['out']
        labels = all_ids_in_out[id]['labels']
        bboxes = all_ids_in_out[id]['boxes']
        bboxes_truck = all_ids_in_out[id]['boxes_truck']
        out_movement, out_frame, out_label = find_movement_frame_label(id, in_ids, out_ids, labels, config_rois, bboxes, bboxes_truck, cam_name)
        if out_movement == 0:
            dis_count_ids['in_and_out'].append(id)
            continue

        index = dict()
        index['movement'] = out_movement
        index['out_frame'] = out_frame
        index['label'] = out_label
        index['id_len'] = len(out_ids[out_movement]['frame'])
        index['id'] = id
        index['bboxes'] = bboxes
        movement_counts.append(index)
        if out_movement not in every_movement_counts_track_life:
            every_movement_counts_track_life[out_movement] = []
        every_movement_counts_track_life[out_movement].append(float(out_ids[out_movement]['frame'][-1]) - float(in_ids[out_movement]['frame'][0]))
    return movement_counts, dis_count_ids, every_movement_counts_track_life

def get_id_infor_use_track(all_id_boxes, all_ids, dic_traj_segs_gt, track_infor, cam_name):
    loss_id_infor = dict()
    too_little_tracks = dict()
    for id_type in all_ids:
        for id in all_ids[id_type]:
            # if id == '4027':
            #     a=1
            id_infor = all_id_boxes[id]
            id_infor_bboxes = []
            id_frame_count = []
            for box in id_infor['boxes']:
                id_infor_bboxes.append(box[0:2])
                id_frame_count.append(box[2])
            fps_interval = track_infor['inter_val']
            seg_list = load_one_traj(id_infor_bboxes, fps_interval)
            if len(seg_list) == 0:
                too_little_tracks[id] = 0
                continue
            mm_result = mm(seg_list, dic_traj_segs_gt)
            too_little_tracks[id] = mm_result
            min_dis = track_infor['min_dis']
            min_angle = track_infor['min_angle']
            save_movement = 0
            out_frame_dis = 0
            for movement_id in mm_result:
                dis, angle, left_dis_per = mm_result[movement_id]
                if angle < min_angle:
                    if dis < min_dis:
                        min_dis = dis
                        save_movement = movement_id
                        out_frame_dis = left_dis_per
            if save_movement == 0:
                continue
            if "_" in save_movement:
                save_movement = save_movement.split("_")[0]
            if save_movement not in loss_id_infor:
                loss_id_infor[save_movement] = dict()
            if id_type not in loss_id_infor[save_movement]:
                loss_id_infor[save_movement][id_type] = []
            index_infor = dict()
            index_infor['id'] = id
            index_infor['movement'] = save_movement
            get_label_name = get_label(id, all_id_boxes[id]['labels'], all_id_boxes[id]['boxes_truck'], save_movement, 0, cam_name)
            if get_label_name == 0:
                continue
            index_infor['label'] = get_label_name
            index_infor['first_frame'] = id_frame_count[0]
            index_infor['last_frame'] = id_frame_count[-1]
            index_infor['appear_len'] = len(id_frame_count)
            index_infor['left_dis_per'] = out_frame_dis
            loss_id_infor[save_movement][id_type].append(index_infor)
            too_little_tracks[id] = mm_result
    return loss_id_infor

def get_one_out_infor(infor):
    last_frame = 0
    label = 0
    track = 0
    if len(infor['out']) != 0:
        last_frame = 0
        for track_id in infor['out']:
            if float(last_frame) < float(infor['out'][track_id]['frame'][-1]):
                last_frame = infor['out'][track_id]['frame'][-1]
                track = track_id
                label = Counter(infor['out'][track_id]['label']).most_common(1)[0][0]
    elif len(infor['in']) != 0:
        last_frame = 0
        for track_id in infor['in']:
            if float(last_frame) < float(infor['in'][track_id]['frame'][-1]):
                last_frame = infor['in'][track_id]['frame'][-1]
                track = track_id
                label = Counter(infor['in'][track_id]['label']).most_common(1)[0][0]
    return track, last_frame, label

def concat_tracks(all_ids_in_out, dis_count_ids, every_movement_life_time, max_frame_len, not_care_infor, dic_traj_segs_gt, track_infor, cam_name):
    not_care_infor = not_care_infor[0:-1]
    movement_lift_time = dict()
    movement_life_list = []
    count_num_infors = []
    for movement_id in every_movement_life_time:
        movement_lift_time[movement_id] = np.mean(every_movement_life_time[movement_id])
        movement_life_list.append(movement_lift_time[movement_id] / len(every_movement_life_time[movement_id]))

    for i in range(1, 13):
        if str(i) not in movement_lift_time:
            movement_lift_time[str(i)] = np.mean(movement_life_list)
    loss_id_information = get_id_infor_use_track(all_ids_in_out, dis_count_ids, dic_traj_segs_gt, track_infor, cam_name)
    for movement_num in loss_id_information:
        movement_infor = loss_id_information[movement_num]
        if 'in' not in movement_infor or 'out' not in movement_infor:
            for index, type_str in enumerate(movement_infor):
                if index == 0:
                    loss_id_information[movement_num] = movement_infor[type_str]
                    continue
                loss_id_information[movement_num] += movement_infor[type_str]
            continue
        in_movement_infor = movement_infor['in']
        out_movement_infor = copy.deepcopy(movement_infor['out'])
        new_movement_infor = []
        this_movement_life_time = movement_lift_time[movement_num]
        for in_infor in in_movement_infor:
            indent = 0
            last_frame = in_infor['last_frame']
            left_lift_time = in_infor['left_dis_per']
            if left_lift_time < 0.3:
                new_movement_infor.append(in_infor)
                continue
            for out_infor in out_movement_infor:
                first_frame = out_infor['first_frame']
                #在当前入轨迹剩余生命周期内是否有目标出去，如有，则认为是脏轨迹，kill
                if abs(last_frame - first_frame) < this_movement_life_time*left_lift_time and last_frame < first_frame:
                    out_movement_infor.remove(out_infor)
                    indent = 1
            if indent == 0:
               new_movement_infor.append(in_infor)
        loss_id_information[movement_num] = new_movement_infor
        for type_str in movement_infor:
            if type_str == 'in':
                continue
            loss_id_information[movement_num] += movement_infor[type_str]
    for id in loss_id_information:
        for infor in loss_id_information[id]:
             index = dict()
             track = infor['movement']
             last_frame = infor['last_frame']
             label = infor['label']
             left_lift_frame = infor['left_dis_per'] * movement_lift_time[track] - 1
             #测试使用
             obj_id = infor['id']
             out_frame = last_frame + int(left_lift_frame)
             if out_frame >= max_frame_len:
                 out_frame = last_frame
             if str(int(float(track))) in not_care_infor:
                 continue
             index['movement'] = track
             index['label'] = label
             index['out_frame'] = out_frame
             #测试使用
             index['id'] = obj_id
             count_num_infors.append(index)
    return count_num_infors

def add_infor(movement_counts, every_movement_life_time, max_frame_len, dic_traj_segs_gt):
    movement_life_time = dict()
    for id in every_movement_life_time:
        if id not in movement_life_time:
            movement_life_time[id] = np.mean(every_movement_life_time[id])
    save_results = []
    for infor in movement_counts:
        index = dict()
        bboxes = infor['bboxes']
        movement_id = infor['movement']
        seg_list = [load_one_traj(bboxes, 3)[-1]]
        left_life_per = mm_nearest(seg_list, dic_traj_segs_gt, movement_id)
        add_len = movement_life_time[movement_id] * left_life_per - 1
        if int(float(infor['out_frame']) + add_len) <= max_frame_len:
            infor['out_frame'] = int(float(infor['out_frame']) + add_len)
        index['out_frame'] = infor['out_frame']
        index['movement'] = movement_id
        index['label'] = infor['label']
        index['id'] = infor['id']
        save_results.append(index)
    return save_results

def main(cam_name, track_result_path):
    config_path = './evaluate_code/config/1570_add_no_obj.json'
    length_path = './evaluate_code/config/max_len_num.json'
    not_care_infors = json.load(open('./evaluate_code/config/not_care_movements.json'))
    traj_segs_gt_path = './evaluate_code/config/track_model'
    max_truck_size_path = './evaluate_code/config/truck_min_size.json'
    save_res_path = './count_nums/'
    check_dir(save_res_path)
    track_infor_all = json.load(open('./evaluate_code/config/track_model_paras.json'))
    track_infor = track_infor_all[cam_name]
    cam_res_save_path = os.path.join(save_res_path, '{}.txt'.format(cam_name))
    if cam_name in fz_cam_name:
        dic_traj_segs_gt = load_traj_gt(os.path.join(traj_segs_gt_path, '1604_{}.json'.format(fz_cam_name[cam_name])))
    else:
        dic_traj_segs_gt = load_traj_gt(os.path.join(traj_segs_gt_path, '1604_{}.json'.format(cam_name)))
    cam_label = cam_name_to_id[cam_name]
    track_res_path = os.path.join(track_result_path, '{}.txt'.format(cam_name))
    file_s = open(cam_res_save_path, 'w')
    #相同场景配置文件相同，根据名字取对应的场景信息
    if cam_name in fz_cam_name:
        config_infor = json.load(open(config_path, 'r'))[fz_cam_name[cam_name]]
        max_truck_infor = json.load(open(max_truck_size_path))[fz_cam_name[cam_name]]
    else:
        config_infor = json.load(open(config_path, 'r'))[cam_name]
        max_truck_infor = json.load(open(max_truck_size_path))[cam_name]
    max_frame_len = json.load(open(length_path, 'r'))[cam_name]

    #计算所有ID的进出口并进行记录
    if cam_name in not_care_infors:
        not_care_infor = not_care_infors[cam_name]
    else:
        not_care_infor = [0, 0]
    all_ids_in_out = cal_all_ids_in_out(open(track_res_path, 'r').readlines(), config_infor, not_care_infor)

    movement_counts, dis_count_ids, every_movement_life_time = count_nums_new(all_ids_in_out, config_infor, cam_name)

    #对不足平均生命周期的目标加入对应的长度作为出ROI区域帧数
    movement_counts_infor_write = add_infor(movement_counts, every_movement_life_time, max_frame_len, dic_traj_segs_gt)

    loss_id_infor = concat_tracks(all_ids_in_out, dis_count_ids, every_movement_life_time, max_frame_len, not_care_infor, dic_traj_segs_gt, track_infor, cam_name)

    if cam_name in fz_cam_name:
        add_cam_name = fz_cam_name[cam_name]
    else:
        add_cam_name = cam_name
    add_cam_infor = json.load(open('./evaluate_code/config/add_num_infor.json'))
    every_movement_count_nums = dict()
    if add_cam_name in add_cam_infor:
        #把同时有出入轨迹的计数车辆写入txt
        for infor in movement_counts_infor_write:
            track = infor['movement']
            frame = infor['out_frame']
            label = infor['label']
            id = infor['id']
            if track in add_cam_infor[add_cam_name]:
                if frame + add_cam_infor[add_cam_name][track] < max_frame_len:
                    frame += add_cam_infor[add_cam_name][track]
            file_s.write("{} {} {} {} {}\n".format(cam_label, int(frame), track, label, id))
        #把断裂和丢失未统计的轨迹车辆写入txt
        for infor in loss_id_infor:
            track = infor['movement']
            frame = infor['out_frame']
            label = infor['label']
            id = infor['id']
            if track in add_cam_infor[add_cam_name]:
                if float(frame) + add_cam_infor[add_cam_name][track] < max_frame_len:
                    frame = float(frame) + add_cam_infor[add_cam_name][track]
            file_s.write("{} {} {} {} {}\n".format(cam_label, int(frame), track, label, id))
    else:
        #把同时有出入轨迹的计数车辆写入txt
        for infor in movement_counts_infor_write:
            track = infor['movement']
            frame = infor['out_frame']
            label = infor['label']
            id = infor['id']
            file_s.write("{} {} {} {} {}\n".format(cam_label, int(frame), track, label, id))
        #把断裂和丢失未统计的轨迹车辆写入txt
        for infor in loss_id_infor:
            track = infor['movement']
            frame = infor['out_frame']
            label = infor['label']
            id = infor['id']
            file_s.write("{} {} {} {} {}\n".format(cam_label, int(frame), track, label, id))
    file_s.close()

def count_main(track_res_path,video_name):
    main(video_name, track_res_path)


if __name__ == '__main__':
    track_res_path = '/home/luban/aicity_test/aicity/results/Tracking_time1/0416/AfterRefine/txt/'
    save_path = './'
    for cam_name in cam_name_to_id:
        if cam_name == 'cam_4':
            print(cam_name)
            score = main(cam_name, track_res_path)
