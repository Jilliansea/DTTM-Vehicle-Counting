# coding=utf-8
from __future__ import print_function, absolute_import
import os
import numpy as np
import sys
import argparse
import time
import cv2
import math
from .application_util.image_viewer import ImageViewer
from .application_util.frame_visualization import create_unique_color_uchar

def check_dir(img_path):
    if not os.path.exists(img_path):
        os.makedirs(img_path, 0o0755)

def convert_det_to_tracing(det_path):
    file_save = dict()
    print(det_path)
    information = open(det_path, 'r').readlines()
    for infor in information:
        index_c = []
        info = infor.split(",")
        index_c.append(float(info[2]))
        index_c.append(float(info[3]))
        index_c.append(float(info[4]))
        index_c.append(float(info[5]))
        index_c.append(float(info[6]))
        # index_c.append(float(info[1]))
        # index_c.append(float(info[2]))
        # index_c.append(float(info[3]))
        # index_c.append(float(info[4]))
        # index_c.append(float(info[5]))
        # index_c.append(float(info[6]))
        index_c.append(info[-1].strip('\n'))
        index_c.append(float(1))
        index_c.append(float(1))
        index_c.append(float(1))
        index_c.append(float(1))
        index_c.append(float(1))
        index_c.append(float(1))
        index_c.append(float(1))
        index_c.append(float(1))
        index_c.append(float(1))
        index_c.append(float(1))
        index_c.append(float(1))
        index_c.append(float(1))
        if int(info[0]) not in file_save.keys():
            file_save[int(info[0])] = []
        file_save[int(info[0])].append(index_c)

    return file_save

def write_results(imgs_path, output_file, seq, results, frames):
    check_dir(output_file)
    check_dir(output_file + '/txt')
    f = open(os.path.join(output_file + '/txt/', seq + '.txt'), 'w')
    img_list = sorted([img for img in os.listdir(imgs_path)])

    for img_name in img_list:
        frame = int(img_name.rstrip('.jpg').lstrip('0'))
        if frame in frames:
            data = results[frame]
            for t in data:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1,%s' % (
                    frame, t[0], t[1], t[2], float(t[1] + t[3]), float(t[2] + t[4]), t[-1]), file=f)
    f.close()

def draw_video(imgs_path, output_file, seq, results, frames, polygon):
    '''
    draw tracking results on each frame and save as a video.

    Parameters
    ----------
    imgs_path: str, absolute path of images on each frame
    output_file: str, absolute path of saved video
    seq: str, sequence of the video
    results: dict(list(dict())), first key: frame, second key: id
    frames: list(), sorted frames
    polygon: roi region of this video

    Returns
    -------

    '''
    # img_name = '{:06d}'.format(frames[0]) + '.jpg'
    # im = cv2.imread(os.path.join(imgs_path, img_name))
    # h,w,c = im.shape
    check_dir(output_file)
    check_dir(output_file+'/txt')
    w = 960
    h = 540
    check_dir(output_file+'/avi')
    viewer = ImageViewer(output_file, seq, 5, (w, h))
    f = open(os.path.join(output_file+'/txt/', seq+'.txt'), 'w')
    img_list = sorted([img for img in os.listdir(imgs_path)])

    pts = dict()
    for img_name in img_list:
        frame = int(img_name.rstrip('.jpg').lstrip('0'))
        viewer.image = cv2.imread(os.path.join(imgs_path, img_name), cv2.IMREAD_COLOR).copy()
        if frame in frames:
            # print(img_name)
            data = results[frame]
            for t in data:
                viewer.thickness = 2
                if t[-1] == 'truck':
                    viewer.thickness = 2
                viewer.color = create_unique_color_uchar(t[0])
                if t[0] not in pts.keys():
                    pts[t[0]] = []
                pts[t[0]].append((int(t[1])+int(t[3])/2, int(t[2])+int(t[4])/2))
                # x,y,w,h
                viewer.rectangle(int(t[1]), int(t[2]), int(t[3]), int(t[4]), label=str(t[0]))
                viewer.trajectory(pts[t[0]])
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1,%s' % (
                    frame, t[0], t[1], t[2], float(t[1] + t[3]), float(t[2] + t[4]), t[-1]), file=f)
        viewer.polygen(polygon)
        viewer.video_writer.write(cv2.resize(viewer.image, (w, h)))

    f.close()
    viewer.video_writer.release()

def isInsidePolygon(pt, poly):
    c = False
    i = -1
    l = len(poly)
    j = l - 1
    while i < l-1:
        i += 1
        # print(i, poly[i], j, poly[j])
        if ((poly[i][0] <= pt[0] and pt[0] < poly[j][0]) or (poly[j][0] <= pt[0] and pt[0] < poly[i][0])):
            if (pt[1] < (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (poly[j][0] - poly[i][0]) + poly[i][1]):
                c = not c
        j = i
    return c

def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def calu_moving_distance(start_p, end_p):
    dis_vec = ((end_p[0] - start_p[0]), (end_p[1] - start_p[1]))
    dis = math.sqrt(dis_vec[0]**2 + dis_vec[1]**2)
    return dis

def get_moving_vector(p1, p2):
    vec = p1[0], p1[1], p2[0], p2[1]
    return vec

def analyze_list(track_results):
    tracks = dict()
    for line in track_results:
        frame = line[0]
        id = line[1]
        if id not in tracks.keys():
            tracks[id] = []
        x1 = line[2]
        y1 = line[3]
        x2 = line[4]
        y2 = line[5]
        label = line[-1]
        tracks[id].append([frame, x1, y1, x2-x1, y2-y1, (x1+(x2-x1)/2.0, y1+(y2-y1)/2.0), label])

    for key, values in tracks.items():
        track_dict = {}
        for v in values:
            track_dict[v[0]] = v[1:]
        tracks[key] = track_dict
    return tracks

def analyze(file_path):
    lines = open(file_path, 'r').readlines()
    tracks = {}
    for line in lines:
        infos = line.strip('\n').split(',')
        frame = int(infos[0].lstrip('0'))
        id = int(infos[1])
        if id not in tracks.keys():
            tracks[id] = []
        x1 = float(infos[2])
        y1 = float(infos[3])
        x2 = float(infos[4])
        y2 = float(infos[5])
        label = infos[-1]
        tracks[id].append([frame, x1, y1, x2-x1, y2-y1, (x1+(x2-x1)/2.0, y1+(y2-y1)/2.0), label])

    for key, values in tracks.items():
        track_dict = {}
        for v in values:
            track_dict[v[0]] = v[1:]
        tracks[key] = track_dict
    return tracks

def complete_track(track):
    frames = track.keys()
    start_frame = min(frames)
    end_frame = max(frames)
    for i in range(start_frame+1, end_frame):
        if i not in track.keys():
            x0, y0 = track[i-1][4][0], track[i-1][4][1]
            j = i+1
            while j not in track.keys():
                j += 1
            x1, y1 = track[j][4][0], track[j][4][1]
            distance = (x1-x0)/(j-i+1), (y1-y0)/(j-i+1)
            center = x0 + distance[0], y0 + distance[1]
            mean_w = (track[i-1][2] + track[j][2])/2.0
            mean_h = (track[i-1][3] + track[j][3])/2.0
            track[i] = [center[0] - mean_w/2.0, center[1] - mean_h/2.0, mean_w, mean_h, (center[0], center[1]), track[j][-1]]

    return track

def short_associate(results, frames, tracks, min_distance_threshold, prev_frame, start_frame, max_ang, max_angs):
    # 对每个新目标id，与前3帧所有目标id计算轨迹相似度
    id_list = set()
    for i in frames[0:3]:
        for targets in results[i]:
            id_list.add(targets[0])

    for index, i in enumerate(frames[3:-3]):
        for target in results[i]:
            id = target[0]
            # 新轨迹
            if id not in id_list:
                # 计算新目标的运动方向
                target_next = tracks[id]
                frames_next = sorted(target_next.keys())
                tar_next = target_next[frames_next[-1]][4]
                direction_new = get_moving_vector(target[5], tar_next)

                # 寻找前5帧中与新目标中心距离小于100，且运动方向角度小于90度的轨迹
                min_distance = min_distance_threshold
                old_id = 0
                old_set = set()
                for k in range(i-1, i-prev_frame, -1):
                    if k not in frames: continue
                    # 对前5帧中的每一个目标
                    for old_target in results[k]:
                        if old_target[0] in old_set:
                            continue
                        # 跳过id延续的目标
                        old_set.add(old_target[0])
                        jump = False
                        for j in range(i+start_frame, i+10):
                            if j not in frames:
                                continue
                            if j >= frames[-1]: break
                            if old_target[0] in [m[0] for m in results[j]]:
                                jump = True
                                break
                        if jump:
                            continue
                        # 对于断裂轨迹，计算角度
                        target_prev = tracks[old_target[0]]
                        frames_prev = sorted(target_prev.keys())
                        tar_prev = target_prev[frames_prev[0]][4]
                        # 旧轨迹与新轨迹之间的角度
                        direction_old = get_moving_vector(tar_prev, old_target[5])
                        ang = angle(direction_old, direction_new)
                        # 约束旧轨迹的合理位置
                        direction_old_new = get_moving_vector(old_target[5], target[5])
                        angs = angle(direction_old, direction_old_new)
                        if ang >= max_ang or angs >= max_angs:
                            continue
                        distance = calu_moving_distance(old_target[5], target[5])
                        #print(i, k, id, old_target[0], ang, angs, distance)

                        if distance < min_distance:
                            #print(i, k, id, old_target[0], ang, angs, distance)
                            min_distance = distance
                            old_id = old_target[0]
                # 连接旧轨迹
                if old_id != 0:
                    print("old:" + str(old_id), " new:" + str(id))
                    m = index+3
                    while frames[m] in frames:
                        for new in results[frames[m]]:
                            if new[0] == id:
                                new[0] = old_id
                        m += 1
                        if m >= len(frames):
                            break
                # 开始下一帧新目标
                else:
                    id_list.add(id)
    return results

def is_remove(track, id, min_distance):
    frames = sorted(track.keys())
    start = track[frames[0]][4]
    end = track[frames[-1]][4]
    distance = calu_moving_distance(start, end)
    if distance < min_distance:
        #print(id, frames[0], frames[-1], distance)
        return True
    return False

def track_processing(polygon, seq, imgs_path, track_results, output_path, display, associate_flag,
            min_distance_static=0, min_distance=0, prev_frame=0, start_frame=0, max_ang=0, max_angs=0):

    #解析跟踪结果，保存为以id为key的字典存储
    tracks = analyze_list(track_results)
    start_time = time.time()
    # 提高轨迹质量
    for id in sorted(tracks.keys()):
        # 去掉长度小于两帧的track
        if len(tracks[id]) <= 2:
            #print(id)
            tracks.pop(id)
            continue
        # 补充由于漏检产生的相同轨迹断裂
        tracks[id] = complete_track(tracks[id])
        # 去掉静止目标的轨迹
        if is_remove(tracks[id], id, min_distance=min_distance_static):
            tracks.pop(id)

    # 转换为以frame为key的字典存储
    results = {}
    for key in tracks.keys():
        for frame, targets in tracks[key].items():
            if frame not in results.keys():
                results[frame] = []
            results[frame].append([key, targets[0], targets[1], targets[2], targets[3], targets[4], targets[-1]])
    frames = sorted(results)

    # 短时轨迹关联
    if associate_flag:
        results = short_associate(results, frames, tracks, min_distance, prev_frame, start_frame, max_ang, max_angs)

    end_time = time.time()
    if display:
        draw_video(imgs_path, output_path, seq, results, frames, polygon)
    else:
        write_results(imgs_path, output_path, seq, results, frames)

    print("Run Time:", end_time - start_time)


if __name__ == "__main__":

    pass
