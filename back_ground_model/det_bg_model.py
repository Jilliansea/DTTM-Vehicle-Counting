# coding=utf-8
import os
import sys
import cv2

from back_ground_model.train_bg_subtracktor import train_bg_subtractor
from back_ground_model.bg_detecion import ContourDetection

def bg_detection(video_name, bg_subtractor, i, cfg, frame_nums, img):
    bg_results =  []
    context = {
        'frame': img,
        'frame_number': i,
        'cap_params': [cfg.BG.w,cfg.BG.h, cfg.BG.w_stride, cfg.BG.h_stride],
    }

    Contour_Detector = ContourDetection(bg_subtractor=bg_subtractor, video_name=video_name, cap_params=[cfg.BG.w,cfg.BG.h, cfg.BG.w_stride, cfg.BG.h_stride])

    bg_results = Contour_Detector.run(context)
    bg_results = bg_results['objects']
    return  bg_results
