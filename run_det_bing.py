from __future__ import division, print_function, absolute_import
import os
import sys
import cv2
import json
from PIL import Image
import argparse

from detection.NAS_FPN.configs.detection_config import NAS_FPN_Detection
from detection.NAS_FPN.detection_nas_fpn import det_main_nas_fpn

from detection.YOLOv3.configs.detection_config import YOLOv3_Detection
from detection.YOLOv3.detection_yolov3_main import det_main_yolov3
from PIL import Image
from deep_sort.track_frame_run import MOT

from back_ground_model.train_bg_subtracktor import train_bg_subtractor
from back_ground_model.merge_results import merge_results
from back_ground_model.det_bg_model import bg_detection

def detection_tracking_process(img, img_name, video_name, cfg,  Model, frame_nums):
    #print('In detection tracking process CFG:', cfg)
    det_model_nas_fpn = Model.det_model_nas_fpn
#    det_model_yolov3 = Model.det_model_yolov3
#    det_model_yolov3_model = Model.det_model_yolov3_model

    i = int(img_name.split('.')[0]) 
    if cfg.Detection.model == 'NAS-FPN':
        # Run Detection
        det_results = None
        if cfg.Tracking.sample and i % (cfg.Tracking.sample_frame + 1) == 0:
            print("no detection")
        else:
            # Background detection
            bg_results = []
            if cfg.BG.flag and (int(img_name.split('.')[0]) > cfg.BG.bg_frame_num*frame_nums): # background model
                bg_results = bg_detection(video_name, Model.bg_subtractor,  i, cfg, frame_nums, img) 
            #print('***bg', bg_results) 
            # Detecion  results        
            det_results = det_main_nas_fpn(cfg, det_model_nas_fpn, img, img_name.split('.')[0])
            # Merge detection results
            det_results = merge_results(det_results, bg_results)

    elif cfg.Detection.model == 'YOLOv3':
        det_results = None
        if cfg.Tracking.sample and i % (cfg.Tracking.sample_frame + 1) == 0:
            print("no detection")
        else:
            # Background detection
            bg_results = []
            if cfg.BG.flag and (int(img_name.split('.')[0]) > cfg.BG.bg_frame_num*frame_nums): # background model
                bg_results = bg_detection(video_name, Model.bg_subtractor,  i, cfg, frame_nums, img)
            # Detecion  results
            det_results = det_main_nas_fpn(cfg, det_model_nas_fpn, img)
            # Merge detection results
            det_results = merge_results(det_results, bg_results)

    #track_model.post_process()
    return  det_results
