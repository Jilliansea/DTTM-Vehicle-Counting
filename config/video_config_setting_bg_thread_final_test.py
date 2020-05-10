# -*- coding: utf-8 -*-
import os
import sys
from easydict import EasyDict as edict


MAIN_PATH = './results'
DETECTION_PATH = os.path.join(MAIN_PATH, 'Detection')
TRACKING_PATH = os.path.join(MAIN_PATH, 'Tracking')
COUNTING_PATH = os.path.join(MAIN_PATH, 'Counting')


class cam_1():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 1.0
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 30

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_1_dawn():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 1.0
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 30

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_1_rain():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_2():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        #self.Tracking.Kalman.iou_distance_sample = 0.95
        #self.Tracking.Kalman.max_age_sample = 6
        #self.Tracking.Kalman.n_init_sample = 2
        #self.Tracking.Kalman.velocity_sample = 1. / 60
        self.Tracking.Kalman.iou_distance_sample = 1.0
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 30
        
        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_2_rain():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_3():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 1.0
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 30

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_3_rain():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 1.0
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 30

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_4():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 960
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        #self.Tracking.Kalman.iou_distance_sample = 0.95
        #self.Tracking.Kalman.max_age_sample = 6
        #self.Tracking.Kalman.n_init_sample = 2
        #self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.Kalman.iou_distance_sample = 1.0
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 30
        
        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_4_dawn():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 960
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_4_rain():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 960
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_5():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        #self.Tracking.Kalman.iou_distance_sample = 0.95
        #self.Tracking.Kalman.max_age_sample = 6
        #self.Tracking.Kalman.n_init_sample = 1
        #self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 60
        
        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()
class cam_5_dawn():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample =1.0
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 30

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_5_rain():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_6():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_6_snow():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_7():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 960
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()
class cam_7_dawn():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_7_rain():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall
        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 960
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_8():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_9():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_10():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_11():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_12():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1920
        self.BG.h = 1080
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_13():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_14():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()
class cam_15():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()


class cam_16():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 2
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_17():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_18():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

class cam_19():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True  # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()
class cam_20():
    def __init__(self):
        # Detection params
        self.Detection = edict()
        self.Detection.det_score = 0.3
        self.Detection.nms_thr = 0.8
        self.Detection.model = 'NAS-FPN'
        self.Detection.improve_recall = True  # reduce detection score to improve recall

        self.BG = edict()
        self.BG.flag = False
        self.BG.bg_frame_num = 1./3
        self.BG.w = 1280
        self.BG.h = 720
        self.BG.w_stride = 10
        self.BG.h_stride = 10
        
        # Tracking Params
        self.Tracking = edict()
        self.Tracking.model = 'Karman'
        self.Tracking.sample = True # det every frame if False
        self.Tracking.sample_frame = 1
        self.Tracking.display_1 = False
        self.Tracking.display_2 = False
        self.Tracking.det_confidence = 0.3

        self.Tracking.Kalman = edict()
        self.Tracking.Kalman.iou_distance = 0.85
        self.Tracking.Kalman.max_age = 5
        self.Tracking.Kalman.n_init = 3
        self.Tracking.Kalman.velocity = 1. / 120
        self.Tracking.Kalman.iou_distance_sample = 0.95
        self.Tracking.Kalman.max_age_sample = 6
        self.Tracking.Kalman.n_init_sample = 1
        self.Tracking.Kalman.velocity_sample = 1. / 60

        self.Tracking.KCF = edict()
        self.Tracking.KCF.iou_distance = 0.85
        self.Tracking.KCF.max_age = 10
        self.Tracking.KCF.n_init = 2
        self.Tracking.KCF.iou_distance_sample = 0.95
        self.Tracking.KCF.max_age_sample = 10
        self.Tracking.KCF.n_init_sample = 2

        # Counting Params
        self.Counting = edict()

def video_config(video_name, main_path):

    config = eval("{}".format(video_name))()
    # Public Config
    config.roi_path = os.path.join(main_path, 'ROIs')
    config.sequence_path = os.path.join(main_path, 'Dataset_A')
    
    
    print(config.Detection, config.Tracking )
    # Config Detection results folder
    dir_name = 'results'

    det_output_path = os.path.join(DETECTION_PATH, dir_name)
    track_org_output_path = os.path.join(TRACKING_PATH, dir_name, 'BeforeRefine')
    track_refine_output_path = os.path.join(TRACKING_PATH, dir_name, 'AfterRefine')

    # Config Counting params
    count_output_path = os.path.join(COUNTING_PATH, dir_name, video_name)

    config.Detection.det_output_path = det_output_path
    config.Tracking.track_org_output_path = track_org_output_path
    config.Tracking.track_refine_output_path = track_refine_output_path
    config.Counting.count_output_path = count_output_path

    return config

