from __future__ import division
import cv2

from .models import *
from .utils.utils import *
from .utils.datasets import *

import os
import sys
import time
import datetime
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from .configs.detection_config import YOLOv3_Detection
from detection.tools.nms_score_scale_filter import detection_results_filter

def det_main_yolov3(config, det_model_main, det_model, org_image):
    image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    cfg = det_model_main.build_config(config)

    input_imgs = det_model_main.image_preprocess(image)
    input_imgs = Variable(input_imgs.type(det_model_main.Tensor))
    # Get detections
    start = time.time()
    with torch.no_grad():
        detections = det_model(input_imgs)
        detections = non_max_suppression(detections,  det_model_main.conf_thres,  det_model_main.nms_thres)
    # label filter
    bboxes = det_model_main.label_filter(detections, org_image)
    # post process (scale, score filter & nms)
    bboxes = detection_results_filter(cfg, bboxes, org_image)
    return bboxes


if __name__ == "__main__":
    det_model_main = YOLOv3_Detection()
    det_model = det_model_main.build_model()
    image_path = '/nfs/cold_project/data/AICityChallenge/2020/Track1/AIC20_track1/Dataset_A/cam_1/000001.jpg'
    import cv2
    image = cv2.imread(image_path)
    results = det_main_yolov3(det_model_main, det_model, image)
 
