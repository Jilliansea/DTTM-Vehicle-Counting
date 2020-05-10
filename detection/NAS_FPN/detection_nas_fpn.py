from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import time
import numpy as np
import sys
from .configs.detection_config import NAS_FPN_Detection
import cv2
from detection.tools.nms_score_scale_filter import detection_results_filter


def det_main_nas_fpn(config, det_model, img, img_name):
    model = det_model.model

    cfg = det_model.build_config(config)
    # inference 
    bboxes = inference_detector(model, img)
    # label filter
    #np.save('/nfs/cold_project/data/AICityChallenge/2020/Track1/AIC20_track1/SAMESAMESAME/Detection/NAS-FPN_Karman_sample1_0.2/cam_1/'+img_name.split('.')[0]+'txt', bboxes )
    #print('****bboxes',bboxes)
    bboxes = det_model.bbs_filter(bboxes, cfg, img)
    #print(bboxes)
    # post process (scale, score filter & nms)
    #bboxes = detection_results_filter(cfg, bboxes, img)
    return bboxes
    

if __name__ == "__main__":
    det_model = Detection()
    
    image_path = '/nfs/cold_project/data/AICityChallenge/2020/Track1/AIC20_track1/Dataset_A/cam_1/000001.jpg'
    img = cv2.imread(image_path)
    results = det_main(det_model, img)
    
