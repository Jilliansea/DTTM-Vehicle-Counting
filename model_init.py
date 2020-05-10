# coding=utf-8
from detection.NAS_FPN.configs.detection_config import NAS_FPN_Detection
from detection.YOLOv3.configs.detection_config import YOLOv3_Detection
from easydict import EasyDict as edict
import cv2

def model_initializer():

    #Load detection model
    det_model_nas_fpn = NAS_FPN_Detection()
#    det_model_yolov3 = YOLOv3_Detection()
#    det_model_yolov3_model = det_model_yolov3.build_model()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=500, detectShadows=True)
 
    Model = edict()
    Model.det_model_nas_fpn =det_model_nas_fpn
#    Model.det_model_yolov3 = det_model_yolov3
#    Model.det_model_yolov3_model = det_model_yolov3_model
    Model.bg_subtractor = bg_subtractor
    #return det_model_nas_fpn, det_model_yolov3, det_model_yolov3_model, track_model
    return Model
