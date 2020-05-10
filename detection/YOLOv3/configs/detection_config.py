from __future__ import division
import cv2

from detection.YOLOv3.models import *
from detection.YOLOv3.utils.utils import *
from detection.YOLOv3.utils.datasets import *

import os
import sys
import time
import datetime
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from PIL import Image
from easydict import EasyDict as edict

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class YOLOv3_Detection():
    def __init__(self):
        self.model_def = './detection/YOLOv3/configs/yolov3.cfg'
        self.weights_path = './detection/YOLOv3/weights/yolov3.weights'
        self.class_path = './detection/YOLOv3/data/coco.names'
        self.conf_thres = 0.1
        self.nms_thres = 0.4
        self.batch_size = 1
        self.n_cpu = 0
        self.img_size = 960
        self.classes = load_classes(self.class_path)
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def build_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Darknet(self.model_def, img_size=self.img_size).to(device)
        model.load_darknet_weights(self.weights_path)
        model.eval()
        
        return model

    def build_config(self, cfg):
        config = edict()
        config.det_score = cfg.Detection.det_score
        config.nms_thr = cfg.Detection.nms_thr
        config.improve_recall = cfg.Detection.improve_recall
        return  config
   
    def image_preprocess(self, img):
        # Extract image as PyTorch tensor
        org_img = transforms.ToTensor()(img)
        # Pad to square resolution
        img, _ = pad_to_square(org_img, 0)
        # Resize
        input_imgs = resize(img, self.img_size)
        # To tensor
        input_imgs = Variable(input_imgs.type(self.Tensor)).unsqueeze(0)
        return input_imgs

    def label_filter(self, detections, img):
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, self.img_size, img.shape[:2])
            results_box = []
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if self.classes[int(cls_pred)] != 'car' and self.classes[int(cls_pred)] != 'truck' and self.classes[int(cls_pred)] != 'bus':
                    continue
                results_box.append([x1.item(),  y1.item(), x2.item(), y2.item(), self.classes[int(cls_pred)],  round(cls_conf.item(),3)])
        return results_box
        






