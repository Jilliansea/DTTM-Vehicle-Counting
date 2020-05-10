import sys
import os
import cv2

from  back_ground_model.frame_preprocess import frame_preprocess

def train_bg_subtractor(inst, frame, cfg):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    w = cfg.BG.w
    h = cfg.BG.h
    w_stride = cfg.BG.w_stride
    h_stride = cfg.BG.h_stride

    _, frame = frame_preprocess(frame, w, h, w_stride, h_stride)
    inst.apply(frame, None, 0.001)
