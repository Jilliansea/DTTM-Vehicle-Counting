import sys
import os
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import numpy as np
from easydict import EasyDict as edict
from detection.tools.nms_between_bbs import nms_bbs

class NAS_FPN_Detection():
    def __init__(self):
        self.config_file = './detection/NAS_FPN/configs/retinanet_crop640_r50_nasfpn_50e_640.py'
        self.checkpoint_file = './detection/NAS_FPN/checkpoints/retinanet_crop640_r50_nasfpn_50e_20191225-b82d3a86.pth'
        #self.config_file = './configs/retinanet_r50_fpn_1x.py'
        #self.checkpoint_file = 'checkpoints/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        self.model = init_detector(self.config_file, self.checkpoint_file, device='cuda:0')

        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle',
            'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']
        self.select_cls = ['car', 'truck', 'bus']
        self.select_cls_id = [self.class_names.index(lab) for lab in self.select_cls]
   
    def build_config(self, cfg):
        config = edict()
        config.det_score = cfg.Detection.det_score
        config.nms_thr = cfg.Detection.nms_thr
        config.improve_recall = cfg.Detection.improve_recall
        return  config 

    def bbs_filter(self, imgs_bboxes, cfg, img):
        print('detection cfg', cfg)
        if cfg.improve_recall:
            score_thr = 0.2
        else:
            score_thr = 0.3

        bboxes = imgs_bboxes
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bboxes)
        ]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bboxes)
        assert bboxes.ndim == 2
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

        # score filter
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        # label filter
        inds_lab = np.zeros(labels.shape).astype(bool)
        for id in self.select_cls_id:
            inds_lab = inds_lab | (labels == id)

        # scale filter: bbs_scale < 2/3(image_scale) & car_aspect_ratio(h/w)<2
        bbs = bboxes[inds_lab, :]
        lls = labels[inds_lab]
        lls = np.expand_dims(lls, 1)
        bbs = np.hstack((bbs, lls)).tolist()  # [x1,y1,x2,y2,score, label_ind]

        inds_scale = (((bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1]))/(img.shape[0]*img.shape[1]))<0.6
        inds_car_aspect_ratio  =  (labels !=2) | ((labels == 2) & ((bboxes[:,3]-bboxes[:,1])/(bboxes[:,2]-bboxes[:,0])<1.5))

        filter_inds = inds_lab & inds_scale & inds_car_aspect_ratio
        bboxes = bboxes[filter_inds, :]
        labels = labels[filter_inds]
        labels[labels == self.class_names.index('bus')]=self.class_names.index('car')  # bus --> car

        # 
        labels = np.expand_dims(labels, 1)
        bboxes = np.hstack((bboxes, labels)).tolist() #[x1,y1,x2,y2,score, label_ind]


        # nms between diff classes
        bboxes = nms_bbs(bboxes, score_thr, cfg)        

        # display and save final results
        new_bbs = []
        for bbox in bboxes:
            bb = bbox[:4]
            score = bbox[4]
            label_text = self.class_names[
                int(bbox[-1])] if self.class_names is not None else 'cls {}'.format(int(bbox[-1]))
            bb.append(label_text)
            bb.append(score)
            new_bbs.append(bb)
        return new_bbs 
