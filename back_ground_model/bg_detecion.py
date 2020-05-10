import os
import sys
import cv2
from back_ground_model.frame_preprocess import frame_preprocess
import numpy as np

class ContourDetection():
    '''
        Detecting moving objects.

        Purpose of this processor is to subtrac background, get moving objects
        and detect them with a cv2.findContours method, and then filter off-by
        width and height.

        bg_subtractor - background subtractor isinstance.
        min_contour_width - min bounding rectangle width.
        min_contour_height - min bounding rectangle height.
        save_image - if True will save detected objects mask to file.
        image_dir - where to save images(must exist).
    '''

    def __init__(self, bg_subtractor, video_name, min_contour_width=30, min_contour_height=30, cap_params=None):
        super(ContourDetection, self).__init__()

        self.bg_subtractor = bg_subtractor
        self.min_contour_width = min_contour_width
        self.min_contour_height = min_contour_height
        self.cap_params=cap_params
        self.video_name = video_name

    def filter_mask(self, img, a=None):
        '''
            This filters are hand-picked just based on visual tests
        '''

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        # Fill any small holes
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        # Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, kernel, iterations=2)

        return dilation

    def detect_vehicles(self, fg_mask, context):

        matches = []

        # finding external contours
        contours, hierarchy = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_valid = (w >= self.min_contour_width) and (
                h >= self.min_contour_height)
            if not contour_valid:
                continue
            bb = [x, y, x+w, y+h, 'car', 1.0]
            matches.append(bb)
        matches = self.roi_refine(matches)
        return matches

    def roi_refine(self, bbs):
        roi_mask_path = './back_ground_model/roi_refine/' + self.video_name + '_img_mask.jpg'
        if not os.path.exists(roi_mask_path):
            print('roi_mask_path',roi_mask_path)
            return bbs
        filter_bbs = []
        for bb in bbs:
            roi_img = cv2.imread('./back_ground_model/roi_refine/' + self.video_name + '_img_mask.jpg', 0)
            bb_area = int((bb[2] - bb[0])) * int((bb[3] - bb[1]))
            bb_patch_roi = roi_img[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]

            if len(np.where(bb_patch_roi < 255)[0]) < (0.4 * bb_area):  
                filter_bbs.append(bb)
        return filter_bbs

    def run(self, context):
        frame_org = context['frame'].copy()

        w, h, w_stride, h_stride  = self.cap_params
        frame_number = context['frame_number']

        ## bg extractor preprocess
        _, frame = frame_preprocess(frame_org, w, h, w_stride, h_stride)
        fg_mask = self.bg_subtractor.apply(frame, None, 0.001)

        ## frame  mask fuyuan
        fg_mask = (fg_mask.repeat(w_stride, axis=0)).repeat(h_stride, axis=1)

        # just thresholding values
        fg_mask[fg_mask >= 127] = 255
        fg_mask_reverse = cv2.bitwise_not(fg_mask)


        fg_mask = self.filter_mask(fg_mask_reverse, frame_number)
        fg_mask = cv2.bitwise_not(fg_mask)
        context['objects'] = self.detect_vehicles(fg_mask, context)
        context['fg_mask'] = fg_mask

        return context
