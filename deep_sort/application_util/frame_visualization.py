# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from .image_viewer import ImageViewer
import cv2
import time

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, min_frame):
        self.frame_idx = min_frame
        self.pre_frame = min_frame

    def update_frame(self, idx):
        self.pre_frame = idx


class Visualization(object):

    def __init__(self, output_file, seq_name, min_frame):
        image_shape = 960, 540
        self.viewer = ImageViewer(output_file, seq_name, 5, image_shape)
        self.viewer.thickness = 2
        self.pre_frame = min_frame
        self.frame_idx = min_frame

    def update_frame(self, idx):
        self.pre_frame = idx

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections, p):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        self.viewer.polygen(p)
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # if track.label == 'truck':
            #     self.viewer.thickness = 6
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
            self.viewer.trajectory(track.center)

    def draw_kcf_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # if track.label == 'truck':
            #     self.viewer.thickness = 6
            self.viewer.color = create_unique_color_uchar(track.track_id)
            r = track.bbox
            self.viewer.rectangle(int(r[0]), int(r[1]), int(r[2]), int(r[3]), label=str(track.track_id))