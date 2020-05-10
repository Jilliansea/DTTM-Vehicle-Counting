#--coding:utf-8
from __future__ import division, print_function, absolute_import

from .application_util import frame_visualization
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .deep_sort.kcf_tracker import KCFTracker
from .convert_detection_to_tracking import *
from .tools.cal_IOU import IOU
import copy

def check_dir(img_path):
    if not os.path.exists(img_path):
        os.makedirs(img_path, exist_ok=True)

def create_detections(detections, polygon, min_height):
    if detections is None:
        return []
    detection_list = []

    for row in detections:
        bbox = row[0], row[1], row[2] - row[0], row[3] - row[1]
        label, confidence = row[4], row[5]
        feature = [float(1)]*12
        if bbox[3] < min_height:
            continue
        if not isInsidePolygon((bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2), polygon):
            continue
        detection_list.append(Detection(bbox, confidence, feature, label))
    return detection_list

def run_frame_kcf(frame_idx, vis, display, image,
              detection, polygon, min_detection_height,
              tracker, s_flag, sample_frame):
    pre_frame = vis.pre_frame
    results = []
    print("Processing frame %08d" % frame_idx)
    detections = create_detections(detection, polygon, min_detection_height)
    # Update tracker.
    tracker.predict(image)
    if s_flag and (frame_idx == 1 or frame_idx - pre_frame > sample_frame):
        tracker.update(detections, image)
        vis.update_frame(frame_idx)
    elif not s_flag:
        tracker.update(detections, image)

    # Update visualization.
    if display:
        vis.set_image(image.copy())
        vis.draw_detections(detections, np.asarray(polygon))
        vis.draw_kcf_trackers(tracker.tracks)

    # Store results.
    if not s_flag or (s_flag and (frame_idx == 1 or frame_idx - pre_frame > sample_frame)):
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox_t = copy.deepcopy(track.bbox)
            bbox_t[2:] = bbox_t[:2] + bbox_t[2:]
            track_and_det = 0
            if len(detections) != 0:
                for det in detections:
                    bbox_det = det.to_tlbr()
                    if IOU(bbox_t, bbox_det) > 0.9:
                        track_and_det = 1
                        break
            if track_and_det == 0:
                bbox_det = bbox_t
            results.append([
                frame_idx, track.track_id, bbox_det[0], bbox_det[1], bbox_det[2], bbox_det[3], track.label])

    return results

def run_frame(frame_idx, vis, display, image,
              detection, polygon, min_detection_height,
              tracker, s_flag, sample_frame):
    pre_frame = vis.pre_frame
    results = []
    print("Processing frame %08d" % frame_idx)
    detections = create_detections(detection, polygon, min_detection_height)

    # Update tracker.
    tracker.predict()
    if s_flag and (frame_idx == 1 or frame_idx - pre_frame > sample_frame):
        tracker.update(detections)
        vis.update_frame(frame_idx)
    elif not s_flag:
        tracker.update(detections)

    # Update visualization.
    if display:
        vis.set_image(image.copy())
        vis.draw_detections(detections, np.asarray(polygon))
        vis.draw_trackers(tracker.tracks)

    # Store results.
    if not s_flag or (s_flag and (frame_idx == 1 or frame_idx - pre_frame > sample_frame)):
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox_t = track.to_tlbr()
            track_and_det = 0
            if len(detections) != 0:
                for det in detections:
                    bbox_det = det.to_tlbr()
                    if IOU(bbox_t, bbox_det) > 0.9:
                        track_and_det = 1
                        break
            if track_and_det == 0:
                bbox_det = bbox_t
            results.append([frame_idx, track.track_id, bbox_det[0], bbox_det[1], bbox_det[2], bbox_det[3], track.label])

    return results

class T:
    def __init__(self, cfg):
        self.model = cfg.Tracking.model
        self.roi_path = cfg.roi_path
        self.sequence_dir = cfg.sequence_path
        self.detection_file = cfg.Detection.det_output_path
        self.output_file = cfg.Tracking.track_org_output_path
        self.result_path = cfg.Tracking.track_refine_output_path
        self.sample = cfg.Tracking.sample
        self.sample_frame = cfg.Tracking.sample_frame
        self.display_1 = cfg.Tracking.display_1
        self.display_2 = cfg.Tracking.display_2

        self.kalman_iou_distance = cfg.Tracking.Kalman.iou_distance
        self.kalman_max_age = cfg.Tracking.Kalman.max_age
        self.kalman_n_init = cfg.Tracking.Kalman.n_init
        self.kalman_velocity = cfg.Tracking.Kalman.velocity
        self.kalman_iou_distance_sample = cfg.Tracking.Kalman.iou_distance_sample
        self.kalman_max_age_sample = cfg.Tracking.Kalman.max_age_sample
        self.kalman_n_init_sample = cfg.Tracking.Kalman.n_init_sample
        self.kalman_velocity_sample = cfg.Tracking.Kalman.velocity_sample

        self.kcf_iou_distance = cfg.Tracking.KCF.iou_distance
        self.kcf_max_age = cfg.Tracking.KCF.max_age
        self.kcf_n_init = cfg.Tracking.KCF.n_init
        self.kcf_iou_distance_sample = cfg.Tracking.KCF.iou_distance_sample
        self.kcf_max_age_sample = cfg.Tracking.KCF.max_age_sample
        self.kcf_n_init_sample = cfg.Tracking.KCF.n_init_sample

class MOT:
    def __init__(self, seq_name, min_frame, cfg):
        self.args = T(cfg)
        self.seq_name = seq_name
        #self.min_frame = int(min_frame.split('.')[0].lstrip('0'))
        self.min_frame = 1
        self.polygon = self.__get_roi(seq_name)
        self.min_detection_height = 20
        self.sample_frame = self.args.sample_frame
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, 100)

        # Karman, no detection sample
        self.iou_distance = self.args.kalman_iou_distance
        self.max_age = self.args.kalman_max_age
        self.n_init = self.args.kalman_n_init
        self.velocity = self.args.kalman_velocity
        self.definition_name = 'definition.txt'
        self.tracker = None

        if self.args.model == 'Karman':
            if self.args.sample:
                self.iou_distance = self.args.kalman_iou_distance_sample
                self.max_age = self.args.kalman_max_age_sample
                self.n_init = self.args.kalman_n_init_sample
                self.velocity = self.args.kalman_velocity_sample
                self.definition_name = 'definition_sample.txt'
            self.tracker = Tracker(self.metric, self.args.sample, self.args.sample_frame,
                                   self.iou_distance, self.max_age, self.n_init, self.velocity)
        if self.args.model == 'KCF':
            if self.args.sample:
                self.iou_distance = self.args.kcf_iou_distance_sample
                self.max_age = self.args.kcf_max_age_sample
                self.n_init = self.args.kcf_n_init_sample
                self.definition_name = 'kcf_definition_sample.txt'
            else:
                self.iou_distance = self.args.kcf_iou_distance
                self.max_age = self.args.kcf_max_age
                self.n_init = self.args.kcf_n_init
                self.definition_name = 'kcf_definition.txt'
            self.tracker = KCFTracker(self.metric, self.args.sample, self.args.sample_frame,
                                      self.iou_distance, self.max_age, self.n_init)

        if self.args.display_1:
            check_dir(self.args.output_file + '/avi')
            self.visualizer = frame_visualization.Visualization(self.args.output_file, self.seq_name, self.min_frame)
        else:
            self.visualizer = frame_visualization.NoVisualization(self.min_frame)

        self.track_results = list()
        check_dir(self.args.output_file)

    def __get_roi(self, seq_name):
        seq_list = seq_name.split('_')
        roi_name = os.path.join(self.args.roi_path, seq_list[0] + '_' + seq_list[1] + '.txt')
        lines = open(roi_name, 'r').readlines()
        return [[int(l.split(',')[0]), int(l.split(',')[1])] for l in lines]

    def track_main(self, frame, detection, image):
        #frame = int(frame.split('.')[0].lstrip('0'))
        frame = int(frame.split('.')[0])
        frame_result = list()
        if self.args.model == 'Karman':
            frame_result = run_frame(frame, self.visualizer, self.args.display_1, image,
                                     detection, self.polygon, self.min_detection_height,
                                     self.tracker, self.args.sample, self.sample_frame)
        elif self.args.model == 'KCF':
            frame_result = run_frame_kcf(frame, self.visualizer, self.args.display_1, image,
                                         detection, self.polygon, self.min_detection_height,
                                         self.tracker, self.args.sample, self.sample_frame)
        self.track_results.extend(frame_result)

    def post_process(self):
        print("End tracking, Begin refine")
        d = [data.split(' ') for data in
             open(os.path.join(os.getcwd(), 'deep_sort/' + self.definition_name), 'r').readlines() if
             data.split(' ')[0] == self.seq_name]
        d = d[0]
        check_dir(self.args.result_path)
        if len(d) > 4:
            track_processing(np.asarray(self.polygon), self.seq_name, os.path.join(self.args.sequence_dir, self.seq_name),
                             self.track_results, self.args.result_path,
                             min_distance_static=int(d[1]),
                             min_distance=int(d[2]), prev_frame=int(d[3]), start_frame=int(d[4]),
                             max_ang=int(d[5]), max_angs=int(d[6].replace('\n', '')), display=self.args.display_2,
                             associate_flag=True)
        else:
            track_processing(np.asarray(self.polygon), self.seq_name, os.path.join(self.args.sequence_dir, self.seq_name),
                             self.track_results, self.args.result_path,
                             min_distance_static=int(d[1].replace('\n', '')), display=self.args.display_2,
                             associate_flag=False)
