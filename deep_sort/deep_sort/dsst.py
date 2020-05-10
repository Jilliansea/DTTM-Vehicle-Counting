import cv2
from . import kcf
import numpy as np

class TrackState:
  """
  Enumeration type for the single target track state. Newly created tracks are
  classified as `tentative` until enough evidence has been collected. Then,
  the track state is changed to `confirmed`. Tracks that are no longer alive
  are classified as `deleted` to mark them for removal from the set of active
  tracks.

  """

  Tentative = 1
  Confirmed = 2
  Deleted = 3

class CorrelationTracker:

  def __init__(self, img, bbox, track_id, n_init, max_age,
                 feature=None, label=""):
    self.bbox = bbox
    self.time_since_update = 0
    self.track_id = track_id
    self.hits = 1
    self.age = 1
    self.state = TrackState.Tentative
    self.features = []
    if feature is not None:
      self.features.append(feature)
    self._n_init = n_init
    self._max_age = max_age
    # ret = mean[:4].copy()
    # self.center = [(int(ret[0]), int(ret[1]))]
    self.label = label
    self.kcf = kcf.KCFTracker()
    self.kcf.init(self.bbox, img)

  def predict(self, img):
    self.bbox = np.asarray(self.kcf.update(img, None))
    self.age += 1
    self.time_since_update += 1
    # ret = self.mean[:4].copy()
    # self.center.append((int(ret[0]), int(ret[1])))

  def update(self, detection, img):
    self.features.append(detection.feature)
    self.label = detection.label
    self.hits += 1
    self.time_since_update = 0
    if self.state == TrackState.Tentative and self.hits >= self._n_init:
      self.state = TrackState.Confirmed

    '''re-start the tracker with detected positions (it detector was active)'''
    self.bbox = np.asarray(self.kcf.update(img, detection.tlwh))

    # if detection != []:
    #   self.tracker.start_track(img, rectangle(long(detection[0]), long(detection[1]), long(detection[2]), long(detection[3])))
    '''
    Note: another approach is to re-start the tracker only when the correlation score fall below some threshold
    i.e.: if bbox !=[] and self.confidence < 10.
    but this will reduce the algo. ability to track objects through longer periods of occlusions.
    '''

  def mark_missed(self):
    """Mark this track as missed (no association at the current time step).
    """
    if self.state == TrackState.Tentative:
      self.state = TrackState.Deleted
    elif self.time_since_update > self._max_age:
      self.state = TrackState.Deleted

  def is_tentative(self):
    """Returns True if this track is tentative (unconfirmed).
    """
    return self.state == TrackState.Tentative

  def is_confirmed(self):
    """Returns True if this track is confirmed."""
    return self.state == TrackState.Confirmed

  def is_deleted(self):
    """Returns True if this track is dead and should be deleted."""
    return self.state == TrackState.Deleted