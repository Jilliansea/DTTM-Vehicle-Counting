# coding=utf-8
import os
import sys
import time
import numpy as np
from queue import Queue
import threading
import cv2
import ffmpeg
import imageio
import argparse


from model_init import model_initializer
from config.video_config_setting_bg_thread_final_test import video_config
from video_read.video_read_func import get_video_info,read_frame_as_jpeg
from run_det_bing import detection_tracking_process
from back_ground_model.train_bg_subtracktor import train_bg_subtractor
from deep_sort.track_frame_run import MOT
from evaluate_code.count_num_concat_tracks import count_main


# Model Initialize
print('Load Models...')
Model =  model_initializer()

exitFlag = 0 
queueLock = threading.Lock()

def pipe_line(video_path, video_name, cfg, vid, frame_nums):
    # while True:
    global exitFlag
    global Model
    # global frame_ind
    exitFlag = 0
    print('{} images wait to put in queue.'.format(frame_nums))

    work_queue = Queue(frame_nums)
    thread_num = 2
    threads = []
    # make multi thread
    for id in range(thread_num):
        if id == 0:
            thread = VideoReader(id, 'thread-{}'.format(id), work_queue, frame_nums, vid, video_path)
            thread.start()
            threads.append(thread)
        else:
            thread = RunFramework(id, 'thread-{}'.format(id), work_queue, cfg, frame_nums, video_name)
            thread.start()
            threads.append(thread)

    while not exitFlag or not work_queue.empty():
        time.sleep(1)
        pass
    print("waiting all threads exits.")
    # exitFlag = 1
    time.sleep(1)
    for t in threads:
        t.join()

    print('end*****')

class VideoReader(threading.Thread):
    def __init__(self, thread_id, thread_name, queue, total_frames, vid,  video_path):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.thread_name = thread_name
        self.queue = queue
        self.total_frames  = total_frames
        self.frame_ind = 1
        self.vid = vid
        self.video_path =  video_path

    def run(self):
        # Read Video Frame
        # global frame_ind
        while self.frame_ind < self.total_frames:
            try:
                #self.frame_ind = 1
                image = self.vid.get_data(self.frame_ind-1)
                #out = read_frame_as_jpeg( video_path, self.frame_ind-1)
                #image_array = np.asarray(bytearray(out), dtype="uint8")
                #image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                #print('image', image)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_dict = dict()
                image_dict.setdefault(str(self.frame_ind).rjust(6, '0') + '.jpg', image)  # frame is start from 1
            except Exception as e:
                print(e)
            self.frame_ind += 1
            #print('get image: {}'.format(self.frame_ind))

            while True:
                if self.queue.qsize() < 1000:
                    # 
                    queueLock.acquire()
                    self.queue.put(image_dict)
                    queueLock.release()
                    #print('frame', self.frame_ind)
                    break
                else:
                    time.sleep(0.001)

        # if self.queue.qsize == 0 and self.frame_ind > self.total_frames :
        if self.frame_ind+1 > self.total_frames:
            global exitFlag
            exitFlag = True

class RunFramework(threading.Thread):
    def __init__(self, thread_id, thread_name, queue, cfg, frame_nums, video_name):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.thread_name = thread_name
        self.queue = queue
        self.cfg = cfg
        self.frame_nums = frame_nums
        self.video_name = video_name
        self.track_model = MOT(self.video_name, 1, self.cfg)

    def run(self):
        # Net Init
        # Track Init
        #print('Into RunFramework...')
        global exitFlag
        global Model
        while not exitFlag or self.queue.qsize()>0:
            queueLock.acquire()
            # print('lock2-open')
            if not self.queue.empty():
                data = self.queue.get()
                data_num = self.queue.qsize()
                queueLock.release()
                # detection
#                print('%%%%', data)
                image_name = list(data.keys())[0]
                image = data[image_name]

                # bacgground modeling
                if cfg.BG.flag:
                    if int(image_name.split('.')[0]) <= cfg.BG.bg_frame_num*self.frame_nums:
                        train_bg_subtractor(Model.bg_subtractor, self.image, self.cfg)
                #print('*****image', image)
                if int(image_name.split('.')[0]) ==1:
                    cv2.imwrite('test.jpg', image)
                det_results = detection_tracking_process(image, image_name, self.video_name, self.cfg, Model, self.frame_nums)
                self.track_model.track_main(image_name, det_results, image)
               
                
                if self.thread_name == 'thread-1':
                    print('{}: {} processing data, {} datas left'.format(self.video_name, self.thread_name, data_num))
            else:
                queueLock.release()
                # print('lock3-release')
                time.sleep(0.1)

        if self.thread_name == 'thread-1':

            # tracking post process
            self.track_model.post_process()
            count_main(cfg.Tracking.track_refine_output_path+'/txt', self.video_name)
            print('{} finished'.format(self.thread_name))

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="AICITY")
    parser.add_argument(
        "--video_dir", help="Dirs to save videos",
        default='', type=str, required=True)
    return parser.parse_args()



if __name__ == '__main__':

    #args = parse_args()
    #main_path = args.video_dir
    main_path = '/nfs/cold_project/data/AICityChallenge/2020/Track1/AIC20_track1/'
    video_dir = os.path.join(main_path, 'Dataset_A')
    videos = os.listdir(video_dir)
    videos = [video for video in videos if '.mp4' in video]
    print('**********videos', len(videos),videos)
    obj_videos = ['cam_1_dawn', 'cam_1_rain','cam_2','cam_2_rain','cam_3_rain','cam_4','cam_4_dawn','cam_4_rain', 'cam_5','cam_5_dawn', 'cam_5_rain','cam_7','cam_7_dawn','cam_7_rain','cam_8']
    for video in videos:
        print('******', video)
        if video.split('.')[0] not in ['cam_1']:
            continue
        #if video.split('.')[0] in obj_videos:
        #    continue
        #if video != 'cam_2.mp4' and video != 'cam_7.mp4' and video != 'cam_4.mp4' and video != 'cam_5.mp4':
        #    continue
        with open('time_record.txt', 'a') as f:
            start = time.time()
            video_name = video.split('.')[0]
            video_path = os.path.join(video_dir, video_name+'.mp4')
            print(video_path)
            cfg = video_config(video_name, main_path)
            print('**video_config', cfg)
            cap = cv2.VideoCapture(video_path)
            frame_nums = cap.get(7)
            vid = imageio.get_reader(video_path, 'ffmpeg')
            pipe_line(video_path, video_name, cfg, vid, frame_nums)
            end = time.time()
            f.write('{}: {}\n'.format(video, str(end-start)))
    

