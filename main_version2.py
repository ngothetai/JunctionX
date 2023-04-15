import argparse
import cv2
import numpy as np
import os
from copy import deepcopy

import torch
import time
import cv2
import numpy as np
import matplotlib.cm as cm

from utils.plotting import make_matching_figure
from utils.loftr import LoFTR, default_cfg


WIDTH = 1280
HEIGHT = 960
_default_cfg = deepcopy(default_cfg)
_default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()
'''
weights link : https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp

'''
class VideoCaptureThread():
    def __init__(self, src, width, height):
        self.ret = True
        self.video = cv2.VideoCapture(src)
        self.width = width
        self.height = height
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def run(self):
        ret, frame = self.video.read()
        if not ret:
            self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.ret = False
        else:
            self.frame = cv2.resize(frame, (self.width, self.height))   

class MultiVideoCapture:
    def __init__(self, width, height, *args):
        self.n = len(args)
        self.run = True
        self.width = width
        self.height = height
        self.grid_width = int(self.width/2)
        self.grid_height = int(self.height/2)

        self.background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.background = cv2.resize(self.background, (self.width, self.height))
        
        self.threads = []
        for cap in args:
            self.threads.append(VideoCaptureThread(cap, width=self.grid_width, height=self.grid_height))

    def start(self):
        for thread in self.threads:
            thread.start()

    def display(self):
        while True:
            row = 0
            column = 0
            frames = []
            
            for i in range(len(self.threads)):
                self.threads[i].run()
                frames.append(self.threads[i].frame.copy())
            
            if not self.threads[0].ret:
                break
            
            for i in range(len(self.threads)):
                if i == 0:
                    continue            
                    
                # Khop 2 anh lien tiep
                mkpts0, mkpts1, mconf = match_image(self.threads[i-1].frame, self.threads[i].frame)
                
                print(mkpts0.shape, mkpts1.shape, mconf.shape)
                
                # Hien thi
                ## Bao loi hinh 0
                points_0 = mkpts0.astype(int)
                hull_list_0 = cv2.convexHull(points_0)
                cv2.polylines(frames[i-1], [hull_list_0], isClosed=True, color=(255, 255, 255), thickness=6)
                
                ## Bao loi hinh 1
                points_1 = mkpts1.astype(int)
                hull_list_1 = cv2.convexHull(points_1)
                cv2.polylines(frames[i], [hull_list_1], isClosed=True, color=(255, 255, 255), thickness=6)
                
                
            for i in range(len(self.threads)):
                if row < 2:
                    self.background[:self.grid_height, self.grid_width*row:self.grid_width*(row+1)] = frames[i]
                    row += 1
                else:
                    self.background[self.grid_height:, self.grid_width*(row-2):self.grid_width*(row-2+1)] = frames[i]
                    row += 1

            cv2.imshow('Multi-Video Capture', self.background)

            if cv2.waitKey(500) & 0xFF == ord('q'):
                break

        for thread in self.threads:
            thread.video.release()
        cv2.destroyAllWindows()

def match_image(img0_raw, img1_raw):
    # Load images
    img0_raw = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2GRAY)
    img1_raw = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2GRAY)
    img0_raw = cv2.resize(img0_raw, (WIDTH//2, HEIGHT//2))
    img1_raw = cv2.resize(img1_raw, (WIDTH//2, HEIGHT//2))

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    start_time = time.time()
    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    print(f'runtime: {time.time() - start_time} s')
    return mkpts0, mkpts1, mconf

def run():
    """them toi da 4 cameras"""
    source = "/home/anhalu/anhalu-data/junction_AITrack/Public_Test/videos/scene_dynamic_cam_01"
    multi_video_capture = MultiVideoCapture(WIDTH, HEIGHT, f'{source}/CAM_1.mp4', f'{source}/CAM_2.mp4', f'{source}/CAM_3.mp4', f'{source}/CAM_4.mp4')
    multi_video_capture.display()
    


if __name__ == '__main__':
    run()
