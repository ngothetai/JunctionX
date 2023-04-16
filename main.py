import argparse
import cv2
import numpy as np
import os
from copy import deepcopy
import sys
import torch
import time
import cv2
import numpy as np
import matplotlib.cm as cm
import kornia as K
import kornia.feature as KF
import gc

from utils.plotting import make_matching_figure
from utils.loftr import LoFTR, default_cfg


WIDTH = 1280
HEIGHT = 960
device = torch.device('cuda')
_default_cfg = deepcopy(default_cfg)
_default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt

outdoor = "/home/anhalu/anhalu-data/junction_AITrack/HIT JuntionXHanoi 2023/weights/indoor_ds_new.ckpt"
loftr_outdoor = "/home/anhalu/anhalu-data/junction_AITrack/HIT JuntionXHanoi 2023/weights/loftr_outdoor.ckpt"
indoor = "/home/anhalu/anhalu-data/junction_AITrack/HIT JuntionXHanoi 2023/weights/indoor_ds_new.ckpt"

# matcher = LoFTR(config=_default_cfg)
matcher = KF.LoFTR(pretrained=None)
matcher.load_state_dict(torch.load(loftr_outdoor)['state_dict'])
matcher = matcher.eval().cuda()
'''
weights link : https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp

'''
class VideoCaptureThread():
    def __init__(self, src, width, height, name):
        self.name = name
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
    def __init__(self, width, height, path, videos_name):
        self.n = len(videos_name)
        self.run = True
        self.width = width
        self.height = height
        self.grid_width = int(self.width/2)
        self.grid_height = int(self.height/2)

        self.background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.background = cv2.resize(self.background, (self.width, self.height))
        
        self.threads = []
        
        for name in videos_name:
            self.threads.append(VideoCaptureThread(os.path.join(path, name), width=self.grid_width, height=self.grid_height, name=name))

    def display(self):
        for i in range(len(self.threads)):
            f = open(f'{self.threads[i].name}.txt', mode='r')
            f.close()
        frame_count = 0
        
        while True:
            row = 0
            column = 0
            frames = [] # Chi de hien thi
            
            frame_count += 1
            
            for i in range(len(self.threads)):
                self.threads[i].run() # Cap nhat frame
                frames.append(self.threads[i].frame.copy())
            
            if not self.threads[0].ret:
                break
            
            croped = self.threads[0].frame
            
            """CHI CAM 1 VA CAM 2"""
            
            # Tach nen vung chong lan khi so sanh khung hinh 1 va 2
            mkpts0, mkpts1, time = match_image(croped, self.threads[1].frame)
            
            if len(mkpts0) > 2:
                points_0 = mkpts0.astype(int)
                hull_list_0 = cv2.convexHull(points_0)
                
                ## (2) make mask
                mask = np.zeros(self.threads[0].frame.shape[:2], np.uint8)
                cv2.drawContours(mask, [hull_list_0], -1, (255, 255, 255), -1, cv2.LINE_AA)

                ## (3) do bit-op
                croped = cv2.bitwise_and(self.threads[0].frame, self.threads[0].frame, mask=mask)
            
                ## Bao loi hinh 1
                cv2.polylines(frames[0], [hull_list_0], isClosed=True, color=(255, 255, 255), thickness=2)
            
            if len(mkpts1) > 2:
                ## Bao loi hinh 2
                points_1 = mkpts1.astype(int)
                hull_list_1 = cv2.convexHull(points_1)
                cv2.polylines(frames[1], [hull_list_1], isClosed=True, color=(255, 255, 255), thickness=2)
            
            """SO SANH CROPED VOI CAM 3 VA CAM 4"""
            
            # Xu ly hinh 3 va 4 dua tren croped
            for i in range(2, len(self.threads)):         
                # Khop anh
                mkpts0, mkpts1, time= match_image(croped, self.threads[i].frame)
                
                if len(mkpts1) > 2:
                    ## Bao loi hinh
                    points = mkpts1.astype(int)
                    hull_list = cv2.convexHull(points)
                    cv2.polylines(frames[i], [hull_list], isClosed=True, color=(255, 255, 255), thickness=2)
                
                
            for i in range(len(self.threads)):
                # Gan tung pixel vao background 
                if row < 2: # 2 hang tren
                    self.background[:self.grid_height, self.grid_width*row:self.grid_width*(row+1)] = frames[i]
                    row += 1
                else: # 2 hang duoi 
                    self.background[self.grid_height:, self.grid_width*(row-2):self.grid_width*(row-2+1)] = frames[i]
                    row += 1

            cv2.imshow('Multi-Video Capture', self.background)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        for thread in self.threads:
            thread.video.release()
        cv2.destroyAllWindows()

def load_torch_image(frame, device):
    frame = cv2.resize(frame, (WIDTH//2, HEIGHT//2))
    frame = K.image_to_tensor(frame, False).float() /255.
    frame = K.color.bgr_to_rgb(frame)
    return frame.to(device)

def match_image(img0_raw, img1_raw):
    st = time.time()
    image_1 = load_torch_image(img0_raw, device) 
    image_2 = load_torch_image(img1_raw, device) 
    input_dict = {"image0": K.color.rgb_to_grayscale(image_1), 
            "image1": K.color.rgb_to_grayscale(image_2)}
    
    with torch.no_grad():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    gc.collect()
    nd = time.time()    
    print("Running time: ", nd - st, " s")
    return mkpts0, mkpts1, nd - st

def run():
    """them toi da 4 cameras"""
    
    
    
    
    source = "/home/anhalu/anhalu-data/junction_AITrack/Public_Test/videos/"
    
    for foldername in os.listdir(source) : 
        scene_path = os.path.join(source, foldername) 
        videos_name = []
        
        for filename in os.listdir(scene_path) :
            videos_name.append(filename)
        
        team_name = "HIT.ATLN"
        
        os.mkdir(f'{team_name}/{foldername}')
            
        # multi_video_capture = MultiVideoCapture(WIDTH, HEIGHT, f'{source}/CAM_1.mp4', f'{source}/CAM_2.mp4', f'{source}/CAM_3.mp4', f'{source}/CAM_4.mp4')
        multi_video_capture = MultiVideoCapture(WIDTH, HEIGHT, scene_path, videos_name)
        multi_video_capture.display() 
            
        


if __name__ == '__main__':
    run()
