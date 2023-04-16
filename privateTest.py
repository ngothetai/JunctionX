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
import gc 
from utils.plotting import make_matching_figure
from utils.loftr import LoFTR, default_cfg

WIDTH = 1920 
HEIGHT = 1080 

device = torch.device('cuda')
_default_cfg = deepcopy(default_cfg)
_default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(torch.load("/home/anhalu/anhalu-data/junction_AITrack/github/JunctionX/weights/outdoor_ds.ckpt")['state_dict'])
matcher = matcher.eval().cuda()



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
    with torch.no_grad() :
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    print(f'runtime: {time.time() - start_time} s')
    return mkpts0, mkpts1, mconf 



def process(list_of_videos) : 
    keypoints = []  
    cap = [] 
    for i in list_of_videos: 
        cap.append(cv2.VideoCapture(i))
    while True : 
        
        list_frame = []
        for i in cap : 
            ret, frame = i.read()
            if not ret : 
                frame = np.zeros((WIDTH, HEIGHT))
            
            list_frame.append(frame) 
        
        print(list_frame) 
        

    for i in cap : 
        i.release() 
    cv2.destroyAllWindows() 
    

process(['/home/anhalu/anhalu-data/junction_AITrack/github/JunctionX/data/1.mp4', '/home/anhalu/anhalu-data/junction_AITrack/github/JunctionX/data/3.mp4'])









