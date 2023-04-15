import cv2 
import numpy as np 
import os
import numpy as np
import cv2
import csv
from glob import glob
import torch
import matplotlib.pyplot as plt
import kornia
from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF
import gc
import time 



device = torch.device('cpu')
matcher = KF.LoFTR(pretrained=None)
matcher.load_state_dict(torch.load("/home/anhalu/anhalu-data/junction_AITrack/lib/loftr_outdoor.ckpt")['state_dict'])
matcher = matcher.to(device).eval()



def load_torch_image(frame, device):
    h = 480
    w = 640
    frame = cv2.resize(frame, (w, h))
    frame = K.image_to_tensor(frame, False).float() /255.
    frame = K.color.bgr_to_rgb(frame)
    return frame.to(device)

def key_point(image_1, image_2) : 
    st = time.time()
    image_1 = load_torch_image(image_1, device) 
    image_2 = load_torch_image(image_2, device) 
    input_dict = {"image0": K.color.rgb_to_grayscale(image_1), 
            "image1": K.color.rgb_to_grayscale(image_2)}
    
    with torch.no_grad():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    gc.collect()
    nd = time.time()    
    print("Running time: ", nd - st, " s")
    
    return mkpts0, mkpts1 

def draw_keypoints(list_frame, keypoints) : 
    for i in range(len(list_frame)) : 
        for j in keypoints[i] : 
            # print(j)
            cv2.circle(list_frame[i], (int(j[0][0]) ,int(j[0][1])), color=(0, 0, 255), thickness=3, radius=1) 
            
    # for i in keypoint : 
    #     cv2.circle(frame, (int(i[0]), int(i[1])) , color=(0, 0, 255), thickness=3, radius=1)
    return list_frame 

def draw_line(list_frame, hulls) : 
    print(hulls)
    
    for i in range(len(list_frame)) :  
        for j in hulls[i] : 
            pass
        
        pass
    
    return list_frame
    
    # for i in range(len(list_frame)) :  
    #     for j in range(len(hulls[0]) + 1) : 
             
    #         cv2.line(list_frame[i], tuple(int(hulls[i][j])), tuple(int(hulls[i][(j+1)%len(hulls[0])])) , color= (0,0,255), thickness=1 )  
    
    return list_frame 

def process_video(list) : 
    keypoints = []  
    cap = [] 
    for i in list: 
        cap.append(cv2.VideoCapture(i))
    while True : 
        
        list_frame = []
        for i in cap : 
            ret, frame = i.read()
            if not ret : 
                frame = np.zeros((480, 640))
            frame = cv2.resize(frame, (480, 640))
            list_frame.append(frame) 
        
        if len(keypoints) == 0 : 
            key1, key2 = None, None
            for i in range(len(list_frame)-1):
                key1, key2 = key_point(list_frame[i], list_frame[i+1]) 
                keypoints.append(key1) 
            
            keypoints.append(key2) 
        
        hulls = [] 
        for i in keypoints : 
            hulls.append(cv2.convexHull(i).tolist()) 
        
        
        #list_frame = draw_keypoints(list_frame, hulls) 
        
        list_frame = draw_line(list_frame, hulls)
        
        bg = np.concatenate(list_frame, axis=1)
        
        
        cv2.imshow('bg', bg)
        
        if cv2.waitKey(1000) & 0xFF == ord('q') : 
            break 

    for i in cap : 
        i.release() 
    cv2.destroyAllWindows() 


process_video(['/home/anhalu/anhalu-data/junction_AITrack/Public_Test/videos/scene2cam_03/CAM_2.mp4','/home/anhalu/anhalu-data/junction_AITrack/Public_Test/videos/scene2cam_01/CAM_1.mp4', '/home/anhalu/anhalu-data/junction_AITrack/Public_Test/videos/scene2cam_01/CAM_2.mp4'])