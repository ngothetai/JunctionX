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
from scipy.spatial import ConvexHull, convex_hull_plot_2d


device = torch.device('cuda')
matcher = KF.LoFTR(pretrained=None)
matcher.load_state_dict(torch.load("/home/anhalu/anhalu-data/junction_AITrack/HIT JuntionXHanoi 2023/weights/loftr_outdoor.ckpt")['state_dict'])
matcher = matcher.to(device).eval()
width = 0
height = 0



def load_torch_image(frame, device):
    frame = cv2.resize(frame, (640, 480))
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
            cv2.circle(list_frame[i], (int(j[0]) ,int(j[1])), color=(0, 0, 255), thickness=3, radius=1) 
            
    # for i in keypoint : 
    #     cv2.circle(frame, (int(i[0]), int(i[1])) , color=(0, 0, 255), thickness=3, radius=1)
    return list_frame 

def draw_line(list_frame, keypoints) :  
    for id in range(len(list_frame)):
        points = keypoints[id].astype(int)
        hull_list = []
        hull = cv2.convexHull(points)
        cnt = 0
        for i in hull:
            hull_list.append(hull)
            cv2.circle(list_frame[id], i[0], 10, (cnt*15, cnt*15, cnt*15), -1)
            print(i[0])
            cnt += 1
        # for i in points:
        #     cv2.circle(list_frame[id], i, 10, (255, 255, 255), -1)
        cv2.polylines(list_frame[id], hull_list, isClosed=True, color=(255, 255, 255), thickness=2)
        # print(points) // Chieu kim dong ho
        # cv2.imshow('Result', list_frame[id])
        # cv2.waitKey(20000)
    
    return list_frame
    


def process_video(list):
    global width, height 
    keypoints = []  
    cap = [] 
    for i in list: 
        cap.append(cv2.VideoCapture(i))
    while True : 
        
        list_frame = []
        for i in cap : 
            ret, frame = i.read()
            if not ret : 
                frame = np.zeros((width, height))
                
            if width == 0 and height == 0:
                width, height = frame.shape[0], frame.shape[1]
            
            list_frame.append(frame) 
        
        if len(keypoints) == 0 : 
            key1, key2 = None, None
            for i in range(len(list_frame)-1):
                key1, key2 = key_point(list_frame[i], list_frame[i+1]) 
                keypoints.append(key1) 
            
            keypoints.append(key2) 
        
        list_frame = draw_line(list_frame, keypoints) 
        
        bg = np.concatenate(list_frame, axis=1)
        
        cv2.imshow('bg', bg)
        
        if cv2.waitKey(100000) & 0xFF == ord('q') : 
            break 

    for i in cap : 
        i.release() 
    cv2.destroyAllWindows() 



process_video(['/home/anhalu/anhalu-data/junction_AITrack/Public_Test/videos/scene2cam_06/CAM_1.mp4', '/home/anhalu/anhalu-data/junction_AITrack/Public_Test/videos/scene2cam_06/CAM_2.mp4'])