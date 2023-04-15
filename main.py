import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *


matcher = KF.LoFTR(pretrained='indoor_new')

def load_torch_image(fname):
    #img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.image_to_tensor(fname, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img


def get_overlapping(frame1, frame2):

    fname1 = frame1
    fname2 = frame2


    img1 = K.geometry.resize(load_torch_image(fname1), (480, 640), antialias=True)
    img2 = K.geometry.resize(load_torch_image(fname2), (480, 640), antialias=True)


    input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
                "image1": K.color.rgb_to_grayscale(img2)}

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 1.0, 0.999, 100000)
    inliers = inliers > 0
        
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                'tentative_color': (1.0, 0.5, 1), 
                'feature_color': (0.2, 0.5, 1), 'vertical': False})

class VideoCaptureThread(threading.Thread):
    def __init__(self, src, width, height):
        threading.Thread.__init__(self)
        self.video = cv2.VideoCapture(src)
        self.width = width
        self.height = height
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def run(self):
        ret, frame = self.video.read()
        if not ret:
            self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            self.frame = cv2.resize(frame, (self.width, self.height))   

class MultiVideoCapture:
    def __init__(self, width, height, *args):
        self.n = len(args)
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
            for i in range(len(self.threads)):
                if row < 2:
                    self.background[:self.grid_height, self.grid_width*row:self.grid_width*(row+1)] = self.threads[i].frame
                    row += 1
                else:
                    self.background[self.grid_height:, self.grid_width*(row-2):self.grid_width*(row-2+1)] = self.threads[i].frame
                    row += 1
                ## Code insert model
                # if i == 1:
                #     frames1 = self.threads[0].frame
                # if i == 2:
                #     frames2 = self.threads[1].frame
                # get_overlapping(frames1, frames2)
                ##
                self.threads[i].run()

            cv2.imshow('Multi-Video Capture', self.background)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        for i in self.threads:
            self.threads.video.release()
        cv2.destroyAllWindows()

multi_video_capture = MultiVideoCapture(800, 600, './data/1.mp4', './data/2.mp4', './data/3.mp4', './data/4.mp4')
multi_video_capture.start()
multi_video_capture.display()
