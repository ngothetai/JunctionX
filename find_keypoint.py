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


device = torch.device('cpu')
matcher = KF.LoFTR(pretrained="indoor_new")
# matcher.load_state_dict(torch.load("lib/loftr_outdoor.ckpt")['state_dict'])
matcher = matcher.to(device).eval()



def load_torch_image(fname, device):
    img = cv2.imread(fname)
    scale = 840 / max(img.shape[0], img.shape[1]) 
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)


import time



st = time.time()
image_1 = load_torch_image('data1.png', device)
image_2 = load_torch_image('data2.png', device)
print(image_1.shape)
input_dict = {"image0": K.color.rgb_to_grayscale(image_1), 
            "image1": K.color.rgb_to_grayscale(image_2)}

with torch.no_grad():
    correspondences = matcher(input_dict)
    
mkpts0 = correspondences['keypoints0'].cpu().numpy()
mkpts1 = correspondences['keypoints1'].cpu().numpy()

Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0

gc.collect()
nd = time.time()    


print("Running time: ", nd - st, " s")
draw_LAF_matches(
KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                            torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                            torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                            torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                            torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
K.tensor_to_image(image_1),
K.tensor_to_image(image_2),
inliers,
draw_dict={'inlier_color': (0.2, 1, 0.2),
            'tentative_color': None, 
            'feature_color': (0.2, 0.5, 1), 'vertical': False})



# Load 2 input images
img1 = cv2.imread('data1.png')
img2 = cv2.imread('data2.png')

# Resize 2 images to the same size

# Create a new blank image with the same height and twice the width as the resized images
combined_img = np.zeros((img1.shape[0], img1.shape[1]*2, 3), dtype=np.uint8)

# Set the background color to gray
bg_color = (192, 192, 192)
cv2.rectangle(combined_img, (0, 0), (combined_img.shape[1], combined_img.shape[0]), bg_color, 2)

for i in mkpts0:
    img1 = cv2.circle(img1,(int(i[0]),int(i[1])),2,(0,0,255),2)

for i in mkpts1:
    img2 = cv2.circle(img2,(int(i[0]),int(i[1])),2,(0,0,255),2)

# Copy the first image onto the left half of the combined image
combined_img[:, :img1.shape[1], :] = img1

# Copy the second image onto the right half of the combined image
combined_img[:, img1.shape[1]:, :] = img2

# Display the combined image and wait for a key press
cv2.imshow('Combined Images', combined_img)
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()

