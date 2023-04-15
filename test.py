import matplotlib.pyplot as plt
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *

matcher = KF.LoFTR(pretrained='indoor_new')
def load_torch_image(fname):
    img = K.image_to_tensor(fname, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img


cam1 = cv2.VideoCapture('data/1.mp4')
cam2 = cv2.VideoCapture('data/2.mp4')


while True:

    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    
    fname1 = frame1
    fname2 = frame2

    img1 = K.geometry.resize(load_torch_image(fname1), (480, 640), antialias=True)
    img2 = K.geometry.resize(load_torch_image(fname2), (480, 640), antialias=True)


    matcher = KF.LoFTR(pretrained='indoor_new')

    input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
                "image1": K.color.rgb_to_grayscale(img2)}

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 1.0, 0.999, 100000)
    inliers = inliers > 0

    fig = draw_LAF_matches(
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
                'feature_color': (0.2, 0.5, 1), 'vertical': False}, return_axis=True)
    fig.savefig("out.jpg")
    img = cv2.imread("out.jpg")
    cv2.imshow("overlapping", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyWindow()
    


