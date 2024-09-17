import numpy as np
import glob 
import os
from PIL import Image
import math
from scipy import ndimage
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import RegularGridInterpolator
from nibabel.affines import apply_affine
from scipy.spatial import ConvexHull
from skimage.draw import polygon2mask
import re
import cv2 
from skimage.measure import label, regionprops
import torch
import torch.nn as nn
import torch.nn.functional as F

## important: DICE loss
def customized_dice_loss(pred, mask, num_classes, exclude_index = 10):
    # set valid mask to exclude pixels either equal to exclude index 
    valid_mask = (mask != exclude_index) 

    # one-hot encode the mask
    one_hot_mask = F.one_hot(mask, num_classes = exclude_index + 1).permute(0,3,1,2)


    # softmax the prediction
    pred_softmax = F.softmax(pred,dim = 1)

    pred_probs_masked = pred_softmax[:,1:num_classes,...] * valid_mask.unsqueeze(1)  # Exclude background class
    ground_truth_one_hot_masked = one_hot_mask[:,1:num_classes,...] * valid_mask.unsqueeze(1)
        
    # Calculate intersection and union, considering only the included pixels
    intersection = torch.sum(pred_probs_masked * ground_truth_one_hot_masked, dim=(0,2, 3))
    union = torch.sum(pred_probs_masked, dim = (0,2,3)) + torch.sum(ground_truth_one_hot_masked, dim=(0,2, 3))
        
    # Compute Dice score
    dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)  # Adding a small epsilon to avoid division by zero
        
    # Dice loss is 1 minus Dice score
    dice_loss = 1 - dice_score

    return torch.mean(dice_loss)

# function: define base, mid and apex segments given slice number N
def define_three_segments(N):
    arr = np.arange(0, N)

    # Calculate the lengths of the segments
    # Ensuring the middle segment has the most elements if the array can't be equally divided
    segment_length = N // 3
    segment_extra = N % 3

    if segment_extra == 1 or segment_extra == 0:
        # then assign this extra to mid
        base_segment = arr[:segment_length]
        mid_segment = arr[segment_length:2 * segment_length + segment_extra]
        apex_segment = arr[2 * segment_length + segment_extra:]

    else:
        # then assign 1 to base and assign 1 to mid
        base_segment = arr[:segment_length + 1]
        mid_segment = arr[segment_length + 1: 2 * segment_length + 2*1]
        apex_segment = arr[2 * segment_length + 2*1 :]

    return base_segment, mid_segment, apex_segment


# function: normalize the CMR image
def normalize_image(x, axis=(0,1,2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x-mu)/(sd+1e-8)

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)
