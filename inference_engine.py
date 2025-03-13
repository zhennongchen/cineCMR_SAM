import os
from tqdm import tqdm
import torch
import nibabel as nb
import numpy as np
from typing import Iterable
# from tensorboardX import SummaryWriter
import logging
from einops import rearrange
import utils.misc as misc
import utils.lr_sched as lr_sched
import cineCMR_SAM.functions_collection as ff


def save_predictions(view_type, batch, output,args, save_folder_patient):
    pred_seg = np.rollaxis(output["masks"].argmax(1).detach().cpu().numpy(), 0, 3)
                        
    original_shape = np.array([x.item() for x in batch["original_shape"]])
    centroid = batch["centroid"].numpy().flatten()
              
    crop_start_end_list = []
    for dim, size in enumerate([args.img_size, args.img_size]):
        start = max(centroid[dim] - size // 2, 0)
        end = start + size
        # Adjust the start and end if they are out of bounds
        if end > original_shape[dim]:
            end = original_shape[dim]
            start = max(end - size, 0)
        crop_start_end_list.append([start, end])
     
    final_pred_seg = np.zeros(original_shape)
    final_pred_seg[crop_start_end_list[0][0]:crop_start_end_list[0][1], crop_start_end_list[1][0]:crop_start_end_list[1][1]] = pred_seg

    original_image_file = batch["img_file"][0]          
    slice_index = batch["slice_index"].item()

    affine = nb.load(original_image_file).affine
    original_image = nb.load(original_image_file).get_fdata()
    original_image = original_image[:,:,:,slice_index] if view_type == 'sax' else np.copy(original_image)

    if view_type == 'sax':
        nb.save(nb.Nifti1Image(final_pred_seg, affine), os.path.join(save_folder_patient, 'pred_seg_slice%s.nii.gz' % slice_index))
        nb.save(nb.Nifti1Image(original_image, affine), os.path.join(save_folder_patient, 'img_slice%s.nii.gz' % slice_index))
    elif view_type == 'lax':
        lax_name = 'LAX'+ (batch["lax_type"][0][0])
        nb.save(nb.Nifti1Image(final_pred_seg, affine), os.path.join(save_folder_patient, lax_name + '_seg_pred.nii.gz'))
        nb.save(nb.Nifti1Image(original_image, affine), os.path.join(save_folder_patient, lax_name + '_img.nii.gz'))

