#!/usr/bin/env python
# refer to dataset_zhennong.ipynb
# zhennong's way to load data

import sys
sys.path.append('/workspace/Documents')
import torch
import numpy as np
import os
import cv2
import math
import random
import pandas as pd
import nibabel as nb
import matplotlib.pylab as plt

from torch.utils.data import Dataset, DataLoader

import cineCMR_SAM.Data_processing as Data_processing
import cineCMR_SAM.functions_collection as ff
import cineCMR_SAM.dataset.CMR.random_aug as random_aug

# load pre-saved prompt features
lax_feature = np.load('/mnt/camca_NAS/SAM_for_CMR/data/text_prompt_clip/lax.npy')

# main
class Dataset_CMR_lax(torch.utils.data.Dataset):
    def __init__(
            self, 
            patient_list_spreadsheet_file,
            image_file_list,
            seg_file_list,

            seg_include_lowest_pixel = 100, # only include the segmentation with the lowest pixel number > seg_include_lowest_piexel in one slice
            turn_zero_seg_slice_into = None, # if there is no segmentation in this slice, then turn the pixel value of this slice into 10

            return_arrays_or_dictionary = 'dictionary', # "arrays" or "dictionary"
            relabel_LV  = None, # relabel the segmentation, 0 for background, 1 for LV, 2 for MYO, 3 for RV
            only_myo = None, # only keep MYO segmentation, 0 for background, 1 for MYO, 2 for others
            center_crop_according_to_which_class  = [1],

            image_shape = None, # [x,y], channel =  tf always 15 
            shuffle = None,
            image_normalization = None,
            augment_list = [('brightness' , None),  ('contrast', None), ('sharpness', None), ('flip', None), ('rotate', [-90,90]), ('translate', [-10,10])], # a list of augmentation methods and their range: v range = None for brightness, contrast, sharpness
            augment_frequency = 0., # how often do we do augmentation

            bounding_box_buffer = None,
    ):

        super().__init__()
        self.patient_list_spreadsheet = pd.read_excel(patient_list_spreadsheet_file)

        self.image_file_list = image_file_list
        self.seg_file_list = seg_file_list

        self.seg_include_lowest_pixel = seg_include_lowest_pixel
        self.turn_zero_seg_slice_into = turn_zero_seg_slice_into

        self.relabel_LV = relabel_LV
        self.only_myo = only_myo    
        self.center_crop_according_to_which_class = center_crop_according_to_which_class

        self.image_shape = image_shape
        self.shuffle = shuffle
        self.image_normalization = image_normalization
        self.augment_list = augment_list
        self.augment_frequency = augment_frequency
        self.return_arrays_or_dictionary = return_arrays_or_dictionary
        self.bounding_box_buffer = bounding_box_buffer

        # how many cases we have in this dataset?
        self.num_files = len(self.image_file_list)

        # the following two should be run at the beginning of each epoch
        # 1. get index array
        self.index_array = self.generate_index_array()
        # 2. some parameters
        self.current_image_file = None
        self.current_image_data = None 
        self.current_seg_file = None
        self.current_seg_data = None

    # function: how many cases do we have in this dataset? = total slice number accumulated from all cases
    def __len__(self):
        return self.num_files
        
    # function: we need to generate an index array for dataloader, it's a list, each element is [file_index, slice_index]
    def generate_index_array(self):
        np.random.seed()
        if self.shuffle == True:
            file_index_list = np.random.permutation(self.num_files)
        else:
            file_index_list = np.arange(self.num_files)

        index_array = np.copy(file_index_list).tolist()
        return index_array
    
    # function: load nii file, it will give you a 3D array [x, y, tf], tf is the temporal dimension always equal to 15
    def load_file(self, filename, segmentation_load = False):
        ii = nb.load(filename).get_fdata()
        if segmentation_load is True:
            ii = np.round(ii).astype(int)
         
        if self.relabel_LV is True and segmentation_load is True:
            iii = np.zeros(ii.shape); iii[ii == 1] = 3; iii[ii == 2] = 2; iii[ii == 3] = 1; ii = np.copy(iii)
        if self.only_myo is True and segmentation_load is True:
            iii = np.zeros(ii.shape); iii[ii != 2] = 0; iii[ii == 2] = 1;ii = np.copy(iii)
        return ii

    
    # function: get each item using the index [file_index, slice_index]
    def __getitem__(self, index):
        # print('I am in LAX!!!. in this geiitem, self.index_array is: ', self.index_array, ' and index is :', index)
        f = self.index_array[index]
        # print('index is: ', index, ' now we pick file ', f)
        image_filename = self.image_file_list[f]
        # print('image file is: ', image_filename, ' while current image file is: ', self.current_image_file)
        seg_filename = self.seg_file_list[f]
        # print('seg file is: ', seg_filename, ' while current seg file is: ', self.current_seg_file)
        if os.path.isfile(seg_filename) is False:
            self.have_manual_seg = False
        else:
            self.have_manual_seg = True

        # if it's a new case, then do the data loading; if it's not, then just use the current data
        if image_filename != self.current_image_file or seg_filename != self.current_seg_file:

            image_loaded = self.load_file(image_filename, segmentation_load = False) 

            self.original_shape = image_loaded.shape # [x,y,tf], before center crop

            if self.have_manual_seg is True:
                seg_loaded = self.load_file(seg_filename, segmentation_load=True) 

            # now we need to do the center crop for both image and seg
            # we have data volume as [x, y ,slice_num,tf], for each data volume , we use the [x,y, middle_slice, 0] for the centroid calculation (0 is ED, which always has segmentation)
            if self.have_manual_seg:
                _,_, self.centroid = Data_processing.center_crop( image_loaded[:,:, 0], seg_loaded[:,:, 0], self.image_shape, according_to_which_class = self.center_crop_according_to_which_class , centroid = None)
                # print('self.centroid is: ', self.centroid)

                if any(jjj[0] == 'random_crop' for jjj in self.augment_list) and np.random.uniform(0,1)  < self.augment_frequency:
                    parameter_index = next((i for i, x in enumerate(self.augment_list) if x[0] == 'random_crop'), None)
                    random_centriod_shift_x = np.random.randint(self.augment_list[parameter_index][1][0], self.augment_list[parameter_index][1][1])
                    random_centriod_shift_y = np.random.randint(self.augment_list[parameter_index][1][0], self.augment_list[parameter_index][1][1])
                    centroid_used_for_crop = [self.centroid[0] + random_centriod_shift_x, self.centroid[1] + random_centriod_shift_y]
                else:
                    centroid_used_for_crop = self.centroid
                
                # then for each dim in slice_num and tf, we do the center crop
                image_loaded_tem = np.zeros([self.image_shape[0],self.image_shape[1],image_loaded.shape[-1]])
                seg_loaded_tem = np.zeros([self.image_shape[0],self.image_shape[1],image_loaded.shape[-1]])
                for tf_dim in range(0,image_loaded.shape[-1]):
                    image_loaded_tem[:,:,tf_dim], seg_loaded_tem[:,:,tf_dim], _ = Data_processing.center_crop( image_loaded[:,:,tf_dim], seg_loaded[:,:,tf_dim], self.image_shape, according_to_which_class = self.center_crop_according_to_which_class , centroid = centroid_used_for_crop)
                image_loaded = np.copy(image_loaded_tem)
                seg_loaded = np.copy(seg_loaded_tem)
            else:
                # just do center crop
                self.centroid = np.array([image_loaded.shape[0]//2, image_loaded.shape[1]//2])
                # crop the image regarding the image center, just find the center and take 128x128 ROI
                image_loaded_tem = np.zeros([self.image_shape[0],self.image_shape[1],image_loaded.shape[-1]])
                seg_loaded_tem = np.zeros([self.image_shape[0],self.image_shape[1],image_loaded.shape[-1]])
                for tf_dim in range(0,image_loaded.shape[-1]):
                    image_loaded_tem[:,:,tf_dim] = image_loaded[self.centroid[0]- self.image_shape[0]//2:self.centroid[0] + self.image_shape[0]//2, self.centroid[1] - self.image_shape[1]//2:self.centroid[1] + self.image_shape[1]//2, tf_dim]
                image_loaded = np.copy(image_loaded_tem)
                seg_loaded = np.copy
            
            self.current_image_file = image_filename
            self.current_image_data = np.copy(image_loaded)  
            self.current_seg_file = seg_filename
            self.current_seg_data = np.copy(seg_loaded)

       
        original_image = np.copy(self.current_image_data)
        original_seg = np.copy(self.current_seg_data)
        
        ######## do augmentation: should we do the augmentation, it's based on the self.augment_list
        processed_seg = np.copy(original_seg)
        # (0) add noise
        if any(jjj[0] == 'noise' for jjj in self.augment_list) and np.random.uniform(0,1)  < self.augment_frequency:
            standard_deviation = 5
            processed_image = original_image + np.random.normal(0,standard_deviation,original_image.shape)
            # turn the image pixel range to [0,255]
            processed_image = Data_processing.turn_image_range_into_0_255(processed_image)
        else:
            processed_image = Data_processing.turn_image_range_into_0_255(original_image)
    
        # (1) do brightness
        if any(jjj[0] == 'brightness' for jjj in self.augment_list) and np.random.uniform(0,1)  < self.augment_frequency:
            parameter_index = next((i for i, x in enumerate(self.augment_list) if x[0] == 'brightness'), None)
            processed_image,v = random_aug.random_brightness(processed_image, v = self.augment_list[parameter_index][1])
            # print('did brightness augmentation, v: ', v, ' image max: ', np.max(processed_image), ' image min: ', np.min(processed_image)  )

        # (2) do contrast
        if  any(jjj[0] == 'contrast' for jjj in self.augment_list) and np.random.uniform(0,1)  < self.augment_frequency:
            parameter_index = next((i for i, x in enumerate(self.augment_list) if x[0] == 'contrast'), None)
            processed_image, v = random_aug.random_contrast(processed_image, v = self.augment_list[parameter_index][1])
            # print('did contrast augmentation, v: ', v, ' image max: ', np.max(processed_image), ' image min: ', np.min(processed_image))

        # (3) do sharpness
        if any(jjj[0] == 'sharpness' for jjj in self.augment_list) and np.random.uniform(0,1)  < self.augment_frequency:
            parameter_index = next((i for i, x in enumerate(self.augment_list) if x[0] == 'sharpness'), None)
            processed_image, v = random_aug.random_sharpness(processed_image, v = self.augment_list[parameter_index][1])
            # print('did sharpness augmentation, v: ', v, ' image max: ', np.max(processed_image), ' image min: ', np.min(processed_image))
            
        # (4) do flip
        if any(jjj[0] == 'flip' for jjj in self.augment_list) and np.random.uniform(0,1)  < self.augment_frequency:
            # doing this can make sure the flip is the same for image and seg
            a, selected_option = random_aug.random_flip(processed_image)
            b,_ = random_aug.random_flip(processed_seg, selected_option)
            processed_image = np.copy(a)
            processed_seg = np.copy(b)
            # print('did flip augmentation, selected option: ', selected_option)

        # (5) do rotate
        if any(jjj[0] == 'rotate' for jjj in self.augment_list)  and np.random.uniform(0,1)  < self.augment_frequency:
            parameter_index = next((i for i, x in enumerate(self.augment_list) if x[0] == 'rotate'), None)
            processed_image, z_rotate_degree = random_aug.random_rotate(processed_image, order = 0, z_rotate_range = self.augment_list[parameter_index][1])
            processed_seg,_ = random_aug.random_rotate(processed_seg, z_rotate_degree, fill_val = 0, order = 0)
            # print('did rotate augmentation, z_rotate_degree: ', z_rotate_degree)

        # (6) do translate
        if any(jjj[0] == 'translate' for jjj in self.augment_list) and np.random.uniform(0,1)  < self.augment_frequency:
            parameter_index = next((i for i, x in enumerate(self.augment_list) if x[0] == 'translate'), None)
            processed_image, x_translate, y_translate = random_aug.random_translate(processed_image, translate_range = self.augment_list[parameter_index][1])
            processed_seg,_ ,_= random_aug.random_translate(processed_seg, x_translate, y_translate)
            # print('did translate augmentation, x_translate: ', x_translate, ' y_translate: ', y_translate)

        # add normalization
        if self.image_normalization is True:
            processed_image = Data_processing.normalize_image(processed_image,denormalize = False) # do image normalization

        # find which times frame has segmentation, and put it into annotation frame list
        # also turn the pixel value of the slice without manual segmentation into 10
        processed_seg_no_class_10 = np.copy(processed_seg) # before we turn the pixel value of the slice without manual segmentation into 10, we need to keep a copy of the original processed seg
        if self.have_manual_seg is True:
            annotation_frame_list = []
            for tf in range(0,original_seg.shape[2]):
                s_i = np.copy(original_seg[:,:,tf])
                if np.sum(s_i==1) > self.seg_include_lowest_pixel:
                    annotation_frame_list.append(tf)
                else:
                    # turn the pixel value of this slice in processeed seg all into turn_zero_seg_slice_into
                    if self.turn_zero_seg_slice_into is not None:
                        processed_seg[:,:,tf] = self.turn_zero_seg_slice_into
        else:
            annotation_frame_list = []

        # prepare the bonunding box prompt:
        if self.have_manual_seg is True:
            bbox,z_ED, z_ES = Data_processing.get_bbox_from_mask_all_volumes(processed_seg, annotation_frame_list)
            # print('bounding box is: ', bbox, ' z_ED is: ', z_ED, ' z_ES is: ', z_ES)
        else:
            bbox = [0,0,0,0]; z_ED = 0; z_ES = 0
            
        # now it's time to turn numpy into tensor and collect as a dictionary (this is the final return)
        processed_image_torch = torch.from_numpy(processed_image).unsqueeze(0).float() 
        processed_seg_torch = torch.from_numpy(processed_seg).unsqueeze(0)  
        # print('processed_image_torch shape: ', processed_image_torch.shape, ' processed_seg_torch shape: ', processed_seg_torch.shape)

        # also need to return the original image and seg without the augmentation (with center crop done)
        original_image_torch = torch.from_numpy(original_image).unsqueeze(0).float()
        original_seg_torch = torch.from_numpy(original_seg).unsqueeze(0)
        
        # add the prompt feature (LAX prompt)
        slice_type = "lax"
        prompt_feature = lax_feature; prompt_feature = np.squeeze(prompt_feature)

        # also add infos from patient list spread sheet
        patient_id = os.path.basename(os.path.dirname(os.path.dirname(image_filename)))
        row = self.patient_list_spreadsheet.loc[self.patient_list_spreadsheet['patient_id'] == patient_id]
    
        final_dictionary = { "image": processed_image_torch, "mask": processed_seg_torch, 
                "original_image": original_image_torch,  "original_seg": original_seg_torch,
                'annotation_frame_list': annotation_frame_list,
                "image_file_name" : image_filename, "seg_file_name": seg_filename,
                "original_shape" : self.original_shape,
                "centroid": self.centroid,
                'slice_type' : slice_type,
                'text_prompt_feature': prompt_feature,
                'box_prompt': torch.tensor(bbox),

                # copy infos from patient list spreadsheet
                "patient_id": row.iloc[0]['patient_id'],
                "patient_group": row.iloc[0]['patient_group'],
                "batch_index": row.iloc[0]['batch_index'],
          
                "lax_name": os.path.basename(image_filename)[0:4],
                "img_file": image_filename,
                "seg_file": seg_filename,
                "processed_time_frame_num": row.iloc[0]['processed_time_frame_num'],
                "ED_index_in_processed_time_frame": row.iloc[0]['ED_index_in_processed_time_frame'],
                "ES_index_in_processed_time_frame": row.iloc[0]['ES_index_in_processed_time_frame'],
                "processed_time_frame_index_list": row.iloc[0]['processed_time_frame_index_list'],}

        if self.return_arrays_or_dictionary == 'dictionary':
            return final_dictionary
        elif self.return_arrays_or_dictionary == 'arrays':
            return processed_image_torch, processed_seg_torch # model input and label
        else:
            raise ValueError('return_arrays_or_dictionary should be "arrays" or "dictionary"')
    
    # function: at the end of each epoch, we need to reset the index array
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()

        self.current_image_file = None
        self.current_image_data = None 
        self.current_seg_file = None
        self.current_seg_data = None