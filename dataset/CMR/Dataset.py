#!/usr/bin/env python

import sys
sys.path.append('/workspace/Documents')
import torch
import numpy as np
import os
import pandas as pd
import nibabel as nb


from torch.utils.data import Dataset, DataLoader

import cineCMR_SAM.Data_processing as Data_processing
import cineCMR_SAM.functions_collection as ff
import cineCMR_SAM.dataset.CMR.random_aug as random_aug

# main function:
class Dataset_CMR(torch.utils.data.Dataset):
    def __init__(
            self, 
            view_type , # 'sax' or 'lax'
            patient_list_spreadsheet_file,
            text_prompt_feature,

            image_file_list,
            seg_file_list,
            total_slice_num_list,

            seg_include_lowest_pixel = 100, # only include the segmentation with the lowest pixel number > seg_include_lowest_piexel in one slice
            turn_zero_seg_slice_into = 10, # if there is no segmentation in this slice, then turn the pixel value of this slice into 10

            return_arrays_or_dictionary = 'dictionary', # "arrays" or "dictionary"
            only_myo = None, # only keep MYO segmentation, assume in the manual segmentation, LV = label 1, MYO = label 2
            center_crop_according_to_which_class  = [1],

            image_shape = None, # [x,y], channel =  tf always 15 
            shuffle = None,
            image_normalization = True,
            augment = None,
            augment_frequency = 0.5, # how often do we do augmentation
            ):

        super().__init__()
        self.view_type = view_type; assert self.view_type in ['sax','lax'] 
        self.patient_list_spreadsheet = pd.read_excel(patient_list_spreadsheet_file)
        self.text_prompt_feature = text_prompt_feature

        self.image_file_list = image_file_list
        self.seg_file_list = seg_file_list
        self.total_slice_num_list = total_slice_num_list

        self.seg_include_lowest_pixel = seg_include_lowest_pixel
        self.turn_zero_seg_slice_into = turn_zero_seg_slice_into

        self.only_myo = only_myo    
        self.center_crop_according_to_which_class = center_crop_according_to_which_class

        self.image_shape = image_shape
        self.shuffle = shuffle
        self.image_normalization = image_normalization
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.return_arrays_or_dictionary = return_arrays_or_dictionary

        # how many cases we have in this dataset?
        self.num_files = len(self.image_file_list)

        # how many slices in total we have in this dataset? Note each case has different number of slices
        if self.view_type == 'sax':
            self.num_slices_total = np.sum(self.total_slice_num_list)

        # the following two should be run at the beginning of each epoch
        # 1. get index array
        self.index_array = self.generate_index_array()
        # 2. some parameters
        self.current_image_file = None
        self.current_image_data = None 
        self.current_seg_file = None
        self.current_seg_data = None

    # function: how many cases do we have in this dataset? = total slice number accumulated from all cases for SAX; = total number of cases for LAX
    def __len__(self):
        return self.num_slices_total if self.view_type == 'sax' else self.num_files
        

    # function: we need to generate an index array for dataloader, it's a list, each element is [file_index, slice_index]
    def generate_index_array(self):
        np.random.seed()
        index_array = []
                
        if self.shuffle == True:
            file_index_list = np.random.permutation(self.num_files)
        else:
            file_index_list = np.arange(self.num_files)

        for i in range(0,file_index_list.shape[0]):
            file_index = file_index_list[i]
            num_slices = self.total_slice_num_list[file_index] if self.view_type == 'sax' else 1 # for LAX, we only have 1 slice

            if self.shuffle == True:
                s_list = np.random.permutation(num_slices)
            else:
                s_list = np.arange(num_slices)
    
            for slice_index in s_list:
                index_array.append([file_index, slice_index])
        return index_array
    
    # function: 
    # load nii, for SAX, you will be given a 4D data as [x,y,tf, slice_num], where tf always = 15
    # for LAX, you will be given a 3D data as [x,y,tf], where tf always = 15
    def load_file(self, filename, segmentation_load = False):
        ii = nb.load(filename).get_fdata()
        if segmentation_load is True:
            ii = np.round(ii).astype(int)
        if self.only_myo is True and segmentation_load is True:
            iii = np.zeros(ii.shape); iii[ii != 2] = 0; iii[ii == 2] = 1; ii = np.copy(iii)
        
        # change it to [x,y,slice_num,tf] for SAX for the purpose of coding
        ii = np.transpose(ii, (0,1,3,2)) if self.view_type == 'sax' else ii
        return ii
    

    # function: get each item using the index [file_index, slice_index]
    def __getitem__(self, index):
        f,s = self.index_array[index]
        image_filename = self.image_file_list[f]
        seg_filename = self.seg_file_list[f]

        if os.path.isfile(seg_filename) is False:
            self.have_manual_seg = False
        else:
            self.have_manual_seg = True
            
        # if it's a new case, then do the data loading; if it's not, then just use the current data
        if image_filename != self.current_image_file or seg_filename != self.current_seg_file:
            image_loaded = self.load_file(image_filename, segmentation_load = False) 
          
            image_loaded = image_loaded[:,:,np.newaxis,:] if self.view_type == 'lax' else image_loaded # [x,y,tf] --> [x,y,1,tf]

            h,w,d,tf_num = image_loaded.shape 
            self.original_shape = image_loaded[:,:,0,:].shape

            if self.have_manual_seg is True:
                seg_loaded = self.load_file(seg_filename, segmentation_load=True) 
                seg_loaded = seg_loaded[:,:,np.newaxis,:] if self.view_type == 'lax' else seg_loaded # [x,y,tf] --> [x,y,1,tf]

            # now we need to do the center crop for both image and seg
            # we have data volume as [x, y ,slice_num,tf], for each data volume , we use the [x,y, middle_slice, 0] for the centroid calculation (refers to  ED, which always has segmentation)
            if self.have_manual_seg is True:
                # define the ED time frame (usually the time frame 0)
                for ed in range(0,tf_num):
                    if np.sum(seg_loaded[:,:,d//2,ed] == self.center_crop_according_to_which_class[0]) > 0:
                        break
                if self.view_type == 'sax':
                    _,_, self.centroid = Data_processing.center_crop( image_loaded[:,:,d // 2, ed], seg_loaded[:,:,d//2, ed], self.image_shape, according_to_which_class = self.center_crop_according_to_which_class, centroid = None)
                else:
                    _,_, self.centroid = Data_processing.center_crop( image_loaded[:,:, 0,0], seg_loaded[:,:, 0,0], self.image_shape, according_to_which_class = self.center_crop_according_to_which_class , centroid = None)
  
                # random crop (randomly shift the centroid)
                if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
                    random_centriod_shift_x = np.random.randint(-5,5)
                    random_centriod_shift_y = np.random.randint(-5,5)
                    centroid_used_for_crop = [self.centroid[0] + random_centriod_shift_x, self.centroid[1] + random_centriod_shift_y]
                else:
                    centroid_used_for_crop = self.centroid
                
                # then for each dim in slice_num and tf, we do the center crop
                image_loaded_tem = np.zeros([self.image_shape[0],self.image_shape[1],d,tf_num])
                seg_loaded_tem = np.zeros([self.image_shape[0],self.image_shape[1],d,tf_num])
                for z_dim in range(0,d):
                    for tf_dim in range(0,tf_num):
                        image_loaded_tem[:,:,z_dim,tf_dim], seg_loaded_tem[:,:,z_dim,tf_dim], _ = Data_processing.center_crop( image_loaded[:,:,z_dim,tf_dim], seg_loaded[:,:,z_dim,tf_dim], self.image_shape, according_to_which_class = None , centroid = centroid_used_for_crop)
                image_loaded = np.copy(image_loaded_tem)
                seg_loaded = np.copy(seg_loaded_tem)

            elif self.have_manual_seg is False:
                # crop the image regarding the image center, just find the center and take 128x128 ROI
                self.centroid = np.array([h//2, w//2])
                image_loaded_tem = np.zeros([self.image_shape[0],self.image_shape[1],d,tf_num])
                for z_dim in range(0,d):
                    for tf_dim in range(0,tf_num):
                        image_loaded_tem[:,:,z_dim,tf_dim] = image_loaded[self.centroid[0]- self.image_shape[0]//2:self.centroid[0] + self.image_shape[0]//2, self.centroid[1] - self.image_shape[1]//2:self.centroid[1] + self.image_shape[1]//2, z_dim, tf_dim]
                image_loaded = np.copy(image_loaded_tem)
                seg_loaded = np.zeros(image_loaded.shape) # segmentation is all zeros

            self.current_image_file = image_filename
            self.current_image_data = np.copy(image_loaded)  
            self.current_seg_file = seg_filename
            self.current_seg_data = np.copy(seg_loaded)
        
        # pick the slice
        original_image = np.copy(self.current_image_data)[:,:,s,:] if self.view_type == 'sax' else np.copy(self.current_image_data)[:,:,0,:]
        original_seg = np.copy(self.current_seg_data)[:,:,s,:] if self.view_type == 'sax' else np.copy(self.current_seg_data)[:,:,0,:]
      
        ######## do augmentation
        processed_seg = np.copy(original_seg)
        # (0) add noise
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            standard_deviation = 5
            processed_image = original_image + np.random.normal(0,standard_deviation,original_image.shape)
            # turn the image pixel range to [0,255]
            processed_image = Data_processing.turn_image_range_into_0_255(processed_image)
        else:
            processed_image = Data_processing.turn_image_range_into_0_255(original_image)
       
        # (1) do brightness
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image,v = random_aug.random_brightness(processed_image, v = None)
    
        # (2) do contrast
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, v = random_aug.random_contrast(processed_image, v = None)

        # (3) do sharpness
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, v = random_aug.random_sharpness(processed_image, v = None)
            
        # (4) do flip
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            # doing this can make sure the flip is the same for image and seg
            a, selected_option = random_aug.random_flip(processed_image)
            b,_ = random_aug.random_flip(processed_seg, selected_option)
            processed_image = np.copy(a)
            processed_seg = np.copy(b)

        # (5) do rotate
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, z_rotate_degree = random_aug.random_rotate(processed_image, order = 1, z_rotate_range = [-10,10])
            processed_seg,_ = random_aug.random_rotate(processed_seg, z_rotate_degree, fill_val = 0, order = 0)

        # (6) do translate
        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            processed_image, x_translate, y_translate = random_aug.random_translate(processed_image, translate_range = [-10,10])
            processed_seg,_ ,_= random_aug.random_translate(processed_seg, x_translate, y_translate)

        # add normalization
        if self.image_normalization is True:
            processed_image = Data_processing.normalize_image(processed_image,inverse = False) 

        # find which times frame has segmentation, and put it into annotation frame list, also turn the pixel value of the slice without manual segmentation into 10
        processed_seg_no_class_10 = np.copy(processed_seg) 
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

        # also add infos from patient list spread sheet
        patient_id = os.path.basename(os.path.dirname(image_filename))
        row = self.patient_list_spreadsheet.loc[self.patient_list_spreadsheet['patient_id'] == patient_id]

        # prepare the box feature (bounding box):
        if self.have_manual_seg is True: # automatically generate bounding box from manual segmentation --> it's for training
            bbox,_,_ = Data_processing.get_bbox_from_mask_all_volumes(processed_seg, annotation_frame_list)
        elif self.have_manual_seg is False: # manually defined bounding box --> it's for testing
            if os.path.isfile(os.path.join(os.path.dirname(image_filename), 'bounding_box.npy')) == True:
                bbox = np.load(os.path.join(os.path.dirname(image_filename), 'bounding_box.npy'))
                bbox = bbox[s,:,:]
                # print('no manual segmentation, please define by your own. in this example, we pre-save the bounding box and we will load here')
                # print('the bounding box is: ', bbox)
            else:
                bbox = np.zeros((2,4))
                # print('no pre-saved bounding box, please define by your own')
        
        # now it's time to turn numpy into tensor and collect as a dictionary (this is the final return)
        processed_image_torch = torch.from_numpy(processed_image).unsqueeze(0).float() 
        processed_seg_torch = torch.from_numpy(processed_seg).unsqueeze(0)  
        processed_seg_no_class_10_torch = torch.from_numpy(processed_seg_no_class_10).unsqueeze(0)

        # also need to return the original image and seg without the augmentation (with center crop done)
        original_image_torch = torch.from_numpy(original_image).unsqueeze(0).float()
        original_seg_torch = torch.from_numpy(original_seg).unsqueeze(0).float()

        final_dictionary = { "image": processed_image_torch, 
                            "mask": processed_seg_torch, 
                            "original_image": original_image_torch,  
                            "original_seg": original_seg_torch,
                            'processed_seg_no_class_10': processed_seg_no_class_10_torch, # this is used for loss calculation
                            'annotation_frame_list': annotation_frame_list,
                            "image_file_name" : image_filename, "seg_file_name": seg_filename,
                            "original_shape" : self.original_shape,
                            "centroid": self.centroid,
                            'slice_index': s,
                            'text_prompt_feature': np.squeeze(self.text_prompt_feature),
                            'box_prompt': torch.tensor(bbox),
                            # copy infos from patient list spreadsheet
                            "patient_id": row.iloc[0]['patient_id'],
                            "img_file": row.iloc[0]['img_file'],
                            "seg_file": row.iloc[0]['seg_file'],
                            "lax_type": row.iloc[0]['lax_type'] if self.view_type == 'lax' else 'SAX',}

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