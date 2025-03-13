#!/usr/bin/env python
# build dataloader for CMR SAX slices, you can build your own

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cineCMR_SAM.dataset.CMR.Dataset as Dataset

def from_excel_file(file, index_list = None, view_type = 'sax'): 
    data = pd.read_excel(file)
    if index_list == None:
        c = data
    else:
        c = data.iloc[index_list]

    patient_id_list = np.asarray(c['patient_id'])
    img_file_list = np.asarray(c['img_file'])
    seg_file_list = np.asarray(c['seg_file'])
    total_slice_num_list = np.asarray(c['total_slice_num'])
    lax_name_list = np.asarray(c['lax_type']) if view_type == 'lax' else None
    return patient_id_list, img_file_list, seg_file_list, total_slice_num_list, lax_name_list
       

def build_dataset(args, view_type, patient_list_file, index_list, text_prompt_feature, only_myo = True, shuffle = False, augment = False):

    _, img_file_list, seg_file_list, total_slice_num_list,_ = from_excel_file(patient_list_file, index_list, view_type = view_type)
 
    dataset = Dataset.Dataset_CMR(view_type,
                                  patient_list_file,
                                  text_prompt_feature = text_prompt_feature,
                                  image_file_list = img_file_list,
                                  seg_file_list = seg_file_list,
                                  total_slice_num_list = total_slice_num_list,
                                  seg_include_lowest_pixel = 1,
                                  turn_zero_seg_slice_into = args.turn_zero_seg_slice_into,
                                  only_myo = True,
                                  image_shape = [args.img_size,args.img_size],
                                  shuffle = shuffle, 
                                  augment = augment,)
    return dataset

  

  