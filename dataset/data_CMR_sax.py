#!/usr/bin/env python
# build dataloader for CMR SAX slices, you can build your own

import os
from torch.utils.data import Dataset, DataLoader
import cineCMR_SAM.dataset.Build_list as Build_list
import cineCMR_SAM.dataset.CMR.dataset_SAX as dataset_SAX

sam_dir = "/mnt/camca_NAS/SAM_for_CMR/"
        
def build_data_CMR(args,dataset_name, train_batch_list, train_index_list, full_or_nonzero_slice, shuffle, augment_list, augment_frequency, return_arrays_or_dictionary = 'dictionary', sample_more_base = 0, sample_more_apex = 0, multi_chamber = None):
    
    if dataset_name == 'ACDC':
        data_path = os.path.join(sam_dir,'data/ACDC_database/temporal')
        patient_list_file = os.path.join(sam_dir, 'data/Patient_list/ACDC_Patient_List_training_testing.xlsx')

    if dataset_name == 'STACOM':
        data_path = os.path.join(sam_dir,'data/STACOM_database/temporal')
        patient_list_file = os.path.join(sam_dir, 'data/Patient_list/STACOM_Patient_List_training_testing.xlsx')
    
    if dataset_name == 'HFpEF':
        data_path = os.path.join(sam_dir,'data/HFpEF_database/temporal')
        patient_list_file = os.path.join(sam_dir, 'data/Patient_list/HFpEF_Patient_List_training_testing-v2.xlsx')

    if dataset_name == 'HFpEF_multiT':
        data_path = os.path.join(sam_dir,'data/HFpEF_database/temporal')
        patient_list_file = os.path.join(sam_dir, 'data/Patient_list/HFpEF_multiT_Patient_List_training_testing.xlsx')

    if dataset_name == 'HFpEF_fulldataset':
        data_path = os.path.join(sam_dir,'data/HFpEF_fulldataset_database/temporal')
        patient_list_file = os.path.join(sam_dir, 'data/Patient_list/HFpEF_fulldataset_Patient_List_training_testing.xlsx')

    if dataset_name == 'AS':
        data_path = os.path.join(sam_dir,'data/AS_database/temporal')
        patient_list_file = os.path.join(sam_dir, 'data/Patient_list/AS_Patient_List_training_testing.xlsx')
    
    if dataset_name == 'MM':
        data_path = os.path.join(sam_dir,'data/MM_database/temporal')
        patient_list_file = os.path.join(sam_dir, 'data/Patient_list/MM_Patient_List_training_testing.xlsx')
    
    
    ##### select train and valid dataset (using either batch_list or index_list, if not using one then set it to None)
    _,_,_,image_full_slice_file_list_train ,seg_full_slice_file_list_train ,image_nonzero_slice_file_list_train ,seg_nonzero_slice_file_list_train ,image_nonzero_slice_file_loose_list_train ,seg_nonzero_slice_file_loose_list_train, total_slice_num_list_train ,nonzero_slice_num_list_train, nonzero_slice_num_loose_list_train = Build_list.__build__(patient_list_file, batch_list = train_batch_list, index_list = train_index_list)
    print('image_full_slice_file_list_train num: ',image_full_slice_file_list_train.shape, ', seg_full_slice_file_list_train num: ',seg_full_slice_file_list_train.shape)
 

    if full_or_nonzero_slice[0:4] == 'full': # all slices
        image_file_list_train = image_full_slice_file_list_train
        seg_file_list_train = seg_full_slice_file_list_train
        total_slice_num_list_train = total_slice_num_list_train


    elif full_or_nonzero_slice[0:4] == 'nonz': # only slices with segmentation at ED and ES
        image_file_list_train = image_nonzero_slice_file_list_train
        seg_file_list_train = seg_nonzero_slice_file_list_train
        total_slice_num_list_train = nonzero_slice_num_list_train


    elif full_or_nonzero_slice[0:4] == 'loos': # only slices with segmentation at ED OR ES
        image_file_list_train = image_nonzero_slice_file_loose_list_train
        seg_file_list_train = seg_nonzero_slice_file_loose_list_train
        total_slice_num_list_train = nonzero_slice_num_loose_list_train

    if dataset_name == 'ACDC':
        relabel_LV = True
        only_myo = True
        seg_include_lowest_pixel = 1
    elif dataset_name == 'STACOM':
        relabel_LV = False 
        only_myo = False
        seg_include_lowest_pixel = 100 
    elif dataset_name == 'HFpEF' or dataset_name == 'HFpEF_multiT'  or dataset_name == 'HFpEF_fulldataset' or dataset_name == 'AS' or dataset_name == 'MM':
        relabel_LV = False
        only_myo = True
        seg_include_lowest_pixel = 1

    if multi_chamber is not None:
        only_myo = False
        print('multi_chamber: ',multi_chamber, ', only_myo: ',only_myo)

    dataset_train = dataset_SAX.Dataset_CMR(patient_list_file,
                                                 image_file_list_train,
                                                 seg_file_list_train,
                                                 total_slice_num_list_train,
                                                 return_arrays_or_dictionary = return_arrays_or_dictionary, 

                                                 seg_include_lowest_pixel = seg_include_lowest_pixel,
                                                 turn_zero_seg_slice_into = args.turn_zero_seg_slice_into,
                                                 
                                                 relabel_LV = relabel_LV,
                                                 only_myo = only_myo,
                                                 center_crop_according_to_which_class = [1],
                                                 image_shape = [args.img_size,args.img_size],
                                                 shuffle = shuffle, 
                                                 image_normalization = True,
                                                 augment_list = augment_list, # a list of augmentation methods and their range: v range = None for brightness, contrast, sharpness
                                                 augment_frequency = augment_frequency,
                                                 )
    
    


    return dataset_train
  
