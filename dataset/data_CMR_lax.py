#!/usr/bin/env python
# build dataloader for CMR LAX slices, you can build your own
import os
from torch.utils.data import Dataset, DataLoader
import cineCMR_SAM.dataset.Build_list as Build_list
import cineCMR_SAM.dataset.CMR.dataset_LAX as dataset_LAX

sam_dir = "/mnt/camca_NAS/SAM_for_CMR/"
        
def build_data_CMR_lax(args,dataset_name, train_batch_list, train_index_list, shuffle, augment_list, augment_frequency, return_arrays_or_dictionary = 'dictionary'):

    if dataset_name == 'ACDC':
       ValueError('ACDC dataset does not have lax data')

    if dataset_name == 'STACOM':
        data_path = os.path.join(sam_dir,'data/STACOM_database/temporal')
        patient_list_file = os.path.join(sam_dir, 'data/Patient_list/STACOM_LAX_training_testing.xlsx')

    if dataset_name == 'HFpEF':
        data_path = os.path.join(sam_dir,'data/HFpEF_database/temporal')
        patient_list_file = os.path.join(sam_dir, 'data/Patient_list/HFpEF_LAX_training_testing.xlsx')
    
    ##### select train and valid dataset (using either batch_list or index_list, if not using one then set it to None)
    patient_id_list,patient_group_list,batch_list,lax_index_list,lax_name_list,image_file_list_train,seg_file_list_train = Build_list.build_lax(patient_list_file, batch_list = train_batch_list, index_list = train_index_list)
    print('image_full_slice_file_list_train num: ',image_file_list_train.shape, ', seg_full_slice_file_list_train num: ',seg_file_list_train.shape)
 
    if dataset_name == 'ACDC':
        relabel_LV = True
        only_myo = True
        seg_include_lowest_pixel = 1
    elif dataset_name == 'STACOM':
        relabel_LV = False 
        only_myo = False
        seg_include_lowest_pixel = 100 
    elif dataset_name == 'HFpEF':
        relabel_LV = False
        only_myo = True
        seg_include_lowest_pixel = 1

    dataset_train = dataset_LAX.Dataset_CMR_lax(patient_list_file,
                                                 image_file_list_train,
                                                 seg_file_list_train,
                                
                                                 return_arrays_or_dictionary = return_arrays_or_dictionary, # 'dictionary' or 'arrays'

                                                 seg_include_lowest_pixel = seg_include_lowest_pixel,
                                                 turn_zero_seg_slice_into = args.turn_zero_seg_slice_into,
                                                 
                                                 relabel_LV = relabel_LV,
                                                 only_myo = only_myo,
                                                 center_crop_according_to_which_class = [1],
                                                 image_shape = [args.img_size, args.img_size],
                                                 shuffle = shuffle, 
                                                 image_normalization = True,
                                                 augment_list = augment_list, # a list of augmentation methods and their range: v range = None for brightness, contrast, sharpness
                                                 augment_frequency = augment_frequency,
                                                )
    
    return dataset_train
  
