import numpy as np
import os
import pandas as pd

def __build__(file, index_list = None):

    data = pd.read_excel(file)

    if batch_list is None and index_list is None:
        ValueError('Please provide either batch_list or index_list')
    elif batch_list is not None and index_list is not None:
        ValueError('Please provide either batch_list or index_list')
    elif batch_list is not None:
        for b in range(len(batch_list)):
            cases = data.loc[data['batch_index'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])
    elif index_list is not None:
        c = data.iloc[index_list]

    patient_id_list = np.asarray(c['patient_id'])
    patient_group_list = np.asarray(c['patient_group'])
    batch_list = np.asarray(c['batch_index'])

    image_full_slice_file_list = np.asarray(c['image_full_slice_file'])
    seg_full_slice_file_list = np.asarray(c['seg_full_slice_file'])
    image_nonzero_slice_file_list = np.asarray(c['image_nonzero_slice_file'])
    seg_nonzero_slice_file_list = np.asarray(c['seg_nonzero_slice_file'])
    image_nonzero_slice_file_loose_list = np.asarray(c['image_nonzero_slice_file_loose'])
    seg_nonzero_slice_file_loose_list = np.asarray(c['seg_nonzero_slice_file_loose'])

    total_slice_num_list = np.asarray(c['total_slice_num'])
    nonzero_slice_num_list = np.asarray(c['nonzero_slice_num'])
    nonzero_slice_num_loose_list = np.asarray(c['nonzero_slice_num_loose'])

    return patient_id_list,patient_group_list,batch_list,image_full_slice_file_list,seg_full_slice_file_list,image_nonzero_slice_file_list,seg_nonzero_slice_file_list,image_nonzero_slice_file_loose_list,seg_nonzero_slice_file_loose_list, total_slice_num_list,nonzero_slice_num_list, nonzero_slice_num_loose_list


def build_lax(file, batch_list = None, index_list = None):

    data = pd.read_excel(file)

    if batch_list is None and index_list is None:
        ValueError('Please provide either batch_list or index_list')
    elif batch_list is not None and index_list is not None:
        ValueError('Please provide either batch_list or index_list')
    elif batch_list is not None:
        for b in range(len(batch_list)):
            cases = data.loc[data['batch_index'] == batch_list[b]]
            if b == 0:
                c = cases.copy()
            else:
                c = pd.concat([c,cases])
    elif index_list is not None:
        c = data.iloc[index_list]

    patient_id_list = np.asarray(c['patient_id'])
    patient_group_list = np.asarray(c['patient_group'])
    batch_list = np.asarray(c['batch_index'])

    lax_index_list = np.asarray(c['lax_index'])
    lax_name_list = np.asarray(c['lax_name'])

    image_file_list = np.asarray(c['img_file'])
    seg_file_list = np.asarray(c['seg_file'])


    return patient_id_list,patient_group_list,batch_list,lax_index_list,lax_name_list,image_file_list,seg_file_list
        