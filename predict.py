#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from pathlib import Path
import nibabel as nb
import time

import argparse
from einops import rearrange
from natsort import natsorted
from madgrad import MADGRAD

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
 
from cineCMR_SAM.utils.model_util import *
from cineCMR_SAM.segment_anything.model import build_model 
from cineCMR_SAM.utils.save_utils import *
from cineCMR_SAM.utils.config_util import Config
from cineCMR_SAM.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from cineCMR_SAM.train_engine import train_loop
from cineCMR_SAM.inference_engine import  inference_loop, pred_save, pred_save_lax

from cineCMR_SAM.dataset.data_CMR_sax import build_data_CMR
from cineCMR_SAM.dataset.data_CMR_lax import build_data_CMR_lax
import cineCMR_SAM.functions_collection as ff

def get_args_parser():
    parser = argparse.ArgumentParser('SAM fine-tuning', add_help=True)
    
    ########### important parameters, fill in using your own
    trial_name = 'sam_multiview_prompt_2box_text_HF_5shot'
    pretrained_model_epoch = 81 # fill in the epoch number of the pretrained model

    parser.add_argument('--text_prompt', default=True)#enable text prompt
    parser.add_argument('--box_prompt', default= True) #enable box prompt
    parser.add_argument('--num_classes', type=int, default=2)  ######## important!!!! background + myocardium.
    parser.add_argument('--validation', default=True)

    main_save_model_folder = os.path.join("/mnt/camca_NAS/SAM_for_CMR/", 'models', trial_name)
    # data
    parser.add_argument('--full_or_nonzero_slice', default='nonzero') # full means all the slices, nonzero means only the slices with manual segmentation at both ED and ES, loose means the slices with manual segmentation at either ED or ES or both
    ###########

    # default, no need to change
    parser.add_argument('--output_dir', default = main_save_model_folder, help='path where to save, empty for no saving')
    parser.add_argument('--pretrained_model_epoch', default = pretrained_model_epoch)
   
    if pretrained_model_epoch == None:
        parser.add_argument('--pretrained_model', default = None, help='path where to save, empty for no saving')
    else:
        parser.add_argument('--pretrained_model', default = os.path.join(main_save_model_folder, 'models', 'model-%s.pth' % pretrained_model_epoch), help='path where to save, empty for no saving')

    parser.add_argument('--model_type', type=str, default='sam')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')  # use single GPU
    parser.add_argument('--use_amp', action='store_true', help='If activated, adopt mixed precision for acceleration')
    parser.add_argument(
        "--config", help="Path to the training config file.", default="configs/config.yaml",
    )
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')  

    # Custom parser 
    parser.add_argument('--seed', default=1234, type=int)   
    parser.add_argument('--input_type', type=str, default='2DT') #has to be 2DT
    parser.add_argument('--vit_type', type=str, default="vit_h")
    parser.add_argument('--max_timeframe', default=15, type=int) 
                        
    if pretrained_model_epoch == None:
        parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')
    else:
        parser.add_argument('--start_epoch', default=pretrained_model_epoch+1, type=int, metavar='N', help='start epoch')
    parser.add_argument('--img_size', default=128, type=int) 
    parser.add_argument('--turn_zero_seg_slice_into', default=10, type=int)
    parser.add_argument('--augment_list', default=[('noise', None),('brightness' , None), ('contrast', None), ('sharpness', None), ('flip', None), ('rotate', [-20,20]), ('translate', [-5,5]), ('random_crop', [-5,5])], type=list)
    parser.add_argument('--augment_frequency', default=0.5, type=float)
 
    return parser

        
def run(args, cfg):  
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cudnn.benchmark = True
    
    """Build Model"""
    if args.vit_type == "vit_h":
        args.resume = os.path.join( "/mnt/camca_NAS/SAM_for_CMR/", 'models/pretrained_sam/sam_vit_h_4b8939.pth')
    else:
        args.resume = os.path.join("/mnt/camca_NAS/SAM_for_CMR/", 'models/pretrained_sam/sam_vit_b_01ec64.pth')

    model = build_model(args, device)
    
    sax_or_lax = 'sax'
    save_folder_name = 'predicts_sax'

    pred_index_list = np.arange(4,5,1)
        
    if sax_or_lax == 'sax':
        dataset_pred = build_data_CMR(args, 'HFpEF',
                            None, pred_index_list, full_or_nonzero_slice = args.full_or_nonzero_slice,
                            shuffle = False,
                            augment_list = [], augment_frequency = -0.1,
                            multi_chamber = [True if args.num_classes == 4 else None][0],
                            return_arrays_or_dictionary = 'dictionary',) # 'diction
    else:
        dataset_pred = build_data_CMR_lax(args, 'HFpEF',
                        None, pred_index_list, 
                        shuffle = False,
                        augment_list = [], augment_frequency = -0.1,) 
        
        
    data_loader_pred = torch.utils.data.DataLoader(dataset_pred, batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
       
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            model = build_model(args, device)#skip_nameing = True, chunk = np.shape(np.zeros(0)))

            if args.pretrained_model is not None:
                if os.path.exists(args.pretrained_model):
                    print('loading pretrained model : ', args.pretrained_model)
                    finetune_checkpoint = torch.load(args.pretrained_model)
                    model.load_state_dict(finetune_checkpoint["model"])
                                       
            if sax_or_lax == 'sax':
                for data_iter_step, batch in tqdm(enumerate(data_loader_pred)):
                    
                    patient_id = batch["patient_id"][0]
                    slice_index = batch["slice_index"].item()
                    print('patient_id: ', patient_id, ' slice_index: ', slice_index)
                    save_folder = os.path.join(args.output_dir, save_folder_name)
                    save_folder_sub = os.path.join(save_folder, patient_id, 'epoch-' + str(args.pretrained_model_epoch))
                    ff.make_folder([os.path.dirname(save_folder_sub),save_folder_sub])

                    batch["image"]= batch["image"].cuda()

                    batch["text_prompt_feature"] = batch["text_prompt_feature"].to(torch.float32)
                    
                    output = model(batch, False, args.img_size)

                    torch.cuda.synchronize()
            
                    pred_save(batch, output, args, save_folder_sub)
                   

            else:
                for data_iter_step, batch in tqdm(enumerate(data_loader_pred)):
                   
                    patient_id = batch["patient_id"][0]
                    lax_name = batch["lax_name"][0]
                    save_folder = os.path.join(args.output_dir, 'predicts_lax')
                    save_folder_sub = os.path.join(save_folder, patient_id, 'epoch-' + str(args.pretrained_model_epoch))
                    ff.make_folder([os.path.dirname(save_folder_sub),save_folder_sub])
        
                    batch["image"]= batch["image"].cuda()

                    batch["text_prompt_feature"] = batch["text_prompt_feature"].to(torch.float32)
                    
                    output = model(batch, False, args.img_size)

                    torch.cuda.synchronize()

                    pred_save_lax(batch, output,args, save_folder_sub)
                   



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    cfg = Config(args.config)
    run(args, cfg)
    