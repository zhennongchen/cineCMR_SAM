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
    trial_name = 'cineCMR_SAM'
    pretrained_model_epoch = None

    parser.add_argument('--text_prompt', default=True)#enable text prompt
    parser.add_argument('--box_prompt', default=None) #enable box prompt
    parser.add_argument('--num_classes', type=int, default=2)  ######## important!!!! background + myocardium.
    parser.add_argument('--validation', default=True)
  
    # data
    parser.add_argument('--full_or_nonzero_slice', default='nonzero') # full means all the slices, nonzero means only the slices with manual segmentation at both ED and ES, loose means the slices with manual segmentation at either ED or ES or both
    parser.add_argument('--dataset_names', default=[['STACOM', 'sax'], ['ACDC', 'sax'], ['HFpEF', 'sax'], ['STACOM', 'lax'], ['HFpEF', 'lax'] ], type=list)
    parser.add_argument('--dataset_split',default=[[np.arange(0,3,1) , np.arange(0,3,1)], [np.arange(0,0,1) , np.arange(0,0,1)], [np.arange(0,0,1) , np.arange(0,0,1)], [np.arange(0,0,1), np.arange(0,0,1)], [np.arange(0,2,1), np.arange(0,0,1)]], type=list) # [training_data, validation_data]. for LAX: 0-60 case: 0-224, 60-80: 224-297, 80-100: 297-376
    parser.add_argument('--dataset_train', default= [], type = list)
    parser.add_argument('--dataset_valid', default= [], type = list)

    # model save folder
    main_save_model_folder = os.path.join("/mnt/camca_NAS/SAM_for_CMR/", 'models', trial_name)
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
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--resume', default=os.path.join( "/mnt/camca_NAS/SAM_for_CMR/", "models"), help='resume from checkpoint') 
    # Custom parser 
    parser.add_argument('--seed', default=1234, type=int)   
    parser.add_argument('--input_type', type=str, default='2DT') #has to be 2DT
    parser.add_argument('--vit_type', type=str, default="vit_h")
    parser.add_argument('--max_timeframe', default=15, type=int) 
                        
    if pretrained_model_epoch == None:
        parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')
    else:
        parser.add_argument('--start_epoch', default=pretrained_model_epoch+1, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--accum_iter', default=20, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print_freq', default=50, type = int)
    parser.add_argument('--save_model_file_every_N_epoch', default=1, type = int) 
    parser.add_argument('--lr_update_every_N_epoch', default=200, type = int)
    parser.add_argument('--lr_decay_gamma', default=0.95)
    
    parser.add_argument('--img_size', default=128, type=int)    
    parser.add_argument('--loss_weights', default=[ 1, 0.5])  ######## [BCE loss, DICE loss]
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
    
    """""""""""""""""""""""""""""""""""""""TRAINING"""""""""""""""""""""""""""""""""""""""
    """Load Data from different sources"""
    assert len(args.dataset_names) == len(args.dataset_split), "The length of dataset_names and dataset_split should be the same"
    # define the dataset list for train and validation, separately
    for i in range(len(args.dataset_split)):
        if args.dataset_split[i][0].shape[0] > 0:
            args.dataset_train.append([args.dataset_names[i][0], args.dataset_names[i][1], args.dataset_split[i][0]])
        if args.dataset_split[i][1].shape[0] > 0:
            args.dataset_valid.append([args.dataset_names[i][0], args.dataset_names[i][1], args.dataset_split[i][1]])
 
    
    dataset_train = []
    dataset_valid = []
    # do trianing
    for i in range(len(args.dataset_train)):
        current_dataset_name = args.dataset_train[i][0]
        current_slice_type = args.dataset_train[i][1]
        current_index_list = args.dataset_train[i][2]
        if current_slice_type == 'sax':
            dataset_train.append(build_data_CMR(args, current_dataset_name, 
                                None,  current_index_list, 
                                full_or_nonzero_slice = args.full_or_nonzero_slice,
                                shuffle = True,
                                augment_list = args.augment_list, augment_frequency = args.augment_frequency,
                                return_arrays_or_dictionary = 'dictionary', 
                                sample_more_base = 1,
                                sample_more_apex = 3,
                                multi_chamber = [True if args.num_classes == 4 else None][0],))
        elif current_slice_type == 'lax':
            dataset_train.append(build_data_CMR_lax(args, current_dataset_name, 
                                None,  current_index_list, 
                                shuffle = True,
                                augment_list = args.augment_list, augment_frequency = args.augment_frequency,))
    # do validation
    for i in range(len(args.dataset_valid)):
        current_dataset_name = args.dataset_valid[i][0]
        current_slice_type = args.dataset_valid[i][1]
        current_index_list = args.dataset_valid[i][2]
        if current_slice_type == 'sax':
            dataset_valid.append(build_data_CMR(args, current_dataset_name,
                                None, current_index_list, 
                                full_or_nonzero_slice = args.full_or_nonzero_slice,
                                shuffle = False,
                                augment_list = [], augment_frequency =-0.1,
                                return_arrays_or_dictionary = 'dictionary',
                                multi_chamber = [True if args.num_classes == 4 else None][0],))
        elif current_slice_type == 'lax':
            dataset_valid.append(build_data_CMR_lax(args, current_dataset_name,
                                None, current_index_list, 
                                shuffle = False,
                                augment_list = [], augment_frequency =-0.1,))

    '''Set up data loader for training and validation set'''
    data_loader_train = []
    data_loader_valid = []
    for i in range(len(dataset_train)):
        data_loader_train.append(torch.utils.data.DataLoader(dataset_train[i], batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0))
    for i in range(len(dataset_valid)):
        data_loader_valid.append(torch.utils.data.DataLoader(dataset_valid[i], batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0))

    '''Load model and optimizer'''
    train_keys = []
    freezed_keys = []
        
    # load pretrained sam model vit_h
    if args.model_type.startswith("sam"):
        if args.resume.endswith(".pth"):
            with open(args.resume, "rb") as f:
                state_dict = torch.load(f)
            try:
                model.load_state_dict(state_dict)
            except:
                if args.vit_type == "vit_h":
                    new_state_dict = load_from(model, state_dict, args.img_size,  16, [7, 15, 23, 31])
                elif args.vit_type == "vit_b":
                    new_state_dict = load_from(model, state_dict, args.img_size,  16, [2 ,5, 8, 11])
                model.load_state_dict(new_state_dict)
                   
            freeze_list = [ "norm1", "attn" , "mlp", "norm2"]  
                
            for n, value in model.named_parameters():
                if any(substring in n for substring in freeze_list):
                    freezed_keys.append(n)
                    value.requires_grad = False
                else:
                    train_keys.append(n)
                    value.requires_grad = True
    print ("==================================================================================")
       
    ## Select optimization method
    optimizer = MADGRAD(model.parameters(), lr=args.lr) # momentum=,weight_decay=,eps=)
        
    print('args.pretrained_model: ', args.pretrained_model)
    # Continue training model
    if args.pretrained_model is not None:
        if os.path.exists(args.pretrained_model):
            print('loading pretrained model : ', args.pretrained_model)
            args.resume = args.pretrained_model
            finetune_checkpoint = torch.load(args.pretrained_model)
            model.load_state_dict(finetune_checkpoint["model"])
            optimizer.load_state_dict(finetune_checkpoint["optimizer"])
            torch.cuda.empty_cache()
    else:
        print('new training\n')

    '''Train loop'''
    training_log = []
    args.log_dir = os.path.join(args.output_dir, "logs")
    ff.make_folder([args.output_dir, os.path.join(args.output_dir, "logs"), os.path.join(args.output_dir, "models")])
    valid_results = [np.inf] * 9

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        print('training epoch:', epoch)

        if epoch % args.lr_update_every_N_epoch == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * args.lr_decay_gamma
        print('learning rate now:', optimizer.param_groups[0]["lr"])
        
        loss_scaler = NativeScaler()
            
        train_results = train_loop(
                model, 
                data_loader_train ,
                optimizer, epoch, loss_scaler,
                args=args,
                inputtype = cfg.data.input_type)          
            
        print('in epoch: ', epoch, ' training average_loss: ', train_results[0], ' average_lossCE: ', train_results[1], ' average_lossDICE: ', train_results[2])
        print('in epoch: ', epoch, ' training sax_loss: ', train_results[3], ' sax_lossCE: ', train_results[4], ' sax_lossDICE: ', train_results[5])
        print('in epoch: ', epoch, ' training lax_loss: ', train_results[6], ' lax_lossCE: ', train_results[7], ' lax_lossDICE: ', train_results[8])

        # on_epoch_end:
        for k in range(len(dataset_train)):
            dataset_train[k].on_epoch_end()
    
        if args.output_dir and (epoch % args.save_model_file_every_N_epoch == 0 or (epoch + 1) == args.start_epoch + args.epochs):
            checkpoint_path = os.path.join(args.output_dir, 'models', 'model-%s.pth' % epoch)
            to_save = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,}
            torch.save(to_save, checkpoint_path)

        if args.validation == True and (epoch % args.save_model_file_every_N_epoch == 0 or epoch + 1 == args.start_epoch + args.epochs):
            valid_results = inference_loop(model, data_loader_valid, args)
            print('in epoch: ', epoch, ' average_valid_loss: ', valid_results[0], ' average_valid_lossCE: ', valid_results[1], ' average_valid_lossDICE: ', valid_results[2])

        training_log.append([epoch, train_results[0], train_results[1], train_results[2], train_results[3], train_results[4], train_results[5], train_results[6], train_results[7], train_results[8], optimizer.param_groups[0]["lr"], valid_results[0], valid_results[1], valid_results[2], valid_results[3], valid_results[4], valid_results[5], valid_results[6], valid_results[7], valid_results[8]])
        df = pd.DataFrame(training_log, columns=['epoch', 'average_loss', 'average_lossCE', 'average_lossDICE', 'sax_loss', 'sax_lossCE', 'sax_lossDICE', 'lax_loss', 'lax_lossCE', 'lax_lossDICE', 'lr', 'average_valid_loss', 'average_valid_lossCE', 'average_valid_lossDICE', 'sax_valid_loss', 'sax_valid_lossCE', 'sax_valid_lossDICE', 'lax_valid_loss', 'lax_valid_lossCE', 'lax_valid_lossDICE'])
        df.to_excel(os.path.join(args.log_dir, 'training_log.xlsx'), index=False)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    cfg = Config(args.config)
    run(args, cfg)
    
    