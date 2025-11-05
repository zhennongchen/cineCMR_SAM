# this script defines the parameterse for the experiment
import argparse
import os

def get_args_parser(text_prompt = True, box_prompt = None, pretrained_model = None, original_sam = None, start_epoch = None, total_training_epochs = 1000, vit_type = "vit_h"):
    parser = argparse.ArgumentParser('SAM fine-tuning', add_help=True)

    # for training
    parser.add_argument('--total_training_epochs', default=total_training_epochs, type=int)
    parser.add_argument('--accum_iter', default=20, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--print_freq', default=50, type = int)
    parser.add_argument('--save_model_file_every_N_epoch', default=1, type = int) 
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--lr_update_every_N_epoch', default=200, type = int)
    parser.add_argument('--lr_decay_gamma', default=0.95)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--loss_weights', default=[ 1, 0.5])  ######## [BCE loss, DICE loss]

    # standard
    parser.add_argument('--text_prompt', default = text_prompt)
    parser.add_argument('--box_prompt', default= box_prompt) 
    parser.add_argument('--pretrained_model', default = pretrained_model)
    
    parser.add_argument('--num_classes', type=int, default=2)  ######## important!!!! background + myocardium.
    parser.add_argument('--validation', default=False)
    parser.add_argument('--cross_frame_attention', default=False) # False

    parser.add_argument('--model_type', type=str, default='sam')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')  # use single GPU
    parser.add_argument('--use_amp', action='store_true', help='If activated, adopt mixed precision for acceleration')
    parser.add_argument(
        "--config", help="Path to the training config file.", default="configs/config.yaml",
    )

    parser.add_argument('--seed', default=1234, type=int)   
    parser.add_argument('--input_type', type=str, default='2DT') #has to be 2DT
    parser.add_argument('--vit_type', type=str, default=vit_type)
    parser.add_argument('--max_timeframe', default=15, type=int) 
                        
    if start_epoch == None:
        parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')
    else:
        parser.add_argument('--start_epoch', default= start_epoch, type=int, metavar='N', help='start epoch')

    parser.add_argument('--resume', default = original_sam)

    parser.add_argument('--img_size', default=128, type=int) 
    parser.add_argument('--turn_zero_seg_slice_into', default=10, type=int)
    parser.add_argument('--augment_frequency', default=0.5, type=float)
 
    return parser
