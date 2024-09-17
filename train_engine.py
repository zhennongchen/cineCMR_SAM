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
import torch.nn.functional as F
import cineCMR_SAM.functions_collection as ff


def train_loop(model: torch.nn.Module,
               data_loader_train: list,
               optimizer: torch.optim.Optimizer,
               epoch: int, 
               loss_scaler,
               args=None,
               inputtype = None):
    
    # make some settings
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
     
    accum_iter = args.accum_iter
    
    model.train(True)
    
    if args.turn_zero_seg_slice_into is not None:
        criterionBCE = torch.nn.CrossEntropyLoss(ignore_index=10)
        print('in train loop we have turn_zero_seg_slice_into: ', args.turn_zero_seg_slice_into)
    else:
        criterionBCE = torch.nn.CrossEntropyLoss()

    # start to train
    average_loss = []; average_lossCE = []; average_lossDICE = []
    sax_loss = []; sax_lossCE = []; sax_lossDICE = []
    lax_loss = []; lax_lossCE = []; lax_lossDICE = []

    assert len(data_loader_train) == len(args.dataset_train)

    for i in range(len(data_loader_train)):
        current_data_loader = data_loader_train[i] # this is an iterable
        current_dataset_name = args.dataset_train[i][0]
        current_slice_type = args.dataset_train[i][1]
        print('in training current slice type: ', current_slice_type)
        
        for data_iter_step, batch in enumerate(metric_logger.log_every(current_data_loader ,args.print_freq, header)):
            with torch.cuda.amp.autocast():
                batch["image"]= batch["image"].to(torch.float16).cuda()
                
                output = model(batch, False, args.img_size)

                mask = batch["mask"]
                mask = rearrange(mask, 'b c h w d -> (b d) c h w ').to("cuda")
                   
                lossCE = criterionBCE(output["masks"], torch.clone(mask).squeeze(1).long()) 
                lossDICE = ff.customized_dice_loss(output["masks"], torch.clone(mask).squeeze(1).long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)

                #### total loss: weighted loss
                loss = args.loss_weights[0] * lossCE + args.loss_weights[1] * lossDICE
                   
                
                if torch.isnan(loss):
                    continue

                subset_params = [p for p in model.parameters()]
                
                loss_scaler(loss, optimizer, parameters=subset_params,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)  
            
                if (data_iter_step + 1) % accum_iter == 0 or data_iter_step == len(current_data_loader) - 1:
                    optimizer.zero_grad()

                torch.cuda.synchronize()
                metric_logger.update(loss1=loss.item())
                lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(lr=lr)

            metric_logger.synchronize_between_processes()
        
            average_loss.append(loss.item()); average_lossCE.append(lossCE.item()); average_lossDICE.append(lossDICE.item())
            if current_slice_type == 'sax':
                sax_loss.append(loss.item()); sax_lossCE.append(lossCE.item()); sax_lossDICE.append(lossDICE.item())
            elif current_slice_type == 'lax':
                lax_loss.append(loss.item()); lax_lossCE.append(lossCE.item()); lax_lossDICE.append(lossDICE.item())

        

    average_loss_mean = sum(average_loss)/len(average_loss);average_lossCE_mean = sum(average_lossCE)/len(average_lossCE);average_lossDICE_mean = sum(average_lossDICE)/len(average_lossDICE)
    if len(sax_loss) == 0:
        sax_loss_mean = np.inf; sax_lossCE_mean = np.inf; sax_lossDICE_mean = np.inf
    else:
        sax_loss_mean = sum(sax_loss)/len(sax_loss);sax_lossCE_mean = sum(sax_lossCE)/len(sax_lossCE);sax_lossDICE_mean = sum(sax_lossDICE)/len(sax_lossDICE)
    
    if len(lax_loss) == 0:
        lax_loss_mean = np.inf; lax_lossCE_mean = np.inf; lax_lossDICE_mean = np.inf
    else:
        lax_loss_mean = sum(lax_loss)/len(lax_loss);lax_lossCE_mean = sum(lax_lossCE)/len(lax_lossCE);lax_lossDICE_mean = sum(lax_lossDICE)/len(lax_lossDICE)
    
    return [average_loss_mean, average_lossCE_mean, average_lossDICE_mean, sax_loss_mean, sax_lossCE_mean, sax_lossDICE_mean, lax_loss_mean, lax_lossCE_mean, lax_lossDICE_mean]
