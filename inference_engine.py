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
import cineCMR_SAM.functions_collection as ff


def inference_loop(model, data_loader_valid, args): 
        
    with torch.no_grad():
                                    
        # model.eval()  # empirically we found we need to comment out this line, otherwise it leads to bad results

        if args.turn_zero_seg_slice_into is not None:
            criterionBCE = torch.nn.CrossEntropyLoss(ignore_index=args.turn_zero_seg_slice_into)
        else:
            criterionBCE = torch.nn.CrossEntropyLoss()

        # start to train
        average_valid_loss = []; average_valid_lossCE = []; average_valid_lossDICE = []
        sax_valid_loss = []; sax_valid_lossCE = []; sax_valid_lossDICE = []
        lax_valid_loss = []; lax_valid_lossCE = []; lax_valid_lossDICE = []

        assert len(data_loader_valid) == len(args.dataset_valid)

        # iterate over dataset
        for i in range(len(data_loader_valid)):
            current_data_loader = data_loader_valid[i] # this is an iterable
            current_slice_type = args.dataset_valid[i][1]
            print('in validation: current slice type: ', current_slice_type)
            
            for data_iter_step, batch in tqdm(enumerate(current_data_loader)):
                # Note that our input shape 
                batch["image"]= batch["image"].cuda()

                if args.text_prompt:
                    batch["text_prompt_feature"] = batch["text_prompt_feature"].to(torch.float32)
                    
                output = model(batch, False, args.img_size)

                mask = batch["mask"]
                mask = rearrange(mask, 'b c h w d -> (b d) c h w ').to("cuda")

                #### CE loss
                lossCE = criterionBCE(output["masks"], mask.squeeze(1).long()) 

                #### customized dice loss
                mask_for_dice = batch["mask"]
                mask_for_dice = rearrange(mask_for_dice, 'b c h w d -> (b d) c h w').to("cuda")
                lossDICE = ff.customized_dice_loss(output["masks"], mask_for_dice.squeeze(1).long(), num_classes = args.num_classes, exclude_index = args.turn_zero_seg_slice_into)
                
                #### total loss: weighted loss
                loss = args.loss_weights[0] * lossCE + args.loss_weights[1] * lossDICE
                
                if torch.isnan(loss):
                    continue
                
                average_valid_loss.append(loss.item()); average_valid_lossCE.append(lossCE.item()); average_valid_lossDICE.append(lossDICE.item())
                if current_slice_type == 'sax':
                    sax_valid_loss.append(loss.item()); sax_valid_lossCE.append(lossCE.item()); sax_valid_lossDICE.append(lossDICE.item())
                elif current_slice_type == 'lax':
                    lax_valid_loss.append(loss.item()); lax_valid_lossCE.append(lossCE.item()); lax_valid_lossDICE.append(lossDICE.item())

                torch.cuda.synchronize()

        average_valid_loss_mean = sum(average_valid_loss)/len(average_valid_loss);average_valid_lossCE_mean = sum(average_valid_lossCE)/len(average_valid_lossCE);average_valid_lossDICE_mean = sum(average_valid_lossDICE)/len(average_valid_lossDICE)
        if len(sax_valid_loss) == 0:
            sax_valid_loss_mean = np.inf; sax_valid_lossCE_mean = np.inf; sax_valid_lossDICE_mean = np.inf
        else:
            sax_valid_loss_mean = sum(sax_valid_loss)/len(sax_valid_loss);sax_valid_lossCE_mean = sum(sax_valid_lossCE)/len(sax_valid_lossCE);sax_valid_lossDICE_mean = sum(sax_valid_lossDICE)/len(sax_valid_lossDICE)
        
        if len(lax_valid_loss) == 0:
            lax_valid_loss_mean = np.inf; lax_valid_lossCE_mean = np.inf; lax_valid_lossDICE_mean = np.inf
        else:
            lax_valid_loss_mean = sum(lax_valid_loss)/len(lax_valid_loss);lax_valid_lossCE_mean = sum(lax_valid_lossCE)/len(lax_valid_lossCE);lax_valid_lossDICE_mean = sum(lax_valid_lossDICE)/len(lax_valid_lossDICE)
       
    return [average_valid_loss_mean, average_valid_lossCE_mean, average_valid_lossDICE_mean, sax_valid_loss_mean, sax_valid_lossCE_mean, sax_valid_lossDICE_mean, lax_valid_loss_mean, lax_valid_lossCE_mean, lax_valid_lossDICE_mean]


def pred_save(batch, output,args, save_folder_sub):
    pred_seg = np.rollaxis(output["masks"].argmax(1).detach().cpu().numpy(), 0, 3)
                        
    original_shape = np.array([x.item() for x in batch["original_shape"]])
    centroid = batch["centroid"].numpy().flatten()
              
    crop_start_end_list = []
    for dim, size in enumerate([args.img_size, args.img_size]):
        start = max(centroid[dim] - size // 2, 0)
        end = start + size
        # Adjust the start and end if they are out of bounds
        if end > original_shape[dim]:
            end = original_shape[dim]
            start = max(end - size, 0)
        crop_start_end_list.append([start, end])
     

    final_pred_seg = np.zeros(original_shape)
    final_pred_seg[crop_start_end_list[0][0]:crop_start_end_list[0][1], crop_start_end_list[1][0]:crop_start_end_list[1][1]] = pred_seg

    # save original image and ground truth segmentation
    if args.full_or_nonzero_slice[0:4] == 'full':
        original_image_file = batch["image_full_slice_file"][0]
        original_seg_file = batch["seg_full_slice_file"][0]
    elif args.full_or_nonzero_slice[0:4] == 'nonz':
        original_image_file = batch["image_nonzero_slice_file"][0]
        original_seg_file = batch["seg_nonzero_slice_file"][0]
    elif args.full_or_nonzero_slice[0:4] == 'loos':
        original_image_file = batch["image_nonzero_slice_file_loose"][0]
        original_seg_file = batch["seg_nonzero_slice_file_loose"][0]
                    
    slice_index = batch["slice_index"].item()

    affine = nb.load(original_image_file).affine
    original_image = nb.load(original_image_file).get_fdata()[:,:,slice_index,:]

    patient_id = batch["patient_id"][0]
   
    ff.make_folder([os.path.dirname(save_folder_sub),save_folder_sub])

    nb.save(nb.Nifti1Image(final_pred_seg, affine), os.path.join(save_folder_sub, 'pred_seg_%s.nii.gz' % slice_index))
    nb.save(nb.Nifti1Image(original_image, affine), os.path.join(save_folder_sub, 'original_image_%s.nii.gz' % slice_index))



def pred_save_lax(batch, output,args, save_folder_sub):
    pred_seg = np.rollaxis(output["masks"].argmax(1).detach().cpu().numpy(), 0, 3)
                        
    original_shape = np.array([x.item() for x in batch["original_shape"]])
    centroid = batch["centroid"].numpy().flatten()
              
    crop_start_end_list = []
    for dim, size in enumerate([args.img_size, args.img_size]):
        start = max(centroid[dim] - size // 2, 0)
        end = start + size
        # Adjust the start and end if they are out of bounds
        if end > original_shape[dim]:
            end = original_shape[dim]
            start = max(end - size, 0)
        crop_start_end_list.append([start, end])
     

    final_pred_seg = np.zeros(original_shape)
    final_pred_seg[crop_start_end_list[0][0]:crop_start_end_list[0][1], crop_start_end_list[1][0]:crop_start_end_list[1][1]] = pred_seg

    # save original image and ground truth segmentation
    original_image_file = batch["img_file"][0]
    original_seg_file = batch["seg_file"][0]

    lax_name = batch["lax_name"][0]
                    
    affine = nb.load(original_image_file).affine
    original_image = nb.load(original_image_file).get_fdata()
    save_folder = os.path.join(args.output_dir, 'predicts_lax'); ff.make_folder([save_folder])

    patient_id = batch["patient_id"][0]

    nb.save(nb.Nifti1Image(final_pred_seg, affine), os.path.join(save_folder_sub, lax_name + '_seg_pred.nii.gz'))
    nb.save(nb.Nifti1Image(original_image, affine), os.path.join(save_folder_sub, lax_name + '_img.nii.gz'))