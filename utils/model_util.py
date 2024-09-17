import torch
import random
import torch.nn.functional as F

def load_from(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
    sam_dict = sam.state_dict()
    
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    
    patch_embed = state_dict['image_encoder.patch_embed.proj.weight'] # 1280 3 16 16 average for input ch 1.
    new_state_dict['image_encoder.patch_embed.proj.weight'] = torch.mean(patch_embed, dim=1).unsqueeze(axis=1)
    
    pos_embed = state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = []
        for rel_pos_key in rel_pos_keys:
            num = int(rel_pos_key.split('.')[2])
            if num in encoder_global_attn_indexes:
                global_rel_pos_keys.append(rel_pos_key)
                
        # global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict

def worker_init_fn(worker_id, args):
        random.seed(args.seed + worker_id)