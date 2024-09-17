import argparse
import random
import json
import numpy as np
from typing import List, Tuple

# from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Sampler
from torch.utils.data.sampler import RandomSampler
import matplotlib.pyplot as plt

import typing as T
import os

def count_parameters(model, train_keys):
    
    for name, p in model.named_parameters():
        if p.requires_grad:
            train_keys.append(name)
    return sum(p.numel() for p in model.parameters() if p.requires_grad), train_keys

def custom_collate_fn(instances: List[Tuple]):
    stacked_data = {}

    # Take the keys from the first item in the batch
    keys = instances[0].keys()

    for key in keys:
        # Check if the item associated with the key is a tensor
        if isinstance(instances[0][key], torch.Tensor):
            # Stack the tensors along the 0th dimension
            stacked_data[key] = torch.stack([item[key] for item in instances], dim=0)
        else:
            # If it's not a tensor, just collect the items in a list
            stacked_data[key] = [item[key] for item in instances]

    return stacked_data

def save_png(save_dir, image):
    plt.imsave(save_dir, image)

def mkdirs(dirs: T.List) -> None:
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        elif isinstance(obj, (DictConfig)):
            return dict(obj)
        elif isinstance(obj, (ListConfig)):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class Custom_BatchSampler(Sampler):
    def __init__(self, data_source, batch_size, is_trainset=True, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.num_samples = len(self.data_source)

        self.batch_size = batch_size

        self.is_trainset = is_trainset
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples).tolist()
        else:
            indices = torch.arange(self.num_samples).tolist()

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            if self.is_trainset:
                remainder = self.batch_size - len(batch)
                batch.extend(indices[:remainder])
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


class Custom_BatchSampler_CL(Sampler):
    def __init__(self, data_source, batch_size, is_trainset=True, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.num_samples = len(self.data_source[0]) + len(self.data_source[1])

        self.number_of_datasets = len(data_source)

        self.batch_size = batch_size

        self.is_trainset = is_trainset
        self.shuffle = shuffle
        self.drop_last = drop_last
        
    def __iter__(self):
        
        samplers_list = []        
        #set fixed size for dataset 1 , dataset 2
        len_sample = [3,2]

        for dataset_idx, _ in enumerate(range(self.number_of_datasets)):
            cur_dataset = self.data_source[dataset_idx]
            cur_num_samples = len(cur_dataset)
            if self.shuffle:
                indices = torch.randperm(cur_num_samples).tolist()

            samplers_list.append(indices)
            
        samplers_list[-1] = [x+len(self.data_source[0]) for x in samplers_list[-1]]   
        batch = []
        
        indices = samplers_list[0] + samplers_list[1]
        
        cur_index = []
        idx_0 = 0
        idx_1 = 0
        
        # first 3 : non-sever last 2: severe
        for iidx in range(len(indices)):
            if iidx % self.batch_size < 3:
                cur_index = samplers_list[0]
                
                if idx_0 > len(cur_index) -1 :
                    idx_0 = 0
                    idx = cur_index[idx_0]
                    
                else:
                    idx = cur_index[idx_0]
                    idx_0 += 1
            else:
                cur_index = samplers_list[1]
                
                if idx_1 > len(cur_index) -1 :
                    idx_1 = 0
                    idx = cur_index[idx_1]
                else:
                    idx = cur_index[idx_1]
                    idx_1 += 1
                
            batch.append(idx)
            
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            
        if len(batch) > 0 and not self.drop_last:
            if self.is_trainset:
                remainder = self.batch_size - len(batch)
                batch.extend(indices[:remainder])
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
