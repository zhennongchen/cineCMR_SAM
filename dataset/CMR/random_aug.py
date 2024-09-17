#!/usr/bin/env python
import numpy as np
import random
import SAM_CMR_seg.Data_processing as Data_processing
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image


# help functions:
def random_flip(i, selected_option = None):
    # i is object
    if selected_option is None:
        options = [[0, 0], [0, 1], [1, 0], [1, 1]]
        selected_option = random.choice(options)
   
    return Data_processing.flip_image(np.copy(i), selected_option), selected_option

def random_rotate(i, z_rotate_degree = None, z_rotate_range = [-10,10], fill_val = None, order = 0):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = random.uniform(z_rotate_range[0], z_rotate_range[1])

    if fill_val is None:
        fill_val = np.min(i)
    
    return Data_processing.rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree

def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-10,10]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))

    return Data_processing.translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate


def random_brightness(i, v = None):
    # i is the object
    # if v is given, then assert v is from 0.1 to 1.9
    if v is not None:
        assert v >= 0.1 and v <= 1.9
    if v is None:
        v = np.random.uniform(0.1,1.9)
    
    if i.ndim == 3:
        new_i = np.zeros(i.shape)
        for k in range(0, i.shape[-1]):
            im = np.copy(i[:,:,k])
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im.astype('uint8'), mode='L')  # need to convert to PIL image
            
            im = PIL.ImageEnhance.Brightness(im).enhance(v)

            new_i[:,:,k] = np.array(im) # convert back to numpy array
    
    elif i.ndim == 2:
        im = np.copy(i)
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im.astype('uint8'), mode='L')
        im = PIL.ImageEnhance.Brightness(im).enhance(v)
        new_i = np.array(im)

    return new_i,v


def random_sharpness(i, v = None):
    # i is the object
    # if v is given, then assert v is from 0.1 to 1.9
    if v is not None:
        assert v >= 0.1 and v <= 1.9
    if v is None:
        v = np.random.uniform(0.1,1.9)
    

    if i.ndim == 3:
        new_i = np.zeros(i.shape)
        for k in range(0, i.shape[-1]):
            im = np.copy(i[:,:,k])
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im.astype('uint8'), mode='L')  # need to convert to PIL image
            
            im = PIL.ImageEnhance.Sharpness(im).enhance(v)

            new_i[:,:,k] = np.array(im) # convert back to numpy array
    elif i.ndim == 2:
        im = np.copy(i)
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im.astype('uint8'), mode='L')
        im = PIL.ImageEnhance.Sharpness(im).enhance(v)
        new_i = np.array(im)

    return new_i,v


def random_contrast(i, v = None):
    # i is the object
    # if v is given, then assert v is from 0.1 to 1.9
    if v is not None:
        assert v >= 0.1 and v <= 1.9
    if v is None:
        v = np.random.uniform(0.1,1.9)


    if i.ndim == 3:
        new_i = np.zeros(i.shape)
        for k in range(0, i.shape[-1]):
            im = np.copy(i[:,:,k])
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im.astype('uint8'), mode='L')  # need to convert to PIL image
            
            im = PIL.ImageEnhance.Contrast(im).enhance(v)

            new_i[:,:,k] = np.array(im) # convert back to numpy array

    elif i.ndim == 2:
        im = np.copy(i)
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im.astype('uint8'), mode='L')
        im = PIL.ImageEnhance.Contrast(im).enhance(v)
        new_i = np.array(im)

    return new_i,v



