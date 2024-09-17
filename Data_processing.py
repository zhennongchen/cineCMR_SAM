import numpy as np
import nibabel as nb
import os
from skimage.measure import block_reduce
from scipy import ndimage
import cineCMR_SAM.functions_collection as ff


# function: basic crop or pad
def crop_or_pad(image, target_size, padding_value):
    # Pad each axis to the target size or crop each axis to the target size
    # target: e.g. [256,256,256]
    # value is the constant value for padding
    margin = target_size - np.array(image.shape)
    padding = [(0, max(x, 0)) for x in margin]
    image = np.pad(image, padding, mode="constant", constant_values = padding_value)

    for i, x in enumerate(margin):
        image = np.roll(image, shift=+(x // 2), axis=i)

    if type(target_size) == int:
        target_size = [target_size] * image.ndim

    ind = tuple([slice(0, t) for t in target_size])
    return image[ind]

# function: correction for the above function
def correct_shift_caused_in_pad_crop_loop(img):
    # if an image goes from [a,b,c] --> pad --> [A,B,c] --> crop --> [a,b,c], when a,b is even, it goes back to original image, but when a,b is odd, it need to shift by 1 pixel in x and y
    if img.shape[0] % 2 == 1:

        img = np.roll(img, shift = 1, axis = 0)
        img = np.roll(img, shift = 1, axis = 1)
    else:
        img = np.copy(img)
    return img

# function:center crop (need to provide the segmentation mask)
def center_crop(I, S, crop_size, according_to_which_class, centroid = None):
    # make sure S is integers
    S = S.astype(int)
    # Compute the centroid of the class 1 region in the mask
    assert isinstance(according_to_which_class, list), "according_to_which_class must be a list"
    assert I.shape == S.shape, "Image and mask must have the same shape"
    assert len(crop_size) == len(I.shape), "Crop size dimensions must match image dimensions"
    
    # Find the indices where the mask > 0
    if centroid is None:
        mask_indices = np.argwhere(np.isin(S, according_to_which_class))

        if len(mask_indices) == 0:
            raise ValueError("The mask does not contain any class 1 region")

        # Compute centroid
        
        centroid = np.mean(mask_indices, axis=0).astype(int)

    # Define the crop slices for each dimension
    slices = []
    for dim, size in enumerate(crop_size):
        start = max(centroid[dim] - size // 2, 0)
        end = start + size
        # Adjust the start and end if they are out of bounds
        if end > I.shape[dim]:
            end = I.shape[dim]
            start = max(end - size, 0)
        slices.append(slice(start, end))

    # Crop the image and the mask
    if len(I.shape) == 2:
        cropped_I = I[slices[0], slices[1]]
        cropped_S = S[slices[0], slices[1]]
    elif len(I.shape) == 3:
        cropped_I = I[slices[0], slices[1], slices[2]]
        cropped_S = S[slices[0], slices[1], slices[2]]
    else:
        raise ValueError("Image dimensions not supported")

    return cropped_I, cropped_S, centroid

# function: turn image range into 0-255
def turn_image_range_into_0_255(img):
    # turn image range into 0-255
    img = img.astype(float)
    if np.max(img) != 255:
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    return img


# function: normalization using mu and std
def normalize_image_mu_std(img, mu = 0.5, std = 0.5, denormalize = False):
    if denormalize == False:
        return (img - mu) / std
    else:
        return img * std + mu
    
# function: normalization using min and max
def normalize_image(array, denormalize=False, original_min=None, original_max=None):
    if denormalize:
        if original_min is None or original_max is None:
            raise ValueError("Original min and max values are required for denormalization.")
        
        # Denormalize the array
        denormalized_array = array * (original_max - original_min) + original_min
        return denormalized_array
    else:
        min_value = np.min(array)
        max_value = np.max(array)

        # Avoid division by zero in case all values are the same
        if max_value - min_value == 0:
            return np.zeros(array.shape)

        normalized_array = (array - min_value) / (max_value - min_value)
        return normalized_array

    
# function: translate image
def translate_image(image, shift):
    assert len(shift) in [2, 3], "Shift must be a list of 2 elements for 2D or 3 elements for 3D"
    assert len(image.shape) in [2, 3], "Image must be either 2D or 3D"
    assert len(image.shape) == len(shift), "Shift dimensions must match image dimensions"

    fill_val = np.min(image)  # Fill value is the minimum value in the image
    translated_image = np.full_like(image, fill_val)  # Create an image filled with fill_val

    if image.ndim == 2:  # 2D image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                new_i = i - shift[0]
                new_j = j - shift[1]
                if 0 <= new_i < image.shape[0] and 0 <= new_j < image.shape[1]:
                    translated_image[new_i, new_j] = image[i, j]
    elif image.ndim == 3:  # 3D image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    new_i = i - shift[0]
                    new_j = j - shift[1]
                    new_k = k - shift[2]
                    if 0 <= new_i < image.shape[0] and 0 <= new_j < image.shape[1] and 0 <= new_k < image.shape[2]:
                        translated_image[new_i, new_j, new_k] = image[i, j, k]
    else:
        raise ValueError("Image dimensions not supported")

    return translated_image


# function: rotate image
def rotate_image(image, degrees, order, fill_val = None):

    if fill_val is None:
        fill_val = np.min(image)
        
    if image.ndim == 2:  # 2D image
        assert isinstance(degrees, (int, float)), "Degrees should be a single number for 2D rotation"
        rotated_img = ndimage.rotate(image, degrees, reshape=False, mode='constant', cval=fill_val, order = order)

    elif image.ndim == 3:  # 3D image
        assert len(degrees) == 3 and all(isinstance(deg, (int, float)) for deg in degrees), "Degrees should be a list of three numbers for 3D rotation"
        # Rotate around x-axis
        rotated_img = ndimage.rotate(image, degrees[0], axes=(1, 2), reshape=False, mode='constant', cval=fill_val, order  = order)
        # Rotate around y-axis
        rotated_img = ndimage.rotate(rotated_img, degrees[1], axes=(0, 2), reshape=False, mode='constant', cval=fill_val, order = order)
        # Rotate around z-axis
        rotated_img = ndimage.rotate(rotated_img, degrees[2], axes=(0, 1), reshape=False, mode='constant', cval=fill_val, order = order)
    else:
        raise ValueError("Image must be either 2D or 3D")

    return rotated_img


# function: flip image
def flip_image(image, flip):
    # flip is [0,0] or [0,1] or [1,0] or [1,1], first meaning whether flip along x-axis, second meaning whether flip along y-axis
    assert len(flip) == 2, "Flip should be a list of two elements (0 or 1)"
    assert all(f in [0, 1] for f in flip), "Elements of flip should be either 0 or 1"
    assert image.ndim in [2, 3], "Image must be either 2D or 3D"

    flipped_image = np.copy(image)

    if flip[0] == 1:  # Flip along the x-axis (vertical flip)
        flipped_image = flipped_image[::-1, ...]

    if flip[1] == 1:  # Flip along the y-axis (horizontal flip)
        if image.ndim == 2:  # 2D image
            flipped_image = flipped_image[:, ::-1]
        elif image.ndim == 3:  # 3D image
            flipped_image = flipped_image[:,  ::-1, :]

    return flipped_image



# function: cutoff intensity
def cutoff_intensity(img, cutoff_low = None, cutoff_high = None):
    xx = np.copy(img)

    if cutoff_low is not None and np.min(img) < cutoff_low:
        xx[img <= cutoff_low] = cutoff_low
    
    if cutoff_high is not None and np.max(img) > cutoff_high:
        xx[img >= cutoff_high] = cutoff_high
    return xx


# function: bounding box generation based on ground truth segmentation
def get_bbox_from_mask_2D(mask, class_id = 1, box_buffer =['random',10]):
    '''Returns a bounding box from a mask'''
    y_indices, x_indices = np.where(mask == class_id)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    if box_buffer[0] == 'random':
        x_min = max(0, x_min - np.random.randint(0, box_buffer[1]))
        x_max = min(W, x_max + np.random.randint(0, box_buffer[1]))
        y_min = max(0, y_min - np.random.randint(0, box_buffer[1]))
        y_max = min(H, y_max + np.random.randint(0, box_buffer[1]))
    elif box_buffer[0] == 'fixed':
        x_min = max(0, x_min - box_buffer[1])
        x_max = min(W, x_max + box_buffer[1])
        y_min = max(0, y_min - box_buffer[1])
        y_max = min(H, y_max + box_buffer[1])
    else:
        raise ValueError('box_buffer[0] must be either random or fixed')
   
    return np.array([x_min, y_min, x_max, y_max]).astype(int)

def get_bbox_from_mask_all_volumes(mask,tf_list, class_id = 1, box_buffer =['random',5]):
    assert len(mask.shape) == 3
    # sum_list = np.sum(mask == class_id, axis = (0,1))
    # nonzero_index = np.nonzero(sum_list)
    # z_max = np.argmax(sum_list)
    # z_min = np.argmin(sum_list[nonzero_index]); z_min = nonzero_index[0][z_min]
    if len(tf_list) == 2:
        z_max = tf_list[0]; z_min = tf_list[1]
    elif len(tf_list) == 1:
        z_max = tf_list[0]; z_min = tf_list[0]
    else:
        z_max = tf_list[0]; z_min = tf_list[len(tf_list)//2]
    box_list = []
    for z in [z_max, z_min]:
        box = get_bbox_from_mask_2D(mask[:,:,z], class_id = class_id, box_buffer = box_buffer)
        if z == z_max:
            box_max = box
        
        box_list.append(box)
    return np.stack(box_list,axis = 0), z_max, z_min

