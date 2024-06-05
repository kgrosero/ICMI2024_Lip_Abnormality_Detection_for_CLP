                            #####################
                            # Utility functions #
                            #####################

###########################################################
# Imports

import numpy as np
import torch
import os 
import cv2

###########################################################
# Description: Crop the face based on the provided landmarks

def crop_img_lks(image, landmarks, pad_percentage):
    # Select min and max values to define bounding box
    x_min, x_max = min(landmarks[:, 0]), max(landmarks[:, 0])
    y_min, y_max = min(landmarks[:, 1]), max(landmarks[:, 1])
    # Add 20% padding to the borders
    padX = pad_percentage* (x_max - x_min) 
    padY = pad_percentage* (y_max - y_min)
    x_min = max(0, x_min - padX)
    x_max = min(image.shape[1], x_max + padX)
    y_min = max(0, y_min - 3*padY) # 3*padY to keep forehead area
    y_max = min(image.shape[0], y_max + padY)
    # Cropping the image based on new borders
    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max), :]         
    # Mapping the landmarks to the cropped image
    resized_lndk = [(int(x - x_min), int(y - y_min)) for (x, y) in landmarks]
    resized_lndk = np.array(resized_lndk)  

    return cropped_image, resized_lndk

###########################################################
# Description: Crop the lips area based on the provided face landmarks

def crop_img_lks_lips(image, landmarks, pad_percentage):
    # Select lip landmarks following ibug format
    lip_lks =  np.array(landmarks[48:])
    # Select min and max values to define lips bounding box
    x_min, x_max = min(lip_lks[:, 0]), max(lip_lks[:, 0])
    y_min, y_max = min(lip_lks[:, 1]), max(lip_lks[:, 1])
    # Add padding to the borders
    padX = pad_percentage * abs(x_max - x_min) 
    padY = pad_percentage * abs(y_max - y_min)  
    x_min = max(0, x_min - padX)
    x_max = min(image.shape[1], x_max + padX)
    y_min = max(0, y_min - padY)
    y_max = min(image.shape[0], y_max + padY)
    # Cropping the image based on new borders
    if len(image) ==3:
        cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max), :]   
    else:      
        cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)] 
    # Mapping the landmarks to the cropped image
    resized_lndk = [(int(x - x_min), int(y - y_min)) for (x, y) in landmarks]
    resized_lndk = np.array(resized_lndk)  

    return cropped_image, resized_lndk[48:]


###########################################################
# Description: Crop the lips area based on the provided landmarks

def crop_img_lks_lips_given(image, landmarks, pad_percentage):
    # Select lip landmarks following ibug format
    lip_lks =  landmarks
    # Select min and max values to define lips bounding box
    x_min, x_max = min(lip_lks[:, 0]), max(lip_lks[:, 0])
    y_min, y_max = min(lip_lks[:, 1]), max(lip_lks[:, 1])
    # Add padding to the borders
    padX = pad_percentage * abs(x_max - x_min) # 0.35
    padY = pad_percentage * abs(y_max - y_min)  
    x_min = max(0, x_min - padX)
    x_max = min(image.shape[1], x_max + padX)
    y_min = max(0, y_min - padY)
    y_max = min(image.shape[0], y_max + padY)
    # Cropping the image based on new borders
    if len(image) ==3:
        cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max), :]   
    else:      
        cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)] 
    # Mapping the landmarks to the cropped image
    resized_lndk = [(int(x - x_min), int(y - y_min)) for (x, y) in landmarks]
    resized_lndk = np.array(resized_lndk)  

    return cropped_image, resized_lndk
    

###########################################################
# Description: Resize face landmarks after resizing image

def resize_lnks(target_image, org_image, org_lnks): 
    scale_factor_width = target_image.shape[1] / org_image.shape[1]
    scale_factor_height = target_image.shape[0] / org_image.shape[0]
    resized_lndk2 = np.empty_like(org_lnks)
    resized_lndk2[:,0] = org_lnks[:,0] * scale_factor_width
    resized_lndk2[:,1] = org_lnks[:,1] * scale_factor_height         
    resized_lndk2 = resized_lndk2.astype(int)
    
    return resized_lndk2

###########################################################
# Description: List files in a directory that end in the extension

def list_files(directory, extension):

    return [file for file in os.listdir(directory) if file.endswith(extension)]
    
###########################################################
# Description: Find common files between two directories

def common_files(dir1, dir2, extension):
    files1 = set(list_files(dir1, extension))
    files2 = set(list_files(dir2, extension))
    common_files = files1.intersection(files2)
    
    return list(common_files)

###########################################################
# Description: Add a black frame around the face image

def add_black_frame(image_path, landmarks, scale=0.05):
    landmarks = landmarks.astype(int)
    # Load the original image
    original_image = cv2.imread(image_path)
    # Compute scale parameters
    original_height, original_width = original_image.shape[:2]
    sc_height, sc_width = int(scale*original_height), int(scale*original_width)
    # Calculate the new dimensions
    new_height = original_image.shape[0] + 2 * sc_height
    new_width = original_image.shape[1] + 2 * sc_width
    # Create a black background image with the new dimensions
    black_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    # Copy the original image onto the black background, leaving the border
    black_frame[sc_height:-sc_height, sc_width:-sc_width] = original_image
    # Adapt landmarks to the new boundaries
    adapted_landmarks_x = landmarks[:,0]+sc_height
    adapted_landmarks_y = landmarks[:,1]+sc_width
    adpt_lks = np.vstack((adapted_landmarks_x,adapted_landmarks_y)).T

    return black_frame, adpt_lks
