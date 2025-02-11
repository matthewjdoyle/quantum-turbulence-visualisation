# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:06:25 2024

@author: Matt
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.image as mpimg
from scipy.signal import medfilt2d
import matplotlib.animation as animation
from openpiv import tools, pyprocess, validation, filters, scaling
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.interpolate import griddata


MAIN_DIRECTORY = r"D:/"
SUB_DIRECTORY = r"PhD Storage\Visualization Experiment\Vortex line at 1 K"
OUTPUT_FOLDER = r"D:\PhD Storage\Visualization Experiment\Vortex line at 1 K\PIV images"
DIRECTORY = os.path.join(MAIN_DIRECTORY, SUB_DIRECTORY)



IMAGE_CMAP = "gray_r"     # ["gray", "gray_r", "plasma"]
ROTATE = True

TIMESTAMP = True
SOURCE_FRAME_RATE = 200 # read this in from _paramters instead?
TIME_INTERVAL = 1/SOURCE_FRAME_RATE
PIXEL_SIZE = 12.4e-4 # cm

CUT_IMAGE = True
LEFTCUT = 400
RIGHTCUT = 100
TOPCUT = 100
BOTCUT = 100

# Filters
APPLY_DILATE = False
DILATION_SIZE = 3

APPLY_MEDIAN_SMOOTH = False
MEDIAN_SMOOTH_SIZE = 5

APPLY_RENORMALISE = True


FRAME_START = 10
FRAME_END = 200


def dilate(image, dilation_kernel = 3):
        cv2kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * dilation_kernel + 1,
                                        2 * dilation_kernel + 1)
                    )
        pad_factor = 2
        new_size = [int(image.shape[0] + pad_factor*2*dilation_kernel),
                    int(image.shape[1] + pad_factor*2*dilation_kernel)]
        padded_image = np.ones((new_size[0], new_size[1]))
        padded_image[pad_factor*dilation_kernel:-pad_factor*dilation_kernel, pad_factor*dilation_kernel:-pad_factor*dilation_kernel] = image

        dilated_pixels = cv2.dilate(padded_image, cv2kernel,
                                    iterations=1)
        
        dilated_image = dilated_pixels[pad_factor*dilation_kernel:-pad_factor*dilation_kernel, pad_factor*dilation_kernel:-pad_factor*dilation_kernel]
        return dilated_image
    
def interpolate_minima(image, size = 50):
    fimage = medfilt2d(image, kernel_size = MEDIAN_SMOOTH_SIZE)
    minima_truth = (fimage == minimum_filter(fimage, size, mode = 'reflect'))
    minima_index = np.where(1 == minima_truth)
    minima_image = fimage.copy().astype(np.float32)
    minima_image[np.where(minima_truth != 1)] = np.nan
    
    dummy_matrix = minima_image
    
    x,y = np.indices(dummy_matrix.shape)
    interp = np.array(dummy_matrix)
    interp[np.isnan(interp)] = griddata(
            (x[~np.isnan(dummy_matrix)], y[~np.isnan(dummy_matrix)]), 
            dummy_matrix[~np.isnan(dummy_matrix)],                    
            (x[np.isnan(dummy_matrix)], y[np.isnan(dummy_matrix)]),
            method = 'cubic') 
    
    dummy_matrix = interp
    x,y = np.indices(dummy_matrix.shape)
    interp = np.array(dummy_matrix)
    interp[np.isnan(interp)] = griddata(
            (x[~np.isnan(dummy_matrix)], y[~np.isnan(dummy_matrix)]), 
            dummy_matrix[~np.isnan(dummy_matrix)],                    
            (x[np.isnan(dummy_matrix)], y[np.isnan(dummy_matrix)]),
            method = 'nearest')  
    
    return interp.astype(np.int32)

def renormalise(image):
    minima_image = interpolate_minima(image)

    minima_image = minima_image.astype(np.int32)
    

    fi = image - minima_image 
    
    return fi


image_folder = DIRECTORY

image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if (f.endswith('.png') or f.endswith('.jpg')) and f.startswith('image_')])

# Loop over image pairs (a PIV algorithm that considers 3 or more frames might be useful if possible). 
for i in range(FRAME_START, FRAME_END, 1):
    # Load two consecutive images
    frame_a = tools.imread(os.path.join(image_folder, image_files[i]))
    frame_b = tools.imread(os.path.join(image_folder, image_files[i + 1]))
    
    # Rotate image
    frame_a = np.rot90(frame_a, k=-1)
    frame_b = np.rot90(frame_b, k=-1)
    
    if CUT_IMAGE:
        frame_a = frame_a[LEFTCUT:-RIGHTCUT, BOTCUT:-TOPCUT]
        frame_b = frame_b[LEFTCUT:-RIGHTCUT, BOTCUT:-TOPCUT]
        
    if APPLY_RENORMALISE: 
        frame_a = renormalise(frame_a)
        frame_b = renormalise(frame_b)
        
    if APPLY_MEDIAN_SMOOTH:
        frame_a = medfilt2d(frame_a, kernel_size = MEDIAN_SMOOTH_SIZE)
        frame_b = medfilt2d(frame_b, kernel_size = MEDIAN_SMOOTH_SIZE)
    
    if APPLY_DILATE:
        frame_a = dilate(frame_a, DILATION_SIZE)
        frame_b = dilate(frame_b, DILATION_SIZE)
        

    winsize = 100       # size of the interrogation window in pixels
    searchsize = 100    # size of the search area in pixels
    overlap = 50        # overlap of the windows
    dt = TIME_INTERVAL  # time between frames (in seconds), adjust to your needs

    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method='peak2peak'
    )

    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape, 
        search_area_size=searchsize, 
        overlap=overlap
    )
    
    

    # Filter based on signal to noise ratio 
    mask = validation.sig2noise_val(sig2noise, threshold=1.01)
    
    # Worth looking up how this function works.  
    u, v = filters.replace_outliers(u, v, mask, method='localmean', max_iter=3, kernel_size=2)
    
    # # optional mask filter (decides own mask based on the filter)
    # u[mask == False] = np.nan
    # v[mask == False] = np.nan

    x, y, u, v = tools.transform_coordinates(x, y, u, v)
    

    fig, ax = plt.subplots(figsize=(4, 4), dpi = 300)
    ax.imshow(frame_b, cmap='gray')
    ax.quiver(x, y, u, v, color='r', scale=None)
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'PIV_result_{i:03d}.png'))
    plt.close()
    
    print(f'Processed image pair {i}: {image_files[i]} and {image_files[i+1]}')
    
    
    # Scale velocities and positions by the pixel size. 
    u, v, x, y = scaling.uniform(
        u, v, x, y, scaling_factor=1/PIXEL_SIZE  # adjust scaling factor as necessary
    )
    print(np.mean(u), np.mean(v))

print("PIV analysis complete!")