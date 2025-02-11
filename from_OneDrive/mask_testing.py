# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:35:50 2023

@author: r61659md
"""
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import sys
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.interpolate import griddata

"""MATT-PC"""
# DRIVE = "D:/"
SUPERFLUID_DRIVE = "S:/"

"""MATT-HOME-PC"""
DRIVE = "C:/Users/Matt/The University of Manchester/Quantum fluids group - Documents/Vortex visualisation files/Main data storage/"


"""CAMERA-PC"""
# DRIVE = r"D:/"
# SUPERFLUID_DRIVE = r"Z:/"

# sys.path.append(r"%sChris_Goodwin/Programs/Class and Function Library" % SUPERFLUID_DRIVE)
sys.path.append(r"C:/Users/Matt/The University of Manchester/Quantum fluids group - Documents/Simulations")
import data_handling_library as dh
import plotting_library as pl
pt = pl.Plotter()

sys.path.append(r"%sCommon/Data collection scripts" % SUPERFLUID_DRIVE)
import mask as msk
import data_handling_tools as dht

NAME = '_mask_movie'

# DIRECTORY = "%sCooldown 9\stationary\990 fps temperature sweep/run3" % DRIVE
DIRECTORY = "%sCooldown 11/varying_transducer_output/990_fps_low_temp/run3" % DRIVE
# DIRECTORY = "%sCooldown 11/varying_transducer_output/200_fps_1K/run1" % DRIVE

TEMPORAL_SMOOTHING = None

STANDARD_NOISE_FACTOR = 0.01
STANDARD_SMOOTHING = 3

COMPARE_SMOOTHING = False
COMPARE_NOISE_FACTOR = True

NOISE_FACTORS =  [0.0075, 0.010, 0.0125, 0.015]
SMOOTHINGS = [3,5,7,9,11]

PARAMETER_ARRAY = [0,1] # NOT USED JUST INITILIASE

if COMPARE_SMOOTHING:
    PARAMETER_ARRAY = SMOOTHINGS
    NAME = NAME + "_compare_smoothing_factor"
if COMPARE_NOISE_FACTOR:
    PARAMETER_ARRAY = NOISE_FACTORS
    NAME = NAME + "_compare_noise_factor"
NAME = NAME + '.mp4'
print(DIRECTORY)
print(NAME)

if COMPARE_SMOOTHING == False and COMPARE_NOISE_FACTOR == False:
    masking = msk.Mask(DIRECTORY)



class Animator:
    def __init__(self, directory, number_frames, frame_rate):
        self.directory = directory
        self.number_frames = int(number_frames)
        self.frame_rate = frame_rate
        if COMPARE_NOISE_FACTOR or COMPARE_SMOOTHING:
            self.setup_multi_mask_figure()
        else:
            self.setup_figure()
        # self.setup_multi_mask_figure()
        self.start_time = time.time()
        self.create_animation()
        

    def setup_figure(self):
        self.fig, self.axis = plt.subplots(1,2,figsize = (4, 10))
        self.fig.tight_layout()
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        # plt.margins(0,0)
        # pt.new_subplot_figure(' ', ['Smoothed image', 'Mask'] ,[' ', ' '], [' ', ' '], 1, 2, figsize=(4,8))
        pt.define_ax(self.axis)
        

        
    def setup_multi_mask_figure(self, num_subplots = len(PARAMETER_ARRAY) + 1):
        self.fig, self.axis = plt.subplots(num_subplots,1,figsize = (8, (2*num_subplots)))
        self.fig.tight_layout()
        pt.define_ax(self.axis)
        
    
    def create_multi_image(self, frame_number):
        time.sleep(0.01)
        current_time = time.time() - self.start_time
        print("\r frame number: %d | time/frame: %.3f s | total_elapsed: %.3f s" % (frame_number, current_time/(frame_number+1), current_time), end = '', flush=True)
        
        for i in range(len(PARAMETER_ARRAY)+1):
            self.axis[i].clear()
            
        for i in range(len(PARAMETER_ARRAY)):
            if COMPARE_NOISE_FACTOR:
                masking = msk.Mask(DIRECTORY, noise_factor = PARAMETER_ARRAY[i],
                                   smoothing = STANDARD_SMOOTHING,
                                   num_summed = TEMPORAL_SMOOTHING)
                self.axis[i].set_title('Noise factor = %.4f' % PARAMETER_ARRAY[i])
            if COMPARE_SMOOTHING:
                masking = msk.Mask(DIRECTORY, smoothing = PARAMETER_ARRAY[i], 
                                   noise_factor = STANDARD_NOISE_FACTOR,
                                   num_summed = TEMPORAL_SMOOTHING)
                self.axis[i].set_title('Smoothing = %.4f' % PARAMETER_ARRAY[i])
            
            if masking.same_numbering:
                image = masking.prepare_image(frame_number + masking.first_number)
                image_with_transducer = masking.load_summed_image(frame_number + masking.first_number)
            else:
                image = masking.prepare_image(frame_number)
                image_with_transducer = masking.load_summed_image(frame_number)
                
            mask = masking.create_mask(image)
            
            pt.show_image(mask, ax = self.axis[i])
            
            
        # pt.show_image(dh.clahe_image(image_with_transducer), ax = self.axis[len(PARAMETER_ARRAY)])
        
        pt.show_image(dh.clahe_image(image_with_transducer[:,:790]), ax = self.axis[len(PARAMETER_ARRAY)])
        
        del image
        del mask
        
        self.axis[len(PARAMETER_ARRAY)].set_title('Frame number %d' % frame_number)
            
        
        
    def create_image(self, frame_number):
        time.sleep(0.01)
        current_time = time.time() - self.start_time
        print("\r frame number: %d | time/frame: %.3f s | total_elapsed: %.3f s" % (frame_number, current_time/(frame_number+1), current_time), end = '', flush=True)
        # n = str(frame_number) # (image for testing)
        self.axis[0].clear()
        self.axis[1].clear()
        
        image = masking.prepare_image(frame_number)
        mask = masking.create_mask(image)
        
        pt.show_image(dh.clahe_image(image), ax = self.axis[0])
        pt.show_image(mask, ax = self.axis[1])
        self.axis[0].set_title('Frame number %d' % frame_number)
        
        del image
        del mask
        
    def create_animation(self):
        
        if COMPARE_NOISE_FACTOR or COMPARE_SMOOTHING:
            anim = animation.FuncAnimation(self.fig, self.create_multi_image,
                                           frames=self.number_frames,
                                           interval=self.number_frames / self.frame_rate,
                                           cache_frame_data=False)
        else:
            anim = animation.FuncAnimation(self.fig, self.create_image,
                                           frames=self.number_frames,
                                           interval=self.number_frames / self.frame_rate,
                                           cache_frame_data=False)
        filename = os.path.join(self.directory.path, NAME)
        anim.save(filename, fps=self.frame_rate, extra_args=['-vcodec', 'libx264'])
        
        
def video_one_run():
    filenames = sorted(dh.find_all_images(DIRECTORY+'/summed frames', '.png'))
    directory = dht.Directory(DIRECTORY)
    animator = Animator(directory, len(filenames)/4, 8)
    
# video_one_run()



###########################################################################################################################################
masking = msk.Mask(DIRECTORY, smoothing = 3, noise_factor = 0.0019, transducer_cut = 800, num_summed=1)    



N = 100
image = masking.prepare_image(N)
plt.figure()
plt.imshow(image, interpolation = 'none')
plt.show()
minima = masking.interpolate_minima_no_box(image)

binary = masking.create_mask(image)

binary = np.uint8(binary)

# binary  = cv2.cvtColor(image1copy, cv2.COLOR_HSV2BGR)

plt.figure()
plt.imshow(binary, interpolation = 'none')
plt.show()

binary = cv2.threshold(binary, 0, 1, cv2.THRESH_BINARY)[1]
num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity = 4)
plt.figure()
plt.imshow(binary, interpolation = 'none')
plt.show()

sizes = stats[:,-1]
sizes = sizes[1:]
num_labels -= 1
min_size = 7**2 +2
im_result = np.zeros_like(labels_im)
for blob in range(num_labels):
    if sizes[blob] >= min_size:
        im_result[labels_im == blob + 1] = 1

plt.figure()
plt.imshow(im_result, interpolation = 'none')
plt.show()

masking.plot_mask_with_image(N)

def imshow_components(ax, labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    
    im = ax.imshow(labeled_img[:,100:], interpolation = 'none')
    ax.axis('off')

    # cv2.waitKey()

def get_minima(image):
    minima_truth = (image == minimum_filter(image, 7, mode = 'reflect'))
    minima_index = np.where(1 == minima_truth)
    minima_image = image.copy()
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
    
    return minima_image, interp



fig, ax = plt.subplots(nrows = 2, figsize = (7,4), dpi = 300)
imshow_components(ax[0], labels_im)
print(np.amax(labels_im))
num_labels, labels_im2, stats, _ = cv2.connectedComponentsWithStats(np.uint8(im_result), connectivity = 4)
imshow_components(ax[1], labels_im2)
print(np.amax(labels_im2))

fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\ConnectedComponents.pdf', format='pdf')
plt.show()
###########################################################################################################################################

""" MINIMAS and INTERPOLATED MINIMAS """
minimas, interpolated_minimas = get_minima(image)

vmin = np.min(image)
vmax = np.max(image)

cmap = 'jet'

fig, ax = plt.subplots(nrows = 3, figsize = (7,6), dpi = 300, layout = 'constrained')


im = ax[0].imshow(image[:,100:], interpolation = 'none', vmin = vmin, vmax = vmax, cmap = cmap)
# ax[0].axis('off')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title(r'$\bf{a}$', loc = 'left', fontsize = 16)

im = ax[1].imshow(minimas[:,100:], interpolation = 'none', vmin = vmin, vmax = vmax, cmap = cmap)
# ax[1].axis('off')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title(r'$\bf{b}$', loc = 'left', fontsize = 16)

im = ax[2].imshow(interpolated_minimas[:,100:], interpolation = 'none', vmin = vmin, vmax = vmax, cmap = cmap)
# ax[2].axis('off')
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title(r'$\bf{c}$', loc = 'left', fontsize = 16)


cb = fig.colorbar(im, ax=ax[:], shrink=1, ticks = [])
cb.set_label(label = 'Intensity', size = 15)
# fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\InterpolatedMinima.pdf', format='pdf')
plt.show()



#####################################################################################################################
""" Convolution and dilation """
image_path = DIRECTORY + '/image_%05d.png' % N
raw_image = dh.read_image(image_path)
clahe_image = dh.clahe_image(raw_image, clahe_clip=4, clahe_tile=10)

kernel_size = 3
KERNEL_SIZE = kernel_size
kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1)
        )

filtered_pixels = cv2.filter2D(raw_image, -1, kernel)
filtered_pixels[filtered_pixels < 0] = 0


def dilate_pixels(filtered_pixels, kernel, iterations=1):
    
    pad_factor = 2
    new_size = [int(filtered_pixels.shape[0] + pad_factor*2*KERNEL_SIZE),
                int(filtered_pixels.shape[1] + pad_factor*2*KERNEL_SIZE)]
    padded_image = np.ones((new_size[0], new_size[1]))
    padded_image[pad_factor*KERNEL_SIZE:-pad_factor*KERNEL_SIZE, pad_factor*KERNEL_SIZE:-pad_factor*KERNEL_SIZE] = filtered_pixels
    # self.dilated_pixels = cv2.dilate(self.filtered_pixels, kernel,
    #                                  iterations=iterations)
    # print(padded_image.shape)
    dilated_pixels = cv2.dilate(padded_image, kernel,
                                iterations=iterations)
    
    dilated_pixels = dilated_pixels[pad_factor*KERNEL_SIZE:-pad_factor*KERNEL_SIZE, pad_factor*KERNEL_SIZE:-pad_factor*KERNEL_SIZE]
    print(dilated_pixels.shape)
    return dilated_pixels

dilated_image = dilate_pixels(filtered_pixels, kernel)

fig, ax = plt.subplots(nrows = 3, figsize = (8,6), dpi = 300)

ax[0].imshow(raw_image, cmap = 'gray')
ax[0].axis('off')
ax[0].set_title(r'$\bf{a}$ (raw image)', loc = 'left', fontsize = 16)

ax[1].imshow(filtered_pixels, cmap = 'gray')
ax[1].axis('off')
ax[1].set_title(r'$\bf{b}$ (filtered)', loc = 'left', fontsize = 16)

ax[2].imshow(dilated_image, cmap = 'gray')
ax[2].axis('off')
ax[2].set_title(r'$\bf{c}$ (dilated)', loc = 'left', fontsize = 16)
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\FilterAndDilation.pdf', format='pdf')
plt.show()


####################################################################################################################
""" Mask and detections """
image_path = DIRECTORY + '/image_%05d.png' % N
raw_image = dh.read_image(image_path)
clahe_image = dh.clahe_image(raw_image, clahe_clip=4, clahe_tile=10)

import pandas as pd
df = pd.read_csv(DIRECTORY + '/_masked_coordinates.csv')
dets = df.loc[df["frame"] == N]
dets2x = dets['y'] - 100
dets2y = dets['x'] 
normm0 = (np.max(dets['m_0']) - dets['m_0']) / (np.max(dets['m_0']) - np.min(dets['m_0']))
colors = plt.cm.jet(normm0)

fig, ax = plt.subplots(nrows = 2, figsize = (7,4), dpi = 300)

ax[0].imshow(clahe_image[:,100:800], cmap = 'gray')
ax[0].axis('off')

ax[1].imshow(im_result[:,100:], cmap = 'gray')
ax[1].axis('off')
ax[1].scatter(dets2x, dets2y, color = 'r', marker = 'x', s = 19)


fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\MaskAndDections.pdf', format='pdf')
plt.show()
# image = image[:,11:]

# bg = 0.02
# too_small = np.where(image < interp + bg)
# big_enough = np.where(image > interp + bg)
# binary_image = np.zeros(np.shape(image))
# binary_image[too_small] = 0
# binary_image[big_enough] = 1
# plt.imshow(binary_image, cmap = 'Greys_r')
# plt.show()


# strips = masking.image_strips(image, 10)
# mins = masking.get_strip_minimas(strips)
# min_image = masking.interpolate_minima_image(image, mins)
# masking.get_local_minima(image)


# masking.plot_mask(100)
# masking.plot_mask(25)
# masking.plot_mask(300)
# masking.plot_blocks(100)
# for i in [20,22]:
#     DIRECTORY = "%sCooldown 10/stationary/temperature sweep 1.0 kV 360 us/run%d" % (DRIVE, i)
#     print(DIRECTORY)
#     video_one_run()

# MAIN_DIRECTORY = "%sCooldown 10/stationary/temperature sweep 1.0 kV 360 us/" % DRIVE
# for run in os.listdir(MAIN_DIRECTORY):
#     if 'run' in run:
#         DIRECTORY = MAIN_DIRECTORY + run
#         print('\n', DIRECTORY)
#         filenames = sorted(dh.find_all_images(DIRECTORY + '/summed frames', '.png'))
#         directory = dht.Directory(DIRECTORY)
#         animator = Animator(directory, len(filenames)/2, 12)

# masking = msk.Mask(DIRECTORY, smoothing = 3, noise_factor = 0.01)    
# masking.plot_mask(50)

