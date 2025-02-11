# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:06:06 2023

@author: Matt
"""
import sys
# sys.path.append(r"Y:/Chris_Goodwin/Programs/Class and Function Library")
sys.path.append(r"C:/Users/Matt/The University of Manchester/Quantum fluids group - Documents/Simulations")
import os
import numpy as np
import matplotlib.pyplot as plt
import data_handling_library as dh
import time as time
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.interpolate import griddata
import data_handling_tools as dht

BACKGROUND_SMOOTHING = 7
BLOCK_SIZE = 40

class Mask:
    def __init__(self, directory, smoothing = 3, noise_factor = 0.01, transducer_cut = 790, num_summed = None):
        self.directory = directory
        self.smoothing = smoothing
        self.noise_factor = noise_factor
        self.transducer_cut = transducer_cut
        self.num_summed = num_summed
        if num_summed is not None:
            print("Generating %d summed images per image for mask\n" % num_summed)
        self.summed_frame_paths = dh.find_all_images(directory+ "/summed frames", '.png')
        self.make_directory_for_plots()
        self.check_image_numbering()
        self.frame_numbers = range(int(self.first_number), 
                                   int(self.first_number + len(self.summed_frame_paths) - 1))
        # self.noise = self.determine_time_averaged_noise()
        # self.load_background_noise()
        # self.load_background_from_csv()
        self.print_tau()
        # self.plot_noise_estimations()
        # self.plot_some_masks()
        
    def mask_image(self, original_image):
        if self.same_numbering:
            frame_number = original_image.frame_number
        else:
            frame_number = original_image.frame_number - self.first_number
        smoothed_image = self.prepare_image(frame_number)
        binary_image = self.create_mask(smoothed_image)
        masked_image = original_image.pixels * binary_image
        return masked_image
    
    def prepare_image(self, image_number):
        if self.num_summed is not None and self.same_numbering:
            image = self.generate_summed_image(image_number)
        else:
            image = self.load_summed_image(image_number)
        # image[:,self.transducer_cut:] = 0
        image = image[:,:self.transducer_cut]
        smoothed_image = medfilt2d(image, kernel_size = self.smoothing)
        return smoothed_image
    
        
    def load_summed_image(self, image_number):
        image_name = r"/summed frames/summed_image_%05d.png" % image_number
        image_path = self.directory + image_name
        image = dh.read_image(image_path)
        return image
    
    def generate_summed_image(self, image_number):
        
        count = 0
        for i in range(image_number - self.num_summed, image_number + self.num_summed + 1):
            image_name = r"\image_%05d.png" % i
            image_path = self.directory + image_name
            image = dh.read_image(image_path)
            if i == image_number - self.num_summed:
                summed_image = np.zeros(np.shape(image))
            summed_image += image
            count += 1
        summed_image = summed_image / count
        return summed_image
    
    def create_mask(self, image):
        # block_width = BLOCK_SIZE
        # blocks, minima = self.make_blocks(image, box_width = block_width)
        # interp = self.interpolate_minima(minima, image, block_width)
        # trimmed_image = image.copy()
        # cut = int(image.shape[1] - interp.shape[1])
        # trimmed_image = trimmed_image[:,cut:]
        # image = image[:,cut:]       
        
        trimmed_image = image
        
        interp = self.interpolate_minima_no_box(image)
        
        too_small = np.where(trimmed_image < interp + self.noise_factor)
        big_enough = np.where(trimmed_image > interp + self.noise_factor)
        binary_image = np.zeros(np.shape(trimmed_image))
        binary_image[too_small] = 0
        binary_image[big_enough] = 1
        
        return binary_image

    
    def image_strips(self, image, num_strips):
        self.num_strips = num_strips
        strips = []
        strip_size = int(np.shape(image)[1] / num_strips)
        self.strip_size = strip_size
        for i in range(num_strips):
            strip = image[:, i*strip_size:(i*strip_size+strip_size)]
            strips.append(strip)
        return strips
    
    def get_strip_minimas(self, strips):
        
        all_minima_indices = []
        for i in range(len(strips)):
            strip = strips[i]
            bool_minima = (strip == minimum_filter(strip, (3,3), mode = 'nearest'))
            index_minima = list(np.where(1 == bool_minima))
            index_minima[1] += i*self.strip_size
            all_minima_indices.append(index_minima)
        return all_minima_indices
    
    def interpolate_minima_image(self, image, indices):
        
        minima_image = np.full(image.shape, np.nan)
        for i in range(self.num_strips):
            strip_indices = indices[i]
            minima_image[tuple(strip_indices)] = image[tuple(strip_indices)]
            
        plt.figure()
        plt.imshow(minima_image)
        plt.show()
        
        # interpolate
        x,y = np.indices(minima_image.shape)
        interp = np.array(minima_image)
        interp[np.isnan(interp)] = griddata(
                (x[~np.isnan(minima_image)], y[~np.isnan(minima_image)]), 
                minima_image[~np.isnan(minima_image)],                    
                (x[np.isnan(minima_image)], y[np.isnan(minima_image)]),
                method = 'nearest')  
        
        plt.figure()
        plt.imshow(interp)
        plt.show()
        
        return minima_image
    
    def get_local_minima(self, summed_image):
        
        plt.figure()
        plt.imshow(summed_image)
        plt.show()
        plt.figure()
        plt.imshow(minimum_filter(summed_image, (3,3), mode = 'nearest'))
        plt.show()
        
        bool_minima = (summed_image == minimum_filter(summed_image, (3,3), mode = 'nearest'))
        plt.figure()
        plt.imshow(bool_minima)
        plt.show()
    
    def create_mask_new(self, image):
        strips = self.image_strips(image, 10)
        indices = self.get_strip_minimas(strips)
        minima_image = self.interpolate_minima_image(image, indices)
        image = image - minima_image
        
        too_small = np.where(image < 0.01)
        big_enough = np.where(image > 0.01)
        binary_image = np.zeros(np.shape(image))
        binary_image[too_small] = 0
        binary_image[big_enough] = 1
        return binary_image
    
    def create_mask_old_old(self, image):
        # not in use
        too_small = np.where(image < self.noise)
        big_enough = np.where(image > self.noise)
        binary_image = np.zeros(np.shape(image))
        binary_image[too_small] = 0
        binary_image[big_enough] = 1
        return binary_image
    
    
    def determine_time_averaged_noise(self):
        print("Determining time-averaged noise level::")
        standard_deviations, means = [], []
        start_time = time.time()
        for frame in self.frame_numbers:
            image = self.prepare_image(frame)
            standard_deviations.append(np.std(image))
            means.append(np.mean(image))
            current_time = time.time() - start_time
            print("\r frame number: %d | time/frame: %.3f s | total_elapsed: %.3f s" % (frame, current_time/(frame+1), current_time), end = '', flush=True)
        noise = np.mean(means) + self.noise_factor * np.mean(standard_deviations)
        self.standard_deviations = standard_deviations
        self.means = means
        return noise
    
    def create_mask_old(self, image):
        too_small = np.where(image < self.background)
        big_enough = np.where(image > self.background)
        binary_image = np.zeros(np.shape(image))
        binary_image[too_small] = 0
        binary_image[big_enough] = 1
        return binary_image
    
    def load_background_from_image(self):
        # not in use
        path_to_background = self.directory[:3] + "Cooldown 10/initial_low_temperature_tests/noise estimation/background QSW = 360 us.png"
        background = dh.read_image(path_to_background)
        background = background[:,:self.transducer_cut]
        background = gaussian_filter(background, 
                                     BACKGROUND_SMOOTHING,
                                     mode = 'nearest')
        self.background = background * self.noise_factor
        
    def load_background_from_csv(self):
        path_to_background = self.directory[:3] + "Cooldown 10/initial_low_temperature_tests/noise estimation/background QSW = 360 us.csv"
        path_to_background = r"C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 10\initial_low_temperature_tests\noise estimation\QSW = 360 us\_total_summed_image.csv"
        raw_background = np.loadtxt(path_to_background, delimiter = ",")
        smoothed_background = gaussian_filter(raw_background, 
                                              BACKGROUND_SMOOTHING,
                                              mode = 'nearest')
        trimmed_background = smoothed_background[:,:self.transducer_cut]
        self.background = trimmed_background * self.noise_factor
    
    def trim_image_for_blocks(self, image, box_width):
        shape = image.shape
        n_rows = int(shape[0] / box_width)
        n_cols = int(shape[1] / box_width)
        row_trim = shape[0] % n_rows
        col_trim = shape[1] % n_cols
        image = image[row_trim:,col_trim:]
        return image, n_rows, n_cols
    
    def make_blocks(self, image, box_width = 40):
        image, n_rows, n_cols = self.trim_image_for_blocks(image, box_width)
        blocks = np.empty((n_rows, n_cols), dtype=object)
        minima = np.empty((n_rows, n_cols))
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                blocks[i,j] = image[i * box_width : (i + 1) * box_width, 
                                    j * box_width : (j + 1) * box_width]
                minima[i,j] = self.find_block_minima(blocks[i,j])
        return blocks, minima
    
    def plot_blocks(self, image, blocks, normalise_cmap = False):
        n_rows, n_cols = blocks.shape
        fig, axis = plt.subplots(n_rows, n_cols, figsize = (n_cols, n_rows))
        
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                if normalise_cmap:
                    axis[i,j].imshow(blocks[i,j], 
                                      vmin = np.min(image), 
                                      vmax = np.max(image),
                                      cmap = 'magma')
                else:
                    axis[i,j].imshow(blocks[i,j], cmap = 'magma')
                axis[i,j].xaxis.set_tick_params(labelbottom=False)
                axis[i,j].yaxis.set_tick_params(labelleft=False)
                axis[i,j].set_xticks([])
                axis[i,j].set_yticks([])
        fig.tight_layout()
        plt.show()
        return None
    
    def find_block_minima(self, block):
        minima_truth = (block == minimum_filter(block, 3, mode = 'reflect'))
        minima_index = np.where(1 == minima_truth)
        minima = block[minima_index]
        return np.mean(minima)
    
    def plot_minima(self, minima):
        plt.figure()
        plt.title("Block minima")
        plt.imshow(minima, cmap = 'magma')
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        return 
    
    def interpolate_minima(self, minima, image, box_width):
        image_cut, n_rows, n_cols = self.trim_image_for_blocks(image, box_width)
        block_centre = box_width // 2
        dummy_matrix = np.full(image_cut.shape, np.nan)
        for i in range(n_rows):
            for j in range(n_cols):
                row_index = (i * box_width) + block_centre
                col_index = (j * box_width) + block_centre
                dummy_matrix[row_index, col_index] = minima[i,j]
        
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
        
        return interp
    
    def plot_interpolated_minima(self, interp):
        plt.figure()
        plt.title("Cubic interpolated minima")
        plt.imshow(interp, cmap = 'magma')
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        return 
    
    def interpolate_minima_no_box(self,image):
        minima_truth = (image == minimum_filter(image, BACKGROUND_SMOOTHING, mode = 'reflect'))
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
        
        
        # plt.imshow(minima_image)
        # plt.title('mins')
        # plt.show()
        
        # plt.imshow(interp)
        # plt.title('interpolated')
        # plt.show()
        return interp
        
    
    def plot_interpolated_minima_with_summed_image(self, interp, image):
        fig, axis = plt.subplots(2,1)
        ax = axis[0]
        ax.set_title("Summed image")
        im = ax.imshow(image, vmin = np.min(image), vmax = np.max(image), cmap = 'magma')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = axis[1]
        ax.set_title("Interpolated minima, max color value = %.3f" % np.max(interp))
        ax.imshow(interp, vmin = np.min(image), vmax = np.max(image), cmap = 'magma')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()
        return None
    
    def remove_ticks(self, ax):
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        return None
        
    def print_tau(self):
        if self.num_summed is not None:
            window = 2 * self.num_summed + 1
            print("Using temporary summed frames with window size %d frames" % window)
    
    def check_image_numbering(self):
        first_image_path = self.summed_frame_paths[0]
        # self.first_number = int(first_image_path[-5])
        self.first_number = dht.extract_number_from_string(
            os.path.basename(first_image_path))
        if self.first_number == 0: 
            self.same_numbering = False
        else:
            self.same_numbering = True
            
    def check_if_summed_frame_exists(self, image_number):
        if isinstance(image_number, bool):
            return False
        if self.same_numbering:
            image_number = image_number
        else:
            image_number = image_number - self.first_number
        if image_number in self.frame_numbers:
            return True
        else:
            return False       
            
    def make_directory_for_plots(self):
        dir_name = '/summed frames/mask_plots'
        dir_path = self.directory + dir_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
     
    def plot_noise_estimations(self):
        fig = plt.figure()
        plt.plot(self.frame_numbers, self.means, label = r'$\mu$')
        plt.plot(self.frame_numbers, self.standard_deviations, label = r'$\sigma$')
        plt.plot(self.frame_numbers, np.array(self.means) + np.array(self.standard_deviations), label = r'$\mu+\sigma$')
        plt.plot(self.frame_numbers, np.array(self.means) + self.noise_factor * np.array(self.standard_deviations), label = r'$\mu +$%.1f$\sigma$' % self.noise_factor )
        plt.plot(self.frame_numbers, self.noise * np.ones(len(self.frame_numbers)), label = r'Noise cut')
        plt.xlabel("Frame number")
        plt.ylabel("Pixel Statistic")
        plt.minorticks_on()
        plt.grid(which = 'major', linestyle = '--')
        plt.grid(which = 'minor', linestyle = ':')
        plt.xlim(left = 0)
        plt.ylim(bottom = 0)
        plt.legend()
        plt.savefig(self.directory + '/summed frames/mask_plots/_pixel_stats.png', dpi = 500)
        plt.close(fig)
        # plt.show()
        return None
    
    def plot_mask(self, image_number):
        image = self.prepare_image(image_number)
        mask = self.create_mask(image)
        fig, axis = plt.subplots(2,1, sharex = True, sharey = True, figsize = (10,4))
        plt.subplots_adjust(wspace = 0, hspace = 0.1)
        axis[0].imshow(dh.clahe_image(image), cmap = 'Greys_r')
        axis[1].imshow(mask, cmap = 'Greys_r')
        self.remove_ticks(axis[0])
        self.remove_ticks(axis[1])
        
        plt.savefig(self.directory + '/summed frames/mask_plots/_mask_%05d.png' % image_number, dpi = 500)
        plt.close(fig)
        # plt.show()
        return None
    
    def plot_some_masks(self):
        frame_numbers = [20,25,30,35,40, 100, 110, 120, 150, 200, 500, 800]
        for frame in frame_numbers:
            self.plot_mask(frame)
            self.plot_mask_with_image(frame)
            
    def plot_mask_with_image(self, image_number, hits = None):
        # check numbering
        summed_image = self.prepare_image(image_number) 
        mask = self.create_mask(summed_image)

        image_path = self.directory + "/image_%05d.png" % image_number
        image = dh.read_image(image_path)
        image = dh.clahe_image(image)
        
        mask_extended = np.zeros(image.shape)
        mask_extended[:mask.shape[0], :mask.shape[1]] = mask
        
        fig = plt.figure()
        plt.imshow(image * mask_extended, cmap = 'gray', interpolation = 'none')
        if hits is not None:
            plt.plot(hits["y"], hits["x"], "rx", markersize = 4)
        plt.title('Frame: %d' % image_number)
        if hits is not None:
            plt.savefig(self.directory + '/summed frames/mask_plots/_detections_image_mask_%05d.png' % image_number, dpi = 500)
        else:
            plt.savefig(self.directory + '/summed frames/mask_plots/_image_mask_%05d.png' % image_number, dpi = 500)
        # plt.close(fig)
        plt.show()
        



def smooth_minima(array):
    index, values = dh.get_local_minima(array)
    return np.mean(values)