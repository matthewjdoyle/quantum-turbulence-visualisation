# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:08:06 2022

@author: Helium
"""

import matplotlib.image as mpimg
import cv2
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import data_handling_tools as dht
import mask as msk

import sys
sys.path.append(r"S:/Chris_Goodwin/Programs/Class and Function Library")
import data_handling_library as dh
from matplotlib.lines import Line2D

# ROOT = r"X:/"
# ROOT = r'C:/Users/Matt/Documents/Vortex Visualisation/Local Working 03-07-2023 (Masking)/Local Working 03-07-2023 (Masking)/'
ROOT = r"C:\Users/Matt/The University of Manchester/Quantum fluids group - Documents/Vortex visualisation files/Main data storage/"
FOLDER = r"Cooldown 11/qsw 350us/T = 150 mK 0.5 kV/run2"

DIRECTORY = os.path.join(ROOT, FOLDER)

KERNEL_SIZE = 8
REJECTION_PERCENTILE = 0
RESCALE_IMAGES = False
PARALLEL = True
MINIMUM_PIXEL_VALUE = 0.0018
NUMBER_OF_CLUSTERS = 2
CLUSTERS_TO_REMOVE = [0]
M_2_CUT = 40
REMOVE_TRANSDUCER = True
TRANSDUCER_CUT = 800
MASK_SMOOTHING = 5
NOISE_FACTOR = 0.0018
APPLY_MASK = True


class Image:
    def __init__(self, path, min_rescale=None, max_rescale=None,
                 minimum_pixel=None):
        self.pixels = None
        self.filtered_pixels = None
        self.dilated_pixels = None
        self.path = path
        self.min_rescale = min_rescale
        self.max_rescale = max_rescale
        self.minimum_pixel = minimum_pixel
        self.find_frame_number()

    def read_image(self):
        self.pixels = mpimg.imread(self.path)
        self.remove_transducer()
        self.normalise_pixel_values()
        self.remove_low_pixel_values()
        
    
    def remove_transducer(self):
        if REMOVE_TRANSDUCER:
            width = self.pixels.shape[0]
            self.pixels = self.pixels[0:width, 0:TRANSDUCER_CUT]

    def clahe_image(self, clahe_clip=5, clahe_tile=8):
        clahe_object = cv2.createCLAHE(clipLimit=clahe_clip,
                                       tileGridSize=(clahe_tile,
                                                     clahe_tile))
        self.pixels = clahe_object.apply(self.pixels)

    def find_frame_number(self):
        filename = os.path.basename(self.path)[:-4]
        self.frame_number = dht.extract_number_from_string(filename)

    def close_image(self):
        self.pixels = None
        self.filtered_pixels = None
        self.dilated_pixels = None

    def normalise_pixel_values(self):
        if self.min_rescale is not None and self.max_rescale is not None:
            self.pixels = ((self.pixels - self.min_rescale)
                           / (self.max_rescale - self.min_rescale))

    def remove_low_pixel_values(self):
        if self.minimum_pixel is not None:
            self.pixels[self.pixels < self.minimum_pixel] = 0

    def histogram_equalisation(self, bin_limit=40, kernel_size=8):
        clahe_object = cv2.createCLAHE(
            clipLimit=bin_limit, tileGridSize=(kernel_size, kernel_size)
        )
        return clahe_object.apply(self.pixels)

    def find_extremes(self):
        minimum = self.pixels.min()
        maximum = self.pixels.max()
        return minimum, maximum

    def filter_pixels(self, kernel):
        self.filtered_pixels = cv2.filter2D(self.pixels, -1, kernel)
        # self.filtered_pixels = signal.fftconvolve(self.pixels, kernel, mode='same')

    def dilate_pixels(self, kernel, iterations=1):
        
        pad_factor = 2
        new_size = [int(self.filtered_pixels.shape[0] + pad_factor*2*KERNEL_SIZE),
                    int(self.filtered_pixels.shape[1] + pad_factor*2*KERNEL_SIZE)]
        padded_image = np.ones((new_size[0], new_size[1]))
        padded_image[pad_factor*KERNEL_SIZE:-pad_factor*KERNEL_SIZE, pad_factor*KERNEL_SIZE:-pad_factor*KERNEL_SIZE] = self.filtered_pixels
        # self.dilated_pixels = cv2.dilate(self.filtered_pixels, kernel,
        #                                  iterations=iterations)
        # print(padded_image.shape)
        dilated_pixels = cv2.dilate(padded_image, kernel,
                                    iterations=iterations)
        
        self.dilated_pixels = dilated_pixels[pad_factor*KERNEL_SIZE:-pad_factor*KERNEL_SIZE, pad_factor*KERNEL_SIZE:-pad_factor*KERNEL_SIZE]
        # print(self.dilated_pixels.shape)
        
        

    def plot_histogram(self):
        self.read_image()
        rescaled_image = 255 * self.pixels
        vals = rescaled_image.flatten()
        counts, bins = np.histogram(vals, range(257))
        plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        plt.xlim([-0.5, 255.5])
        plt.show()
        self.close_image()

    def plot_side_intensity(self, minimum_pixel=None):
        self.minimum_pixel = minimum_pixel
        self.read_image()
        maxima = self.pixels.max(axis=0)
        plt.plot(maxima, label=f"frame={self.frame_number}")
        self.close_image()

class ParticleFinder:
    def __init__(self, directory, kernel_size, rejection_percentile):
        self.kernel_size = kernel_size
        self.rejection_percentile = rejection_percentile
        if APPLY_MASK:
            self.masking = msk.Mask(directory, smoothing=MASK_SMOOTHING, 
                                    noise_factor=NOISE_FACTOR, transducer_cut=TRANSDUCER_CUT, num_summed = 1)
        self.create_masks()

    def build_filter_kernel(self):
        w = self.kernel_size
        x = np.linspace(-w, w, 2 * w + 1)
        y = x.copy()

        X, Y = np.meshgrid(x, y)
        B = np.sum(np.exp(-(x ** 2) / 4)) ** 2
        K_0 = 1 / B * np.sum(np.exp(-(x ** 2) / 2)) ** 2 - B / (2 * w + 1) ** 2

        kernel_blur = 1 / B / K_0 * np.exp(-(X ** 2 + Y ** 2) / 4)
        kernel_low_pass = -1 / K_0 / (2 * w + 1) ** 2
        kernel = kernel_blur + kernel_low_pass
        return kernel

    def apply_filter_kernel(self, image):
        image.filter_pixels(self.filter_kernel)
        image.filtered_pixels[image.filtered_pixels < 0] = 0

    def dilate_image(self, image):
        image.dilate_pixels(self.dilation_kernel)

    def find_local_maxima(self, image):
        maxima_indicies = image.filtered_pixels == image.dilated_pixels
        maxima = np.zeros(image.pixels.shape)
        maxima[maxima_indicies] = image.filtered_pixels[maxima_indicies]

        if maxima.any():
            threshold = np.percentile(maxima[maxima != 0], self.rejection_percentile)
            maxima[maxima <= threshold] = 0
            coordinates = np.array(np.nonzero(maxima != 0)).T.astype(int)

            self.coordinates = pd.DataFrame({"x": coordinates[:, 0],
                                                  "y": coordinates[:, 1]})
        else:
            self.coordinates = pd.DataFrame({"x": pd.Series(dtype=int),
                                                  "y": pd.Series(dtype=int)})

    def find_intial_coordinates(self, image):
        self.apply_filter_kernel(image)
        self.dilate_image(image)
        # now mask
        if APPLY_MASK:
            image.pixels = self.masking.mask_image(image)
        self.find_local_maxima(image)


    def create_circular_mask(self, diameter):
        center = (int(diameter / 2), int(diameter / 2))
        radius = min(center[0], center[1], diameter - center[0], diameter - center[1])
        Y, X = np.ogrid[:diameter, :diameter]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= radius
        return mask

    def create_m_0_mask(self):
        diameter = self.kernel_size * 2 + 1
        self.m_0_mask = self.create_circular_mask(diameter)

    def create_m_2_mask(self):
        self.m_2_mask = self.x_mask** 2 + self.y_mask ** 2

    def calculate_particle_moment(self, particle_area, mask):
        return np.sum(particle_area * mask)

    def create_offset_masks(self,):
        diameter = self.kernel_size * 2 + 1
        circular_mask = self.create_circular_mask(diameter)
        size = np.linspace(-self.kernel_size, self.kernel_size,
                           2 * self.kernel_size + 1)
        y_mesh, x_mesh = np.meshgrid(size, size)
        self.x_mask = x_mesh * circular_mask
        self.y_mask = y_mesh * circular_mask

    def create_masks(self):
        self.filter_kernel = self.build_filter_kernel()
        self.dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * self.kernel_size + 1, 2 * self.kernel_size + 1)
            )
        self.create_m_0_mask()
        self.create_offset_masks()
        self.create_m_2_mask()

    def refine_coordinate(self, image, x, y):
        x_offset, y_offset = 1, 1
        counter = 0
        k = self.kernel_size
        while abs(x_offset) >= 0.5 and abs(y_offset) >= 0.5:
            try:
                particle = image.pixels[x - k : x + k + 1, y - k : y + k + 1]

                m_0 = self.calculate_particle_moment(particle, self.m_0_mask)
                if m_0 == 0:
                    return None

                m_2 = self.calculate_particle_moment(particle, self.m_2_mask) / m_0

                if m_2 == 0:
                    return None
                x_offset = self.calculate_particle_moment(particle, self.x_mask) / m_0
                y_offset = self.calculate_particle_moment(particle, self.y_mask) / m_0
            except ValueError:
                return None

            if abs(x_offset) >= 0.5:
                x += int(1 * np.sign(x_offset))

            if abs(y_offset) >= 0.5:
                y += int(1 * np.sign(y_offset))

            if counter > self.kernel_size / 2:
                return None

            counter += 1

        new_x = x + x_offset
        new_y = y + y_offset
        output = pd.DataFrame({"x": [new_x], "y": [new_y], "m_0": [m_0], "m_2": [m_2]})

        return output

    def remove_duplicates(self):
        remove_index = pd.Series(np.zeros(len(self.coordinates)), dtype=bool)
        for i in range(len(self.coordinates)):
            coord = self.coordinates.iloc[i]
            distance = ((coord.x - self.coordinates.x) ** 2
                        + (coord.y - self.coordinates.y) ** 2) ** 0.5
            distance_index = distance < self.kernel_size
            distance_index.iloc[i] = False
            if distance_index.any():
                if coord.m_0 < self.coordinates.loc[distance_index].m_0.min():
                    remove_index[i] = True

        self.coordinates = self.coordinates.loc[~remove_index]

    def refine_coordinates(self, image):
        coordinates = pd.DataFrame({"x": pd.Series(dtype=float),
                                    "y": pd.Series(dtype=float),
                                    "m_0": pd.Series(dtype=float),
                                    "m_2": pd.Series(dtype=float)})

        for index, (x, y) in enumerate(zip(self.coordinates["x"], self.coordinates["y"])):
            refined = self.refine_coordinate(image, x, y)
            if refined is not None:
                coordinates = pd.concat((coordinates, refined), ignore_index=True)

        self.coordinates = coordinates

    def find_coordinates(self, image):
        image.read_image()
        if image.pixels.any() and (len(np.shape(image.pixels)) == 2):
            # print('\n', np.shape(image.pixels), image.frame_number, self.masking.check_if_summed_frame_exists(image.frame_number))
            self.find_intial_coordinates(image)
            self.refine_coordinates(image)
            # self.remove_duplicates()
        else:
            self.coordinates = pd.DataFrame({"x": pd.Series(dtype=float),
                                    "y": pd.Series(dtype=float),
                                    "m_0": pd.Series(dtype=float),
                                    "m_2": pd.Series(dtype=float)})
        # image.close_image()
        return self.coordinates


class CoordinateExtractor:
    def __init__(self, directory, find_extremes=False, minimum_pixel_value=None):
        self.directory = directory
        self.find_extremes = find_extremes
        self.minimum_pixel_value = minimum_pixel_value
        self.image_paths = dht.find_all_images(self.directory)
        self.find_global_extremes(find_extremes)
        self.load_images()
        self.get_max_image_number()

    def find_global_extremes(self, find_extremes):
        if not self.find_extremes:
            self.min_value = None
            self.max_value = None
        else:
            minimum, maximum = 1e10, 0
            for path in self.image_paths:
                image = Image(path)
                image.read_image()
                image_min, image_max = image.find_extremes()

                if image_min < minimum:
                    minimum = image_min
                if image_max > maximum:
                    maximum = image_max

            self.min_value = minimum
            self.max_value = maximum

    def load_images(self):
        self.images = []
        for path in self.image_paths[3:]:
            image = Image(path,
                          min_rescale=self.min_value,
                          max_rescale=self.max_value,
                          minimum_pixel=self.minimum_pixel_value)
            self.images.append(image)
            
    def get_max_image_number(self):
        summed_image_folder = os.path.join(self.directory, '')
        files = os.listdir(summed_image_folder)
        files = [f for f in files if '.png' in f]
        # self.starting_summed_frame = dht.extract_number_from_string(files[0])
        self.starting_summed_frame = 0
        self.number_of_summed_images = len(files)
        self.max_frame_number = (self.number_of_summed_images - 6)
        

    def setup_finder(self, kernel_size, rejection_percentile):
        self.finder = ParticleFinder(self.directory, 
                                     kernel_size, rejection_percentile)

    def find_image_coordinate(self, image, frame_number):
        im_coords = self.finder.find_coordinates(image)
        im_coords["frame"] = image.frame_number
        return im_coords


    def find_coordinates_serial(self):
        for frame, image in enumerate(self.images):
            if frame < self.max_frame_number:
                im_coords = self.find_image_coordinate(image, frame)
                self.coordinates = pd.concat((self.coordinates, im_coords),
                                             ignore_index=True)

    def find_coordinates_parallel(self):
        mp.freeze_support()
        pool = mp.Pool(mp.cpu_count())
        out = [
            pool.apply_async(self.find_image_coordinate, args=(image, frame))
            for frame, image in enumerate(self.images) if frame < self.max_frame_number
        ]
        results = [p.get() for p in out]
        pool.close()
        pool.join()
        for result in results:
            self.coordinates = pd.concat((self.coordinates, result),
                                          ignore_index=True)
    


    def find_coordinates(self, parallel=False):
        self.coordinates = pd.DataFrame({"x": pd.Series(dtype=float),
                                         "y": pd.Series(dtype=float),
                                         "m_0": pd.Series(dtype=float),
                                         "m_2": pd.Series(dtype=float),
                                         "frame": pd.Series(dtype="int")})
        if parallel:
            self.find_coordinates_parallel()
        else:
            self.find_coordinates_serial()
        return self.coordinates

    def save_coordinates(self, directory):
        filepath = os.path.join(directory, "_coordinates.csv")
        self.coordinates.to_csv(filepath)


class CoordinateFilterer:
    def __init__(self, extractor):
        self.extractor = extractor
        self.coordinates = extractor.coordinates.copy()
        self.unfiltered_coordinates = self.coordinates.copy()
        self.images = extractor.images.copy()
        self.sort_images()

    def filter_coordinates(self):
        print("Starting filtering, detection count = %d" % len(self.coordinates['m_0']))
        self.remove_high_m2_particles(m_2_max = M_2_CUT)
        print("Removed high m_2 detections, detection count = %d" % len(self.coordinates['m_0']))
        self.remove_cluster("m_0", NUMBER_OF_CLUSTERS, CLUSTERS_TO_REMOVE)
        print("%d clusters removed out of total %d, detection count = %d" % (len(CLUSTERS_TO_REMOVE), NUMBER_OF_CLUSTERS, len(self.coordinates['m_0'])))
        self.coordinates.reset_index(inplace=True, drop=True)
        self.coordinates["detection_number"] = self.coordinates.index
        # self.coordinates.drop("index", inplace=True)

    def show_filtering(self, image_number):
        fig = plt.figure(figsize=(18, 12))
        plt.subplot(211)
        self.label_image(image_number, clahe=False)
        self.filter_coordinates()
        # plt.ylim(900, 300)
        # plt.xlim(400, 950)
        plt.subplot(212)
        self.label_image(image_number, clahe=False)
        # plt.ylim(900, 300)
        # plt.xlim(400, 950)
        plt.savefig(os.path.join(self.extractor.directory, '_coords filtering.png'))
        plt.close(fig)

    def plot_moments(self, frame=None):
        # plt.figure()
        if frame is None:
            plt.plot(self.coordinates["m_0"], self.coordinates["m_2"], '.', markersize=2)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel("$m_0$")
        plt.ylabel("$m_2$")

    def sort_images(self):
        images_sorted = {}
        for image in self.images:
            images_sorted[image.frame_number] = image

        self.sorted_images = images_sorted

    def label_image(self, frame_number, clahe=False):

        image = self.sorted_images[frame_number]
        image.minimum_pixel = None
        image.read_image()
        if clahe:
            image.pixels = (image.pixels * 255).astype("uint8")
            image.clahe_image()
            # image.pixels = dh.clahe_image(image.pixels)

        image_coordinates = self.coordinates[(self.coordinates["frame"] == frame_number)]
        plt.imshow(image.pixels, cmap="gray")
        plt.plot(image_coordinates["y"], image_coordinates["x"], "rx")
        image.close_image()

    def perform_clustering(self, number_clusters):
        cluster = GaussianMixture(n_components=number_clusters)
        self.cluster_fit = cluster.fit_predict(self.coordinates[["m_0", "m_2"]])

    def find_average_param(self, parameter):
        average = []
        groups = list(set(self.cluster_fit))
        for group in groups:
            indicies = list(self.cluster_fit == group)
            values = self.coordinates[parameter]
            average.append(np.mean(values.iloc[indicies]))

        index = np.argsort(average)
        groups = np.array(groups)[index]
        average = np.array(average)[index]
        return average, groups

    def remove_points(self, groups_to_remove):
        for group in groups_to_remove:
            indicies = list(self.cluster_fit == group)
            inverted_indicies = np.invert(indicies)
            self.coordinates = self.coordinates.iloc[inverted_indicies]
            self.cluster_fit = np.delete(self.cluster_fit, indicies)

    def remove_high_m2_particles(self, m_2_max=55):
        self.coordinates = self.coordinates.loc[self.coordinates.m_2<m_2_max]

    def remove_cluster(self, parameter, number_clusters, remove_index):
        self.perform_clustering(number_clusters)
        average, groups = self.find_average_param(parameter)
        self.remove_points(groups[remove_index])

    def plot_cluster(self, number_clusters):
        plt.figure()
        self.perform_clustering(number_clusters)
        unique_groups = set(self.cluster_fit)
        for group in unique_groups:
            indicies = self.cluster_fit == group
            plt.plot(self.coordinates["m_0"].iloc[indicies],
                     self.coordinates["m_2"].iloc[indicies],
                     '.', label=group, markersize=2)
        plt.legend()
        plt.xlabel("$m_0$")
        plt.ylabel("$m_2$")
        
    def plot_cluster2(self, number_clusters, ax, color = None):
        self.perform_clustering(number_clusters)
        unique_groups = set(self.cluster_fit)
        for group in unique_groups:
            indicies = self.cluster_fit == group
            if color:
                ax.plot(self.coordinates["m_0"].iloc[indicies],
                         self.coordinates["m_2"].iloc[indicies],
                         '.', label=group, markersize=2,
                         color = color)
            else:
                ax.plot(self.coordinates["m_0"].iloc[indicies],
                         self.coordinates["m_2"].iloc[indicies],
                         '.', label=group, markersize=2, rasterized=True)


        
    def image_with_detections(self, frame_number):
        fig, axis = plt.subplots(2,1, sharex=True, sharey=True)
        
        
        image_path = self.extractor.directory + '/image_%05d.png' % frame_number
        image = dh.read_image(image_path)
        
        # image.pixels = dh.rgb_image_to_grayscale(image.pixels)
        image = dh.clahe_image(image)
        # image = np.rot90(image, k=-1)

        image_coordinates = self.coordinates[(self.coordinates["frame"] == frame_number)]
        axis[1].imshow(image, cmap="gray")
        axis[1].plot(image_coordinates["y"], image_coordinates["x"], "rx")
        axis[1].set_title("Filtered detections")
        
        image_coordinates = self.unfiltered_coordinates[(self.unfiltered_coordinates["frame"] == frame_number)]
        axis[0].imshow(image, cmap="gray")
        axis[0].plot(image_coordinates["y"], image_coordinates["x"], "rx")
        axis[0].set_title("Raw detections")
        
        plt.show()
        
    def mask_and_image_with_detections(self, frame_number):
        
        image_coordinates = self.coordinates[(self.coordinates["frame"] == frame_number)]
        masking = msk.Mask(self.extractor.directory, smoothing=MASK_SMOOTHING, 
                                    noise_factor=NOISE_FACTOR, transducer_cut=TRANSDUCER_CUT)
        masking.plot_mask_and_image_with_coordinates(frame_number, image_coordinates)
        
        return None
    
    def mask_and_image_with_unfiltered_detections(self, frame_number):
        
        image_coordinates = self.coordinates[(self.coordinates["frame"] == frame_number)]
        unfiltered_coords = self.unfiltered_coordinates[(self.unfiltered_coordinates["frame"] == frame_number)]
        masking = msk.Mask(self.extractor.directory, smoothing=MASK_SMOOTHING, 
                                    noise_factor=NOISE_FACTOR, transducer_cut=TRANSDUCER_CUT)
        masking.plot_mask_and_image_with_unfiltered_coordinates(frame_number, image_coordinates, unfiltered_coords)
        
        return None
        


    def save_coordinates(self, directory):
        if APPLY_MASK:
            filepath = os.path.join(directory, "_masked_coordinates.csv")
            # for i in [80, 90, 115, 120, 125, 130, 150]:
            #     self.extractor.finder.masking.plot_mask_with_image(i, self.coordinates[(self.coordinates["frame"] == i)])
        else:
            filepath = os.path.join(directory, "_coordinates.csv")
        self.coordinates.to_csv(filepath, index=False)


    

# def find_and_save_coordinates(directory, kernel_size=KERNEL_SIZE,
#                               minimum_pixel_value=MINIMUM_PIXEL_VALUE,
#                               rejection_percentile=REJECTION_PERCENTILE,
#                               parallel=True):
#     extractor = CoordinateExtractor(directory, find_extremes=RESCALE_IMAGES,
#                                    minimum_pixel_value=minimum_pixel_value)
#     extractor.setup_finder(kernel_size, rejection_percentile)
#     extractor.find_coordinates(parallel=parallel)
#     filterer = CoordinateFilterer(extractor)
#     filterer.filter_coordinates()
#     filterer.save_coordinates(directory)

def run_coordinate_finder(directory=DIRECTORY):
    start = time.time()
    extractor = CoordinateExtractor(directory, find_extremes=RESCALE_IMAGES,
                                   minimum_pixel_value=MINIMUM_PIXEL_VALUE)
    extractor.setup_finder(KERNEL_SIZE, REJECTION_PERCENTILE)
    coordinates = extractor.find_coordinates(parallel=PARALLEL)
    print(f"Coordinates found in {time.time() - start:.2f} s")

    filterer = CoordinateFilterer(extractor)
    # filterer.plot_cluster(NUMBER_OF_CLUSTERS)
    fig, ax = plt.subplots(ncols = 3, figsize = (9,4), dpi = 300,
                           sharey = True,gridspec_kw={'wspace' : 0})
    
    filterer.plot_cluster2(NUMBER_OF_CLUSTERS, ax[0])
    # filterer.show_filtering(20)
    filterer.filter_coordinates()
    
    filterer.plot_cluster2(NUMBER_OF_CLUSTERS , ax[1])

    filterer.filter_coordinates()
    # filterer.plot_cluster(NUMBER_OF_CLUSTERS)
    filterer.plot_cluster2(NUMBER_OF_CLUSTERS, ax[2], color = 'cyan')
    
    
    ax[0].hlines(40, xmin = 0, xmax = 8, color = 'k', ls = '--', lw = 0.8)
    ax[1].hlines(40, xmin = 0, xmax = 8, color = 'k', ls = '--', lw = 0.8)
    ax[2].hlines(40, xmin = 0, xmax = 8, color = 'k', ls = '--', lw = 0.8)
    
    ax[0].set_xlabel(r'$m_0$', fontsize = 'x-large')
    ax[0].set_ylabel(r'$m_2$', fontsize = 'x-large')
    ax[1].set_xlabel(r'$m_0$', fontsize = 'x-large')
    # ax[1].set_ylabel(r'$m_2$')
    ax[2].set_xlabel(r'$m_0$', fontsize = 'x-large')
    # ax[2].set_ylabel(r'$m_2$')
    
    ax[0].set_xlim(0, 7.9)
    ax[0].set_ylim(0, 70)
    
    ax[1].set_xlim(0, 7.9)
    # ax[1].set_ylim(0, 70)
    
    ax[2].set_xlim(0, 8)
    # ax[2].set_ylim(0, 70)
    
    legend = ax[0].legend(title = 'Cluster ID', fancybox = False, edgecolor = 'k', framealpha=1, markerscale = 4)
    legend.get_frame().set_linewidth(0.8)
    legend = ax[1].legend(title = 'Cluster ID', fancybox = False, edgecolor = 'k', framealpha=1, markerscale = 4)
    legend.get_frame().set_linewidth(0.8)
    
    legend = ax[2].legend([Line2D([], [], color='cyan', marker='.', linestyle='None')], ['Filtered'],
                          fancybox = False, edgecolor = 'k', framealpha=1, markerscale = 2)
    legend.get_frame().set_linewidth(0.8)
    
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\GaussianCLuster.pdf', format='pdf')
    
    plt.show()
    
    # filterer.plot_moments()
    # filterer.filter_coordinates()
    # filterer.image_with_detections(20)
    # filterer.image_with_detections(21)
    # filterer.image_with_detections(22)
    # filterer.image_with_detections(100)
    # for i in range(10, 200, 10):
    #     filterer.mask_and_image_with_detections(i)
    # for i in range(10, 200, 10):
    #     filterer.mask_and_image_with_unfiltered_detections(i)
    # for i in range(200, 900, 50):
    #     filterer.mask_and_image_with_detections(i)
    
    
    
    
    # filterer.save_coordinates(directory)
    return coordinates

if __name__ == "__main__":
        
    # start = time.time()
    # extractor = CoordinateExtractor(DIRECTORY, find_extremes=RESCALE_IMAGES,
    #                                minimum_pixel_value=MINIMUM_PIXEL_VALUE)
    # extractor.setup_finder(KERNEL_SIZE, REJECTION_PERCENTILE)
    # coordinates = extractor.find_coordinates(parallel=PARALLEL)
    # print(f"{time.time() - start:.2f} s")


    # # filterer.plot_cluster(NUMBER_OF_CLUSTERS)
    # filterer = CoordinateFilterer(extractor)
    # filterer.show_filtering(14)
    # # filterer.plot_moments()
    # filterer.filter_coordinates()



    # filterer.save_coordinates(DIRECTORY)
    
    
    coordinates = run_coordinate_finder(DIRECTORY)