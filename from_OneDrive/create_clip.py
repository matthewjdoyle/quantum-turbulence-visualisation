# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.image as mpimg
from scipy.signal import medfilt2d
# import collect_data as cd


# CHANGE TO DESIRED RUN FOLDER
MAIN_DIRECTORY = r"D:/"
SUB_DIRECTORY = r"PhD Storage\Visualization Experiment\Vortex line at 1 K"
DIRECTORY = os.path.join(MAIN_DIRECTORY, SUB_DIRECTORY)


FRAME_RATE = 15

# Standard settings
INVERT_CONTRAST = False
ROTATE = True

# Leave CLAHE flase to apply dilation
CLAHE = True
CLAHE_CLIP_LIMIT = 4
CLAHE_TILE_SIZE = 10

TIMESTAMP = True
# SHOW_BRIGHTNESS = True
SOURCE_FRAME_RATE = 200 # read this in from paramters instead

# Filters / time-cutting
FRAME_START = 1
FRAME_END = 100
TIME_SMOOTHING = 1 # number of frames either side of target frame to sum (0 default)
SPACE_SMOOTHING = 0  # kernel size for median filter (0 default, no smoothing)

# broken
DILATION_KERNEL = 0 # kernel size for dilation (0 default, no dilation)


FILENAME = "_clip_video"

class VideoWriter:
    def __init__(self, directory, frame_rate, invert_contrast=False,
                 clahe=False, clahe_clip=None, clahe_tile=None, rotate=ROTATE,
                 timestamp=TIMESTAMP, filename=FILENAME, frame_start = None, 
                 frame_end = None, time_smoothing = None, 
                 space_smoothing = None, dilation_kernel = None):
        self.directory = directory
        self.invert_contrast = invert_contrast
        self.if_clahe = clahe
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.if_timestamp = timestamp
        self.rotate = rotate
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.time_smoothing = time_smoothing
        self.space_smoothing = space_smoothing
        self.dilation_kernel = dilation_kernel
        
        
        # self.total_luminosity()
        
        self.find_images(self.directory, self.frame_start, self.frame_end)
        
        print("Animating ", len(self.files), " frames...")
        
        self.calculate_times()
        
        self.write_video(frame_rate, filename=filename)
    
    def find_images(self, directory, frame_start, frame_end):
        files = []
        for file in os.listdir(directory):
            if file.endswith(".png"):
                if file.startswith("image"):
                    files.append(os.path.join(directory, file))
                    
        if frame_start and frame_end:
            self.files = sorted(files)[frame_start:frame_end]

        elif frame_start:
            self.files = sorted(files)[frame_start:]

        elif frame_end:
            self.files = sorted(files)[:frame_end]
            
        else:
            self.files = sorted(files)
            return sorted(files)
            
    def read_image(self, path):
        image = mpimg.imread(path)
        return image

    def read_images(self):
        paths = self.find_images(self.directory, None, None)
        image_shape = self.read_image(paths[0]).shape
        self.image_shape = image_shape
    
        images = np.zeros((len(paths), *image_shape))
        
        for i, path in enumerate(paths):
            image = self.read_image(path)
            images[i] = image

        self.images = images
        print(f"Read {self.images.shape[0]} images")
        
    def total_luminosity(self):
        
        self.read_images()
        
        total_pixel_values = np.sum(self.images, axis = 0) # sum all pixels in frames
        
        self.max_lumuinosity = np.max(total_pixel_values)
        self.max_time = np.where(total_pixel_values == self.max_lumuinosity)[0]
        self.max_pixel_value = np.max(self.images)
        self.percent_brightness = total_pixel_values / self.max_lumuinosity * 1e2
        
        print("Max brightness (total single frame) = ", self.max_lumuinosity, 
              " occured at ", self.max_time/SOURCE_FRAME_RATE, " s")
        print("Max pixel value (single pixel) = ", self.max_pixel_value)
        
        # make plot
        fig, ax = plt.subplots(figsize = (4,4), dpi = 500)
        
        ax.plot(np.arange(len(total_pixel_values))/SOURCE_FRAME_RATE, 
                total_pixel_values, 'r-')
        
        ax.set_xlim(left = 0)
        ax.set_ylim(bottom = 0)
        
        ax.set_ylabel('Total Brightness')
        ax.set_xlabel('Time (s)')
        
        ax.grid(which = 'major', linestyle = '--')
        ax.minorticks_on()
        ax.grid(which = 'minor', linestyle = ':')
        
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        plt.savefig(os.path.join(self.directory, "_Brightness_Curve.pdf"), format = 'pdf', bbox_inches = 'tight')

        
        del self.images
        
    
    def normalise_image(self, invert_contrast):
        self.image *= 255
        if invert_contrast:
            self.image = -1 * self.image + 255
        self.image = self.image.astype("uint8")

    def clahe_image(self, clahe, clahe_clip, clahe_tile):
        if clahe:
            clahe_object = cv2.createCLAHE(clipLimit=clahe_clip,
                                           tileGridSize=(clahe_tile,
                                                         clahe_tile))
            
            self.image= clahe_object.apply(self.image)
    
    def rotate_image(self):
        if self.rotate:
            self.image = np.rot90(self.image, k=-1)
    
    def time_sum(self, num):
        if self.time_smoothing:
            start_num = num - self.time_smoothing
            end_num = num + self.time_smoothing
            if start_num > 0 and end_num < len(self.files):
                
                summed_image = np.zeros(self.image.shape)
                for i in np.arange(start_num, num + self.time_smoothing + 1):
                    current = self.read_image(self.files[i])
                    summed_image += current * 255
                summed_image = summed_image / (1 + 2*self.time_smoothing)
                self.image = summed_image.astype("uint8")
                # print(self.image[500,500])
                
            
    
    def median_filter(self, num = 0):
        if self.space_smoothing:
            self.image = medfilt2d(self.image, kernel_size = self.space_smoothing)
            
    def dilate(self):
        if self.dilation_kernel:
            cv2kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * self.dilation_kernel + 1,
                                            2 * self.dilation_kernel + 1)
                        )
            pad_factor = 2
            new_size = [int(self.image.shape[0] + pad_factor*2*self.dilation_kernel),
                        int(self.image.shape[1] + pad_factor*2*self.dilation_kernel)]
            padded_image = np.ones((new_size[0], new_size[1]))
            padded_image[pad_factor*self.dilation_kernel:-pad_factor*self.dilation_kernel, pad_factor*self.dilation_kernel:-pad_factor*self.dilation_kernel] = self.image

            dilated_pixels = cv2.dilate(padded_image, cv2kernel,
                                        iterations=1)
            
            self.image = dilated_pixels[pad_factor*self.dilation_kernel:-pad_factor*self.dilation_kernel, pad_factor*self.dilation_kernel:-pad_factor*self.dilation_kernel]
            
    
    
    def load_image(self, filepath, num=0):
        self.image = self.read_image(filepath)
        
        self.normalise_image(self.invert_contrast)
        
        self.time_sum(num)
        
        self.rotate_image()
        
        self.median_filter()
        self.dilate()
        
        # self.rotate_image()
        # self.normalise_image(self.invert_contrast)
        
        self.clahe_image(self.if_clahe, self.clahe_clip, self.clahe_tile)
        
        # self.add_brightness_stamp(num)
        self.add_time_stamp(num)
        return self.image
        
    def write_video(self, frame_rate, filename):
        filepath = os.path.join(self.directory, filename + ".mp4")
        self.load_image(self.files[0])
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(filepath, 0, frame_rate,
                              (self.image.shape[1], self.image.shape[0]))
        self.load_image(self.files[0])
    
        for num, image_path in enumerate(self.files):
            out.write(self.load_image(image_path, num))
        out.release()
        print(f"Wrote video to '{filepath}'")
    
    def surface_plot(self, image_number, downsample=None):
        image = self.images[image_number].copy()
        if downsample is not None:
            width = int(image.shape[1] * downsample)
            height = int(image.shape[0] * downsample)
            image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        
        xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, image, rstride=1, cstride=1, cmap=plt.cm.gray,
                        linewidth=0)
        plt.show()
    
    def calculate_times(self):
        self.times = np.linspace(0, len(self.files) / SOURCE_FRAME_RATE,
                                 len(self.files))
    
    def add_time_stamp(self, image_num):
        bottom_corner_pos = (10, 90)
        timestamp = f'{self.times[image_num]:.3f} s'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 2
        if self.invert_contrast:
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)
        if self.if_timestamp:
            cv2.putText(self.image,
                        timestamp, 
                        bottom_corner_pos, 
                        font, 
                        font_scale,
                        color,
                        thickness)

def create_video(directory=DIRECTORY):
    writer = VideoWriter(directory=directory, frame_rate=FRAME_RATE, clahe=CLAHE,
                         clahe_clip=CLAHE_CLIP_LIMIT, clahe_tile=CLAHE_TILE_SIZE,
                         invert_contrast=INVERT_CONTRAST, frame_start = FRAME_START, 
                         frame_end = FRAME_END, time_smoothing = TIME_SMOOTHING, 
                         space_smoothing = SPACE_SMOOTHING, dilation_kernel = DILATION_KERNEL)
    return writer
         
if __name__ == "__main__":
    
    # for run in np.linspace(1, 16, 16):
    #     SUB_DIRECTORY = fr"Cooldown 10/initial_room_temperature_tests/cell alignment/run{int(run)}"
    #     DIRECTORY = os.path.join(MAIN_DIRECTORY, SUB_DIRECTORY)
    #     writer = create_video(DIRECTORY)
        
    writer = create_video()
