# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.image as mpimg
from scipy.signal import medfilt2d
import matplotlib.animation as animation



# CHANGE TO DESIRED RUN FOLDER
MAIN_DIRECTORY = r"D:/"
SUB_DIRECTORY = r"PhD Storage\Visualization Experiment\Vortex line at 1 K"
DIRECTORY = os.path.join(MAIN_DIRECTORY, SUB_DIRECTORY)


FRAME_RATE = 25         # OUTPUT FRAME RATE
FIGSIZE = (4,4)         # SIZE IN INCHES OF VIDEO AREA
OUTPUT_DPI = 500        # DOTS (pixels) PER INCH

IMAGE_CMAP = "gray"     # ["gray", "gray_r", "plasma"]
ROTATE = True

TIMESTAMP = True
SOURCE_FRAME_RATE = 200 # read this in from _paramters instead?



# START AND END FRAME NUMBERS FOR CLIP
FRAME_START = 50
FRAME_END = 150

# SETTINGS FOR FILTERS (set to 0 to turn off)
TIME_SMOOTHING = 0   # number of frames either side of target frame to sum
SPACE_SMOOTHING = 0  # kernel size for median filter 
DILATION_KERNEL = 5  # kernel size for dilation (0 default, no dilation)

STAT_NOISE_CUT = 0  # sets pixels with value < (mean_pixel - STAT_NOISE_CUT*pixel_standard_deviation) to zero
ROWWISE_STAT_NOISE_CUT = -0.5   # same as above but mean and standard deviation are sampled from each row of pixels independently

# Save filename. 
FILENAME = "rowcut-5e-1_dilated5"

class VideoWriter:
    def __init__(self, directory, frame_rate, rotate=ROTATE,
                 timestamp=TIMESTAMP, filename=FILENAME, frame_start = None, 
                 frame_end = None, time_smoothing = None, 
                 space_smoothing = None, dilation_kernel = None):
        self.directory = directory
        self.frame_rate = frame_rate
        self.if_timestamp = timestamp
        self.rotate = rotate
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.time_smoothing = time_smoothing
        self.space_smoothing = space_smoothing
        self.dilation_kernel = dilation_kernel
        self.savepath = os.path.join(self.directory, FILENAME + ".mp4")
        
        # self.total_luminosity()
        
        self.find_images(self.directory, self.frame_start, self.frame_end)
        
        print("Animating ", len(self.files), " frames...")

        self.setup_figure()
        print('Figure set up. \n')
        self.create_animation()
    
    def find_images(self, directory, frame_start, frame_end):
        files = []
        for file in os.listdir(directory):
            if file.endswith(".png"):
                if file.startswith("image"):
                    files.append(os.path.join(directory, file))
                    
        if frame_start and frame_end:
            self.files = sorted(files)[frame_start:frame_end]
            return sorted(files)

        elif frame_start:
            self.files = sorted(files)[frame_start:]
            return sorted(files)

        elif frame_end:
            self.files = sorted(files)[:frame_end]
            return sorted(files)
            
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
        
    
    def normalise_image(self):
        # self.image *= 255
        # self.image = self.image.astype("uint8")
        self.image = (self.image - self.image.min()) / (self.image.max() - self.image.min())

    
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
                    summed_image += current #* 255
                summed_image = summed_image / (1 + 2*self.time_smoothing)
                self.image = summed_image#.astype("uint8")
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
            
    def noise_cut(self):
        if STAT_NOISE_CUT:
            mean = np.mean(self.image)
            std = np.std(self.image)
            cut_level = mean - STAT_NOISE_CUT * std
            self.image[np.where(self.image < cut_level)] = 0
    
    def rowwise_noise_cut(self):
        if ROWWISE_STAT_NOISE_CUT:
            for i in range(len(self.image)):
                mean = np.mean(self.image[i,:])
                std = np.std(self.image[i,:])
                cut_level = mean - ROWWISE_STAT_NOISE_CUT * std
                self.image[i,np.where(self.image[i,:] < cut_level)] = 0 
    
    def load_image(self, num=0):
        filepath = self.files[num]
        self.image = self.read_image(filepath)
        
        self.normalise_image()
        self.time_sum(num)
        self.rotate_image()
        self.median_filter()
        self.dilate()
        
        self.noise_cut()
        
        return self.image
        
    def setup_figure(self):
        # setup the figure dimensions 
        self.fig, self.axis = plt.subplots(figsize = FIGSIZE, dpi = OUTPUT_DPI)
        self.axis.set_axis_off()
        self.fig.patch.set_facecolor('k')


    def create_image(self, num):
        # adds the plot to the current video frame's axis and adds time
        print("\rrendering: ", num, flush=True,  end = '')
        self.load_image(num)
        self.axis.clear()
        self.axis.imshow(self.image, cmap = 'gray', interpolation = 'none')
        self.axis.set_axis_off()
        
        current_time = (num + self.frame_start) / SOURCE_FRAME_RATE
        
        props = dict(boxstyle='round', facecolor='purple', alpha=1)
        
        if TIMESTAMP:
            self.axis.text(0.05, 0.95, r'$t=%.0f$ ms' % (current_time*1e3, ), transform=self.axis.transAxes, fontsize=10, color = 'w',
                           verticalalignment='top', bbox=props)
        
        self.fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        

    def create_animation(self):
        
          
            anim = animation.FuncAnimation(self.fig, self.create_image,
                                           frames=len(self.files),
                                           interval=len(self.files) / self.frame_rate)
            
            anim.save(self.savepath, fps=self.frame_rate, extra_args=['-vcodec', 'libx264'], dpi = 500)
            



def create_video(directory=DIRECTORY):

    animator = VideoWriter(directory=directory, frame_rate=FRAME_RATE, frame_start = FRAME_START, 
                         frame_end = FRAME_END, time_smoothing = TIME_SMOOTHING, 
                         space_smoothing = SPACE_SMOOTHING, dilation_kernel = DILATION_KERNEL)
    return animator
         
if __name__ == "__main__":
    
    # # PROCESS ACROSS A COLLECTION OF RUNS
    # for run in np.linspace(1, 16, 16):
    #     SUB_DIRECTORY = fr"Cooldown 10/initial_room_temperature_tests/cell alignment/run{int(run)}"
    #     DIRECTORY = os.path.join(MAIN_DIRECTORY, SUB_DIRECTORY)
    #     writer = create_video(DIRECTORY)
        
    writer = create_video()
