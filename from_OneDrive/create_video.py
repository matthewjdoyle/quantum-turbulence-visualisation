# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.image as mpimg
# import collect_data as cd

MAIN_DIRECTORY = r"D:/"
SUB_DIRECTORY = r"Cooldown 11\initial_room_temperature_tests\testing_1-5_um_particles\run16"
DIRECTORY = os.path.join(MAIN_DIRECTORY, SUB_DIRECTORY)

# DIRECTORY = cd.WRITE_DIRECTORY

FRAME_RATE = 8

INVERT_CONTRAST = False
ROTATE = True

CLAHE = True
CLAHE_CLIP_LIMIT = 4
CLAHE_TILE_SIZE = 20

TIMESTAMP = True
SOURCE_FRAME_RATE = 200



class VideoWriter:
    def __init__(self, directory, frame_rate, invert_contrast=False,
                 clahe=False, clahe_clip=None, clahe_tile=None, rotate=ROTATE,
                 timestamp=TIMESTAMP, filename="_video"):
        self.directory = directory
        self.invert_contrast = invert_contrast
        self.if_clahe = clahe
        self.clahe_clip = clahe_clip
        self.clahe_tile = clahe_tile
        self.if_timestamp = timestamp
        self.rotate = rotate
        
        self.find_images(self.directory)
        self.calculate_times()
        self.write_video(frame_rate, filename=filename)
    
    def find_images(self, directory):
        files = []
        for file in os.listdir(directory):
            if file.endswith(".png"):
                if file.startswith("image"):
                    files.append(os.path.join(directory, file))
        self.files = sorted(files)
    
    def read_image(self, path):
        image = mpimg.imread(path)
        return image

    def read_images(self):
        paths = self.find_images(self.directory)
        image_shape = self.read_image(paths[0]).shape
    
        images = np.zeros((len(paths), *image_shape))
        
        for i, path in enumerate(paths):
            image = self.read_image(path)
            images[i] = image

        self.images = images
        print(f"Read {self.images.shape[0]} images")
    
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
    
    def load_image(self, filepath, num=0):
        self.image = self.read_image(filepath)
        self.rotate_image()
        self.normalise_image(self.invert_contrast)
        self.clahe_image(self.if_clahe, self.clahe_clip, self.clahe_tile)
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
                         invert_contrast=INVERT_CONTRAST)
    return writer
         
if __name__ == "__main__":
    
    # for run in np.linspace(1, 16, 16):
    #     SUB_DIRECTORY = fr"Cooldown 10/initial_room_temperature_tests/cell alignment/run{int(run)}"
    #     DIRECTORY = os.path.join(MAIN_DIRECTORY, SUB_DIRECTORY)
    #     writer = create_video(DIRECTORY)
        
    writer = create_video()
