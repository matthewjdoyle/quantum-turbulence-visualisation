# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:45:39 2022

@author: r61659md

VORTEX VIDEO

TURN IMAGES INTO VIDEO
"""


DATAFOLDER = r'D:\PhD Storage\Visualization Experiment\Vortex line at 1 K\PIV images'



import os
import moviepy.video.io.ImageSequenceClip
image_folder= DATAFOLDER
fps=15

image_files = [os.path.join(image_folder,img)
               for img in os.listdir(image_folder)
               if img.endswith(".png")]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(os.path.join(image_folder, 'my_video.mp4'))

# CAN OPEN VIDEO WITH VLC MEDIA PLAYER AND WORKS ON YOUTUBE

