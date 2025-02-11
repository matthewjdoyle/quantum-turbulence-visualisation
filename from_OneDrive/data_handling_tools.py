# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:08:35 2022

@author: w05876jh
"""
import re
import os
# import coordinate_finder
# import link_coordinates
import pandas as pd
import numpy as np


def extract_number_from_string(string):
    try:
        number = re.findall(r"[-+]?\d*\.\d+|\d+", string)[0]
        if "." in number:
            return float(number)
        else:
            return int(number)
    except IndexError:
        if string.lower() == "false":
            return False
        else:
            return True


def read_parameters(directory):
    parameter_path = os.path.join(directory, "_parameters.txt")
    param_dict = {}
    with open(parameter_path) as parameter:
        for line in parameter:
            split_line = line.split("=")
            if len(split_line) == 1:
                continue
            key = split_line[0].strip()
            value = split_line[1].strip()
            value = extract_number_from_string(value)
            param_dict[key] = value
    return param_dict


def find_all_images(directory, extension=".png"):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension):
                if 'sum' not in file.lower():
                    path = os.path.join(root, file)
                    filenames.append(path)
    return filenames


class Directory:
    def __init__(self, path, masked=False):
        self.path = path
        self.masked = masked
        self.parameters_path = self.check_if_file_exists("_parameters.txt")
        self.paths_path = self.check_if_file_exists("_paths.csv")
        self.get_coordinates_path()
        self.find_parameters()
        self.find_images()

    def find_parameters(self):
        if self.parameters_path is not None:
            self.parameters = read_parameters(self.path)
        else:
            self.parameters = None

    def check_if_file_exists(self, filename):
        filepath = os.path.join(self.path, filename)
        if os.path.exists(filepath):
            return filepath
        else:
            return None
    
    def get_coordinates_path(self):
        if self.masked:
            self.coordinates_path = self.check_if_file_exists("_masked_coordinates.csv")
        else:
            self.coordinates_path = self.check_if_file_exists("_coordinates.csv")
        

    def find_images(self):
        if self.parameters_path is not None:
            self.image_paths = find_all_images(self.path)


    def load_coordinates(self):
        return pd.read_csv(self.coordinates_path)


class FileStructure:
    def __init__(self, base_path):
        self.base_path = base_path
        self.find_all_directories()

    def find_all_directory_paths(self):
        directory_paths = [x[0] for x in os.walk(self.base_path)]
        return directory_paths

    def find_all_directories(self):
        paths = self.find_all_directory_paths()
        self.directories = []
        for path in paths:
            candiate_dir = Directory(path)
            if candiate_dir.parameters is not None:
                self.directories.append(candiate_dir)

    def check_directory(self, test_dict, directory):
        test_keys = test_dict.keys()
        for key in test_keys:
            if directory.parameters[key] != test_dict[key]:
                return False
        return True

    def get_dir(self, parameters):
        dirs = []
        for directory in self.directories:
            if self.check_directory(parameters, directory):
                dirs.append(directory)

        if len(dirs) == 1:
            return dirs[0]
        elif len(dirs) == 0:
            return None
        else:
            return dirs

    def find_unique_parameters(self, parameter_name):
        parameters = []
        for directory in self.directories:
            parameters.append(directory.parameters[parameter_name])

        return np.unique(parameters)


    # def track_particles(self, directory):
    #     coordinate_finder.find_and_save_coordinates(directory.path)
    #     link_coordinates.link_paths(directory.path)

    def extract_all_coordinates(self, redo=False):
        for directory in self.directories:
            if redo:
                self.track_particles(directory)
                print(f"Extracted {directory.path}")
            elif directory.coordinates_path is None:
                self.track_particles(directory)
                print(f"Extracted {directory.path}")


def moving_average(array, window_size):
    array = array.copy()
    array = np.concatenate(([array[0]] * int(np.floor(window_size/2) - 1), array,
                            [array[-1]] * int(np.ceil(window_size/2))))
    smoothed_array = np.convolve(array, np.ones((window_size,))/window_size, mode='valid')
    return smoothed_array


if __name__ == "__main__":
    MAIN_DIRECTORY = r"X:/"
    SUB_DIRECTORY = r"Cooldown 10\stationary\temperature sweep 1.0 kV 360 us\run1"
    PATH = os.path.join(MAIN_DIRECTORY, SUB_DIRECTORY)
    
    d = Directory(PATH, masked=True)
    coords = d.load_coordinates()