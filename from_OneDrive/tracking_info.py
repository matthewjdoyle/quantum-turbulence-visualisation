# -*- coding: utf-8 -*-
"""
Displays the information contained in _coordinates.csv and _paths.csv files. 
Refer to each csv as the dataframes C and P
Compares between the files to find the linking rate. 
"""

import numpy as np
import os
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats

from tabulate import tabulate

test_run = r"D:\PhD Storage\Visualization Experiment\Vortex line at 1 K"

def load_data_files(path_to_run):
    try:
        C = pd.read_csv(os.path.join(path_to_run, "_masked_coordinates.csv"))
        P = pd.read_csv(os.path.join(path_to_run, "_masked_paths.csv"))
    except:
        C = pd.read_csv(os.path.join(path_to_run, "_coordinates.csv"))
        P = pd.read_csv(os.path.join(path_to_run, "_paths.csv"))
    # print("Read data from ", path_to_run)
    return C, P

def read_parameters_file(path_to_run):
    parameters = {}
    
    # Open the file and read its content
    with open(os.path.join(path_to_run, "_parameters.txt"), 'r') as file:
        for line in file:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
        
            # Convert value to a number if possible (float or int)
            try:
                if '.' in value or 'e' in value or 'E' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep value as string if conversion fails
            parameters[key] = value
    return parameters

def read_frame_rate(path_to_run):
    return read_parameters_file(test_run)["Frame rate (Hz)"]

def get_all_images(run_path):
    image_files = sorted([os.path.join(run_path, f) for f in os.listdir(run_path) if (f.endswith('.png') or f.endswith('.jpg')) and f.startswith('image_')])
    print("\nReading images to measure total luminosity")
    images = []
    for i in range(len(image_files)):
        # image_name = r"image_%05d.png" % i
        # image_path = os.path.join(directory, image_name)
        image_path = image_files[i]
        image = image = mpimg.imread(image_path)
        images.append(image)
    print("Images read.")
    return images

def luminosity_with_time(run_path, frame_rate, show_plot = False):
    images = get_all_images(run_path)
    luminosity = []
    for i in range(len(images)):
        luminosity.append(np.sum(images[i]))
    time = np.arange(len(images)) / frame_rate
    
    if show_plot:
        fig, ax = plt.subplots(1, figsize = (4,4), dpi = 500)
        ax.plot(time, luminosity, c='r')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Luminosity")
        ax.set_xlim(left = 0, right = round(time.max()))
        ax.set_ylim(bottom = 0)
        plt.show()
    return np.array(luminosity), time

x = read_frame_rate(test_run)

C, P = load_data_files(test_run)


def counts_and_rates(C, P):
    Ndetected = C["x"].count()
    Nlinked = P["x"].count()
    Npaths = len(np.unique(P["particle"]))
    rate = Nlinked / Ndetected
    print(tabulate([['Detection count', Ndetected], 
                    ['Linked count', Nlinked],
                    ['Linking rate', rate],
                    ['Path count', Npaths]]))
    return Ndetected, Nlinked, Npaths, rate

counts_and_rates(C, P)

# apply to detection count, path count and linked particle count
def count_per_second(run_path, frame_rate, option = 'c', show_plot = False):
    C, P = load_data_files(run_path)
    if option == 'c':
        df = C
        vertical_label = "Detection count"
        color = "teal"
    elif option == 'p':
        df = P
        vertical_label = "Path count"
        color = 'navy'
    else:
        sys.exit(1)
    # metric is the column name e.g. "particle" or "frame"
    frame_counts_df = df['frame'].value_counts().reset_index()
    frame_counts_df.columns = ['frame', 'count']
    frame_counts_df = frame_counts_df.sort_values(by="frame")
    frame_counts_df['time'] = frame_counts_df['frame'] / frame_rate
    time = frame_counts_df['time'] 

    vals = frame_counts_df['count']
    if show_plot:
        fig, ax = plt.subplots(1, figsize = (3.5,3), dpi = 500)
        ax.plot(time, vals, color = color)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(vertical_label)
        ax.set_xlim(left = 0, right = round(time.max()))
        ax.set_ylim(bottom = 0)
        plt.show()
        
    return vals, time


def compare_time_counts(run_path):
    dets, dt = count_per_second(test_run, read_frame_rate(test_run))
    paths, pt = count_per_second(test_run, read_frame_rate(test_run), option = 'p')
    
    fig, ax = plt.subplots(1, figsize = (3.5,3), dpi = 500)
    ax.plot(dt, dets, color = 'navy', label = "Detections")
    ax.plot(pt, paths, color = 'maroon', label = "Linked paths")
    
    ax.set_ylabel("Count")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(left = 0, right = round(dt.max()))
    ax.set_ylim(bottom = 0)
    
    legend = ax.legend(fontsize = 'medium', loc = 'best', fancybox = False, labelspacing = 1, edgecolor = 'k', framealpha = 1)
    legend.get_frame().set_linewidth(0.8)
    plt.show()

def compare_detections_and_links(run_path):
    fr = read_frame_rate(run_path)
    C, P = load_data_files(run_path)
    detections, dtime = count_per_second(run_path, fr)
    links, ltime = count_per_second(run_path, fr, option = 'p')
    
    
    common_times = np.intersect1d(dtime, ltime)
    dtime_indices = np.isin(dtime, common_times)
    ltime_indices = np.isin(ltime, common_times)
    matched_detections = np.array(detections)[dtime_indices]
    matched_links = np.array(links)[ltime_indices]
        
    fig, ax = plt.subplots(1, figsize = (4,4), dpi = 500)
    ax.plot(matched_detections, matched_links, 'm.')
    ax.set_xlabel("Detection count")
    ax.set_ylabel("Link count")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(left = 0)
    plt.show()
    
    fig, ax = plt.subplots(1, figsize = (3.5,3), dpi = 500)
    ax.plot(common_times, matched_links/matched_detections, '.', color = 'limegreen')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linking rate")
    ax.set_ylim(bottom = 0, top = 1)
    ax.set_xlim(left = 0, right = round(common_times.max()))
    plt.show()
    
    
    
def compare_detections_to_luminosity(run_path):
    fr = read_frame_rate(run_path)
    C, P = load_data_files(run_path)
    counts, ctime = count_per_second(run_path, fr)
    lumins, ltime = luminosity_with_time(run_path, fr)
    
    fig, ax1 = plt.subplots(1, figsize = (8,4), dpi = 500)
    ax2 = ax1.twinx()
    
    ax1.plot(ctime, counts, 'b')
    ax2.plot(ltime, lumins, 'r')
    
    ax1.set_xlim(left = 0, right = round(ctime.max()))
    ax1.set_ylim(bottom = 0)
    ax2.set_ylim(bottom = round(lumins.min()))
    
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Detection count", color = 'b')
    ax2.set_ylabel("Luminosity", color = 'r')
    
    plt.show()
    return None
count_per_second(test_run, read_frame_rate(test_run), show_plot=True)
count_per_second(test_run, read_frame_rate(test_run), show_plot=True, option = 'p')
# luminosity_with_time(test_run, read_frame_rate(test_run), show_plot=True)
compare_detections_and_links(test_run)
# compare_detections_to_luminosity(test_run)
compare_time_counts(test_run)


# also do path statistics on the combined velocity data
def path_count(run_path, first_cut = 10, second_cut = 20):
    C, P = load_data_files(run_path)
    unique_path_labels = np.unique(P["particle"])
    path_length = []
    m0 = []
    m2 = []
    for i in unique_path_labels:
        path = P.loc[P["particle"] == i]
        path_length.append(len(path))
        
        m0.append(np.mean(P.loc[P["particle"] == i, "m_0"]))  # Average m0 for each particle
        m2.append(np.mean(P.loc[P["particle"] == i, "m_2"]))
    path_length = np.array(path_length)
    m0 = np.array(m0)
    m2 = np.array(m2)
    
    unique_lengths = np.unique(path_length)
    avg_m0_per_length = [np.mean(m0[path_length == length]) for length in unique_lengths]
    std_m0_per_length = [np.std(m0[path_length == length])/np.sqrt(np.count_nonzero(m0[path_length == length])) for length in unique_lengths]
    avg_m2_per_length = [np.mean(m2[path_length == length]) for length in unique_lengths]
    std_m2_per_length = [np.std(m2[path_length == length])/np.sqrt(np.count_nonzero(m2[path_length == length])) for length in unique_lengths]
    
    total_path_count = len(path_length)
    
    maximum_length = np.max(path_length)
    minimum_length = np.min(path_length)
    
    mean_length = np.mean(path_length)
    std_length = np.std(path_length)
    
    first_cut_path_lengths = path_length[np.where(path_length > first_cut)]
    second_cut_path_lengths = path_length[np.where(path_length > second_cut)]
    
    first_cut_count = len(first_cut_path_lengths)
    second_cut_count = len(second_cut_path_lengths)
    
    first_cut_mean = np.mean(first_cut_path_lengths)
    first_cut_std = np.std(first_cut_path_lengths)
    second_cut_mean = np.mean(second_cut_path_lengths)
    second_cut_std = np.std(second_cut_path_lengths)
    
    output = [["Path count", total_path_count, first_cut_count, second_cut_count], 
              ["Max path length", maximum_length],
              ["Min path length", minimum_length],
              ["Mean of path lengths", mean_length, first_cut_mean, second_cut_mean],
              ["Standard deviation of path lengths", std_length, first_cut_std, second_cut_std]
              ]
    
    print(tabulate(output, headers=["-/-", "Full distributions", "Path length > %d" % first_cut, "Path length > %d" % second_cut]))
    
    bins = np.arange(0, path_length.max() + 1.5) - 0.5
    fig, ax = plt.subplots(1, figsize = (3.5,3), dpi = 500)
    ax.hist(path_length, density = False, bins = bins, ec = 'w')

    ax.set_ylabel("Path count")
    ax.set_xlabel("Links")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(left = 0, right = 30)

    plt.show()
    
    # links vs average m0 and m2
    fig, ax = plt.subplots(1, figsize = (3.5,3), dpi = 500)
    # ax.plot(unique_lengths, avg_m0_per_length, marker='o', color='b')
    ax.errorbar(unique_lengths, avg_m0_per_length, yerr = std_m0_per_length, fmt = 'bo', capsize = 4)
    ax.set_ylabel(r"$\langle m_0 \rangle$")
    ax.set_xlabel("Path Length (Links)")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(left = 0)
    plt.show()
    
    fig, ax = plt.subplots(1, figsize = (3.5,3), dpi = 500)
    # ax.plot(unique_lengths, avg_m2_per_length, marker='o', color='g')
    ax.errorbar(unique_lengths, avg_m2_per_length, yerr = std_m2_per_length, fmt = 'go', capsize = 4)
    ax.set_ylabel(r"$\langle m_2 \rangle$")
    ax.set_xlabel("Path Length (Links)")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(left = 0)
    plt.show()
    
    return np.array(path_length)


def find_unlinked(C, P):
    merged = C.merge(P, on=['x', 'y'], how='left', indicator=True)
    unlinked_indexes = merged[merged['_merge'] == 'left_only'].index
    
    unlinked = C.loc[unlinked_indexes]
    return unlinked

def brightness_moments(run_path):
    C, P = load_data_files(test_run)
    U = find_unlinked(C, P)
    
    # population averages, mins, maximums, deviations
    Cm0_mean = np.mean(C["m_0"])
    Cm0_std = np.std(C["m_0"])
    Cm2_mean = np.mean(C["m_2"])
    Cm2_std = np.std(C["m_2"])
    
    Pm0_mean = np.mean(P["m_0"])
    Pm0_std = np.std(P["m_0"])
    Pm2_mean = np.mean(P["m_2"])
    Pm2_std = np.std(P["m_2"])
    
    Um0_mean = np.mean(U["m_0"])
    Um0_std = np.std(U["m_0"])
    Um2_mean = np.mean(U["m_2"])
    Um2_std = np.std(U["m_2"])
    
    output = [["m_0 mean", Cm0_mean, Pm0_mean, Um0_mean],
              ["m_0 std", Cm0_std, Pm0_std, Um0_std],
              ["m_2 mean", Cm2_mean, Pm2_mean, Um2_mean],
              ["m_2 std", Cm2_std, Pm2_std, Um2_std]
              ]
    
    print(tabulate(output, headers = ["-/-", "All detections", "Linked detections", "Unlinked detections"]))
    
    # m0 histogram of each population
    fig, ax = plt.subplots(1, figsize = (3.5,3), dpi = 500)
    ax.hist(C["m_0"], density = False, bins = 50, range = [0, 10], histtype = 'stepfilled', label = "All", color = 'grey')
    ax.hist(P["m_0"], density = False, bins = 50, range = [0, 10], histtype = 'step', label = "Linked", color = 'b')
    ax.hist(U["m_0"], density = False, bins = 50, range = [0, 10], histtype = 'step', label = "Unlinked", color = 'r')
    legend = ax.legend(fontsize = 'medium', loc = 'best', fancybox = False, labelspacing = 1, edgecolor = 'k', framealpha = 1)
    legend.get_frame().set_linewidth(0.8)
    ax.set_ylabel("Count")
    ax.set_xlabel("$m_0$")
    # ax.set_ylim(bottom = 0)
    ax.set_xlim(left = 0)
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_ylim(bottom = 1)
    plt.show()
    # m2 histogram of each population
    fig, ax = plt.subplots(1, figsize = (3.5,3), dpi = 500)
    ax.hist(C["m_2"], density = False, bins = 50, histtype = 'stepfilled', label = "All", color = 'grey')
    ax.hist(P["m_2"], density = False, bins = 50, histtype = 'step', label = "Linked", color = 'b')
    ax.hist(U["m_2"], density = False, bins = 50, histtype = 'step', label = "Unlinked", color = 'r')
    legend = ax.legend(fontsize = 'medium', loc = 'best', fancybox = False, labelspacing = 1, edgecolor = 'k', framealpha = 1)
    legend.get_frame().set_linewidth(0.8)
    ax.set_ylabel("Count")
    ax.set_xlabel("$m_2$")
    ax.set_ylim(bottom = 0)
    ax.set_xlim(left = 0)
    plt.show()
    
    # plot m0 m2 space heatmap for unlinked
    fig, ax = plt.subplots(1, figsize = (3.5,3), dpi = 500)
    ax.plot(P["m_0"], P["m_2"], 'b,')
    ax.plot(U["m_0"], U["m_2"], 'r,')
    # legend = ax.legend(fontsize = 'medium', loc = 'best', fancybox = False, labelspacing = 1, edgecolor = 'k', framealpha = 1)
    # legend.get_frame().set_linewidth(0.8)
    ax.set_ylabel("$m_2$")
    ax.set_xlabel("$m_0$")
    # ax.set_ylim(bottom = 0)
    # ax.set_xlim(left = 0)
    plt.show()
path_count(test_run)

brightness_moments(test_run)