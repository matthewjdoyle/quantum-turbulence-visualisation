
"""
Created on Fri Dec 17 10:48:35 2021

@author: mbcx4cg2
"""

import os
import re
import time
import cv2
import shutil
import pathlib
import numpy as np
import pandas as pd
import threading
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData
from scipy.signal import argrelextrema
from scipy import interpolate as it
from itertools import compress
    
    
""" Decorator functions """
def catch_errors(func):
    def wrapper(*args, **kwargs):
        try:
            vals = func(*args, **kwargs)
        except ValueError as e:
            print('ValueError occured:')
            print(e)
            vals = None
        except TypeError as e:
            print('TypeError occured:')
            print(e)
            vals = None
        except RuntimeError as e:
            print('RuntimeError occured:')
            print(e)
            vals = None
        return vals
    return wrapper


def function_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        vals = func(*args, **kwargs)
        name = func.__name__
        run_time = time.time() - start
        if run_time < 60:
            print(f'{name} finished in {run_time:.9f} seconds')
        else:
            run_time = [int(run_time/60), run_time % 60]
            if run_time[0] < 60:
                print(f'{name} finished in {run_time[0]:.0f} minutes {run_time[1]:.0f} seconds')
            else:
                run_time = [int(run_time[0]/60), run_time[0] % 60, run_time[1]]
                print(f'{name} finished in {run_time[0]:.0f} hours {run_time[1]:.0f} minutes')
        return vals
    return wrapper


def allow_keyboard_escape(func):
    def wrapper(*args, **kwargs):
        try:
            vals = func(*args, **kwargs)
            return vals
        except KeyboardInterrupt:
            print('Process cancelled by user\n')
            pass
    return wrapper


def run_if_true(bool_flag):
    def outer_wrapper(func):
        def wrapper(*args, **kwargs):
            if bool_flag:
                vals = func(*args, **kwargs)
            else:
                vals = None
            return vals
        return wrapper
    return outer_wrapper


""" Data reading """
def read_data(path, index_column=None, names=None, columns=None, sep='\t'):
    name = os.path.basename(path)
    if name.endswith(".csv"):
        df = pd.read_csv(path, index_col=index_column, usecols=columns)
    if name.endswith(".xlsx"):
        df = pd.read_excel(path, index_col=index_column, usecols=columns)
    if name.endswith(".txt") or name.endswith(".dat"):
        df = pd.read_table(path, index_col=index_column, names=names, sep=sep)
    return df


def read_all_data(directory, index_column=None, columns=None,  names=None,
                  N=None, strip_units=False, date_range=None):
    data_base = []
    paths = os.listdir(directory)
    paths = sort_paths_by_date(directory, paths)
    if N is not None:
        paths = paths[-N:]
    if date_range is not None:
        paths = filter_paths_by_modified_time(paths, date_range)
    for path in paths:
        if '.csv' in path or '.txt' in path or '.dat' in path:
            data = read_data(path, index_column=index_column, 
                             names=names, columns=columns)
            if strip_units:
                data = data[1:]
            data_base.append(data)
    return data_base


def read_text_file_as_dictionary(path):
    dictionary = {}
    with open(path) as p:
        for line in p:
            split_line = line.split("=")
            if len(split_line) == 1:
                continue
            key = split_line[0].strip()
            value = split_line[1].strip()
            value = extract_number_from_string(value)
            dictionary[key] = value
    return dictionary


""" Data writing """
def save_dataframe(directory, filename, df, columns=None, write_index=False, mode='w'):
    if mode == 'a':
        df.to_csv(os.path.join(directory, filename), index=write_index, 
                  columns=columns, mode=mode, header=False)
    else:
        df.to_csv(os.path.join(directory, filename), index=write_index, 
                  columns=columns, mode=mode)
    
    
def save_data_as_dataframe(directory, filename, data_list, names, mode='w'):
    product = {}
    for data, name in zip(data_list, names):
        product[name] = data
    df = pd.DataFrame(product)
    save_dataframe(directory, filename, df, mode=mode)


def create_empty_csv(columns, path):
    product = {}
    for col in columns:
        product[col] = []
    data = new_dataframe(product)
    data.to_csv(path, index=False)
    return data


def locate_or_create_datafile(columns, directory, filename):
    path = os.path.join(directory, filename)
    try:
        data = read_data(path)
    except FileNotFoundError:
        path = todays_filepath(directory, filename)
        data = create_empty_csv(columns, path)
    return data


def save_dictionary_as_text_file(dictionary, filepath):
    with open(filepath, "w") as file:
        for key, value in dictionary.items():
            file.write(f"{key} = {value}\n")
    

""" Label writing """
def now():
    return pd.to_datetime(pd.Timestamp.now())


def today():
    return now().strftime('%Y-%m-%d')


def make_timestamp(now=None, date_only=False, return_as_datetime=False):
    if now is None:
        now = now()
    if date_only:
        TimeStamp = now.strftime('%Y-%m-%d')
    else:
        TimeStamp = now.strftime('%Y-%m-%d %H %M')
    if return_as_datetime:
        TimeStamp = now
    return TimeStamp


def todays_filepath(directory, name, now=None):
    if now is None:
        now = pd.Timestamp.now()
    filename = name + ' ' + make_timestamp(date_only=True, now=now) + '.csv'
    path = os.path.join(directory, filename)
    return path


""" File path handling """
def get_file_modified_time(path):
    return pd.to_datetime(os.path.getmtime(path), unit='s')


def sort_paths_by_date(directory, paths):
    sorted_paths = []
    for path in paths:
        sorted_paths.append(os.path.join(directory, path))
    sorted_paths.sort(key=os.path.getmtime)
    return sorted_paths


def filter_paths_by_modified_time(paths, date_range):
    dates = [get_file_modified_time(p) for p in paths]
    index = [(d > date_range[0]) & (d < date_range[1]) for d in dates]
    return list(compress(paths, index))


def get_n_folders_from_path(path, n, end='last'):
    p = pathlib.Path(path)
    if end == 'first':
        return p.parts[:n]
    if end == 'last':
        return p.parts[-n:]
    

def get_outer_directory(directory, n_folders):
    path = os.path.join(
        *get_n_folders_from_path(
            directory, n_folders, 'first')
        )
    return path


""" File management """
def delete_file(directory, filename):
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        os.remove(path)
    else:
        print(f'No {filename} file in {directory}')
        

def copy_file(filename, source_folder, target_folder, 
              overwrite=False):
    source_path = os.path.join(source_folder, filename)
    target_path = os.path.join(target_folder, filename)
    if overwrite:
        shutil.copy(source_path, target_path)
        print(f'Copied {filename} to {target_folder}')
    else:
        if not os.path.exists(target_path):
            shutil.copy(source_path, target_path)
            print(f'Copied {filename} to {target_folder}')
            
            
def copy_file_or_folder(source_path, target_path):
    if os.path.isdir(source_path):
        shutil.copytree(source_path, target_path, symlinks=False, ignore=None)
    else:
        shutil.copy2(source_path, target_path)
        
        
def make_directory(directory):
    try:
        os.mkdir(directory)
    except FileNotFoundError:
        os.mkdir(os.path.dirname(directory))
        os.mkdir(directory)
        
        
def find_number_of_runs_in_directory(directory):
    run_numbers = [extract_number_from_string(n) for n in os.listdir(directory)]
    run_numbers = [n for n in run_numbers if type(n)==int]
    return np.max(run_numbers)
    

""" Image reading and writing """
def read_image(path, grayscale=True):
    image = mpimg.imread(path)
    if grayscale:
        if len(image.shape) == 3:
            image = rgb_image_to_grayscale(image)
    return image


def find_all_images(directory, file_type):
    filenames = []
    for file in os.listdir(directory):
        if file.endswith(file_type):
            filenames.append(os.path.join(directory, file))
    return filenames


def read_all_images(directory, file_type, N=np.inf):
    images = []
    files = find_all_images(directory, file_type)
    for i, file in enumerate(files):
        if i<N:
            images.append(read_image(os.path.join(directory, file)))
    return images


def save_image(path, image, grayscale=True):
    if grayscale:
        plt.imsave(path, image, cmap='gray')
    else:
        plt.imsave(path, image)
        

""" Image processing """
def rgb_image_to_grayscale(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])


def normalise_image(image, invert_contrast=False):
    image *= 255
    if invert_contrast:
        image = -1 * image + 255
    return image.astype("uint16")


def rescale_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)

        
def clahe_image(image, clahe_clip=4, clahe_tile=20):
    new_image = image
    new_image = normalise_image(new_image)
    clahe_object = cv2.createCLAHE(clipLimit=clahe_clip,
                                   tileGridSize=(clahe_tile,
                                                 clahe_tile))
    return clahe_object.apply(new_image)


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    if h != w:
        """do nothing"""
        #print('image not square - will be clipped!')
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


""" Data management """
def new_dataframe(product=None):
    if product is not None:
        return pd.DataFrame(product)
    else:
        return pd.DataFrame()
    

def merge_dataframes(data_base, drop_first_row=False, axis=0):
    for df in data_base:
        if drop_first_row:
            df.drop(index=0, inplace=True)
    df = pd.concat(data_base, axis=axis)
    return df


def combine_dataframes_by_nearest_timestamp(df1, df2, timestamp_name):
    return pd.merge_asof(df1, df2.sort_values(timestamp_name),
                         on=timestamp_name, direction='nearest')

    
def set_index_variable(df, variable, drop=False):
    df.sort_values(by=variable, inplace=True)
    df.index = df[variable]
    if drop:
        df.drop(columns=[variable], inplace=True)
    return df


def rename_df_columns(df, names, path=None):
    name_change = {}
    for name, old_name in zip(names, df.columns):
        name_change[old_name] = name
    df.rename(columns = name_change, inplace=True)
    if path is not None:
        df.to_csv(path, index=False)
    return df


def unpack_lists(list_of_lists):
    return [x for l in list_of_lists for x in l]


def duplicate_list_entries(l, N_duplicates):
    return [x for x in l for _ in [1 for i in range(N_duplicates)]]


def split_data_by_variable(data, variable):
    data, variable = np.array(data), np.array(variable)
    split_data = []
    unique_variable = np.unique(variable)
    for u in unique_variable:
        split_data.append(data[variable == u])
    return split_data


def split_dataframe_by_variable(data, variable, reindex=True):
    split_data = []
    unique_variable = np.unique(data[variable])
    for u in unique_variable:
        df = data.loc[data[variable] == u]
        if reindex:
            df.reset_index(inplace=True)
            df.drop(columns=['index'], inplace=True)
        split_data.append(df)
    return split_data


def reduce_to_unique_data(data, variable):
    reduced_data = data.drop_duplicates(variable)
    reduced_data = reduced_data.reset_index()
    return reduced_data


def find_nearest_value_in_array(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def reorder_data_by_unique_variable(data, variable, get_index=False):
    variable, index = np.unique(variable, return_index=True)
    data = data[index]
    if get_index:
        return data, variable, np.array(index)
    else:
        return data, variable
    

def reorder_dataframe_by_variable(df, variable, reindex=True):
    df = df.sort_values(variable)
    if reindex:
        df.reset_index(inplace=True)
        df.drop(columns=['index'], inplace=True)
    return df


def reorder_dataframe_list_by_mean_variable_value(df_list, variable):
    means = [np.mean(df[variable]) for df in df_list]
    indices = np.argsort(np.array(means))
    sorted_df_list = [df_list[i] for i in indices]
    return sorted_df_list


def split_multiple_data_by_variable(dataset, variable):
    split_dataset = []
    for data in dataset:
        split_data = split_data_by_variable(data, variable)
        split_dataset.append(split_data)
    return split_dataset


def reorder_multiple_data_by_unique_variable(dataset, variable):
    variable, index = np.unique(variable, return_index=True)
    ordered_dataset = []
    for data in dataset:
        ordered_dataset.append(data[index])
    return ordered_dataset, variable


def extract_number_from_string(string):
    try:
        number = re.findall(r"[-+]?\d*\.\d+|\d+", string)[0]
        if "." in number:
            return float(number)
        else:
            return int(number)
    except:
        return np.nan
    

def convert_number_to_string(number, n_digits=None):
    #easier to write f'{number:05}'
    number_length = len(f'{number}')
    if n_digits is None:
        n_digits = number_length
    zeros = n_digits - number_length
    pad = '0' * zeros
    return pad + f'{number}'


def strip_times(df):
    time = []
    for t in df['Time']:
        time.append(t.strftime('%X'))
    return time
    
    
def string_to_datetime(string):
    return pd.to_datetime(string)


def timedelta(value, unit='days'):
    return pd.Timedelta(value, unit)
    
    
def datetime_to_unixtime(dt):
    return time.mktime(dt.timetuple()) + dt.microsecond/1e6


def number_to_seconds(seconds):
    return pd.to_timedelta(seconds, unit='s')


def datetime_series_to_unixtime(dt_series):
    ut = []
    for dt in dt_series:
        ut.append(time.mktime(dt.timetuple()) + dt.microsecond/1e6)
    return ut


def create_datetime_from_date_and_time(df, date, time):
    df['datetime'] = pd.to_datetime(df[date] + ' ' + df[time])
    df.drop(columns=[date, time], inplace=True)
    df.rename(columns={'datetime':'Time'}, inplace=True)
    return df
        

def select_by_limit(df, var, lower, upper, in_range=True, reindex=True):
    df_new = pd.DataFrame
    if in_range:                         # select data between limits
        df_new = df.loc[df[var] <= upper]
        df_new = df_new.loc[df_new[var] >= lower]
    else:                                # select data outside of limits
        df_low = df.loc[df[var] < lower]
        df_high = df.loc[df[var] > upper]
        df_new = pd.concat((df_low, df_high))
    if reindex:
        df_new = df_new.reset_index()
    return df_new


def select_by_subset(df, var, subset, in_range=True):
    if in_range:                         # select data in subset
        df_new = df.loc[df[var].isin(subset)]
    else:                                # select data outside of subset
        df_new = df.loc[~df[var].isin(subset)]
    df_new = df_new.reset_index()
    return df_new


def select_by_substring(df, var, string, contains=True):
    if contains:
        return df.loc[df[var].str.contains(string)]
    else:
        return df.loc[~df[var].str.contains(string)]


def remove_data_by_index_range(df, subrange):
    df1 = df.loc[:subrange[0]]
    df2 = df.loc[subrange[1]:]
    return merge_dataframes([df1, df2])


""" Numerical operations """
def moving_average(data, n=3, center=True):
    data = pd.Series(data)
    moving_average = data.rolling(n, center=center).mean()
    return np.array(moving_average)


def second_order_difference(data):
    data = pd.Series(data)
    diff = (data.diff() + data.diff(-1))/2
    return np.array(diff)


def second_order_fluctuation(data):
    data = pd.Series(data)
    abs_diff = (abs(data.diff()) + abs(data.diff(-1)))/2
    sign_diff = data.diff()
    sign = sign_diff / abs(sign_diff)
    diff = abs_diff * sign
    return np.array(diff)


def smooth_data(values, errors, N, preserve_length=True):
    new_values = moving_average(values, n=int(N))
    if preserve_length:
        new_values[0:N-1] = values[0:N-1]
    new_err = errors/np.sqrt(N)
    return new_values, new_err


def smooth_dataset(values, errors, N, preserve_length=True):
    new_vals = []
    new_errs = []
    for vals, errs in zip(values, errors):
        new_values, new_errors = smooth_data(vals, errs, N, preserve_length=preserve_length)
        new_vals.append(new_values)
        new_errs.append(new_errors)
    return new_vals, new_errs


def add_noise_to_data(data, noise_amplitude):
    noisy_data = (data + noise_amplitude * 
                  np.random.randn(*np.array(data).shape))
    return noisy_data


def get_local_minima(array):
    index = argrelextrema(array, np.less)[0]
    values = array[index]
    return index, values


def get_local_maxima(array):
    index = argrelextrema(array, np.greater)[0]
    values = array[index]
    return index, values


def get_nth_percentile(array, n):
    return np.percentile(array, n)


def histogram_data(data, bins=None, weights=None, pdf=False,
                   normalise=False, polar_normalise=False):
    if bins is None:
        bins = 100
    data = data[~np.isnan(data)]
    hist, edges = np.histogram(data, bins, weights)
    bin_centers = moving_average(edges, 2)[1:]
    bin_width = bin_centers[2] - bin_centers[1]
    area = np.sum(bin_width*hist[~np.isnan(hist)])
    if pdf:
        hist = hist / np.sum(hist)
    if normalise:
        hist = hist / area
    elif polar_normalise:
        area = np.pi / bins * np.sum(hist[~np.isnan(hist)]**2)
        hist = hist / area**0.5
        
    return np.array(bin_centers), np.array(hist), area


def histogram_2d_data(xdata, ydata, bins=None, weights=None,
                      normalise=False):
    if bins is None:
        bins = 100
    hist, xedges, yedges = np.histogram2d(xdata, ydata, bins, 
                                          weights=weights)
    x_bin_centers = moving_average(xedges, 2)[1:]
    y_bin_centers = moving_average(yedges, 2)[1:]
    
    x_bin_width = x_bin_centers[2] - x_bin_centers[1]
    y_bin_width = y_bin_centers[2] - y_bin_centers[1]
    volume = np.sum(x_bin_width * y_bin_width * hist[~np.isnan(hist)])
    if normalise:
        hist = hist / volume
    
    return np.array(x_bin_centers), np.array(y_bin_centers), hist.T
    

""" Signal processing """
def fourier_transform(time, signal):
    N = len(signal)
    T = np.max(time) - np.min(time)
    freq = np.fft.fftfreq(N, d=T/N)
    fft_raw = np.fft.fft(signal)
    fft_true = 2 * np.abs(fft_raw / N)
    mask = freq > 0
    freq_true = freq[mask]
    fft_true = fft_true[mask]
    return freq_true, fft_true
    

""" Fitting functions """
@catch_errors
def fit_data(func, x, y, yerr=None, p0=None, 
             bounds=(-np.inf, np.inf)):
    x = x[~np.isnan(y)]
    if yerr is not None:
        yerr = yerr[~np.isnan(y)]
    y = y[~np.isnan(y)]
    params, pcov = curve_fit(func, x, y, sigma=yerr, p0=p0, 
                             bounds=bounds, absolute_sigma=False)
    params_err = np.sqrt(np.abs(np.diag(pcov)))
    return params, params_err


@catch_errors
def fit_with_2d_errors(func, x, y, xerr, yerr, p0):
    linear = Model(func)
    mydata = RealData(x, y, sx=xerr, sy=yerr)
    odr = ODR(mydata, linear, beta0=p0)
    result = odr.run()
    params = result.beta
    params_err = result.sd_beta
    return params, params_err


@catch_errors
def fit_2d_data(func, X, Y, Z,
                p0=None):
    # X, Y = np.meshgrid(X, Y)
    xdata = np.vstack((X.ravel(), Y.ravel()))
    params, pcov = curve_fit(func, xdata, 
                            Z.ravel(), p0=p0)
    params_err = np.sqrt(np.diag(pcov))
    fit = np.zeros(Z.shape)
    fit += func((X, Y), *params)
    return fit, params, params_err


""" 1D f(x) curves """
def straight_line(x, m, c):
    return m*x + c


def straight_line_through_origin(x, m):
    return m*x


def polynomial(x, coeffs):
    poly = 0
    for i, c in enumerate(coeffs):
        poly += c*x**i
    return poly


def quadratic(x, c0, c1, c2):
    coeffs = [c0, c1, c2]
    poly = 0
    for i, c in enumerate(coeffs):
        poly += c*x**i
    return poly


def cubic(x, c0, c1, c2, c3):
    coeffs = [c0, c1, c2, c3]
    poly = 0
    for i, c in enumerate(coeffs):
        poly += c*x**i
    return poly


def quartic(x, c0, c1, c2, c3, c4):
    coeffs = [c0, c1, c2, c3, c4]
    poly = 0
    for i, c in enumerate(coeffs):
        poly += c*x**i
    return poly


def rank_9_polynomial(x, x0, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    x = x - x0
    coeffs = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    poly = 0
    for i, c in enumerate(coeffs):
        poly += c*x**i
    return poly


def polynomial_error(x, errs):
    error_squared = 0
    for i, e in enumerate(errs):
        error_squared += (e*x**i) **2
    return error_squared ** (1/2)


def power_law(x, A, n):
    return A * x**n


def one_third_power_law(x, A):
    return A * x**(1/3)


def exponential(x, A, x0, R):
    return A * np.exp(R * (x-x0))


def exponential_asymptote(x, A, x0, R):
    return A * (1 - np.exp(-R * (x-x0)))


def gaussian(x, A, x0, var, c):
    return A * np.exp(-((x-x0)/var)**2) + c


def lorentzian(x, gamma, x0):
    return 1/(2*np.pi) * gamma / ((x-x0)**2 + (gamma/2)**2) 


def sin_wave(x, A, w, p, c):
    return A * np.sin(w*x + p) + c


def double_sin_wave(x, A1, w1, p1, c1, A2, w2, p2, c2):
    return sin_wave(x, A1, w1, p1, c1) * sin_wave(x, A2, w2, p2, c2)


def cos_sin_series(x, a1, a2, a3, b1, b2, b3, w1, w2, w3, p):
    x = x-p
    cos_series = a1*np.cos(w1*x) + a2*np.cos(w2*x) + a3*np.cos(w3*x)
    sin_series = b1*np.sin(w1*x) + b2*np.sin(w2*x) + b3*np.sin(w3*x)
    return cos_series + sin_series


def quartic_sin_wave(x, A, w, p, c, c0, c1, c2, c3, c4):
    return sin_wave(x, A, w, p, c) * quartic(x, c0, c1, c2, c3, c4)


""" Spline fitting functions """
def make_knots_and_coeffs_from_data(x, y, degree=3, weights=None):
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    knots, coeffs, deg = it.splrep(x, y, w=weights, k=degree)
    return knots, coeffs


def make_spline_from_knots_and_coeffs(x, knots, coeffs, degree=3):
    spl = it.BSpline(knots, coeffs, degree)
    spline = spl(x)
    return spline


def make_spline_from_data(x, y, x_map=None, degree=3, weights=None):
    knots, coeffs = make_knots_and_coeffs_from_data(x, y, degree, weights)
    if x_map is not None:
        spline = make_spline_from_knots_and_coeffs(x_map, knots, coeffs, degree)
    else:
        spline = make_spline_from_knots_and_coeffs(x, knots, coeffs, degree)
    return spline


def determine_degree_for_spline(x, y, weights):
    plt.figure()
    s1 = make_spline_from_data(x, y, degree=1, weights=weights)
    s3 = make_spline_from_data(x, y, degree=3, weights=weights)
    s5 = make_spline_from_data(x, y, degree=5, weights=weights)
    plt.plot(x,y, label='data')
    plt.plot(x,s1, label='degree = 1')
    plt.plot(x,s3, label='degree = 3')
    plt.plot(x,s5, label='degree = 5')
    plt.legend()
    

""" Quality checking functions """
def chi_squared(observed_data, fitted_data, error, Nparams):
    c = []
    observed_data = np.array(observed_data)
    fitted_data = np.array(fitted_data)
    error = np.array(error)
    [fitted_data, error], observed_data = (
        reorder_multiple_data_by_unique_variable([fitted_data, error], observed_data)
        )
    for o, f, e in zip(observed_data, fitted_data, error):
        c.append(((o-f)**2 / e**2)/(len(observed_data) - Nparams))
    return np.sum(c)


""" Error propagation """
def combine_errors_in_quadrature(errors):
    s = 0
    for e in errors:
        s += e**2
    final_error = s**0.5
    return final_error


def combine_fractional_errors_in_quadrature(values, errors):
    s = 0
    for e, v in zip(errors, values):
        s += (e/v)**2
    fractional_error = s**0.5
    return fractional_error


if __name__ == '__main__':
   # xdata = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1]
   # ydata = [5, 6, 7, 8, 5, 6, 7, 5, 6, 5]
   # x, y, z = histogram_2d_data(xdata, ydata, bins=4)
   
   x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
   noisy_x = add_noise_to_data(x, 0.1)