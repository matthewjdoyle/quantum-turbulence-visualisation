# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 09:54:54 2021

@author: mbcx4cg2
"""

import os
import time
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.image as mpimg
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from matplotlib.ticker import MaxNLocator
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import data_handling_library as dh

pio.renderers.default='browser'
# plt.rcParams['legend.title_fontsize'] = 16
# mpl.rcParams['agg.path.chunksize'] = 10000


# COLOUR = ['b', 'r', 'g', 'm', 'c', 'y', 'k',
#           'chocolate', 'lime', 'darkviolet', 'plum', 'lightsteelblue', 'coral', 'grey',
#           'lightcoral', 'gold', 'peru', 'slategrey', 'deepskyblue', 'salmon']
# COLOUR = ['r', 'orange', 'g', 'b', 'k']

CMAP = mpl.cm.nipy_spectral


class Plotter:
    def __init__(self):
        pass
    
    """ Figure creation """
    def new_figure(self, title, xlabel, ylabel,
                 log=False, minor_labels=False, rotate_labels=False, rotate_ylabels=False,
                 limits=False, xMin=None, xMax=None, yMin=None, yMax=None, live_data=False,
                 figsize=(12,9), lw=0, ls='-', markersize=3, alpha=1,
                 title_fontsize=18, axis_fontsize=16, tick_fontsize=16, cmap=CMAP):
        self.title = title 
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = figsize
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = plt.gca()
        self.ax.set_title(self.title, fontsize=title_fontsize)
        self.ax.set_xlabel(self.xlabel, fontsize=axis_fontsize)
        self.ax.set_ylabel(self.ylabel, fontsize=axis_fontsize)
        self.ax.tick_params(axis="both", labelsize=tick_fontsize)
        self.log = log
        self.lw = lw
        self.ls = ls
        self.markersize = markersize
        self.alpha = alpha
        self.axes = None
        self.lines = []
        self.stored_data = []
        self.cmap = cmap
        self.set_colours()
        
        if log:
            self.log_scale(self.ax)
            if minor_labels:
                self.ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
                self.ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
                
        if rotate_labels:
            self.ax.tick_params(axis='x', which='both', rotation=30)
        if rotate_ylabels:
            self.ax.tick_params(axis='y', which='both', rotation=30)
        if limits:
                self.limit_axes(xMin, xMax, yMin, yMax)
        if live_data:
            plt.ion()
            
    
    def new_subplot_figure(self, main_title, sub_titles, xlabels, ylabels, rows, cols,
                           log=False, rotate_labels=False, limits=False, title_offset=False,
                           xMin=None, xMax=None, yMin=None, yMax=None, projection=None, 
                           separate_projections=None, sharex=False, sharey=False, alpha=1,
                           remove_gaps=False, shrink_gaps=False, gap_size=0.1,
                           figsize=(12,9), lw=0, ls='-', markersize=3, 
                           title_fontsize=18, axis_fontsize=16, cmap=CMAP,
                           constrained_layout=True, overflow_warning=True):
        self.figsize = figsize
        self.title = main_title 
        self.sub_titles = sub_titles
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.log = log
        self.lw = lw
        self.ls = ls
        self.markersize = markersize
        self.alpha = alpha
        self.lines = []
        self.cmap = cmap
        self.set_colours()
        self.fig, self.axs = plt.subplots(nrows=rows, ncols=cols, 
                                          subplot_kw=dict(projection=projection),
                                          figsize=self.figsize, 
                                          constrained_layout=constrained_layout,
                                          sharex=sharex, sharey=sharey)
        self.shrink_gaps_between_subplots(remove_gaps, shrink_gaps, gap_size)
        self.line = []
        self.fig.suptitle(self.title, fontsize=20)
        self.axes = self.get_axes_list(rows, cols)
        self.axis_overflow_warning(overflow_warning)
        
        if separate_projections is not None:
            self.change_axes_projections(rows, cols, separate_projections)
        
        for ax, title, xlabel, ylabel in zip(self.axes, self.sub_titles, 
                                             self.xlabels, self.ylabels):
            self.set_title_and_labels(ax, title, xlabel, ylabel, 
                                      title_offset, log, rotate_labels,
                                      title_fontsize, axis_fontsize)
            if limits:
                self.limit_axes(xMin, xMax, yMin, yMax, ax=ax)
    
    
    """ Formatting methods """      
    def get_axes_list(self, rows, cols):
        self.axes = []
        if rows > 1 and cols > 1:
            for ax in self.axs:
                for a in ax:
                    self.axes.append(a)
        elif rows>1 or cols>1:
            for ax in self.axs:
                self.axes.append(ax)
        else:
            self.axes.append(self.axs)
        return self.axes
    
    
    def change_axes_projections(self, rows, cols, projections):
        for i, (ax, p) in enumerate(zip(self.axes, projections)):
            # self.axes[i] = plt.subplot((rows*100 + cols*10 + i + 1), projection=p)
            self.axes[i] = plt.subplot(rows, cols, (i + 1), projection=p)
            ax.grid(True, axis='both')
            
    
    def rotate_polar_plot(self, ax=None, direction=-1, offset=np.pi/2):
        ax = self.define_ax(ax)
        ax.set_theta_direction(direction)
        ax.set_theta_offset(offset)
    
    
    def axis_overflow_warning(self, on=True):
        if on:
            if len(self.axes) < len(self.sub_titles):
                print(f'{len(self.sub_titles) - len(self.axes)} plots will not fit on {self.title} canvas')
            
    
    def set_rows_and_columns(self, N_plots):
        if N_plots == 1 or N_plots == 0:
            rows = 1
            cols = 1
        if N_plots == 2:
            rows = 1
            cols = 2
        if N_plots > 2:
            rows = 2
            if N_plots % 2 == 0:
                cols = int(N_plots/2)
            else:
                cols = int(N_plots/2 + 1)
        if N_plots > 12:
            rows = 3
            if N_plots % 3 == 0:
                cols = int(N_plots/3)
            else:
                cols = int(N_plots/3 + 1)
        if N_plots > 30:
            rows = 4
            if N_plots % 4 == 0:
                cols = int(N_plots/4)
            else:
                cols = int(N_plots/4 + 1)
        return rows, cols
    
    
    def define_ax(self, ax):
        if ax is None:
            ax = self.ax
        return ax
    
    
    def set_title_and_labels(self, ax, title, xlabel, ylabel, 
                             title_offset=False, log=False, rotate_labels=False,
                             title_fontsize=14, axis_fontsize=12):
        if title_offset:
            ax.set_title(title, fontsize=title_fontsize, loc='left')
            ax.set_xlabel(xlabel, fontsize=axis_fontsize, loc='right')
        else:
            ax.set_title(title, fontsize=title_fontsize)
            ax.set_xlabel(xlabel, fontsize=axis_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_fontsize)
        ax.tick_params(axis="both", labelsize=axis_fontsize)
        if log:
            self.log_scale(ax)
        if rotate_labels:
            ax.tick_params(axis='x', which='both', rotation=30)
            
    
    def label_axes(self, xlabel, ylabel, zlabel=None, ax=None,
                   fontsize=16):
        ax = self.define_ax(ax)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if zlabel is not None:
            ax.set_zlabel(zlabel, fontsize=fontsize)
            ax.zaxis.labelpad = 10
    
    
    def autoformat_timestamps(self):
        self.fig.autofmt_xdate()
    
    
    def set_up_dual_axes(self, shared_axis, label1, label2):
        if shared_axis == 'x':
            self.ax2 = self.ax.twinx()
            self.ax.set_ylabel(label1, fontsize=16, color='b')
            self.ax.tick_params(axis='y', colors='b')
            self.ax2.set_ylabel(label2, fontsize=16, color='r')
            self.ax2.tick_params(axis='y', colors='r', labelsize=16)
        if shared_axis == 'y':
            self.ax2 = self.ax.twiny()
            self.ax.set_xlabel(label1, fontsize=16, color='b')
            self.ax.tick_params(axis='x', colors='b')
            self.ax2.set_xlabel(label2, fontsize=16, color='r')
            self.ax2.tick_params(axis='x', colors='r', labelsize=16)
    
    
    def set_up_residuals_axis(self, line=True, colour='k'):
        self.remove_ticks(self.ax)
        gs = self.fig.add_gridspec(4,1)
        self.axes = []
        self.axes.append(self.fig.add_subplot(gs[0:3,:]))
        self.axes.append(self.fig.add_subplot(gs[3,:]))
        if line:
            self.add_horizontal_line(0, colour=colour, ax=self.axes[1])
        for (ax, title, xlabel, ylabel) in zip(self.axes, 
                                               [None, None],
                                               [None, self.xlabel],
                                               [self.ylabel, r'$\Delta$ ' + self.ylabel]):
            self.set_title_and_labels(ax, title, xlabel, ylabel, 
                                      log=self.log, axis_fontsize=14)
    
    
    def limit_axis(self, Min, Max, ax=None, axis='x'):
        ax = self.define_ax(ax)
        if axis == 'x':
            ax.set_xlim(Min,Max)
        if axis == 'y':
            ax.set_ylim(Min,Max)
        
    
    def limit_axes(self, xMin, xMax, yMin, yMax, ax=None):
        ax = self.define_ax(ax)
        ax.set_xlim(xMin,xMax)
        ax.set_ylim(yMin,yMax)
        
    
    def strip_decimal_on_zero(self, ticks):
        labels = []
        for i, tick in enumerate(ticks):
            if i==0:
                labels.append(f'{tick:.0f}') # strip decimal on 0
            else:
                if np.abs(tick)<10:
                    labels.append(f'{tick:.9g}') # gives full accuracy if < 10 ignoring floating point errors
                elif np.abs(tick) < 10000:
                    labels.append(f'{tick:.0f}') # gives integers if > 10
                else:
                    labels.append(f'{tick:.2g}') # gives integers if > 10
        return labels
    
    
    def set_max_number_of_ticks(self, N, ax=None, axis='x'):
        ax = self.define_ax(ax)
        if axis == 'x':
            ax.xaxis.set_major_locator(MaxNLocator(N))
        if axis == 'y':
            ax.yaxis.set_major_locator(MaxNLocator(N))
    
    
    def write_tick_labels(self, ax, ticks, labels, axis='x'):
        if axis == 'x':
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        if axis == 'y':
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
    
    
    def relabel_ticks(self, xlabels, ylabels, ax=None):
        ax = self.define_ax(ax)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        self.write_tick_labels(ax, xticks, xlabels, axis='x')
        self.write_tick_labels(ax, yticks, ylabels, axis='y')
    
    
    def show_origin(self, ax=None, True_Zero=True):
        ax = self.define_ax(ax)
        ax.set_xlim(0,)
        ax.set_ylim(0,)
        if True_Zero:
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            xlabels = self.strip_decimal_on_zero(xticks)
            ylabels = self.strip_decimal_on_zero(yticks)
            self.write_tick_labels(ax, xticks, xlabels, axis='x')
            self.write_tick_labels(ax, yticks, ylabels, axis='y')
        
            
    def zero_single_axis(self, ax=None, axis='x'):
        ax = self.define_ax(ax)
        if axis == 'x':
            ax.set_xlim(0,)
            ticks = ax.get_xticks()
        if axis == 'y':
            ax.set_ylim(0,)
            ticks = ax.get_yticks()
        labels = self.strip_decimal_on_zero(ticks)
        self.write_tick_labels(ax, ticks, labels, axis)
    
    
    def log_scale(self, ax=None, axes='both'):
        ax = self.define_ax(ax)
        if axes == 'both' or axes == 'x':
            ax.set_xscale('log')
        if axes == 'both' or axes == 'y':
            ax.set_yscale('log')
    
        
    def remove_ticks(self, ax=None):
        ax = self.define_ax(ax)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axis('off')
        
    
    def string_multiple(self, variable, N):
        multiples = []
        var_range = np.max(variable)-np.min(variable)
        for n in range(N+1):
            if var_range > 5:
                multiples.append(f'{np.min(variable) + (n * var_range/N):.0f}')
            else:
                multiples.append(f'{np.min(variable) + (n * var_range/N):.2f}')
        return multiples
    
    
    def set_colours(self, N=20):
        self.colour = [self.cmap(i) for i in range(self.cmap.N)]
        space = int(self.cmap.N/N)
        self.colour = [self.colour[i*space] for i in range(N)]
    
    
    def reset_colours(self, data_list):
        self.colour = [self.cmap(i) for i in range(self.cmap.N)]
        space = int(self.cmap.N/len(data_list))
        self.colour = [self.colour[i*space] for i in range(len(data_list))]
        
    
    def set_colour_spectrum(self, zdata):
        norm = mpl.colors.Normalize(vmin=np.min(zdata), vmax=np.max(zdata))
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
        colour = np.array([(mapper.to_rgba(z)) for z in np.unique(zdata)])
        return colour
    
    
    def format_colourbar(self, cbar, zdata, zlabel):
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.set_yticklabels(self.string_multiple(zdata, 5), fontsize=16)
        cbar.set_label(zlabel, fontsize=16)
    
    
    ''' Elements used in plotting methods '''
    def make_grid(self, ax, xincr, yincr, major=True, minor=True, soft=False):
        xminorLocation = MultipleLocator(xincr)
        yminorLocation = MultipleLocator(yincr)
        ax.xaxis.set_minor_locator(xminorLocation)
        ax.yaxis.set_minor_locator(yminorLocation)
        if major:
            if soft:
                ax.grid(True, which = 'major', color='dimgrey', linestyle='--')
            else:
                ax.grid(True, which = 'major', color='k', linestyle='-')
        if minor:
            ax.grid(True, which = 'minor', color='lightgray', linestyle='--')
        if minor and not major:
            ax.grid(True, which = 'major', color='lightgray', linestyle='--')
        
    
    def flexible_grid(self, grid, xincr=1, yincr=1, ax=None,
                      major=True, minor=True, soft=False):
        if grid:
            ax = self.define_ax(ax)
            if self.log:
                if major:
                    if soft:
                        ax.grid(True, which='dimgrey', color='k')
                    else:
                        ax.grid(True, which='major', color='k')
                if minor:
                    ax.grid(True, which='minor', linestyle='--')
            else:
                self.make_grid(ax, xincr, yincr, major, minor, soft)
    
    
    def add_legend(self, ax, legend, separate_legend, 
                   l_title, loc, ncol, data, fontsize):
        self.l_fontsize = fontsize
        self.loc = loc
        self.ncol = ncol
        if legend:
            self.legend = ax.legend(fontsize=fontsize, loc=loc, 
                                    title=l_title, ncol=ncol)   
        if separate_legend:
            self.add_second_legend(ax, data, loc, l_title, ncol)


    def add_second_legend(self, ax, data, loc, l_title, ncol):
        if isinstance(data, list):
            N_curves = len(data)
        else:
            N_curves = 1
        handles, labels = ax.get_legend_handles_labels()
        ax.add_artist(self.legend)
        ax.legend(handles=handles[0:N_curves],
                  labels=labels[0:N_curves],
                  fontsize=14, 
                  loc=loc, title=l_title, ncol=ncol)
        
    
    def move_legend_to_anchor(self, ax, anchor=(1, 1)):
        ax = self.define_ax(ax)
        ax.legend(bbox_to_anchor=anchor, fontsize=self.l_fontsize, ncol=self.ncol)
        
        
    def reorder_legend(self, order, ax=None, loc=None, ncol=None):
        # e.g. order = [0, 1, 2]
        ax = self.define_ax(ax)
        if loc is None:
            loc = self.loc
        if ncol is None:
            ncol=self.ncol
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                  fontsize=self.l_fontsize, loc=loc, ncol=ncol) 

    
    def save_plot(self, directory, filename, save=True):
        if save:
            try:
                self.fig.savefig(os.path.join(directory, filename))
            except FileNotFoundError:
                dh.make_directory(directory)
                self.fig.savefig(os.path.join(directory, filename))
            
            
    def add_text(self, x, y, string, fontsize=14, colour='k', 
                 bkg_colour=None, bkg_alpha=None, edgecolour='black', ax=None):
        ax = self.define_ax(ax)
        if bkg_alpha is None:
            if bkg_colour is None:
                bkg_alpha = 0
            else:
                bkg_alpha = 1
        ax.text(x, y, string, fontsize=fontsize, color=colour, 
                bbox=dict(facecolor=bkg_colour, alpha=bkg_alpha, 
                          edgecolor=edgecolour))
        
        
    def add_arrow(self, x, y, x_length, y_length, ax=None, colour='k', width=None):
        ax = self.define_ax(ax)
        length = (x_length**2 + y_length**2) **0.5
        if width is None:
            width=length/5
        ax.arrow(x, y, x_length, y_length, width=width, head_length=width*5,
                 head_width=width*5, facecolor=colour, edgecolor=colour)
        
    
    def add_circle(self, x, y, radius, colour='r', fill=False, ax=None):
        ax = self.define_ax(ax)
        circle = plt.Circle((x, y), radius, color=colour, fill=False)
        ax.add_patch(circle)
        
    
    def add_horizontal_line(self, placement, colour='k', ax=None, ls='-', lw=1.5):
        ax = self.define_ax(ax)
        ax.axhline(y=placement, color=colour, ls=ls, lw=lw)
        
    
    def add_vertical_line(self, placement, colour='k', ax=None, ls='-', lw=1.5):
        ax = self.define_ax(ax)
        ax.axvline(x=placement, color=colour, ls=ls, lw=lw)
        
    
    def label_points(self, xdata, ydata, labels, ax=None):
        ax = self.define_ax(ax)
        for i, label in enumerate(labels):
            ax.annotate(label, (xdata[i], ydata[i]))
        
    
    # def add_inset_axis(self, pos, title, xlabel, ylabel, alpha, 
    #                    grid, rotate_labels, ax=None):
    #     ax = self.define_ax(ax)
    #     ip = InsetPosition(ax, pos)
    #     ax2 = plt.axes([0,0,1,1])
    #     ax2.set_axes_locator(ip)
    #     if rotate_labels:
    #         ax2.tick_params(axis='x', which='both', rotation=30)
    #     if grid:
    #         ax2.grid(True)
    #     self.set_title_and_labels(ax2, title, xlabel, ylabel)
    #     ax2.patch.set_alpha(alpha)
    #     return ax2
    
    
    def add_inset_axis(self, loc, title, xlabel, ylabel, alpha, 
                       grid, rotate_labels, height='20%', width='20%',
                       ax=None):
        ax = self.define_ax(ax)
        ax2 = inset_axes(ax, height, width, loc)
        if rotate_labels:
            ax2.tick_params(axis='x', which='both', rotation=30)
        if grid:
            ax2.grid(True)
        self.set_title_and_labels(ax2, title, xlabel, ylabel)
        ax2.patch.set_alpha(alpha)
        return ax2
    
    
    def add_arrows(self, xdata, ydata, ax, colour, return_lengths=False):
        d_x = []
        d_y = []
        for i in range(int(xdata.index.min()+1), xdata.index.max()+1):
            d_x.append((xdata[i] - xdata[i-1]) * 0.7)
            d_y.append((ydata[i] - ydata[i-1]) * 0.7)
        for x, y, dx, dy in zip(xdata, ydata, d_x, d_y):
            ax.arrow(x, y, dx, dy, head_width=self.markersize*1.5, color=colour)
        if return_lengths:
            self.stored_data.append((d_x, d_y))
    
    
    def normalise_errors(self, xerr, yerr):
        if xerr is None:
            xerr = [None for err in yerr]
        if yerr is None:
            yerr = [None for err in xerr]
        xerr, yerr = np.array(xerr, dtype=object), np.array(yerr, dtype=object)
        return xerr, yerr
    
    
    def normalise_labels_and_errors(self, labels, fit_labels, xerr, yerr, errorbar):
        if labels is None:
            labels = ['Data' for ax in self.axes]
        if fit_labels is None:
            fit_labels = ['Fit' for ax in self.axes]
        if not errorbar:
            yerr = [None for ax in self.axes]
        xerr, yerr = self.normalise_errors(xerr, yerr)
        return labels, fit_labels, xerr, yerr
    
    
    def add_subplot_grids(self, grid):
        if grid:
            for ax in self.axes:
                ax.grid(True, which='major', color='k')
                ax.grid(True, which='minor', linestyle='--')
    
    
    def add_subplot_legends(self, legend, specific_legends, 
                            ncol, l_title, loc, fontsize=12,
                            cutoff=None):
        self.l_fontsize = fontsize
        self.loc = loc
        self.ncol = ncol
        if cutoff is not None:
            axes = self.axes[:cutoff]
        else:
            axes = self.axes
        if legend:
            for ax in axes:
                ax.legend(ncol=ncol, fontsize=fontsize, title=l_title, loc=loc)  
        
        if specific_legends is not None:
            for ax in specific_legends:
                ax.legend(ncol=ncol, fontsize=fontsize, title=l_title, loc=loc)
                
    
    def shrink_gaps_between_subplots(self, remove, shrink, gap_size=0.1):
        if remove:
            plt.subplots_adjust(wspace=0, hspace=0)
        if shrink:
            plt.subplots_adjust(wspace=gap_size, hspace=gap_size)
            
    
    def align_y_labels(self, axes=None):
        if axes is None:
            axes = self.axes
        self.fig.align_ylabels(axes)


    def define_surface(self, xdata, ydata, zdata, function, func):
        X, Y = np.meshgrid(xdata, ydata)
        if function:
            Z = func(X,Y)
        else: 
            Z = zdata
        return X, Y, Z
    
    
    def setup_live_daq_plot(self, daq, label): 
        self.curves = []
        self.TimeStamp = []
        self.data = [[] for i in range(daq.number_of_channels)]
        for i in range(daq.number_of_channels):
            curve, = self.ax.plot(datetime.datetime.now(), 
                                  daq.read_voltage()[i], 
                                  color=self.colour[i], 
                                  label=label + f' {i+1}')
            self.curves.append(curve,)
            
    
    def collect_live_daq_data(self, daq, save_data=False, names=None,
                              directory=None, filename='data.csv', mode='w'):
        self.TimeStamp.append(datetime.datetime.now())
        for i, data in enumerate(self.data):
            data.append(daq.read_voltage()[i]) 
        if save_data:
            self.save_live_daq_data(names, directory, filename, mode)
    
    
    def save_live_daq_data(self, names, directory, filename, mode='w'):
        data = np.array([self.TimeStamp, *self.data])
        if mode == 'a':
            data = [[i] for i in data[:,-1]]
        if names is None:
            names = ['Time stamp', *[f'Voltage {i+1}' for i in range(len(self.data))]]
        dh.save_data_as_dataframe(directory, filename, data, names, mode)
    
    
    def add_live_data_to_plotter(self, scale_factor=1, subtract_mean=False,
                                 rolling_mean_window=None):
        for i, curve in enumerate(self.curves):
            if subtract_mean:
                ydata = (self.data[i]- np.mean(self.data[i])) * scale_factor
            else:
                ydata = np.array(self.data)[i] * scale_factor
            if rolling_mean_window is not None:
                ydata = dh.moving_average(ydata, rolling_mean_window)
            curve.set_xdata(self.TimeStamp)
            curve.set_ydata(ydata)
            
    
    def update_live_plot(self, ax, grid):
        ax.relim()
        ax.autoscale_view()
        ax.legend(fontsize=16)
        if grid:
            ax.grid(True, which = 'major', color='k', linestyle='-')
            ax.grid(True, which = 'minor', color='lightgray', linestyle='--')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
            
    
    def add_data(self, xdata, ydata, label, colour, 
                 errorbar=False, marker='o', xerr=None, yerr=None, 
                 arrows=False, return_lengths=False, ax=None):
        ax = self.define_ax(ax)
        if errorbar:
            self.lines.append(
                ax.errorbar(xdata, ydata, label=label, fmt=marker, yerr=yerr, xerr=xerr,
                            capsize=3, elinewidth=1, color=colour, alpha=self.alpha,
                            markersize=self.markersize, lw=self.lw, ls=self.ls)
                )
        else:
            self.lines.append(
                ax.plot(xdata, ydata, label=label, marker=marker, markersize=self.markersize, 
                        color=colour, lw=self.lw, ls=self.ls, alpha=self.alpha)
                )
        if arrows:
            self.add_arrows(xdata, ydata, ax, colour, return_lengths)


    @dh.catch_errors
    def populate_data(self, xdata, ydata, label, colour='b', 
                      marker='.', xerr=None, yerr=None, errorbar=False, 
                      arrows=False, return_lengths=False, ax=None):
        
        if isinstance(xdata, list):
            self.reset_colours(xdata)

            if errorbar:
                xerr, yerr = self.normalise_errors(xerr, yerr)
                
                for i, (x, y, l, xe, ye) in enumerate(zip(xdata, ydata, label, xerr, yerr)):
                    self.add_data(x, y, l, self.colour[i], errorbar, marker=marker, 
                                  xerr=xe, yerr=ye, arrows=arrows, 
                                  return_lengths=return_lengths, ax=ax)
            else:
                for i, (x, y, l) in enumerate(zip(xdata, ydata, label)):
                    self.add_data(x, y, l, self.colour[i], 
                                  errorbar, marker=marker, arrows=arrows, 
                                  return_lengths=return_lengths, ax=ax)
        else:
            self.add_data(xdata, ydata, label, colour, errorbar, marker=marker, 
                          xerr=xerr, yerr=yerr, arrows=arrows, 
                          return_lengths=return_lengths, ax=ax)
    
    
    def add_colourbar_data(self, xdata, ydata, zdata, colour, errorbar, 
                           xerr, yerr, ax, alpha, add_bar=True):
        xdata, ydata, zdata = np.array(xdata), np.array(ydata), np.array(zdata)
        if errorbar:
            xerr, yerr = self.normalise_errors(xerr, yerr)
            for (z, col) in zip(np.unique(zdata), colour):
                for index in np.where(zdata == z)[0]:
                    ax.errorbar(xdata[index], ydata[index], yerr=yerr[index],
                                  xerr=xerr[index], marker='o', capsize=3, 
                                  color=col, alpha=alpha)
                    sc = ax.scatter(xdata[index], ydata[index], color=col, 
                                    s=self.markersize, alpha=alpha)
                    
        else:
            for (z, col) in zip(np.unique(zdata), colour):
                for index in np.where(zdata == z)[0]:
                    sc = ax.scatter(xdata[index], ydata[index], color=col, s=self.markersize)
        
        if add_bar:
            cbar = plt.colorbar(mappable=sc, ax=ax)
            return cbar
            
            
    """ Plotting methods """
    def plot(self, xdata, ydata, label, ax=None,
             legend=True, separate_legend=False,
             grid=False, errorbar=False, line=False, 
             arrows=False, return_lengths=False, origin=False, 
             ncol=1, l_title=None, loc=None, l_fontsize=14, 
             xincr=1, yincr=1, major=True, minor=True,
             softgrid=False, yerr=None, xerr=None, 
             colour='b', marker='o'):
        
        ax = self.define_ax(ax)
            
        self.populate_data(xdata, ydata, label, colour, 
                           marker, xerr, yerr, errorbar, 
                           arrows, return_lengths, ax)
        self.flexible_grid(grid, xincr, yincr, ax, major, minor, softgrid)
        self.add_legend(ax, legend, separate_legend, 
                   l_title, loc, ncol, xdata, l_fontsize)
        
        if origin:
            self.show_origin(self.ax)
        
    
    def plot_curve(self, xline, yline, label,
                   ax=None, legend=True, grid=False, colour='k',
                   l_title=None, loc=None, l_fontsize=14,
                   separate_legend=False, ncol=1, xincr=1, yincr=1,
                   major=True, minor=True, softgrid=False):
        
        self.lw = 1.5
        self.markersize = 0
        ax = self.define_ax(ax)
        
        self.populate_data(xline, yline, label, colour, ax=ax)
        self.flexible_grid(grid, xincr, yincr, ax, major, minor, softgrid)
        self.add_legend(ax, legend, separate_legend, 
                        l_title, loc, ncol, xline, l_fontsize)
        self.lw = 0
        self.markersize = 3
            
            
    def plot_histogram(self, data, bins=50, labels=None, ax=None,
                       grid=False, xincr=1, yincr=2, legend=True,
                       major=True, minor=True, softgrid=False,
                       l_title=None, loc='upper right', ncol=1, l_fontsize=14,
                       stacked=False,):
        ax = self.define_ax(ax)
        self.flexible_grid(grid, xincr, yincr, ax, major, minor, softgrid)
        ax.hist(data, bins=bins, label=labels, stacked=stacked)
        self.add_legend(ax, legend, False, l_title, loc, ncol, data, l_fontsize)
        
    
    def plot_2d_histogram(self, xdata, ydata, bins=50, ax=None,
                       grid=False, xincr=1, yincr=2,
                       major=True, minor=True, softgrid=False,
                       plot_range=None):
        ax = self.define_ax(ax)
        ax.hist2d(xdata, ydata, bins=bins, range=plot_range)  
        self.flexible_grid(grid, xincr, yincr, ax, major, minor, softgrid)
        
    
    def plot_vector_field(self, x, y, vx, vy, 
                          colour='k', ax=None):
        ax = self.define_ax()
        ax.quiver(x, y, vx, vy, color=colour)
    
    
    def plot_live_from_daq(self, daq, label='Output', wait_time=None,         # Use with data_acquisition_objects.py Daq class
                           ax=None, grid=True, scale_factor=1, 
                           subtract_mean=False, rolling_mean_window=None, 
                           save_data=False, directory=None, 
                           filename=None, names=None, mode='w'):
        ax = self.define_ax(ax)
        self.autoformat_timestamps()
        self.setup_live_daq_plot(daq, label)
        try:
            while True:
                self.collect_live_daq_data(daq, save_data, names, 
                                           directory, filename, mode)
                self.add_live_data_to_plotter(scale_factor, subtract_mean, 
                                              rolling_mean_window)
                self.update_live_plot(ax, grid)
                if wait_time is not None:
                    time.sleep(wait_time)
        except KeyboardInterrupt:
            pass
        daq.close()
    
    
    def subplot(self, xdata, ydata, labels=None, N0=1, colour=None,
                errorbar=False, grid=False, fitted=False, legend=True,
                yerr=None, xerr=None, xfit=None, yfit=None, fit_labels=None,
                specific_legends=None, l_title=None, l_fontsize=12,
                ncol=1, loc=None, max_N_legends=None):
        
        if colour is None:
            colour = self.colour
            
        labels, fit_labels, xerr, yerr = (
            self.normalise_labels_and_errors(labels, fit_labels, xerr, yerr, errorbar))
            
        for i, (ax, x, y, l, xe, ye) in enumerate(zip(self.axes, xdata, ydata, labels, xerr, yerr)):
            self.populate_data(x, y, l, colour[i], ax=ax,
                               errorbar=errorbar, xerr=xe, yerr=ye)
      
        if fitted:
            for i, (ax, x, y, l) in enumerate(zip(self.axes, xfit, yfit, fit_labels)):
                self.plot_curve(x, y, label=l, ax=ax, colour=colour[i+N0])
                
        self.add_subplot_grids(grid)            
        self.add_subplot_legends(legend, specific_legends, ncol, l_title, 
                                 loc, l_fontsize, max_N_legends)
    
    
    def dual_axes_plot(self, xdata1, ydata1, xdata2, ydata2, label1, label2,
                       legend=False, grid=False, origin=False,
                       ncol=1, locs=[(0.65,0.9), (0.65,0.85)], 
                       major=True, minor=True, softgrid=False,
                       xincr=1, yincr=1, shared_axis='x'):
        
        self.set_up_dual_axes(shared_axis, label1, label2)
        self.populate_data(xdata1, ydata1, label1, ax=self.ax, colour='b')
        self.populate_data(xdata2, ydata2, label2, ax=self.ax2, colour='r')
        self.flexible_grid(grid, xincr, yincr, major, minor, softgrid)
        
        if legend:
            self.ax.legend(fontsize=14, ncol=ncol, loc=locs[0])
            self.ax2.legend(fontsize=14, ncol=ncol, loc=locs[1])
        
        if origin:
            self.show_origin(self.ax)
       
    
    def surface_plot(self, xdata, ydata, zdata, zlabel=None, 
                      function=False, func=None, 
                      projection=False, colourbar=False):
        ax = self.fig.add_subplot(111, projection='3d', label='surface')           
        self.remove_ticks(self.ax)
        self.label_axes(self.xlabel, self.ylabel, zlabel, ax)
        X, Y, Z = self.define_surface(xdata, ydata, zdata, function, func)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.jet)
        ax.view_init(elev=15, azim=45)
        
        if projection:
            ax.contour(X, Y, Z, zdir='x', offset=0)
            ax.contour(X, Y, Z, zdir='y', offset=0)
            ax.contour(X, Y, Z, zdir='z', offset=-np.max(Z/5))
        if colourbar:
            self.fig.colorbar(surf, shrink=0.5, aspect=10)
        self.axes = [ax]
        
    
    def surface_subplot(self, xdata, ydata, zdata, ax, zlabel=None, 
                      function=False, func=None, top_view=False,
                      projection=False, colourbar=False):
        
        for xlabel, ylabel in zip(self.xlabels, self.ylabels):
            self.label_axes(xlabel, ylabel, ax=ax, fontsize=12)
                
        X, Y, Z = self.define_surface(xdata, ydata, zdata, function, func)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.jet) 
        if top_view:
            ax.view_init(elev=90, azim=270)
        if projection:
            # ax.contour(X, Y, Z, zdir='x', offset=0)
            # ax.contour(X, Y, Z, zdir='y', offset=0)
            ax.set_zlim(-np.max(Z/5))
            ax.contour(X, Y, Z, zdir='z', offset=-np.max(Z/5))
        if colourbar:
            self.fig.colorbar(surf, shrink=0.5, aspect=10)
        self.axes = [ax]
            
    
    def image_surface_plot(self, image, zlabel=None, 
                      function=False, func=None, 
                      projection=False, colourbar=False):
        ax = self.fig.add_subplot(111, projection='3d', label='surface') 
        xdata = np.linspace(0, image.shape[1]-1, image.shape[1])
        ydata = np.linspace(0, image.shape[0]-1, image.shape[0])
        zdata = image
        self.remove_ticks(self.ax)
        self.label_axes(self.xlabel, self.ylabel, zlabel, ax)
        X, Y, Z = self.define_surface(xdata, ydata, zdata, function, func)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.jet)
        ax.view_init(elev=15, azim=45)
        
        if projection:
            ax.contour(X, Y, Z, zdir='x', offset=0)
            ax.contour(X, Y, Z, zdir='y', offset=0)
            ax.contour(X, Y, Z, zdir='z', offset=-np.max(Z/5))
        if colourbar:
            self.fig.colorbar(surf, shrink=0.5, aspect=10)
        
        
    def colour_bar_plot(self, xdata, ydata, zdata, zlabel, 
                        errorbar=False, yerrorbar=False, grid=False, origin=False,
                        yerr=None, xerr=None, xincr=1, yincr=1, 
                        major=True, minor=True, softgrid=False, 
                        ax=None, alpha=None):
        ax = self.define_ax(ax)
        self.flexible_grid(grid, xincr, yincr, ax, major, minor, softgrid)
        colour = self.set_colour_spectrum(zdata)
        cbar = self.add_colourbar_data(xdata, ydata, zdata, colour, errorbar, 
                                       xerr, yerr, ax, alpha)
        self.format_colourbar(cbar, zdata, zlabel)
        if origin:
            self.show_origin(ax)
        
    
    def add_inset_plot(self, xdata, ydata, xlabel, ylabel, title, pos=[0.07, 0.65, 0.3, 0.3],
                       errorbar=False, origin=False, grid=False, rotate_labels=False, colour='k',
                       yerr=None, xerr=None, alpha=1, ax=None): # pos = relative positiion = [x_min, y_min, x_width, y_width]
        
        self.markersize=2
        ax2 = self.add_inset_axis(pos, title, xlabel, ylabel, alpha, 
                                  grid, rotate_labels, ax)    
        self.add_data(xdata, ydata, label=None, colour=colour, errorbar=errorbar, 
                      xerr=xerr, yerr=yerr, ax=ax2)
        if origin:
            self.show_origin(ax2)
    
    
    def add_inset_image(self, image, pos):
        ax2 = self.add_inset_axis(pos, title='', xlabel='', ylabel='', alpha=1, 
                                  grid=False, rotate_labels=False)
        ax2.imshow(image)
        ax2.axis('off')
    
    
    """ Image display methods"""
    def show_image(self, image, ax=None, grayscale=True, 
                   remove_ticks=True, interpolation_method=None):
        ax = self.define_ax(ax)
        if isinstance(image, str):
            image = image = mpimg.imread(image)
        if grayscale:
            ax.imshow(image, cmap='gray', 
                      interpolation=interpolation_method)
        else:
            ax.imshow(image, 
                      interpolation=interpolation_method)
        if remove_ticks:
            self.remove_ticks(ax)
        return image
        
    
    def show_images(self, images, grayscale=True):
        if self.axes is not None:
            for image, ax in zip(images, self.axes):
                    self.show_image(image, ax, grayscale)
        else:
            for image in images:
                self.show_image(image)
    
    
    def heat_map(self, image, ax=None):
        ax = self.define_ax(ax)
        ax.imshow(image, cmap='hot', interpolation='nearest')
    
    
    def heat_maps(self, images):
        for image, ax in zip(images, self.axes):
            self.heat_map(image, ax)
    
    """ Close figures """
    def close_figures(self, which='all'):
        plt.close(which)
                
            

class Interactive_Plotter:
    def __init__(self):
        pass
    
    """ New figure methods """
    def new_figure(self, title, xlabel, ylabel):
        self.title = title 
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig = go.Figure()
        self.fig.update_layout(
                title_text= title,
                xaxis_title = xlabel,
                yaxis_title = ylabel)
        
        
    def new_subplot_figure(self, title, subtitles, rows, cols,
                           shared_xaxes=False, v_gap=0.3, h_gap=0.2):
        self.title = title 
        self.subtitles = subtitles
        self.fig = make_subplots(rows=rows, cols=cols,
                                 subplot_titles=subtitles,
                                 shared_xaxes=shared_xaxes,
                                 vertical_spacing=v_gap,
                                 horizontal_spacing=h_gap)
        self.fig.update_layout(title_text = title)
                
    
    """ Formatting and data plotting methods """
    def set_scale(self, fig=None, log=False, axis='yaxis'):
        if fig is None:
            fig = self.fig
        if log:
            fig['layout'][axis]['type']='log'
        else:
            fig['layout'][axis]['type']='linear'
        
    
    def set_margin_widths(self, l=20, r=20, t=20, b=20):
        self.fig.update_layout(
        margin=dict(l=l, r=r, t=t, b=b)
        )
            

    def change_axis_label(self, axis, label):
        self.fig['layout'][axis]['title']=label
        
    
    def label_axes(self, axes, labels):
        for axis, label in zip(axes, labels):
            self.change_axis_label(axis, label)
        
    
    def add_data(self, xdata, ydata, label, 
                 subplot=False, show_legend=True, colour=None,
                 leg_title=None, group='1', row=1, col=1):
        if subplot:
            self.fig.add_trace(go.Scattergl(x=xdata, y=ydata, name=label, 
                                            legendgrouptitle_text=leg_title,
                                            showlegend=show_legend, 
                                            line=dict(color=colour)),
                               row=row, col=col)
        else:
            self.fig.add_trace(go.Scattergl(x=xdata, y=ydata, name=label, 
                                            line=dict(color=colour)))
    

    def add_text(self, text, x, y, colour, size=18,
                 bkg='rgba(0,0,0,0)', x_coord_type='x', y_coord_type='y'):
        self.fig.add_annotation(x=x,
                                y=y,
                                xref=x_coord_type,
                                yref=y_coord_type,
                                xanchor='left',
                                showarrow=False,
                                bgcolor=bkg,
                                font={'size':size,
                                      'color':colour},
                                text=text)
    
    
    def add_horizontal_line(self, height, colour):
        self.fig.add_hline(y=height, line=dict(color=colour))
    
    
    def add_vertical_line(self, placement, colour):
        self.fig.add_vline(x=placement, line=dict(color=colour))
            
    
    def load_plot(self):
        self.fig.show()
        
    
    """ Interactive elements and methods """
    def add_range_slider(self, set_initial_range=False,
                         start=None, end=None, axis='xaxis', scale='date'):
        self.fig['layout'][axis].update(dict(
            rangeslider=dict(
                visible=True),
            type=scale))
        self.fig['layout']['yaxis'].update(dict(
                  fixedrange = False))
        if set_initial_range:
            initial_range = [start, end]
            self.fig['layout'][axis].update(range=initial_range)
            
            
    def add_range_buttons(self, days=[], labels=[], mode='todate', inc_all=True):
        buttons = list([])
        for day, label in zip(days, labels):
            buttons.append(dict(count=day,
                                  label=label,
                                  step="day",
                                  stepmode="todate"))
        if inc_all:
            buttons.append(dict(step="all"))
        self.fig.update_layout(
             xaxis=dict(
                 rangeselector=dict(
                     buttons=buttons)))  
    
    
    def add_scale_buttons(self, axes=[], lin_label='Linear Scale', log_label='Log Scale'):
        lin_args = {}
        log_args = {}
        for axis in axes:
            lin_args[axis + '.type'] = 'linear'
            log_args[axis + '.type'] = 'log'
        updatemenus = [
            dict(
                type="buttons",
                buttons=list([
                    dict(
                        args=[lin_args],
                        label=lin_label,
                        method="relayout"
                    ),
                    dict(
                        args=[log_args],
                        label=log_label,
                        method="relayout"
                    )]))]
        self.fig.update_layout(updatemenus=updatemenus)
        
        
    def add_scale_buttons_for_separate_plots(self, axes, lin_labels, log_labels):
        lin_buttons = list([])
        log_buttons = list([])
        for axis, label in zip(axes, lin_labels):
            lin_buttons.append(dict(
                args=[{axis + '.type' : 'linear'}],
                label=label,
                method="relayout"))
        for axis, label in zip(axes, log_labels):
            log_buttons.append(dict(
                args=[{axis + '.type' : 'log'}],
                label=label,
                method="relayout"))
        updatemenus = [
            dict(
                type="buttons",
                buttons=list([
                    *lin_buttons,
                    *log_buttons
                    ]))]
        self.fig.update_layout(updatemenus=updatemenus)
        
    
    def save_plot(self, directory, filename, static_copy=False):
        self.fig.write_html(os.path.join(directory, filename + ".html")) # save as browser link
        if static_copy:
            self.fig.write_image(os.path.join(directory, filename + ".png"),
                                         width=4*300, height=2*300, scale=1) # save as png without animations

            
    
if __name__ == '__main__':
    x = [np.linspace(0, 10, 11) for i in range(20)]
    y = [a for a in x]
    y = [i*0.5*b for i,b in enumerate(y)]
    l = [f'{i*0.5} x' for i in range (len(x))]
    pt = Plotter()
    pt.new_figure(' ', 'xlabel', 'ylabel', markersize=0, lw=2)
    pt.plot(x, y, l)
    
    xdata = [x for i in range(3)]
    ydata = [y for i in range(3)]
    labels = [l for x in xdata]
    pt.new_subplot_figure(' ', [' ' for x in xdata], 
                          ['x' for x in xdata], ['y' for x in xdata], 
                          1, 3, markersize=0, lw=2)
    pt.subplot(xdata, ydata, labels)
    
    