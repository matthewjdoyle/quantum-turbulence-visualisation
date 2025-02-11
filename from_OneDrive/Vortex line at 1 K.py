# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:05:36 2024

@author: Matt
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

RUNPATH = r'D:\PhD Storage\Visualization Experiment\Vortex line at 1 K'

import data_handling_library


START = 0
END = 999

LINESTART = 18
LINEEND = 22


FRAME19X = []
FRAME19Y = []

def get_all_images(start, end, directory = RUNPATH):
    images = []
    for i in range(start, end):
        image_name = r"image_%05d.png" % i
        image_path = os.path.join(directory, image_name)
        image = data_handling_library.read_image(image_path)
        if ('Cooldown 11' in directory) and ('200 fps' not in directory):
            rot_image = image
        else:
            rot_image = data_handling_library.rotate_image(image, -90)
        clahe_image = data_handling_library.clahe_image(rot_image, clahe_clip=5, clahe_tile=10)
        images.append(clahe_image)
    return images



def summed_image(images, start, end, plot = False):
    
    summed = np.zeros(np.shape(images[0]))
    for i in range(start, end):
        image = images[i]
        summed += image
    
    final = summed / (end - start)
    
    if plot:
        fig, ax = plt.subplots(figsize = (5,4), dpi = 500)
        
        ax.imshow(final)
        
        ax.axis('off')
        
        plt.show()
        
    return summed / (end - start)




def plot_line(images):
    
    image_numbers = [17,18, 19, 20, 21]
    timestep = 5 # ms
    
    crop_left = 250
    crop_right = 750
    crop_top = 1000
    crop_bot = 500   
    
    fig, ax = plt.subplots(ncols = len(image_numbers), figsize = (16,4), dpi = 300)
    for i in range(len(image_numbers)):
        ax[i].imshow(images[image_numbers[i]][crop_bot:crop_top, crop_left:crop_right],
                     cmap = 'gray', interpolation = None)
        
        ax[i].axis('off')
        
        # ax[i].set_xticks([])
        # ax[i].set_yticks([])
        
        ax[i].set_title(r'$t = %d$ ms' % (timestep*image_numbers[i]), fontsize = 24)
        if i == 0:
            ax[i].hlines(135, xmin = 0, xmax = 250, color = 'r', linestyle = '--')
        if i == len(image_numbers) - 1:
            ax[i].hlines(315, xmin = 0, xmax = 250, color = 'r', linestyle = '--')
            ax[i].hlines(425, xmin = 200, xmax = 300, color = 'limegreen', linestyle = '-')
        
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\VortexLineAt1K.pdf', format='pdf')
    plt.show()





def plot_line(images):
    
    image_numbers = [17,18, 19, 20, 21]
    timestep = 5 # ms
    
    crop_left = 250
    crop_right = 750
    crop_top = 1000
    crop_bot = 500   
    
    fig, ax = plt.subplots( ncols = len(image_numbers), figsize = (16,4), dpi = 300)
    
    

    for i in range(len(image_numbers)):
        ax[i].imshow(images[image_numbers[i]][crop_bot:crop_top, crop_left:crop_right],
                     cmap = 'gray', interpolation = None)
        
        ax[i].axis('off')
        
        # ax[i].set_xticks([])
        # ax[i].set_yticks([])
        
        ax[i].set_title(r'$t = %d$ ms' % (timestep*image_numbers[i]), fontsize = 24)
        if i == 0:
            ax[i].hlines(135, xmin = 0, xmax = 250, color = 'r', linestyle = '--')
        if i == len(image_numbers) - 1:
            ax[i].hlines(315, xmin = 0, xmax = 250, color = 'r', linestyle = '--')
            ax[i].hlines(425, xmin = 200, xmax = 300, color = 'limegreen', linestyle = '-')
        
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\VortexLineAt1K.pdf', format='pdf')
    plt.show()
    
    
from matplotlib.gridspec import GridSpec
    
def plot_line_with_big(images):
    
    image_numbers = [17,18, 19, 20, 21]
    timestep = 5 # ms
    
    crop_left = 250
    crop_right = 750
    crop_top = 1000
    crop_bot = 500   
    
    
    
    fig = plt.figure(layout="constrained", figsize = (11, 10), dpi = 500)

    gs = GridSpec(5, 5, figure=fig)
    ax1 = fig.add_subplot(gs[0:4, :])
    ax2 = fig.add_subplot(gs[4,0])
    ax3 = fig.add_subplot(gs[4,1])
    ax4 = fig.add_subplot(gs[4,2])
    ax5 = fig.add_subplot(gs[4,3])
    ax6 = fig.add_subplot(gs[4,4])
    
    axs = [ax2, ax3, ax4, ax5, ax6]
    
    ax1.imshow(images[20][crop_bot+200:crop_top, crop_left:crop_right],
                 cmap = 'gray', interpolation = None)
    ax1.hlines(275, xmin = 25, xmax = 75, lw = 3, color = 'w')
    ax1.axis("off")

    for i in range(len(image_numbers)):
        axs[i].imshow(images[image_numbers[i]][crop_bot:crop_top, crop_left:crop_right],
                     cmap = 'gray', interpolation = None)
        
        axs[i].axis('off')
        
        # ax[i].set_xticks([])
        # ax[i].set_yticks([])
        
        axs[i].set_title(r'$t = %d$ ms' % (timestep*image_numbers[i]), fontsize = 14)
        if i == 0:
            axs[i].hlines(135, xmin = 0, xmax = 250, color = 'r', linestyle = '--')
        if i == len(image_numbers) - 1:
            axs[i].hlines(315, xmin = 0, xmax = 250, color = 'r', linestyle = '--')
            axs[i].hlines(425, xmin = 200, xmax = 300, color = 'limegreen', linestyle = '-')
        
    # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\VortexLineAt1Kbig.pdf', format='pdf')
    plt.show()
        
        
    
images = get_all_images(START, END)
summed_image(images, START, END, plot = True)
summed_image(images, LINESTART, LINEEND, plot = True)


plot_line(images) # swtich to qt and choose data points for line curvature
plot_line_with_big(images)


def ring_velocity(radius):
    return 9.97e-4 / (4 * np.pi * radius) * np.log(8 * radius / 1e-8)
def r_of_curvature(X, a, b):
    R = (1 + (2*a*X + b)**2)**1.5 / np.abs(2*a)
    return R
def calc_radius(): 
    """

    Vortex ring velocity: v = kappa/4piR ln(8R/a)

    """
    data = np.genfromtxt(r'D:\PhD Storage\Visualization Experiment\Vortex line at 1 K\_100ms_line_points.csv', delimiter=',')
    x = data[1:,0] * 12.4e-4
    y = data[1:,1] * 12.4e-4

    plt.plot(x, y, 'o-')
    plt.show()
    
    # Calculate the first derivatives using central differences
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    # Calculate the second derivatives using central differences
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    # Calculate the curvature (kappa)
    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
    
    # Calculate the radius of curvature
    radius_of_curvature = 1 / curvature
    
    # Print the results
    print("Mean radius of curvature:", np.mean(radius_of_curvature), " cm")
    print("Min radius of curvature:", np.max(radius_of_curvature), " cm")
    print("Max radius of curvature:", np.min(radius_of_curvature), " cm")
    print("Std radius of curvature:", np.std(radius_of_curvature), " cm")
    print()
    print("Velocity based on average radius: ", ring_velocity(np.mean(radius_of_curvature)), " cm/s")
    print("Velocity based on maximum radius: ", ring_velocity(np.max(radius_of_curvature)), " cm/s")
    print("Velocity based on minimum radius: ", ring_velocity(np.min(radius_of_curvature)), " cm/s")
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', label='Data points and curve')
    for i in range(len(x)):
        plt.annotate(f'{radius_of_curvature[i]:.2f}', (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Curve and Radius of Curvature at Each Point')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    X = y
    Y = x
    
    
    coefficients = np.polyfit(X, Y, 2)
    a, b, c = coefficients
    
    # Polynomial function
    polynomial = np.poly1d(coefficients)
    

    
    # Calculate the radius of curvature at each point
    radii_of_curvature = r_of_curvature(X, a, b)
    
    # Print results
    print("Coefficients of the fitted polynomial: a = {}, b = {}, c = {}".format(a, b, c))
    print("Radii of curvature at each point:", radii_of_curvature)
    print(np.mean(radii_of_curvature), np.std(radii_of_curvature))
    print("Velocity based on average radius: ", ring_velocity(np.mean(radii_of_curvature)), " cm/s")
    print("Velocity error based on average radius: ", np.std(ring_velocity(radii_of_curvature)), " cm/s")
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'o', label='Data points')
    x_curve = np.linspace(min(X), max(X), 100)
    y_curve = polynomial(x_curve)
    plt.plot(x_curve, y_curve, '-', label='Fitted curve')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.title('Polynomial Fit and Radius of Curvature')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    

calc_radius()

def summed_image_and_paths(directory, start, end,
                           figsize = (10,10), min_links = 4, save_suffix = ''):
    images = get_all_images(start, end, directory = directory)
    summed = np.zeros(np.shape(images[0]))

    for i in range(len(images)):
        image = images[i]
        summed += image
    
    final = summed / (end - start)
    if ('Cooldown 11' in directory) and ('200 fps' in directory):
        final[800:,:] = np.mean(final[:700, :])
    if ('Cooldown 11' in directory) and ('200 fps' not in directory):
        final = final.T
        final[800:,:] = np.mean(final[:700, :])
    try:
        df = pd.read_csv(directory + '/_paths.csv')
    except FileNotFoundError:
        df = pd.read_csv(directory + '/_masked_paths.csv')
    
    dat = df.loc[((df['frame'] >= start) & (df['frame'] <= end))]
    
    k = np.shape(final)[1]
    if ('Cooldown 8' or '200 fps') in directory:
        k =  np.shape(final)[0]
    

    fig, ax = plt.subplots(figsize = figsize, dpi = 500)
    
    ax.imshow(final, cmap = 'gray')
    ids = np.unique(dat['particle'])
    for i in range(len(np.unique(dat['particle']))):
        cur = dat.loc[dat['particle'] == ids[i]]
        # norm_time = (1-np.min(cur['frame'])) / (np.max(cur['frame'])-np.min(cur['frame'])) * cur['frame'] + 0.2
        if len(cur['x']) >= 12:
            ax.plot(k-cur['x'], cur['y'], 'g.-', lw = 1, markevery=len(cur), markersize = 4)
        elif len(cur['x']) >= 6:
            ax.plot(k-cur['x'], cur['y'], 'b.-', lw = 0.5, markevery=len(cur), markersize = 2)
        elif len(cur['x']) >= min_links:
            ax.plot(k-cur['x'], cur['y'], 'r.-', lw = 0.5, markevery=len(cur), markersize = 2)
            
    
    ax.axis('off')
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\DrawnTrajectories1K%s.pdf' % save_suffix, format='pdf')
    
    plt.show()
        
    return summed / (end - start)



""" T = 1 K large particles """
# newpath = r'S:\Common\Vortex visualization project data\Cooldown 8\temperature_sweep\omega = 0.0 rads Qsw = 355 us\run1'
newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 8\temperature_sweep\omega = 0.0 rads Qsw = 355 us\run1'

summed_image_and_paths(newpath, 200, 220, min_links = 4)



newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 8\temperature_sweep\omega = 0.0 rads Qsw = 355 us\run1'

summed_image_and_paths(newpath, 200, 300, min_links = 6)
# summed_image_and_paths(newpath, 200, 220, min_links = 5)
# summed_image_and_paths(newpath, 200, 220, min_links = 10)

""" T = 350 mK large particles """
# newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 8\temperature_sweep\omega = 0.0 rads Qsw = 355 us\run13'
# summed_image_and_paths(newpath, 30, 60)


# """ T = 1 K small mixed particles """
# newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 11\laser_sheet_raised\T = 1.0 K 1.0 kV 200 fps\run1'
# summed_image_and_paths(newpath, 200, 220, min_links=3)


# """ T = 600 mK small mixed particles """
# newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 11\laser_sheet_raised\T = 600 mK 0.75 kV 200 fps\run1'
# summed_image_and_paths(newpath, 20, 40, min_links=3)

# """ T = 150 mK small mixed particles """
# newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 11\high_power_bursts\T = 150 mK 1.5 kV 200 fps\run1'
# summed_image_and_paths(newpath, 30, 60, min_links=3)

newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 11\laser_sheet_raised\T = 150 mK 0.5 kV\run1'
summed_image_and_paths(newpath, 10, 70, min_links=2)


def summed_image_and_paths_ax(ax,directory, start, end,
                           figsize = (10,10), min_links = 4, save_suffix = ''):
    images = get_all_images(start, end, directory = directory)
    summed = np.zeros(np.shape(images[0]))

    for i in range(len(images)):
        image = images[i]
        summed += image
    
    final = summed / (end - start)
    if ('Cooldown 11' in directory) and ('200 fps' in directory):
        final[800:950,200:800] = np.mean(final[:700, :])
    if ('Cooldown 11' in directory) and ('200 fps' not in directory):
        final = final.T
        final[800:950,20:180] = np.mean(final[:700, :])
    try:
        df = pd.read_csv(directory + '/_paths.csv')
    except FileNotFoundError:
        df = pd.read_csv(directory + '/_masked_paths.csv')
    
    dat = df.loc[((df['frame'] >= start) & (df['frame'] <= end))]
    
    k = np.shape(final)[1]
    if ('Cooldown 8' or '200 fps') in directory:
        k =  np.shape(final)[0]
    

    
    ax.imshow(final, cmap = 'gray')
    ids = np.unique(dat['particle'])
    for i in range(len(np.unique(dat['particle']))):
        cur = dat.loc[dat['particle'] == ids[i]]
        if len(cur['x']) >= 11:
            ax.plot(k-cur['x'], cur['y'], 'g.-', lw = 1, markevery=len(cur), markersize = 4)
        elif len(cur['x']) >= 6:
            ax.plot(k-cur['x'], cur['y'], 'b.-', lw = 0.5, markevery=len(cur), markersize = 2)
        elif len(cur['x']) >= min_links:
            ax.plot(k-cur['x'], cur['y'], 'r.-', lw = 0.5, markevery=len(cur), markersize = 2)
            
    
    ax.axis('off')

        
    return summed / (end - start)



from matplotlib.lines import Line2D

fig, ax = plt.subplots(1,2 ,figsize = (20,10), dpi = 500)

newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 8\temperature_sweep\omega = 0.0 rads Qsw = 355 us\run1'
summed_image_and_paths_ax(ax[0], newpath, 200, 220, min_links=4)
ax[0].set_title(r'$T \approx 1$ K', fontsize = 50)

newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 8\temperature_sweep\omega = 0.0 rads Qsw = 355 us\run13'
summed_image_and_paths_ax(ax[1], newpath, 30, 60)
ax[1].set_title(r'$T \approx 0.4$ K', fontsize = 50)


legend = ax[0].legend(handles=[
                        Line2D([0], [0], label='3', color='r', ls = '-', lw=2),
                        Line2D([0], [0], label='5', color='b', ls = '-', lw=2)       ,
                        Line2D([0], [0], label='10', color='g', ls = '-', lw=2)       
                               ],
                      fontsize = 20, loc = 'lower left', fancybox = False, labelspacing = 1, 
                   edgecolor = 'w', framealpha = 1, title = r"$N_\mathrm{links}>$", title_fontsize = 20, borderaxespad=1., labelcolor='linecolor', facecolor = 'k' )
legend.get_frame().set_linewidth(0.8)
legend._legend_title_box._text.set_color('#FFFFFF')
fig.tight_layout(h_pad = -20)

plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\LargeParticlesTraced.pdf', format='pdf', bbox_inches='tight')
  
plt.show()





fig, ax = plt.subplots(1,2 ,figsize = (20,10), dpi = 500)

newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 11\laser_sheet_raised\T = 1.0 K 1.0 kV 200 fps\run1'
summed_image_and_paths_ax(ax[0], newpath, 200, 220, min_links=2)
ax[0].set_title(r'$T \approx 1$ K', fontsize = 50)

newpath = r'C:\Users\Matt\The University of Manchester\Quantum fluids group - Documents\Vortex visualisation files\Main data storage\Cooldown 11\high_power_bursts\T = 150 mK 1.5 kV 200 fps\run1'
summed_image_and_paths_ax(ax[1], newpath, 30, 60, min_links=2)
ax[1].set_title(r'$T \approx 150$ mK', fontsize = 50)

legend = ax[0].legend(handles=[
                        Line2D([0], [0], label='2', color='r', ls = '-', lw=2),
                        Line2D([0], [0], label='5', color='b', ls = '-', lw=2)       ,
                        Line2D([0], [0], label='10', color='g', ls = '-', lw=2)       
                               ],
                      fontsize = 20, loc = 'lower left', fancybox = False, labelspacing = 1, 
                   edgecolor = 'w', framealpha = 1, title = r"$N_\mathrm{links}>$", title_fontsize = 20, borderaxespad=1., labelcolor='linecolor', facecolor = 'k' )
legend.get_frame().set_linewidth(0.8)
legend._legend_title_box._text.set_color('#FFFFFF')
fig.tight_layout(h_pad = -20)


fig.tight_layout()

plt.savefig(r'C:\Users\Matt\Desktop\Thesis figures i made here\Visualisation\MixedParticlesTraced.pdf', format='pdf',bbox_inches='tight')
  
plt.show()



