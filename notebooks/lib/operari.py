#!/bin/env python3
#Operations input/output
#Tim Tyree
#6.6.2020
#for spiral tip track processing
import os, matplotlib.pyplot as plt, numpy as np


def count_tips(x_list):
    return str(x_list).count('.')

def find_files(filename, search_path):
    result = []
    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result

def find_file(**kwargs):
    return find_files(**kwargs)[0]
# def plot_buffer(img_nxt, img_inc, contours_raw, contours_inc, tips, dpi, figsize=(15,15)):
def plot_buffer(img_nxt, img_inc, contours_raw, contours_inc, tips, figsize=(15,15), max_marker_size=800, lw=2):
    '''computes display data; returns fig.'''
    #plot figure
    fig, ax = plt.subplots(1,figsize=figsize)
    ax.imshow(img_nxt,cmap='Reds', vmin=0, vmax=1)
    ax.axis('off')

    #plot contours, if any.  type 1 = contours_raw (blue), type 2 = contours_inc (green)
    for n, contour in enumerate(contours_inc):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=lw, c='g', zorder=2)
    for n, contour in enumerate(contours_raw):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=lw, c='b', zorder=2)

    #plot tips, if any
    n_values, y_values, x_values = tips
    #     if len(n_values)>0:
    for j in range(len(x_values)): 
        ax.scatter(x = x_values[j], y = y_values[j], c='yellow', s=int(max_marker_size/(j+1)), zorder=3, marker = '*')
    return fig

def get_lifetime(trajectory_list):
    '''trajectory_list is a list of lists.  
    return np.mean( [ len(trajectory) for trajectory in trajectory_list ], axis=0 )'''
    return np.mean( [ len(trajectory) for trajectory in trajectory_list ], axis=0 )
#    TODO: for a given .csv of tip positions, make their trajectories naively in trackpy


def make_log_folder(folder_name='Data/log-tmp/'):
    try:
        os.mkdir(folder_name)
    except:
        print('^that folder probs existed already.')
def compress_log_folder_to(folder_name='Data/log-tmp/'):
    print("TODO: make compress_log_folder_to(folder_name='Data/log-tmp/')")
    return False

def remove_log_folder(folder_name='Data/log-tmp/'):
    try:
        os.rkdir(folder_name)
    except:
        print('^that folder probs existed already.')