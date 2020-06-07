#!/bin/env python3
####################################################
# Plotting functionality for highlighting odd birth/death events
# Author/Copywrite - Timothy Tyree
# Date - 5/28/2020, 3:49pm
# University of Calif., San Diego
####################################################

import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from ast import literal_eval

width = 512
height = 512
channel_no = 3

from lib.minimal_model import *
from lib.TexturePlot import *
from lib.get_tips import *
from lib.minimal_model import *
from lib.intersection import *
from lib.ProgressBar import *

#automate the boring stuff
from IPython import utils
import time, os, sys, re
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
	nb_dir = os.getcwd()
width = 512
height = 512
channel_no = 3

####################################################
# Plots
####################################################
# Plot temporal occurance of odd births/deaths 
def plot_odd_timeseries(t_values, n_values, todd_b_values, nodd_b_values, 
	todd_d_values, nodd_d_values, figsize = (16,8), fontsize = 24,
	save_dir=None):
	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(t_values, n_values, c='g', alpha=0.8, lw=2)
	ax.scatter(x=todd_b_values, y=nodd_b_values, label=f"odd births", c='b', s=100)
	ax.scatter(x=todd_d_values, y=nodd_d_values, label=f"odd deaths", c='r', s=100)
	
	ax.legend(fontsize=fontsize-5)
	ax.set_xlabel('time', fontsize=fontsize)
	ax.set_ylabel('number of spiral tips', fontsize=fontsize)
	ax.tick_params(axis='both', labelsize= fontsize)

	print(f"saved odd_timeseries plot to:\n{save_dir}")
	if save_dir != None:
		plt.tight_layout()
		fig.savefig(save_dir)
		print(f"saved odd_histogram plot to:\n{save_dir}")
		try:
			fig.savefig(save_dir.replace('.png','.svg'))
		except:
			print('oops! no copy saved as .svg.  if you want this, try saving as .png or using plt.savefig!')
	return fig

# Plot histogram of odd births/deaths 
def plot_odd_histogram(dn_even, dn_odd_births, dn_odd_deaths, bins_even = 32, bins_odd = 4
	, figsize=(12,7), fontsize = 24, xticks=list(range(-10,10,1)), save_dir=None):
	fig, ax = plt.subplots(figsize=figsize)
	ax.hist(dn_even, bins = bins_even,      color='g', label='even')
	ax.hist(dn_odd_births, bins = bins_odd, color='b', label='odd births')
	ax.hist(dn_odd_deaths, bins = bins_odd, color='r', label='odd deaths')

	ax.legend(fontsize=fontsize-5)
	ax.set_xticks(xticks)
	ax.tick_params(axis='both', labelsize= fontsize)
	ax.set_xlabel('frequency', fontsize=fontsize)
	ax.set_ylabel('change in spiral tip number', fontsize=fontsize)

	if save_dir != None:
		plt.tight_layout()
		fig.savefig(save_dir)
		print(f"saved odd_histogram plot to:\n{save_dir}")
		try:
			fig.savefig(save_dir.replace('.png','.svg'))
		except:
			print('oops! no copy saved as .svg.  if you want this, try saving as .png or using plt.savefig!')
	return fig