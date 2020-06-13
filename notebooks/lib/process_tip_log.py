#!/bin/env python3
####################################################
# Processing for tip log - 2D Simulation Results for Atrial Fibrillation 
#            for the development of reliable tip tracking
# Author/Copywrite - Timothy Tyree
# Date - 5/28/2020, 2:40pm
# University of Calif., San Diego
####################################################
from lib.OddBirthDeathPlots import *
from ast import literal_eval
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt

#automate the boring stuff
import time, os, sys, re
# beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
# if not 'nb_dir' in globals():
#     nb_dir = os.getcwd()

#delete these if everything works without them
# from lib.TexturePlot import *
# from lib.get_tips import *
# from lib.minimal_model import *
# from lib.intersection import *
# from lib.ProgressBar import *
# from pylab import imshow, show
# import skimage as sk
# from skimage import measure, filters
# from PIL import Image
# import imageio
# import seaborn as sns
# width = 512
# height = 512
# channel_no = 3

#From end of Identifying Odd Births and Deaths.ipynb
# _TODO(later)_ in new notebook for the hopefully minimal minutia of particle tracking...
# TODO: delete duplicates. print percent of rows deleted this way.
# TODO: track particles with trackpy
# TODO: track consistency of particle number with topological number over time
# TODO: locate any inconsistencies.  plot them.  do you see a tendency?
# TODO: can I identify spiral tips by grouping by t then by s, and keeping only even numbered groups? No! Not as long as the contour numbering fails to obey pbc
# TODO: with this cleaned data, compute the apparent birth-death rates

#TODO: use cuda via numba to do this post processing.  maybe modin. first profile how long this step actually takes.  focus on the slowest step first!
# from numba import jit, njit, vectorize, cuda, uint32, f8, uint8
# from lib.contours_to_tips import *

#some example tip files for data_dir
# data_dir = 'Data/tip_log_circle6_at_time_3012.8.csv' # 2 tips in uniform circular motion
# data_dir = 'Data/tip_log_circle6_params_changed_at_time_1800.0.csv' # chaos - sigma = 1, threshold = 0.9, no wrapping
# data_dir = 'Data/tip_log_chaos_circle6_sigma_2_at_time_3600.2.csv'  # chaos - sigma = 2, threshold = 0.9, no wrapping
# data_dir = 'Data/tip_log_chaos_circle6_sigma_2_wrapping_at_time_1800.0.csv'
# data_dir = 'Data/tip_log_chaos_circle6_sigma_3_threshold_0.8_wrapping_true_at_time_1800.0.csv'
# data_dir = 'Data/tip_log_chaos_circle6_sigma_4_threshold_0.9_wrapping_true_at_time_1800.0.csv'

####################################################
# Summarize Tip Log
####################################################
def _summarize_log(data_dir, save_dir_summary=None, verbose=True, dropna=True):
	'''description_plots is a string that will be turned into a save directory'''
	#compute the number of spiral tips
	df = pd.read_csv(data_dir)
	t_values = []
	n_values = []
	for j in range(df.t.size):
		t = df.iloc[j].t
		t_values.append(t)
		x = df.iloc[j].x
		n = x.count('.')
		n_values.append(n)
	df['n'] = n_values
	df['dn'] = df['n'].diff()
	if dropna:
		df = df.dropna().copy()
	df = df.astype({'dn': 'int32'}).copy()

	#find/select the birth/deaths values that are odd
	dn_values = np.diff(np.array(n_values))
	set_bd = list(set(dn_values))
	boo = [dn % 2 == 1 for dn in set_bd]
	set_odd = np.array(set_bd)[boo]  # the set of all odd birth/death types
	boo_odd = df.t.isnull().values
	for odd in set_odd:
		boo_odd |= (df.dn == odd)

	#a time series of when/where nontrivial birth/deaths occur
	series_bd = df.loc[df.dn != 0].dropna().dn
	num_odds = len(df[boo_odd].t.values)
	num_total = len(series_bd.values)
	percent_odd = 100*num_odds/num_total
	num_odd_events = sum(df.dn%2!=0)
	num_odd_frames = sum(df.n%2!=0)
	percent_odd_frames_that_are_events = sum(df.dn%2!=0)/sum(df.n%2!=0)

	tmin = np.min(df.t.values)
	tmax = np.max(df.t.values)

	if verbose:
		#bluf - fraction of odd births/deaths
		print(f"> odd birthdeaths occured with # of births/deaths per event = {set_odd}.<br>")
		print(f"> total # of odd birth/death events detected {num_odds}.<br>")
		print(f"> total # of birth/death events detected {num_total}.<br>")
		print(f"> the percent of birth/deaths that were odd was {percent_odd:.1f}%.<br>")

		#bluf - stability of odd spiral tip numbers
		print('')
		print(f"> # of frames with an odd # of spiral tips created/destroyed = {num_odd_events}<br>")
		print(f"> # of frames with an odd # of spiral tips existing = {num_odd_frames}<br>")
		print(f"> fraction of such frames that are odd births/deaths = {percent_odd_frames_that_are_events:.4f}<br>")

	#TODO: append ^these values to a csv
	df_summary = pd.DataFrame([{
		'data_dir': data_dir, 'tmin':tmin, 'tmax':tmax, 'set_odd':set_odd, 'num_odds':num_odds, 'num_total':num_total, 'percent_odd':percent_odd, 
		'num_odd_events':num_odd_events, 'num_odd_frames':num_odd_frames, 
		'percent_odd_frames_that_are_events':percent_odd_frames_that_are_events}])

	# discriminate odd births from odd deaths
	todd_values, nodd_values = df[boo_odd][['t','n']].values.T
	todd_b_values, nodd_b_values = df[(df.dn>0) & boo_odd][['t','n']].values.T
	todd_d_values, nodd_d_values = df[(df.dn<0) & boo_odd][['t','n']].values.T
	
	if save_dir_summary != None:
		if not os.path.exists(save_dir_summary):
			df_summary.to_csv(save_dir_summary, mode='w', header=True, index=True)
		else:
			df_summary.to_csv(save_dir_summary, mode='a', header=False, index=True)
		print(f"df_summary saved to {save_dir_summary}")

	dn_even       = df.loc[(~boo_odd) & (df.dn!=0)].dropna().dn
	dn_odd_births = df.loc[(df.dn>0) & boo_odd].dropna().dn
	dn_odd_deaths = df.loc[(df.dn<0) & boo_odd].dropna().dn

	args_plot1 = (t_values, n_values, todd_b_values, nodd_b_values, todd_d_values, nodd_d_values)
	args_plot2 = (dn_even, dn_odd_births, dn_odd_deaths)
	return df, args_plot1, args_plot2

####################################################
# Format tip log so each tip location gets a row
####################################################
# __Tip Features:__
# - **t** : time (dimensionless)
# - **x** : x coordinate (pixels)
# - **y** : y coordinate (pixels)
# - **s** : topological number (not consistent between frames and not 
#           reliable until contour numbering obeys periodic boundary conditions)

def _log_to_table(df, save_dir):
	tip_list = []
	start = time.time()# t_list, n_list, dn_list = df[['t', 'n', 'dn']].values.T
	for index, d in df.reset_index().iterrows():
		#assume d to be a pandas series instance
		x_list, y_list = literal_eval(d.x),literal_eval(d.y)
		t, n, dn = tuple(d[['t', 'n', 'dn']].values[:].T)
		for s, (xx, yy) in enumerate(zip(x_list, y_list)):
			for x,y in zip(xx,yy):
				tip_list.append((t, x, y, s, n, dn))
	print(f"{len(tip_list)} rows appended in {time.time()-start:.3f} seconds.")
	df_tips = pd.DataFrame(tip_list).rename(columns={0:'t',1:'x',2:'y',3:'s',4:'n',5:'dn'})


	print(f"the number of null observations in output dataset is {sum(sum(df_tips.isnull().values))}.")
	df_tips.to_csv(save_dir)
	print(f"tip locations saved to:\n\t{save_dir}")
	return df_tips

# process the initial data run regarding how to not detect an odd number of tips with periodic boundary conditions
def process_tip_log(data_dir, descrip = None, plot_figs=True, 
	save_summary_dir = 'Data/tip_log_summaries/odd_tip_log_summary.csv', 
	save_tip_locations_dir = None, verbose = False):
	'''if save_tip_locations_dir = None then
		save_tip_locations_dir = data_dir.replace('tip_log','tip_positions')
		save_tip_locations_dir = save_tip_locations_dir.replace('input','output')
		'''
	if descrip == None:
		descrip = data_dir[data_dir.find('_sigma'):-4]
	if save_tip_locations_dir == None:
		save_tip_locations_dir = data_dir.replace('tip_log','tip_positions')
		save_tip_locations_dir = save_tip_locations_dir.replace('input','output')

	#make a single summary entry for the entire data log, adding some global features to df_new
	df_new, args_plot1, args_plot2 = _summarize_log(data_dir, save_dir_summary=save_summary_dir, verbose=verbose, dropna=True)
	
	#call plotting functions if desired
	if plot_figs:
		t_values, n_values, todd_b_values, nodd_b_values, todd_d_values, nodd_d_values = args_plot1
		dn_even, dn_odd_births, dn_odd_deaths = args_plot2
		save_dir_fig1 = f'Figures/odd/odd_timeseries_{descrip}.png'
		save_dir_fig2 = f'Figures/odd/odd_histogram_{descrip}.png'
		fig1 = plot_odd_timeseries(t_values, n_values, todd_b_values, nodd_b_values, todd_d_values, nodd_d_values, save_dir=save_dir_fig1, figsize = (16,8), fontsize = 24)
		fig2 = plot_odd_histogram(dn_even, dn_odd_births, dn_odd_deaths, save_dir=save_dir_fig2, bins_even = 32, bins_odd = 4, figsize=(12,7), fontsize = 24, xticks=list(range(-10,10,1)))
		plt.close('all')
	#change the log to a table and save it
	df_tips = _log_to_table(df=df_new, save_dir=save_tip_locations_dir)
	return df_tips

#TODO: make a command line interface for process_tip_log
# def main():
#   save_dir = data_dir.replace('tip_log','tip_positions')

# if __name__ == '__main__':
#   main()