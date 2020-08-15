#! /usr/bin/env python
import pandas as pd, numpy as np, matplotlib.pyplot as plt

#automate the boring stuff
# from IPython import utils
import time, os, sys, re
beep = lambda x: os.system("echo -n '\\a';sleep 0.2;" * x)
if not 'nb_dir' in globals():
	nb_dir = os.getcwd()
	
#load the libraries
from lib import *

class y_axis_struct():
	def __init__(self):
		self.fields = {"x_values", "y_values", "y_err_1" "y_err_2",
		self.label = "default y_axis_struct",
		self.color = 'black'}
		return self

###########################################################
####### Plot results of birth-death data pipeline #########
###########################################################
def _get_termination_times_from_file(folder_name, file_name, verbose=False):
	#sort a df into the rows with termination times and everything else
	os.chdir(folder_name)
	df = pd.read_csv(file_name)
	na_loc = df.isna().T.any()
	df_term = df[na_loc].copy()
	termination_times = df_term['t'].values
	# df_bd = df[~na_loc].copy() 
	if verbose:
		print(f"""the mean termination time is 
				{np.mean(termination_times):.0f} ± {np.std(termination_times):.0f} ms
				""")
		print(f"""the median termination time is 
						{np.median(termination_times):.0f} ms (IQR:  {np.quantile(termination_times, 0.25):.0f} - {np.quantile(termination_times, 0.75):.0f} ms)
						""")
	return True

def plot_histogram_of_termination_times(termination_times, saving = True, fontsize = 16, savefig_dir = None, savefig_fn = None, dpi=300):
	'''plot the histogram of termination times'''
	if savefig_dir is None:
		savefig_dir = f'{nb_dir}/Figures/birth_death_analysis'
	if savefig_fn is None:
		savefig_fn = 'histogram_termination_times.png'

	fig, ax = plt.subplots(figsize=(5, 5))
	ax.hist(termination_times, bins = 3)

	#format plot
	# plt.title(f'''termination times for 9 200x200 patches''', fontsize=fontsize)
	ax.set_ylabel('freq.', fontsize=fontsize)
	ax.set_xlabel('termination times (ms)', fontsize=fontsize)
	ax.tick_params(axis='both', which='both', labelsize=fontsize)

	if not saving: 
		plt.show() 
	else:
		plt.tight_layout()
		os.chdir(savefig_dir)
		plt.savefig(savefig_fn, dpi=300)
		print(f"saved figure in \n\t{savefig_fn}.")
	return fig

def _get_birth_death_data_from_file(folder_name, file_name, verbose=False, dn_list = None, testing=True):
	'''consider passing dn_list = [-2,2]
	returns errorbar_data_list, scatter_data_list, formatting_data_list'''
	#import data
	os.chdir(folder_name)
	df = pd.read_csv(file_out)
	df.dropna(inplace=True)
	if dn_list is None:
		dn_list = sorted(set(df.dn.values))

	#compute median rates and IQR for the error bars corresponding to each category in dn_list
	errorbar_data_list = []
	for dn in dn_list:  
		#iterate over n for each dn
		df2 = df.loc[df.dn==dn].copy()
		n_list = sorted(set(df2.n.values))
		y_val_list = []
		y_err_1_list = []
		y_err_2_list = []
		for n in n_list:
			df3 = df2.loc[df2.n == n].copy()
			y_val, y_err_1, y_err_2 = df3.describe().T[['50%', '25%', '75%']].loc['rates'].values
			y_err_2 = float(y_err_2 - y_val)
			y_err_1 = float(y_val - y_err_1)
			y_val = float(y_val)
			y_val_list.append(y_val)
			y_err_1_list.append(y_err_1)
			y_err_2_list.append(y_err_2)
		# errorbar_data = (dn, n_list, y_val_list, y_err_1_list, y_err_2_list)
		errorbar_data = {'dn':dn, 
						 'n_list':n_list, 
						 'y_val_list':y_val_list, 
						 'y_err_1_list':y_err_1_list, 
						 'y_err_2_list':y_err_2_list
						}
		errorbar_data_list.append(errorbar_data)    

	#retrieve scatter plot data points for each category in dn_list
	scatter_data_list = []
	for dn in dn_list:  
		df2 = df.loc[df.dn==dn].copy()
		x_values = df2.n.values
		y_values = df2.rates.values
		scatter_data = {'dn':dn, 
						 'x_values':x_values, 
						 'y_values':y_values, 
						}
		scatter_data_list.append(scatter_data)

	#assign a color/label/other formatting to each category in dn_list
	color_list_raw = ['red', 'blue', 'green', 'orange', 'brown', 'purple']
	formatting_data_list = []
	for i, dn in enumerate(dn_list):
		formatting_data = {
			'dn': dn,
			'color': color_list_raw[i],
			'label':f'$W_{{{int(dn):+d}}}$'
		}
		formatting_data_list.append(formatting_data)

	if testing:
		assert (len(formatting_data_list) is len(scatter_data_list ))
		assert (len(formatting_data_list) is len(errorbar_data_list))
	return errorbar_data_list, scatter_data_list, formatting_data_list

def plot_histogram_of_termination_times(termination_times, saving = True, fontsize = 20, savefig_dir = None, savefig_fn = None, figsize=(6,5), use_log_scale = False, dpi=300):
	'''plot the histogram of termination times'''
	if savefig_dir is None:
		savefig_dir = f'{nb_dir}/Figures/birth_death_analysis'
	if savefig_fn is None:
		savefig_fn = 'birth_death_rates.png'

	# plot birth death rates with IQR y error bars with n on the x axis
	fig, ax = plt.subplots(figsize=figsize)
	for scatter_data, errorbar_data, formatting_data in zip(
		scatter_data_list, errorbar_data_list, formatting_data_list):

		dn, x_values, y_values = scatter_data.values()
		dn, n_list, y_val_list, y_err_1_list, y_err_2_list = errorbar_data.values()
		dn, color, label = formatting_data.values()
		yerr = np.array(list(zip(y_err_1_list,y_err_2_list))).T
		
		ax.scatter(x=x_values,y=y_values, c=color, s=10, alpha=0.5, label=label)
		ax.errorbar(n_list, y_val_list, yerr=yerr, c=color)
		
	ax.legend()

	ax.legend(loc='lower right', fontsize= fontsize-8)
	ax.tick_params(axis='both', labelsize= fontsize)
	ax.set_ylabel('birth/death rate (ms$^{-1}$)', fontsize=fontsize)
	ax.set_xlabel('n', fontsize=fontsize) 
	if use_log_scale:
		ax.set_yscale('log')
		  
	if not saving: 
		plt.show() 
	else:
		plt.tight_layout()
		os.chdir(savefig_dir)
		plt.savefig(savefig_fn, dpi=dpi)
		print(f"saved figure in \n\t{savefig_fn}.")
	return fig
