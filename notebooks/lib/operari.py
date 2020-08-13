#! /usr/bin/env python
#Operations input/output
#Tim Tyree
#6.6.2020
#for spiral tip track processing
import os, re, sys, matplotlib.pyplot as plt, numpy as np, pandas as pd
from tkinter import filedialog, Tk
from glob import glob

def get_trailing_number(search_text):
	search_obj = re.search(r"([0-9]+)$", search_text)
	if not search_obj:
		return 0
	else:
		return int(search_obj.group(1))

def get_unique_file_name(path, width=3, ext_string='.csv'):
	'''example usage: get_unique_file_name("output")'''
	# if it doesn't exist, create
	if not os.path.exists(path):
		print("file name is unused - {}".format(path))
		return path
	
	# otherwise, increment the highest number folder in the series
	n = len(ext_string)
	dirs = glob(path[:-n] + "*")
	print(dirs)
	num_list = sorted([get_trailing_number(d) for d in dirs])
	highest_num = num_list[-1]
	next_num = highest_num + 1
	new_path = "{0}_{1:0>{2}}".format(path[:-n], next_num, width)+ext_string

	print("incremented file name is - {}".format(new_path))
	return new_path    
	
def get_unique_dir(path, width=3):
	# if it doesn't exist, create
	if not os.path.isdir(path):
		print("Creating new directory - {}".format(path))
		os.makedirs(path)
		return path

	# if it's empty, use
	if not os.listdir(path):
		print("Using empty directory - {}".format(path))
		return path

	# otherwise, increment the highest number folder in the series
	dirs = glob(path + "*")
	num_list = sorted([get_trailing_number(d) for d in dirs])
	highest_num = num_list[-1]
	next_num = highest_num + 1
	new_path = "{0}_{1:0>{2}}".format(path, next_num, width)

	print("Creating new incremented directory - {}".format(new_path))
	os.makedirs(new_path)
	return new_path

def search_for_file (currdir = os.getcwd()):
	'''#make functions for save file name, input cell frames, and input cell trajectories'''
	#TODO: eventually make this ^take cell trajectories or cell positions
	root = Tk()
	tempdir = filedialog.askopenfilename(parent=root, 
										 initialdir=currdir, 
										 title="Please select a file")#, 
										 # filetypes = (("all files","*.*")))
	root.destroy()
	if len(tempdir) > 0:
		print ("File: %s" % tempdir)
	return tempdir

def load_buffer(data_dir):
	if data_dir[-4:]=='.npy':
		txt = np.load(data_dir)
		return txt
	elif data_dir[-4:]=='.npz':
		txt = np.load(data_dir)
		txt = txt[txt.files[0]]  #only take the first buffer because there's typically one
		return txt
	else:
		print(f"\tWarning: Failed to load {data_dir}.")
		raise Exception(f"\tWarning: Failed to load {data_dir}.")

#deprecated/broken
# def get_incremented_output_filename(output_dir, output_fn):
#     path = os.path.join(output_dir, output_fn)
#     if not os.path.exists(path):
#         return path
#     string = output_fn[::-1][output_fn[::-1].find('.')+1:]
#     m = re.search(r'[0-9]+',string)
#     if not m:
#         raise('match not found and output file name already exists.')
#     else:
#         #increment the numerical match
#         num_string = m.group(0)[::-1]
#         base_string = output_fn[:output_fn.find(num_string)]
#         end_string = output_fn[output_fn.find('.'):]
#         fn =  base_string + str(eval(num_string)+1) + end_string
#         return get_incremented_output_filename(output_dir,fn)

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

# def plot_buffer(img_nxt, img_inc, contours_raw, contours_inc, tips, figsize=(15,15), max_marker_size=800, lw=2):
#     '''computes display data; returns fig.'''
#     #plot figure
#     fig, ax = plt.subplots(1,figsize=figsize)
#     ax.imshow(img_nxt,cmap='Reds', vmin=0, vmax=1)
#     ax.axis('off')

#     #plot contours, if any.  type 1 = contours_raw (blue), type 2 = contours_inc (green)
#     for n, contour in enumerate(contours_inc):
#         ax.plot(contour[:, 1], contour[:, 0], linewidth=lw, c='g', zorder=2)
#     for n, contour in enumerate(contours_raw):
#         ax.plot(contour[:, 1], contour[:, 0], linewidth=lw, c='b', zorder=2)

#     #plot tips, if any
#     s1_values, s2_values, y_values, x_values = tips
#     #     if len(n_values)>0:
#     for j in range(len(x_values)): 
#         ax.scatter(x = x_values[j], y = y_values[j], c='yellow', s=int(max_marker_size/(j+1)), zorder=3, marker = '*')
#     return fig

# def get_lifetime(trajectory_list):
#     '''trajectory_list is a list of lists.  
#     return np.mean( [ len(trajectory) for trajectory in trajectory_list ], axis=0 )'''
#     return np.mean( [ len(trajectory) for trajectory in trajectory_list ], axis=0 )
# #    TODO: for a given .csv of tip positions, make their trajectories naively in trackpy



def process_tip_log_file(input_fn, include_EP=False, include_nonlinear_EP=True):
	'''input = file of tip log. returns pandas dataframe of tips.'''
	assert(os.path.exists(input_fn))
	df_input = pd.read_csv(input_fn)
	data_list = []
	time_sig_figs = 3
	#for each row of the input log
	for j in df_input.index:
		row = df_input.iloc[j]
		t = row['t']
		x_lst = eval(row['x'])
		y_lst = eval(row['y'])
		s1_lst = eval(row['s1'])
		s2_lst = eval(row['s2'])
		if include_EP:
			if include_nonlinear_EP:
				states_nearest_lst = eval(row['states_nearest'])
				states_interpolated_linear_lst = eval(row['states_interpolated_linear'])
				states_interpolated_cubic_lst = eval(row['states_interpolated_cubic'])
			else:
				states_interpolated_linear_lst = eval(row['states_interpolated_linear'])
		n = len(x_lst)
		if not include_EP:
			for x,y,s1,s2 in zip(x_lst,y_lst,s1_lst,s2_lst):
				s = (s1,s2)
				datum = {'t': t,
				# datum = {'t': float(np.around(t, time_sig_figs)),
							'x': x,
							'y': y,
							's1': tuple(s1),
							's2': tuple(s2),
							'n':n}
				data_list.append(datum)
		else:
			if include_nonlinear_EP:
				for x,y,s1,s2, sn,sil,sic in zip(x_lst,y_lst,s1_lst,s2_lst, 
					states_nearest_lst, states_interpolated_linear_lst,
					states_interpolated_cubic_lst):
					# s = (s1,s2)
					# datum = {'t': float(np.around(t, time_sig_figs)),
					datum = {'t': t,
								'x': x,
								'y': y,
								's1': tuple(s1),
								's2': tuple(s2),
								'n':n,
								'states_nearest': tuple(sn),
								'states_interpolated_linear':tuple(sil),
								'states_interpolated_cubic':tuple(sic)
								}
					data_list.append(datum)
			else:
				for x,y,s1,s2,sil in zip(x_lst,y_lst,s1_lst,s2_lst, states_interpolated_linear_lst):
					# s = (s1,s2)
					# datum = {'t': float(np.around(t, time_sig_figs)),
					datum = {'t': t,
								'x': x,
								'y': y,
								's1': tuple(s1),
								's2': tuple(s2),
								'n':n,
								'states_interpolated_linear':tuple(sil)
								}
					data_list.append(datum)
	df_output = pd.DataFrame(data_list)
	return df_output

def make_log_folder(folder_name='Data/log-tmp/'):
	try:
		os.mkdir(folder_name)
	except:
		print('^that folder probs existed already.')


def remove_log_folder(folder_name='Data/log-tmp/'):
	try:
		os.rkdir(folder_name)
	except:
		print('^that folder probs existed already.')

def is_csv(file_name):
	return file_name[-4:]=='.csv'

def produce_one_csv(list_of_files, file_out):
   # Consolidate all csv files into one object
   result_obj = pd.concat([pd.read_csv(file) for file in list_of_files])
   # Convert the above object into a csv file and export
   result_obj.to_csv(file_out, index=False, encoding="utf-8")
   return True
  
def combine_csv_in_folder_to_one(folder_name, file_out = "../consolidated_rates.csv"):
	os.chdir(folder_name)
	# get all .csv files in the current working directory
	retval = os.system('ls')
	file_name_list = list(retval)
	# check each file if it ends in .csv before merging it
	def is_csv(file_name):
		return file_name[-4:]=='.csv'
	file_name_list = [f for f in file_name_list if is_csv(f)]
	produce_one_csv(list_of_files=file_name_list, file_out=file_out)
	return True
# def compress_log_folder_to(folder_name='Data/log-tmp/'):
#     print("DONE(see combine_csv_in_folder_to_one): make compress_log_folder_to(folder_name='Data/log-tmp/')")
#     return False

######################################################
# Convert empty IPython notebook to a sphinx doc page
######################################################
def convert_nb(nbname):
	os.system("runipy --o %s.ipynb --matplotlib --quiet" % nbname)
	os.system("ipython nbconvert --to rst %s.ipynb" % nbname)
	os.system("tools/nbstripout %s.ipynb" % nbname)

#a CLI for convert_nb
# if __name__ == "__main__":
# 	for nbname in sys.argv[1:]:
# 		convert_nb(nbname)