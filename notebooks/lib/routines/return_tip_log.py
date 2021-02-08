#! /usr/bin/env python
# Generate Birth Death Rates for Given Initial Conditions
# Tim Tyree
# 1.13.2021
from .. import *
from ..measure.measure import *
#^this import is not the best practice. however, minimalism reigns the pythonic.
import numpy as np, pandas as pd, matplotlib.pyplot as plt


from ..my_initialization import *
from ..controller.controller_LR import *#get_one_step_explicit_synchronous_splitting as get_one_step
# from ..model.LR_model_optimized import *
from ..utils.utils_traj import *
from ..utils.stack_txt_LR import stack_txt, unstack_txt
from ..routines.bdrates import *
from ..measure.utils_measure_tips_cpu import *
from ..viewer import *
import trackpy
from ..utils import get_txt
from ..model.LR_model_optimized import *
from ..controller.controller_LR import *#get_one_step_explicit_synchronous_splitting as get_one_step
from ..model.LR_model_optimized_w_Istim import *

# from ..utils.get_txt import load_buffer

#automate the boring stuff
# from IPython import utils
import time, os, sys, re

def return_tips_from_txt(txt, h, tmax, V_threshold,dsdpixel,
	tmin_early_stopping, save_every_n_frames, **kwargs):
	'''generates a log of tip locations on 2D grid with periodic boundary conditions.
	default key word arguments are returned by lib.routines.kwargs.get_kwargs(initial_condition_dir).'''
	nb_dir=os.getcwd()
	level1 = V_threshold
	level2 = 0.
	dt=h
	#initialize records
	t = 0.
	dict_out_lst = []
	num_steps = int(np.around((tmax)/dt))
	#initialize simulation
	# zero_txt = txt.copy()*0.
	width, height, channel_no = txt.shape
	#precompute anything that needs precomputing
	compute_all_spiral_tips= get_compute_all_spiral_tips(mode='simp',width=width,height=height)
	dt, one_step_map = get_one_step_map(nb_dir,dt,dsdpixel,width,height,**kwargs)
	txt_Istim_none=np.zeros(shape=(width,height), dtype=np.float64, order='C')
	# kwargs.update({'width':width,'height':height})

	while (t<tmin_early_stopping):
	    one_step_map(txt,txt_Istim_none)
	    t+=dt
	t=np.around(t,decimals=1)

	#check for any tips being present
	inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt)
	img=inVc[...,0]
	dimgdt=dVcdt[...,0]
	dict_out=compute_all_spiral_tips(t,img,dimgdt,level1,level2)#,width=width,height=height)
	n_tips=dict_out['n']#skip this trial if no spiral tips are present
	if n_tips==0:
	    dict_out_lst.append(dict_out)
	# n_tips=1 #to initialize loop invarient (n_tips > 0) to True

	##########################################
	#run the simulation, measuring regularly
	##########################################
	t_values=np.arange(t,tmax,dt)
	for step_count,t in enumerate(t_values):
		if step_count%save_every_n_frames == 0:
			#update texture namespace
			inVc,outVc,inmhjdfx,outmhjdfx,dVcdt=unstack_txt(txt)
			img=inVc[...,0]
			dimgdt=dVcdt[...,0]
			#compute tip locations in dict_out
			dict_out=compute_all_spiral_tips(t,img,dimgdt,level1,level2)#,width=width,height=height)
			#save tip data
			n_tips=dict_out['n']
			if n_tips>0:
				dict_out_lst.append(dict_out)
		if n_tips==0:
			break
		#forward integration in time
		one_step_map(txt,txt_Istim_none)

	if len(dict_out_lst)==0:
		#no spiral tips observed, print that this ic yielded no spiral tips
		return None
	else:
		df = pd.concat([pd.DataFrame(dict_out) for dict_out in dict_out_lst])
		df.reset_index(inplace=True, drop=True)
		#if any spiral tips have been observed
		#if the end of AF was indeed reachded, append a row recording this
		if n_tips==0:
			next_id = df.index.values[-1]+1
			df = pd.concat([df,pd.DataFrame({'t': float(save_every_n_frames*h+t),'n': int(n_tips)}, index = [next_id])])
		#return the recorded data
		return df.astype('float32')

if __name__=='__main__':
	from ..utils import get_txt
	nb_dir='/Users/timothytyree/Documents/GitHub/care/notebooks/lib/routines'
	txt=get_txt(3,0,100,100,nb_dir)
	# width,height,chno=txt.shape
	df=return_tips_from_txt(
	    txt=txt,
	    h=0.1,
	    tmax=150,
	    V_threshold=-50,
	    dsdpixel=0.025,
	    tmin_early_stopping=100,
	    save_every_n_frames=40)#,
	print(df.head())
