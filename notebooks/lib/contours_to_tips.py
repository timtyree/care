#!/bin/bash/env/python3
#Tim Tyree
#Rappel Group
#UCSD
#8.4.2020
import pandas as pd, numpy as np
from scipy.interpolate import interp2d



#########################################################################
# Interpolating Spiral tip locations from contours determined by marching squares
#########################################################################
# contours_raw  = np.load(f'Data/buffer_time_origin.npy', image)
# contours_edge =  np.load(f'Data/buffer_time_origin_primed.npy', image)
def interp(x_values, y_values):
	"""linear interpolation for y = a*x + b.  returns a,b"""
	xbar = x_values.mean()
	ybar = y_values.mean()
	SSxy = np.dot(x_values,y_values) - x_values.size*xbar*ybar
	SSxx = np.dot(x_values,x_values) - x_values.size*xbar**2
	a = SSxy/SSxx
	b = ybar - a*xbar
	return a,b

def intersect(a1, b1, a2, b2):
	"""finds the intersection of two lines"""
	x = (b1 - b2)/(a2 - a1)
	y = a1*x + b1
	return x,y


	#  '''
	# returns list_tips, a list of spiral tips.  Returns [] for no spiral tips.
	# contours_raw is the list of contour points for cond1, contours_edge is the list of 
	# contour points with cond2''' 
	# # if not contours_raw.any():
	# #   return []  
	# #  if not contours_edge.any():
	# #   return []   
	# # if not contours_edge.any():
	# # return []   
	# #put contour sample points into a dataframe

def contours_to_tips(contours_raw, contours_edge, inc):
	'''TODO: update this with the tested pbc() periodic boundary conditions '''
	df = pd.concat([pd.DataFrame(c,columns =['y', 'x']) for c in contours_raw], axis=0)   
	df = df.reset_index()
	# df = df.reset_index(drop=True)
	#color each pixel as increasing (1) or nonincreasing (0)
	#rounding to nearest pixel  #lookup pixel on inc
	lst = []
	for i in range(len(df)):
		x = int(np.around(df.x.iloc[i]))
		y = int(np.around(df.y.iloc[i]))
		lst.append([i, inc[y,x]])
	#return indices where 0 maps to 1 or 1 maps to 0
	ids = np.argwhere(np.abs(np.diff(np.array(lst)[:,1]))==1).flatten()

	#get displacements with pbc with previous step
	df['tmp'] = (df.x.shift(-1) - df.x)
	df.loc[df.tmp<-500, 'dx']        = df['tmp']+600
	df.loc[df.tmp>500, 'dx']         = df['tmp']-600
	df.loc[(500>=df.tmp) | (df.tmp>=-500), 'dx']  = df['tmp']
	df = df.drop(columns=['tmp'])
	df['tmp'] = (df.y.shift(-1) - df.y)
	df.loc[df.tmp<-500, 'dy']        = df['tmp']+600
	df.loc[df.tmp>500, 'dy']         = df['tmp']-600
	df.loc[(500>=df.tmp) | (df.tmp>=-500), 'dy']  = df['tmp']
	df = df.drop(columns=['tmp'])
	#get distances to next neighbor
	df['ds_prev'] = np.sqrt(df.dx**2 + df.dy**2)

	#get displacements with pbc next step
	df['tmp'] = (df.x.shift(1) - df.x)
	df.loc[df.tmp<-500, 'dx']        = df['tmp']+600
	df.loc[df.tmp>500, 'dx']         = df['tmp']-600
	df.loc[(500>=df.tmp) | (df.tmp>=-500), 'dx']  = df['tmp']
	df = df.drop(columns=['tmp'])
	df['tmp'] = (df.y.shift(1) - df.y)
	df.loc[df.tmp<-500, 'dy']        = df['tmp']+600
	df.loc[df.tmp>500, 'dy']         = df['tmp']-600
	df.loc[(500>=df.tmp) | (df.tmp>=-500), 'dy']  = df['tmp']
	df = df.drop(columns=['tmp'])
	#get distances to next neighbor
	df['ds'] = np.sqrt(df.dx**2 + df.dy**2)

	#return positions for those nearest two pixels
	#also check that the distance between these two adjacent points are not bigger than np.sqrt(2) pixels (same contour condition)
	tips = df.iloc[ids].query('ds<5 and ds_prev<5').copy()

	#linear interpolation of those pixels is too dependent on the smoothing parameters (i.e. the gaussian filter)
	#Ythis can be used to get smoother results.  In order to get results robust to parameter choice, we simply average the two nearest pixels
	#put contour sample points into a dataframe for the boundary of the increasing regiob
	ef = pd.concat([pd.DataFrame(c, columns = ['y', 'x']) for c in contours_edge], axis=0)
	ef = ef.reset_index(drop=True)
	lst = []
	for i in range(len(tips)):
		#for the ith tip,
		tip    = tips.iloc[i]   
		#get ds = distances from current pixel with pbc
		tip = tips.iloc[i]
		ef['tmp'] = (ef.x - tip.x)
		ef.loc[ef.tmp<-500, 'dx']        = ef['tmp']+600
		ef.loc[ef.tmp>500, 'dx']         = ef['tmp']-600
		ef.loc[(500>=ef.tmp) | (ef.tmp>=-500), 'dx']  = ef['tmp']
		ef = ef.drop(columns=['tmp'])
		ef['tmp'] = (ef.y - tip.y)
		ef.loc[ef.tmp<-500, 'dy']        = ef['tmp']+600
		ef.loc[ef.tmp>500, 'dy']         = ef['tmp']-600
		ef.loc[(500>=ef.tmp) | (ef.tmp>=-500), 'dy']  = ef['tmp']
		ef = ef.drop(columns=['tmp'])
		#get distances to tip
		ef['ds'] = np.sqrt(ef.dx**2 + ef.dy**2)

		try:
			#return the three or six contour points closest to the change in inc
			xy12   = df.iloc[tip.name-1:tip.name+2][['x','y']].values
			#get a least squares fit for those values
			a1, b1 = interp(xy12[:,1], xy12[:,0])
			#return the three or six contour points of in closest to the change in inc
			xy34   = ef.sort_values('ds').head(4).sort_index()[['x','y']].values
			#replace inc line with a least squares fit to the six nearest values
			a2, b2 = interp(xy34[:,1], xy34[:,0])

			#return the two closest members and use linear interpolation for subpixel accuracy
			y5,x5 = intersect(a1, b1, a2, b2)
			lst.append([x5,y5])
		# except:#(RuntimeError, TypeError, NameError):
			# #return the three or six contour points closest to the change in inc
			# xy12   = df.iloc[tip.name-1:tip.name+2][['x','y']].values
			# #get a least squares fit for those values
			# a1, b1 = interp(xy12[:,1], xy12[:,0])
			# #return the three or six contour points of in closest to the change in inc
			# xy34   = ef.sort_values('ds').head(4).sort_index()[['x','y']].values
			# #replace inc line with a least squares fit to the six nearest values
			# a2, b2 = interp(xy34[:,1], xy34[:,0])
			# pass
			# #return the two closest members and use linear interpolation for subpixel accuracy
			# y5,x5 = intersect(b1, a1, b2, a2)
			# lst.append([x5,y5])
			# pass#print("Tell Tim to update linear interpolation of spiral tips with xy-->yx exception handling.")
			#TODO: if errors are thrown for dividing by zero, repeat linear interpolation with xy-->yx
			#TODO: if both of ^those yield an error, then plot and check for a perfect cross.  
			#      ff that cross exists, average the isoline contour values used
		except:
			pass

	if len(lst)==0:
		return []
	#return linearly interpolated spiral tips
	lst = np.array(lst)
	xtips = lst[:,0]
	ytips = lst[:,1]
	# print('%s seconds elapsed detecting peaks for this frame' % str(np.around(end-start,2)))

	return list(zip(xtips,ytips))


#TODO: (later) dump any preexisting ../tmp/tip_log.csv (put this in an initialization script.py)
#TODO: initialize header of ../tmp/tip_log.csv
#TODO: open log in append mode
# with open('../tmp/tip_log.csv', 'a') as f:
#     ff.to_csv(f, header=True)

#TODO: add an append xytips to file command before returning tip positions
# frameno = 791
# out_data = dict({'x':xtips,'y':ytips})


