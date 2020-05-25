# import time

# start = time.time()
import numpy as np
import pandas as pd
import skimage as sk
from skimage import measure

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
# end = time.time()
# print('%s seconds elapsed loading packages' % str(np.around(end-start,2)))

# start = time.time()

# #ASSUMING the /mat/red cmap is in use
# canvas_dir = '../data/redsample000792.png'
# canvas_old_dir = '../data/redsample000788.png'
# img = sk.io.imread(canvas_dir, as_gray=False)
# img_old = sk.io.imread(canvas_old_dir, as_gray=False)

# # map an rgb from '/mat/red' cmap to it's scalar value between 0 and 1 by just selecting the red channel"""
# image = img[:,:,0]/255
# image_old = img_old[:,:,0]/255

#detect whether a pixel is increasing or decreasing.  
ifilter = np.vectorize(lambda x,x_old: 1. if x>x_old else 0.)
inc = ifilter(image,image_old) #mask of increasing cells between a few frames

#get contours by a marching squares algorithm (which can be parallelized to increase performance in theory)
ithresh = 0.3
contours_raw = measure.find_contours(image, level=ithresh, fully_connected='low', positive_orientation='low')
contours_edge = measure.find_contours(inc, 0.5)

#put contour sample points into a dataframe
df = pd.concat([pd.DataFrame(c, columns = ['y', 'x']) for c in contours_raw], axis=0)
df = df.reset_index(drop=True)
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
df.loc[df.tmp<-500, 'dx']        = df['tmp']+512
df.loc[df.tmp>500, 'dx']         = df['tmp']-512
df.loc[(500>=df.tmp) | (df.tmp>=-500), 'dx']  = df['tmp']
df = df.drop(columns=['tmp'])
df['tmp'] = (df.y.shift(-1) - df.y)
df.loc[df.tmp<-500, 'dy']        = df['tmp']+512
df.loc[df.tmp>500, 'dy']         = df['tmp']-512
df.loc[(500>=df.tmp) | (df.tmp>=-500), 'dy']  = df['tmp']
df = df.drop(columns=['tmp'])
#get distances to next neighbor
df['ds_prev'] = np.sqrt(df.dx**2 + df.dy**2)

#get displacements with pbc next step
df['tmp'] = (df.x.shift(1) - df.x)
df.loc[df.tmp<-500, 'dx']        = df['tmp']+512
df.loc[df.tmp>500, 'dx']         = df['tmp']-512
df.loc[(500>=df.tmp) | (df.tmp>=-500), 'dx']  = df['tmp']
df = df.drop(columns=['tmp'])
df['tmp'] = (df.y.shift(1) - df.y)
df.loc[df.tmp<-500, 'dy']        = df['tmp']+512
df.loc[df.tmp>500, 'dy']         = df['tmp']-512
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
    ef.loc[ef.tmp<-500, 'dx']        = ef['tmp']+512
    ef.loc[ef.tmp>500, 'dx']         = ef['tmp']-512
    ef.loc[(500>=ef.tmp) | (ef.tmp>=-500), 'dx']  = ef['tmp']
    ef = ef.drop(columns=['tmp'])
    ef['tmp'] = (ef.y - tip.y)
    ef.loc[ef.tmp<-500, 'dy']        = ef['tmp']+512
    ef.loc[ef.tmp>500, 'dy']         = ef['tmp']-512
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
    except (RuntimeError, TypeError, NameError):
        print("Tell Tim to update linear interpolation of spiral tips with xy-->yx exception handling.")
        #TODO: if errors are thrown for dividing by zero, repeat linear interpolation with xy-->yx
        #TODO: if both of ^those yield an error, then plot and check for a perfect cross.  
        #      ff that cross exists, average the isoline contour values used

#return linearly interpolated spiral tips
lst = np.array(lst)
xtips = lst[:,0]
ytips = lst[:,1]
# end = time.time()
print('%s seconds elapsed detecting peaks for this frame' % str(np.around(end-start,2)))


#TODO: (later) dump any preexisting ../tmp/tip_log.csv (put this in an initialization script.py)
#TODO: initialize header of ../tmp/tip_log.csv
#TODO: open log in append mode
# with open('../tmp/tip_log.csv', 'a') as f:
#     ff.to_csv(f, header=True)

#TODO: add an append xytips to file command before returning tip positions
frameno = 791
out_data = dict({'frame':frameno+0*xtips,'x':xtips,'y':ytips})

#TODO: make ^that append statement appear at the end of a callable script that I can import into a separate ipython notebook.
#TODO: update all directories to search explicitely from the current script directory.  A more 'hacky' solution is to just update the directories for the new script destination within the app.

# out_data = np.vstack([frameno+0*xtips,xtips,ytips])

ff = pd.DataFrame(out_data)

with open('../tmp/tip_log.csv', 'a') as f:
    ff.to_csv(f, header=True, index=False)
# gf = pd.read_csv('../tmp/tip_log.csv', index_col=0)