# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:11:59 2020

@author: Jacqueline Coesfeld, GFZ German Research Centre for Geosciences.

This script is used for selecting grid locations and exporting them to a csv file.
More information is available at http://doi.org/10.5880/GFZ.1.4.2020.003
"""

# LOAD IN MODULES
import numpy as np
import gdal
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

import seaborn as sns
sns.set_style('darkgrid')

# FUNCTION DEFINITIONS

# function to load in tiff file using gdal
def singleTifToArray(inRas):
    ds = gdal.Open(inRas, gdal.GA_ReadOnly)   
    cols = ds.RasterXSize
    rows = ds.RasterYSize        
    array = ds.ReadAsArray(0, 0, cols, rows) 
    gt = ds.GetGeoTransform()
    return array , gt
    
# Function to convert latitude and longitude to matrix
def lat_to_matrix(x, y , gt):
    X = int(round((x - gt[0] - gt[1]/2 ) / gt[1]))
    Y = int(round((y - gt[3] - gt[5]/2 ) / gt[5]))
    return X, Y

# Define directories

# LOCAL PC
base_dir = 'C:/Users/jacqu/Documents/GFZ/DNB_correction/'
GHS_dir = 'C:/Users/jacqu/Documents/GFZ/Light_per_Capita/GHS_POP_tiles/' # this is still the old one
DNB2015_dir = 'C:/Users/jacqu/Documents/GFZ/Light_per_Capita/VIIRS_DNB/' 
DNB_dir = base_dir + 'VIIRS DNB/'

# Define tile names and beginning and end of GHS file names
tiles = ['00N060E' , '00N060W' , '00N180W' , '75N060E' , '75N060W' , '75N180W']
GHS_start , GHS_end = GHS_dir+'GHS_POP_E2015_GLOBE_30ss_res_norm_' , '_Q.tif'
DNB_start, DNB_end = DNB2015_dir+'SVDNB_npp_20150101-20151231_' , '_vcm-orm-ntl_v10_c201701311200.avg_rade9.tif'


# %%
# GENERATE POINTS - EQUALLY SPACED GRID

n = 28 # number of rows
m = 72 # number of columns
sigmas = [4, 20, 100, 20] # sigmas for gaussian filter

index = np.arange(n*m) # number of points
columns = ['x matrix' , 'y matrix' , 'lon' , 'lat' , 'lon grid' , 'lat grid' , 'tile'] # headers for df
df = pd.DataFrame(index=index, columns=columns) # pandas datarfame to store values

# create list of latitudes & longitudes
step_lat = 140 / n # 65°S until 75°N = 140
step_lon = 360 / m # 180°E until 180°W = 360
lats = np.linspace(75 - step_lat*0.5, -65 + step_lat*0.5,n) # lists of lats
lons = np.linspace(-180 + step_lon*0.5, 180 - step_lon*0.5,m) # list of lons
coords = np.array(np.meshgrid(lats, lons)).T.reshape(-1,2) # matrix with all coordinate pairs
# add lat & lon entries to dataframe
df['lat'] = coords[:,0]
df['lon'] = coords[:,1]
df['lon grid'] = coords[:,1]
df['lat grid'] = coords[:,0]

# Text strings for plots
titles = ['GHS', r'GHS, $\sigma$=%0d' % sigmas[0], r'GHS, $\sigma$=%0d' % sigmas[1], 
          r'GHS, $\sigma$=%0d' % sigmas[2],'DNB', r'DNB, $\sigma$=%0d' % sigmas[3], 'Sum']
vmaxs = [1,1,1,1,2,2,7] # color maxima for subplots
com_stack = []

# OPTIMIZE LOCATION FOR SPECIFIC POINTS
for i in range(6): # loop through all tiles        
    # Open GHS raster
    print('Working on tile' , tiles[i] , '......')
    GHS, gt  = singleTifToArray(GHS_start + tiles[i] + GHS_end)
    GHS[GHS > 0] = 1 # create binary raster with 1 where there is population & 0 where there isn't
    GHS[GHS < 0] = 0
    
    # Load in VIIR DNB annual composite of 2015
    DNB, gt  = singleTifToArray(DNB_start + tiles[i] + DNB_end)
    DNB[DNB > 10] = 10 # turn all values above 10 to 10
    DNB = DNB / 5 # divide DNB by in order to have a maximum value of 2
       
    # Coordinate grid of DNB/GHS tile
    xmin , xmax = gt[0] , gt[0]+gt[1]*GHS.shape[1]
    ymin , ymax = gt[3]+gt[5]*GHS.shape[0] , gt[3]
    lon , lat = np.arange(xmin,xmax,gt[1]) , np.arange(ymax,ymin,gt[5])
    xc, yc = lon+gt[1]/2 , lat+gt[1]/2
    
    # define rows in dataframe of current DNB/GHS tile
    rows = (df['lat'] < np.max(yc)) & (df['lat'] > np.min(yc)) & (df['lon'] < np.max(xc)) & (df['lon'] > np.min(xc))
    df['tile'][rows] = tiles[i] # append tile name to df
    
    idx = df.index[rows] # indicies of current tile
    
    # loop through all points within current tile
    for j in idx:
        X,Y = lat_to_matrix(df['lon'][j] , df['lat'][j], gt) # convert from lat lon to matrix coordinates
        df['x matrix'][j] , df['y matrix'][j] = X, Y # append coordinates to df
        
        r = 250 # radius around inspected pixel
        subset = GHS[Y-r : Y+r , X-r : X+r] # create subset with radius r around inspected pixel  
        subset_DNB = DNB[Y-r : Y+r , X-r : X+r]
        
        if j == 155:
            subset[:,:] = 0 #(ID 155 is an exception since pop is at 0.0000004)
        if (np.sum(subset) != 0): # if entire subset has population inside modify position
            df['x matrix'][j] = np.nan # reset coordinates for all locations
            df['y matrix'][j] = np.nan 
            df['lat'][j] = np.nan 
            df['lon'][j] = np.nan 
                    
            # turn corners into maximum values (1 for GHS, 2 for DNB) in order to move new pixel location towards center
            cut=10 # corner width
            subset[0:r*2,0:cut] = 1
            subset[0:cut,0:r*2] = 1
            subset[0:r*2, r*2-cut:r*2] = 1
            subset[r*2-cut:r*2 , 0:r*2] = 1    
            
            subset_DNB[0:r*2,0:cut] = 2
            subset_DNB[0:cut,0:r*2] = 2
            subset_DNB[0:r*2, r*2-cut:r*2] = 2
            subset_DNB[r*2-cut:r*2 , 0:r*2] = 2 
        
            # use gaussian filter in order to smooth subset with differnt sigma values
            blurred = np.zeros((subset.shape[0],subset.shape[1],len(sigmas))) # empty array to store data
            for k in range(len(sigmas)-1): # loop through defined sigma values
                blurred[:,:,k] = gaussian_filter(subset, sigma=sigmas[k]) # filtered GHS subsets
            blurred[:,:,k+1] = gaussian_filter(subset_DNB, sigma=sigmas[-1]) # filtered DNB subset
            
            # determine new coordinate
            sumj = subset + blurred[:,:,0] + blurred[:,:,1]+ blurred[:,:,2] + subset_DNB + blurred[:,:,3] # calculate sum of all datasets
            y_new, x_new = np.where(sumj == np.min(sumj)) # find coordinate with lowest value (in sum dataset)
            x_new, y_new = x_new[0], y_new[0] # if minimum is found in multiple pixels, take the first one
            
            # Save sum of all subsets for example location (for illustration in paper)
            if (j==450) or (j== 1750) or (j==183) or (j==522) or (j==539) or j==(1394):
                 np.savetxt(base_dir + "grid_locations_subsets_csv/grid_locations_subset_ID_%01d.csv" % j, sumj, delimiter=",")
            
            # Create plot for example location showing all 6 subsets (for illustration in paper) 
            if (j == 621):
                 # stack all subplots to 3 dimensional data cube (for plotting purposes)
                 stack = np.stack((subset, blurred[:,:,0],blurred[:,:,1],blurred[:,:,2], subset_DNB, blurred[:,:,3], sumj))
                
                 label= ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
                
                 fig, axs = plt.subplots(2,4, figsize=(12,7.7))#, sharey=True)
                 axs = axs.ravel()
                
                 rad = 100 # radius around new coordinate for zoom plot
                 if x_new < rad or y_new < rad: # if new point is close to the edge, reduce radius for zoomed plot
                     rad = np.nanmin((x_new, y_new))
                 for k in range(8): # loop through all 8 subplots
                     if k == 7: # in last sublot show zoomed region around new point
                         zoom = stack[k-1,:,:][int(y_new)-rad:int(y_new)+rad+1, int(x_new)-rad:int(x_new)+rad+1]
                         im = axs[k].imshow(zoom, cmap="viridis",norm = LogNorm())
                         axs[k].plot(rad+1, rad+1, marker=r'$\ast$', markersize=10, color='orangered') # add new pixel location to last subplot
                         axs[k].set_xticks([])
                         axs[k].set_yticks([])
                         axs[k].set_title("Sum (zoom)")
                     else: # all other subplots cover same extent
                         im = axs[k].imshow(stack[k,:,:], vmin=0, cmap="viridis")#vmax=vmaxs[k] # show k-th subset
                         # Define coordinate grid
                         axs[k].tick_params(reset=False,labelcolor='k', labelsize='small', width=1, length=2, 
                                     grid_color='grey',color='grey', direction='out', grid_linewidth=0.5)
                         if df['lat grid'][j] > 0:
                             ylabel = "%0.1f °N" % df['lat grid'][j]
                         elif df['lat grid'][j] < 0:
                             ylabel = "%0.1f °S" % abs(df['lat grid'][j])
                         if df['lon grid'][j] > 0:
                             xlabel = "%0.1f °E" % df['lon grid'][j]
                         elif df['lon grid'][j] < 0:
                             xlabel = "%0.1f °W" % abs(df['lon grid'][j])
                         axs[k].set_xticks([250])
                         axs[k].set_yticks([250])
                         axs[k].set_xticklabels([xlabel])
                         axs[k].set_yticklabels([ylabel], rotation=90, va='center')
                         axs[k].set_title(titles[k]) # add title to subplot
                     axs[k].text(-0.09,0.93,label[k]+')',transform=axs[k].transAxes, fontsize=12,ha="center", weight="bold")
                     cb = fig.colorbar(im, ax=axs[k], orientation="horizontal", aspect=10, pad=0.1)
                    
                 axs[6].plot(x_new, y_new, marker=r'$\ast$', markersize=10, color='orangered') # add new pixel location to last subplot
                 x1,x2, y1, y2 = x_new-rad , x_new+rad, y_new-rad , y_new+rad,
                 axs[6].plot((x1,x2,x2,x1,x1), (y1,y1,y2,y2,y1), color='black')
    
                 plt.savefig(base_dir + 'plots/subsets/subset_ID_%01d.png' % j, dpi=300, bbox_inches='tight') # save image
                 plt.show()
            
            # update df with new coodinates
            if np.min(sumj) < 4: # if lowest value of sum is 4 everything is populated & we exclude location completely
                df['x matrix'][j] = X -r + x_new # append new x & y coordinades
                df['y matrix'][j] = Y -r + y_new
                df['lat'][j] = yc[df['y matrix'][j]] # fill in converted new lat & lon
                df['lon'][j] = xc[df['x matrix'][j]]
       
    print('Finished with current tile')
print('FINISHED!!!!')  

df.to_csv(base_dir + 'grid_locations.csv') # save processed df to csv file   

#%%################################################################################################ 

# PLOT OF SELECTED SUBSET LOCATIONS

df = pd.read_csv(base_dir + 'grid_locations.csv',  index_col=0)
label= ['a', 'b', 'c', 'd', 'e', 'f']
titles = ["Chicago, USA", "Argentina", "Sweden", "Eastern USA", "Spain", u"São Paulo, Brazil"]

idx = [450, 1750, 183, 522, 539, 1394]
fig, axs = plt.subplots(2,3, figsize=(10,5.5))
axs = axs.ravel()
for i in range(6):
    j = idx[i]
    subset = np.genfromtxt(base_dir + "grid_locations_subsets_csv/grid_locations_subset_ID_%01d.csv" % j, delimiter=",")
    y_new, x_new = np.where(subset == np.min(subset)) # find coordinate with lowest value (in sum dataset)
    
    im = axs[i].imshow(subset, norm = LogNorm(0.1,10), cmap="viridis") # show j-th sum
    axs[i].plot(x_new, y_new, marker=r'$\ast$', markersize=8, color='orangered') # add new pixel location 
    axs[i].text(-0.11,0.93,label[i]+')',transform=axs[i].transAxes,va="center", fontsize=12, weight="bold")
    axs[i].set_title(titles[i], fontsize=11) #add title to subplot
    
    # Define coordinate grid
    axs[i].tick_params(reset=False,labelcolor='k', labelsize='small', width=1, length=2, 
                grid_color='grey',color='grey', direction='out', grid_linewidth=0.5)
    if df['lat grid'][j] > 0:
        ylabel = "%0.1f °N" % df['lat grid'][idx[i]]
    elif df['lat grid'][j] < 0:
        ylabel = "%0.1f °S" % abs(df['lat grid'][j])
    if df['lon grid'][j] > 0:
        xlabel = "%0.1f °E" % df['lon grid'][j]
    elif df['lon grid'][j] < 0:
        xlabel = "%0.1f °W" % abs(df['lon grid'][j])
    axs[i].set_xticks([250])
    axs[i].set_yticks([250])
    axs[i].set_xticklabels([xlabel])
    axs[i].set_yticklabels([ylabel], rotation=90, va='center')
cb = fig.colorbar(im, ax=axs.flat)  
cb.set_label('Unsuitability factor') 
plt.savefig(base_dir + 'plots/subset_comparison_with_titles.png', dpi=300, bbox_inches='tight') # save image
plt.show()