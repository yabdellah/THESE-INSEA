# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:04:07 2020

@author: Jacqueline Coesfeld, GFZ German Research Centre for Geosciences.

This script is used for extracting VIIRS DNB radiance of specified grid locations, 
removing outliers, filling missing data and smoothing it. The final monthly 
correction maps are exported as csv files.
More information is available at http://doi.org/10.5880/GFZ.1.4.2020.003
"""

# LOAD IN MODULES
import os
import numpy as np
import gdal
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import datetime
import matplotlib.dates as mdates
from scipy.ndimage.filters import gaussian_filter
import cartopy.crs as ccrs
import scipy
import random
import sys

import seaborn as sns
sns.set_style('darkgrid')

# FUNCTION DEFINITIONS

# Function to load in tiff file using gdal
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
    

# INPUT ARGUMENTS
LOCAL_PC = True
GFZ_SERVER = False
# LOCAL_PC = False
# GFZ_SERVER = True

# Define directories
if LOCAL_PC:
    base_dir = 'C:/Users/jacqu/Documents/GFZ/DNB_correction/' # base directory
elif GFZ_SERVER:
    base_dir = '/misc/nacht2/coesfeld/DNB_correction/'


# %%
# ADD VIIRS DNB RADIANCE FOR ALL LOCATIONS
# Load in each VIIRS DNB tile and access brightness value for each location

if LOCAL_PC:
    print('Working on LOCAL PC')      
    DNB_dir = base_dir + 'VIIRS DNB/'
    
    # Define list with all VIIRS DNB files within defined directory
    DNB_list_full = sorted(os.listdir(DNB_dir)) # list of all files in directory
    DNB_list = [] # new list where only DNB rasters are listed
    cloud_list = [] # new list where only number of cloud free acquisition rasters are listed
    for names in DNB_list_full: # delte all unnecessary files from file list
        if names.endswith("rade9.tif") or names.endswith("rade9h.tif"):
            DNB_list.append(names)
        elif names.endswith("cvg.tif"):
            cloud_list.append(names)
            
elif GFZ_SERVER:
    print('Working on GFZ SERVER')
    DNB_dir = '/misc/nacht1/VIIRS_DNB/'
    #DNB_dir = 'Y:/VIIRS_DNB/'
    DNB_dirs = [] 
    for i in range(2012,2020): # create list of subfolders
        for j in range(1,13):
            subfolder = DNB_dir + '%04d%02d' %(i,j) + '/'
            DNB_dirs.append(subfolder)
    DNB_dirs = DNB_dirs[3:96] # delete non-existent subfolders (first three months of 2012)
    
    # Define list with all VIIRS DNB files within defined directorie
    DNB_list = [] # new list where only DNB rasters are listed
    cloud_list = [] # new list where only number of cloud free acquisition rasters are listed
    for i in range(len(DNB_dirs)): # loop through all subfolders
        DNB_list_full = sorted(os.listdir(DNB_dirs[i])) # list of all files in directory
        for names in DNB_list_full: # delte all unnecessary files from file list
            if (names[35:43] == "_vcmcfg_"): # exclude stray light corrected rasters
                if names.endswith("rade9.tif") or names.endswith("rade9h.tif"): # exclude rasters with number of cloud free observation
                    DNB_list.append(names) 
                elif names.endswith("cvg.tif"):
                    cloud_list.append(names)

#%%

# Loop through all DNB rasters
for YEAR in range(2012,2020):
    print("NOW PROCESSING RASTERS OF YEAR %01d!" %YEAR )

    idx_yr = [i for i, v in enumerate(DNB_list) if  "SVDNB_npp_" + str(YEAR) in v] # all file names of defined year
    
    # Load in df with correction grid locations
    if YEAR == 2012:
        df2 = pd.read_csv(base_dir + 'grid_locations.csv',  index_col=0) # open dataframe of locations
    else: # load in data of previous years (after 2012)
        df2 = pd.read_csv(base_dir + 'grid_locations_median_monthly_values_'+str(YEAR-1)+'.csv',  index_col=0)

    # Create new columns for all months of specified year
    for month in range(1,13):
        col_name = '%04d-%02d' %(YEAR,month)
        df2[col_name] = np.nan
     
    for i in idx_yr: # loop through all files for specified year
        name_i = DNB_list[i] # access i-th filename
        tile = name_i[28:35] # access tile information
        year, month = name_i[10:14] , name_i[14:16] # access date information
        date_col = year + '-' + month
        percent = (i - idx_yr[0]) / (idx_yr[-1] - idx_yr[0])*100
        print('Working on', date_col, tile, '..... %.01f'% percent, '% done')
        
        # Load in VIIRS DNB tile
        if LOCAL_PC:
            DNB, gt  = singleTifToArray(DNB_dir + name_i) # LOCAL PC
            cloud, gt = singleTifToArray(DNB_dir + cloud_list[i])
        elif GFZ_SERVER:
            DNB, gt  = singleTifToArray(DNB_dir + year + month +'/'+ name_i) # GFZ SERVER
            cloud, gt = singleTifToArray(DNB_dir + year + month +'/'+ cloud_list[i])
        
        # Find indicies for inspected tile
        idx = df2.index[df2['tile'] == tile]
    
        # Loop through all locations for corresponding tile
        for j in idx:
            xr , yr = df2['x matrix'][j] , df2['y matrix'][j] # j-th x & y matrix coordiante
            if ~np.isnan(xr): # skip undefined coords
                cloud_subset = cloud[int(yr)-2:int(yr)+3, int(xr)-2:int(xr)+3].flatten()
                DNB_subset = DNB[int(yr)-2:int(yr)+3, int(xr)-2:int(xr)+3].flatten()
                cloud_idx = [cloud_subset > 1]
                
                df2[date_col][j] = np.nanmedian(DNB_subset[cloud_idx]) # append average DNB radiance of non-cloudy observations to df
            
    df2.to_csv(base_dir + 'grid_locations_median_monthly_values_'+str(YEAR)+'.csv') # export df


#%%#############################################################################

# DATA INPUT 

# Load in processed monthly grid location values
df3 = pd.read_csv(base_dir + 'grid_locations_median_monthly_values_2019.csv', index_col=0) # load in dataframe
cols = df3.columns.tolist() # column names

# Define coordinate grid
n = 28 # number of rows
m = 72 # number of columns
# create list of latitudes & longitudes
step_lat = 140 / n # 65°S until 75°N = 140
step_lon = 360 / m # 180°E until 180°W = 360
lats = np.linspace(75 - step_lat*0.5, -65 + step_lat*0.5,n) # lists of lats
lons = np.linspace(-180 + step_lon*0.5, 180 - step_lon*0.5,m) # list of lons
Y, X = np.array(np.meshgrid(lats, lons))#.T.reshape(-1,2) # matrix with all coordinate pairs
X, Y = X.T, Y.T

# Transform 2d list of DNB rad to 3d maps for each time step
months = 93 # number of inspected months (93  = April 2012 until December 2019)
img_stack = np.zeros((n,m,months)) # empty array to stack all correction images

# Loop through all months & turn each column storing monthly values to 2d arrays
for i in range(months):
    img = df3.pivot_table(values=cols[10+i], index='lat grid', columns='lon grid', dropna=False)
    img = img.iloc[::-1]
    img_stack[:,:,i] = img # append 2d array to image cube
    
# VISUAL INSPECTION OF UNPROCESSED CORRECTION IMAGES
for i in range(9):
    f = plt.figure(figsize=(13,4))
    ax = f.add_subplot(111, projection=ccrs.PlateCarree())
    ax.coastlines()
    im = plt.pcolormesh(X,Y,img_stack[:,:,i], cmap="rainbow", vmin=0,vmax=2)
    #im = plt.scatter(df3['lon grid'], df3['lat grid'], c=df3[cols[10+i]], cmap="rainbow",vmin=0,vmax=1)
    ax.set_title(cols[10+i])
    f.colorbar(im)
    plt.show()    

#%%#############################################################################

# OUTLIER REMOVAL

# PIXEL THRESHOLD DEFINITION
threshold_calculation = False
if threshold_calculation: 
    thr = np.zeros((n,m)) # empty array to store threshold values for each pixel
    for i in range(n):
        for j in range(m):
            pxl = img_stack[i,j,:93] # time series from April 2012 until December 2019
            pxl = np.copy(pxl)
            pxl[57:] = pxl[57:] - 0.15 # correct data acquired after 2017 by 0.15
            std = (np.nanpercentile(pxl, 84.1) - np.nanpercentile(pxl,15.9))/2
            cut1 = np.nanmedian(pxl) + 4* std
            cut2 = 1 
            
            thr[i,j] = np.nanmax((cut1,cut2))
    
    np.savetxt(base_dir + "pixel_thresholds.csv", thr, delimiter=",")
    plt.figure(figsize=(10,3))
    plt.imshow(thr, vmin=1, vmax=1.5)
    plt.colorbar()

# REMOVE PIXELS ABOVE DEFINED THRESHOLD
plots = False

# Define datetime object of acquisition times
dates = []
for i in range(months):
    year, month = np.array(cols)[10:][i].split('-') # break time string into pieces
    date = datetime.datetime(int(year), int(month), int(1)) # convert time to integers
    dates.append(date)
    
# Open matrix with threshold definitions
thr = np.genfromtxt(base_dir + 'pixel_thresholds.csv', delimiter=',')

count = 0
img_stack_olr = np.copy(img_stack) # img stack with outliers removed
for i in range(n): # good example: i in range(1,2) & j in range(48,52)
    for j in range(m):
        pxl = img_stack[i,j,:]
        pxl = np.copy(pxl)
        pxl[57:] = pxl[57:] - 0.15 # correct data acquired after 2017 by 0.15
                        
        if np.nanmax(pxl) > thr[i,j]:
            count = count + 1
            
            idx = (pxl > thr[i,j])
            idx = np.where(idx)[0].astype(int)
            
            if plots:
                fig, axs = plt.subplots(1,2, figsize=(8,2.5))
                axs[0].plot(dates,pxl,'o')
                axs[0].xaxis.set_major_locator(mdates.YearLocator())
                axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                axs[0].axhline(y=thr[i,j], color="orangered", linestyle="dashed")
                
                axs[1].hist(pxl)
                axs[1].axvline(x=thr[i,j], color="orangered", linestyle="dashed")
                for k in idx:
                    axs[0].plot(dates[k],pxl[k],'o', color="orangered")
                
                axs[0].text(1.1,1.1, 'Coordinates: %0.2f, %0.2f' % (X[i,j] , Y[i,j]), transform=axs[0].transAxes, ha="center")
                plt.show()

            for k in idx:
                img_stack_olr[i,j,k] = np.nan
                
print("Number of locations with outliers:", count)


# VISUAL COMPARISON OF ORIGINAL & OUTLIER REMOVED IMAGES

titles = ["Original" , "Outlier removed"]
for i in range(1):
    f = plt.figure(figsize=(13,5))
    axs_list = []
    for j in range(2):
        ax = f.add_subplot(2,1,(j+1),projection=ccrs.PlateCarree())
        axs_list.append(ax)
        ax.coastlines()
        ax.set_title(titles[j])
        if j == 0:
            im = plt.pcolormesh(X,Y,img_stack[:,:,i], cmap="rainbow", vmin=0,vmax=2)
            ax.text(0.5,1.2, cols[10+i], transform=ax.transAxes, ha="center", fontsize=13)
        elif j == 1:
            im = plt.pcolormesh(X,Y,img_stack_olr[:,:,i], cmap="rainbow", vmin=0,vmax=2)
    f.colorbar(im, ax=axs_list)
    plt.show()    
    
#%%#############################################################################

# FILLING IN DATA

img_stack_olr_fill = np.copy(img_stack_olr)
for i in range(months):
    img_i = img_stack_olr[:,:,i]
    nany, nanx = np.where(np.isnan(img_i))
    for j in range(len(nanx)):
        left, right = 8,9
        if nanx[j] < left+1: 
            sub1 = img_i[nany[j]-1:nany[j]+2, 0:nanx[j]+right] 
            sub2 = img_i[nany[j]-1:nany[j]+2, nanx[j]-left:] 
            sub = np.concatenate((sub2, sub1),axis=1)
        elif nanx[j] > img_i.shape[1] - right:
            sub1 = img_i[nany[j]-1:nany[j]+2, nanx[j]-left:] 
            sub2 = img_i[nany[j]-1:nany[j]+2, 0:(nanx[j]+right - img_i.shape[1])] 
            sub = np.concatenate((sub1, sub2),axis=1)
        else:
            sub = img_i[nany[j]-1:nany[j]+2, nanx[j]-left:nanx[j]+right]
                
        if sub.shape[0] > 0 and sub.shape[1] > 0: # skip subsets that consist only of NaNs  
            if (sub.shape[0] * sub.shape[1]) - sum(np.isnan(sub).flatten()) > 17: # skip subsets with less than 18 values
                
                img_stack_olr_fill[nany[j],nanx[j],i] = np.nanmedian(sub) # fill missing values with median of subset
                   
                    
# VISUAL COMPARISON OF ORIGINAL, OUTLIER REMOVED & FILLED IMAGES
titles = ["Original" , "Outlier removed", "Filled"]
for i in range(1):
    f = plt.figure(figsize=(13,10))
    axs_list = []
    for j in range(3):
        ax = f.add_subplot(3,1,(j+1),projection=ccrs.PlateCarree())
        axs_list.append(ax)
        ax.coastlines()
        ax.set_title(titles[j])
        if j == 0:
            im = plt.pcolormesh(X,Y,img_stack[:,:,i], cmap="rainbow", vmin=0,vmax=0.5)
            ax.text(0.5,1.15, cols[10+i], transform=ax.transAxes, ha="center", fontsize=13)
        elif j == 1:
            im = plt.pcolormesh(X,Y,img_stack_olr[:,:,i], cmap="rainbow", vmin=0,vmax=0.5)
        elif j == 2:
            im = plt.pcolormesh(X,Y,img_stack_olr_fill[:,:,i], cmap="rainbow", vmin=0,vmax=0.5)
    f.colorbar(im, ax=axs_list)
    plt.show()    
    
#%%#############################################################################

# DATA SMOOTHING

img_stack_olr_fill_flt = np.copy(img_stack_olr_fill) # filtered image stack
for t in range(months):
    img_i = img_stack_olr_fill_flt[:,:,t]
    for i in range(n):
        for j in range(m):
            if j == 0: 
                flt = (img_i[i, j]*2 + img_i[i, img_i.shape[1]-1] + img_i[i, j+1])/4
            elif j == (m-1):
                flt = (img_i[i, j]*2 + img_i[i, j-1] + img_i[i, 0])/4
            else:
                flt = (img_i[i, j]*2 + img_i[i, j-1] + img_i[i, j+1])/4
            
            img_stack_olr_fill_flt[i,j,t] = flt
            
# Comparison of original, outlier removed imgs, filled and filtered imgs
titles = ["Original" , "Outlier removed", "Filled", "Smoothed"]
for i in range(5):
    f = plt.figure(figsize=(13,10))
    axs_list = []
    for j in range(4):
        ax = f.add_subplot(4,1,(j+1),projection=ccrs.PlateCarree())
        axs_list.append(ax)
        ax.coastlines()
        ax.set_title(titles[j])
        if j == 0:
            im = plt.pcolormesh(X,Y,img_stack[:,:,i], cmap="rainbow", vmin=0,vmax=0.5)
            ax.text(0.5,1.15, cols[10+i], transform=ax.transAxes, ha="center", fontsize=13)
        elif j == 1:
            im = plt.pcolormesh(X,Y,img_stack_olr[:,:,i], cmap="rainbow", vmin=0,vmax=0.5)
        elif j == 2:
            im = plt.pcolormesh(X,Y,img_stack_olr_fill[:,:,i], cmap="rainbow", vmin=0,vmax=0.5)
        elif j == 3:
            im = plt.pcolormesh(X,Y,img_stack_olr_fill_flt[:,:,i], cmap="rainbow", vmin=0,vmax=0.5)   
    f.colorbar(im, ax=axs_list)
    plt.show()
    
#%%

# PLOT FOR PAPER
# Comparison of original and outlier removed, filled and filtered imgs
titles = ["Raw DNB dark values" , "DNB values after outlier removal, filling and smoothing"]
labels = ["a", "b"]
for i in range(40,48):
    f = plt.figure(figsize=(12,7))
    axs_list = []
    for j in range(2):
        ax = f.add_subplot(2,1,(j+1),projection=ccrs.PlateCarree())
        axs_list.append(ax)
        ax.coastlines()
        ax.set_title(titles[j])

        if j == 0:
            im = plt.pcolormesh(X,Y,img_stack[:,:,i], cmap='viridis', vmin=0,vmax=0.5)
            #ax.text(0.5,1.12, cols[10+i], transform=ax.transAxes, ha="center", fontsize=13)
        elif j == 1:
            im = plt.pcolormesh(X,Y,img_stack_olr_fill_flt[:,:,i], cmap='viridis', vmin=0,vmax=0.5)  
        ax.text(0,1.05, labels[j]+')', transform=ax.transAxes,va="center", fontsize=12, weight="bold")
    cb = f.colorbar(im, ax=axs_list)
    cb.set_label("Radiance [nW/cm²sr]")#[nanoWatts/cm²/sr]")
    plt.savefig(base_dir + 'plots/correction_processing/correction_processing_'+cols[10+i]+'.png', dpi=300, bbox_inches='tight') # save image
    plt.show()

#%%

# EXPORT CORRECTION IMAGE TO CSV FILES

# Create list of all DNB file names
#DNB_dir = '/misc/nacht1/VIIRS_DNB/'
DNB_dir = 'Y:/VIIRS_DNB/'
DNB_dirs = [] 
for i in range(2012,2020): # create list of subfolders
    for j in range(1,13):
        subfolder = DNB_dir + '%04d%02d' %(i,j) + '/'
        DNB_dirs.append(subfolder)
DNB_dirs = DNB_dirs[3:96] # delete non-existent subfolders

# Define list with all VIIRS DNB files within defined directorie
DNB_list = [] # new list where only DNB rasters are listed
for i in range(len(DNB_dirs)): # loop through all subfolders
    DNB_list_full = sorted(os.listdir(DNB_dirs[i])) # list of all files in directory
    for names in DNB_list_full: # delte all unnecessary files from file list
        if (names[35:43] == "_vcmcfg_"): # exclude stray light corrected rasters
            if names.endswith("rade9.tif") or names.endswith("rade9h.tif"): # exclude rasters with number of cloud free observation
                DNB_list.append(names) 

# Loop through all correction slices and export each to csv file
for i in range(months):
    fname = DNB_list[i*6]
    print(fname)
    start, middle1, middle2, end = fname[0:28],"zero_correction", fname[35:60], ".csv"
    outname = (start + middle1 + middle2 + end)
    
    np.savetxt(base_dir + "zero_correction_csv/"+ outname , img_stack_olr_fill_flt[:,:,i], delimiter=",")
    
