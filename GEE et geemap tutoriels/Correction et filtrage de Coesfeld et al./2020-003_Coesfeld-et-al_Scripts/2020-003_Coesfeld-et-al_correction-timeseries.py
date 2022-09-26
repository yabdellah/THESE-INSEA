# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:59:08 2020

@author: Jacqueline Coesfeld, GFZ German Research Centre for Geosciences.

This script is used to generate time series of corrected and uncorrected DNB 
radiance for specified locations and its visualization.
More information is available at http://doi.org/10.5880/GFZ.1.4.2020.003
"""

# LOAD IN MODULES
import os
import numpy as np
import gdal
from matplotlib import pyplot as plt
import sys
import cv2
import pandas as pd
import datetime
import matplotlib.dates as mdates
from scipy import stats

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
    
def DNB_correction(corr_dir, fname):
    """
    Function to perfom zero point correction for specified DNB raster
    
    Parameters
    ----------
    corr_dir : str
        Directory where correction csv's are stored
    fname : str
        File name of VIIRS DNB raster

    Returns
    -------
    correction : Array of float64
        DNB correction array which needs to be subtracted from original DNB array
    """
   
    # Access tile information from file name
    tile = fname[28:35]
    
    # Open corresponding zero correcion file
    corr_name = fname[0:28] + "zero_correction" + fname[35:60] + ".csv"
    corr = np.genfromtxt(corr_dir+ corr_name, delimiter=',')
    
    # CORRECTION IMAGE PREPARATION
    
    # Pad correction image with zeros
    corr_pad = np.zeros((corr.shape[0]+2, corr.shape[1]+2)) # empty image filled with zeros
    corr_pad[1:-1,1:-1] = corr # add correction image inside newly created image
    
    # Fill additional columns with real values
    corr_pad[1:-1,0] = corr[:,-1] # fill first column with values of last corr column
    corr_pad[1:-1,-1] = corr[:,0] # fill last column with values of first corr column
    corr_pad[0,:] = corr_pad[1,:] # dublicate first available row
    corr_pad[-1,:] = corr_pad[-2,:] # dublicate last available row
    
    # Slice correction image according to loaded in DNB tile
    if tile == "75N180W":
        corr_pad_tile = corr_pad[0:17,0:26]
    elif tile == "75N060W":
        corr_pad_tile = corr_pad[0:17,24:50]
    elif tile == "75N060E":
        corr_pad_tile = corr_pad[0:17,48:]
    elif tile == "00N180W":
        corr_pad_tile = corr_pad[15:,0:26]
    elif tile == "00N060W":
        corr_pad_tile = corr_pad[15:,24:50]
    elif tile == "00N060E":
        corr_pad_tile = corr_pad[15:,48:]
    
    # Double the dimension of sliced correction image in order to remove surplus pixels
    scale_percent = 200 # percent of original size
    width = int(corr_pad.shape[1] * scale_percent / 100)
    height = int(corr_pad.shape[0] * scale_percent / 100)
    dim = (width, height)
    corr_pad_tile = cv2.resize(corr_pad_tile, dim, interpolation = cv2.INTER_LINEAR) # resize image to twice the resolution
    corr_tile = corr_pad_tile[1:-1,1:-1] # remove surplus pixels
    
    # EXPAND CORRECTION IMAGE TO SIZE OF DNB RASTER
    if tile == "75N180W" or tile == "75N060W" or tile == "75N060E":
        dim = (28800, 18000)
    elif tile == "00N180W" or tile == "00N060W" or tile == "00N060E":
        dim = (28800, 15600)
    correction = cv2.resize(corr_tile, dim, interpolation = cv2.INTER_LINEAR)
        
    return correction


##################################################################################################

# INPUT ARGUMENTS

LOCAL_PC = True
GFZ_SERVER = False
# LOCAL_PC = False
# GFZ_SERVER = True

# Define directories
if LOCAL_PC:
    base_dir = 'C:/Users/jacqu/Documents/GFZ/DNB_correction/' # base directory
    countries_dir = base_dir + 'Earth_map_raster/'
    DNB2015_dir = 'C:/Users/jacqu/Documents/GFZ/Light_per_Capita/VIIRS_DNB/' 
elif GFZ_SERVER:
    base_dir = '/misc/nacht2/coesfeld/DNB_correction/'
    countries_dir =  '/misc/nacht1/Earth_map_raster/'
    DNB2015_dir = '/misc/nacht1/VIIRS_DNB/2015/'
corr_dir = base_dir + 'zero_correction_csv/' # directory where correction csvs are stored

# Define tile names and beginning and end of DNB file names
tiles = ['00N060E' , '00N060W' , '00N180W' , '75N060E' , '75N060W' , '75N180W']
DNB_start, DNB_end = DNB2015_dir+'SVDNB_npp_20150101-20151231_' , '_vcm-orm-ntl_v10_c201701311200.avg_rade9.tif'


#%%#%%#############################################################################

# CREATE TABLE OF LOCATIONS TO INSPECT  

lats = [52.690530, 52.691583, 23.087500, 21.545834,  8.812500, 
        40, 40, 12, # Oceans
        28.741667, -25.345834, 30.958333, 43.866667, # Countryside
        43.073771, -8.891115, # Village/ Town
        52.516667, 52.391667, # Urban
        62.183334] #Greenland
lons = [12.455052, 12.464729,  -8.525000, 21.404167, 168.208335,
        -40, -20, 88, # Oceans
        76.362501, 131.037501, 116.062500,  -97.303500, # Country side
        11.153816, 116.279259, # Village/Town
        13.379167, 13.062500, # Urban
        -45.458334] # Greenland
names = ["Verlust der Nacht - lit field", 
         "Verlust der Nacht - unlit field", 
         "Western Sahara",
         "Eastern Sahara",
         "Pacific Ocean",
         "Atlanic Ocean", "Atlantic Ocean", "Bay of Bengal",
         "Farmland West of New Dehli, India","Uluru / Ayers Rock, Australia", 
         "Countryside Hubei province, China","Farmland, South Dakota, USA",
         "Village of Torniella, Italy" ,"Village of Kuta, Lombok, Indonesia",
         "Berlin", "Potsdam",
         "Southern Greenland"]

index = np.arange(len(lats)) # number of rows
columns = ['name','lon' , 'lat','tile','x matrix','y matrix'] # headers
df = pd.DataFrame(index=index, columns=columns) # pandas datarfame to store values
df['name'] = names
df['lon'] = lons
df['lat'] = lats 

# Loop through annual DNB composites to extract x & y matrix location of specified lats & lons
for i in range(6):
    # Load in VIIR DNB annual composite of 2015
    DNB, gt  = singleTifToArray(DNB_start + tiles[i] + DNB_end)
    
    # Coordinate grid of DNB tile
    xmin , xmax = gt[0] , gt[0]+gt[1]*DNB.shape[1]
    ymin , ymax = gt[3]+gt[5]*DNB.shape[0] , gt[3]
    lon , lat = np.arange(xmin,xmax,gt[1]) , np.arange(ymax,ymin,gt[5])
    xc, yc = lon+gt[1]/2 , lat+gt[1]/2    
    
    # define rows in dataframe of current DNB tile
    rows = (df['lat'] < np.max(yc)) & (df['lat'] > np.min(yc)) & (df['lon'] < np.max(xc)) & (df['lon'] > np.min(xc))
    df['tile'][rows] = tiles[i] # append tile name to df

    idx = df.index[rows] # indicies of current tile
    
    # loop through all points within current tile
    for j in idx:
        X,Y = lat_to_matrix(df['lon'][j] , df['lat'][j], gt) # convert from lat lon to matrix coordinates
        df['x matrix'][j] , df['y matrix'][j] = X, Y # append coordinates to df    
    
df.to_csv(base_dir + 'locations_timeseries.csv') # save df to csv file 


#%%#############################################################################

# ADD VIIRS DNB RADIANCE TO FOR ALL LOCATIONS
# Load in each VIIRS DNB tile and access brightness value for each location

if LOCAL_PC:
    print('Working on LOCAL PC')      
    DNB_dir = base_dir + 'VIIRS DNB2/'
    
    # Define list with all VIIRS DNB files within defined directory
    DNB_list_full = os.listdir(DNB_dir) # list of all files in directory
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
    DNB_dirs = DNB_dirs[3:96] # delete non-existent subfolders
    
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

# Open dataframe with locations
df = pd.read_csv(base_dir + 'locations_timeseries.csv',  index_col=0) # point locations

# Create new df to store North Korean values
columns = ['name','tile'] # headers
df_NK = pd.DataFrame(index=np.arange(3), columns=columns) # pandas datarfame to store values
df_NK['name'] = ["North Korea (mean)", "North Korea (sum)", "Number of cloudfree pixels"]
df_NK['tile'] = "75N060E"

# Add columns of all months to dataframe
col_names = []
for i in range(2012,2020):
    for j in range(1,13):
        col_name = '%04d-%02d' %(i,j)
        col_names.append(col_name)
        df[col_name] = np.nan
        df_NK[col_name] = np.nan

# Create copies of df to store different types of values
df_pixel = df.copy()
df_area_mean = df.copy()
df_area_median = df.copy()
df_pixel_corr = df.copy()
df_area_mean_corr = df.copy()
df_area_median_corr = df.copy()
df_NK_corr = df_NK.copy()

# Remove DNB rasters from list who's tile is not utilized
DNB_list_tiles, DNB_list_NK = [], [] # empty lists to store required DNB names
cloud_list_tiles, cloud_list_NK = [], [] # empty lists to store required DNB cloud names
used_tiles = np.unique(df["tile"])
for i in range(len(DNB_list)):
    tile_i = DNB_list[i][28:35]
    if tile_i in used_tiles:
        DNB_list_tiles.append(DNB_list[i]) 
        cloud_list_tiles.append(cloud_list[i])
    if tile_i == "75N060E":
        DNB_list_NK.append(DNB_list[i]) 
        cloud_list_NK.append(cloud_list[i])        

#%%

# EXTRACT DNB RADIANCE FOR **POINT** LOCATIONS

# Loop through all DNB rasters
for YEAR in range(2012,2020):

    idx_yr = [i for i, v in enumerate(DNB_list_tiles) if  "SVDNB_npp_" + str(YEAR) in v] # all files names of defined year
    
    # load in data of previous years
    if (YEAR != 2012):        
        df_pixel = pd.read_csv(base_dir + 'processed_locations/timeseries_pixel_'+str(YEAR-1)+'.csv',  index_col=0)
        df_pixel_corr =pd.read_csv(base_dir + 'processed_locations/timeseries_pixel_corr_'+str(YEAR-1)+'.csv',  index_col=0)
        
        df_area_mean =pd.read_csv(base_dir + 'processed_locations/timeseries_area_mean_'+str(YEAR-1)+'.csv',  index_col=0)
        df_area_mean_corr= pd.read_csv(base_dir + 'processed_locations/timeseries_area_mean_corr_'+str(YEAR-1)+'.csv',  index_col=0)
        
        df_area_median=pd.read_csv(base_dir + 'processed_locations/timeseries_area_median_'+str(YEAR-1)+'.csv',  index_col=0)
        df_area_median_corr=pd.read_csv(base_dir + 'processed_locations/timeseries_area_median_corr_'+str(YEAR-1)+'.csv',  index_col=0)
        
    for i in idx_yr: # loop through rasters of specfied year
        name_i = DNB_list_tiles[i] # access i-th filename
        tile = name_i[28:35] # access tile information
        year, month = name_i[10:14] , name_i[14:16] # access date information
        date_col = year + '-' + month
        percent = (i - idx_yr[0]) / (idx_yr[-1] - idx_yr[0])*100
        print('Working on', date_col, tile, '..... %.01f'% percent, '% done')
        
        # Load in VIIRS DNB tile
        if LOCAL_PC:
            DNB, gt  = singleTifToArray(DNB_dir + name_i) # LOCAL PC
            cloud, gt = singleTifToArray(DNB_dir + cloud_list_tiles[i])
        elif GFZ_SERVER:
            DNB, gt  = singleTifToArray(DNB_dir + year + month +'/'+ name_i) # GFZ SERVER
            cloud, gt = singleTifToArray(DNB_dir + year + month +'/'+ cloud_list_tiles[i])
        
        # Correct DNB file
        correction = DNB_correction(corr_dir, name_i)
        DNB_corr = DNB - correction
        
        # Find indicies for inspected tile
        idx = df_pixel.index[df_pixel['tile'] == tile]
    
        # Loop throgh all locations for corresponding tile
        for j in idx:
            xr , yr = df_pixel['x matrix'][j] , df_pixel['y matrix'][j] # j-th x & y matrix coordiante
            if ~np.isnan(xr): # skip undefined coords
                
                # Extract radiance for single pixel
                if cloud[int(yr), int(xr)] > 1:
                    df_pixel[date_col][j] = DNB[int(yr), int(xr)]
                    df_pixel_corr[date_col][j] = DNB_corr[int(yr), int(xr)]
                
                # Extract average radiance for 5x5 pixel area    
                cloud_subset = cloud[int(yr)-2:int(yr)+3, int(xr)-2:int(xr)+3].flatten()
                cloud_idx = [cloud_subset > 1]
                
                DNB_subset = DNB[int(yr)-2:int(yr)+3, int(xr)-2:int(xr)+3].flatten()
                df_area_mean[date_col][j] = np.nanmean(DNB_subset[cloud_idx]) # append average DNB radiance to df
                df_area_median[date_col][j] = np.nanmedian(DNB_subset[cloud_idx]) # append average DNB radiance to df
                
                DNB_corr_subset = DNB_corr[int(yr)-2:int(yr)+3, int(xr)-2:int(xr)+3].flatten()
                df_area_mean_corr[date_col][j] = np.nanmean(DNB_corr_subset[cloud_idx]) # append average DNB radiance to df
                df_area_median_corr[date_col][j] = np.nanmedian(DNB_corr_subset[cloud_idx]) # append average DNB radiance to df
    
    # Export df to csv files
    df_pixel.to_csv(base_dir + 'processed_locations/timeseries_pixel_'+str(YEAR)+'.csv')
    df_pixel_corr.to_csv(base_dir + 'processed_locations/timeseries_pixel_corr_'+str(YEAR)+'.csv')
    
    df_area_mean.to_csv(base_dir + 'processed_locations/timeseries_area_mean_'+str(YEAR)+'.csv')
    df_area_mean_corr.to_csv(base_dir + 'processed_locations/timeseries_area_mean_corr_'+str(YEAR)+'.csv')
    
    df_area_median.to_csv(base_dir + 'processed_locations/timeseries_area_median_'+str(YEAR)+'.csv')
    df_area_median_corr.to_csv(base_dir + 'processed_locations/timeseries_area_median_corr_'+str(YEAR)+'.csv')


#%%###################################################################################

# EXTRACT DNB RADIANCE FOR **NORTH KOREAN** LOCATIONS

# Open country mask (to find pixels within North Koreas boundaries)
countries, gt = singleTifToArray(countries_dir + "countries_75N060E.bsq")
countries = countries[0:-1,:] # delte last row to have same dimensions as DNB tile
countries = countries.flatten() # flatten array to 1-dimentsional list
country_ID = 127 # North Korea

# Loop through all DNB rasters
for YEAR in range(2012,2020):

    idx_yr = [i for i, v in enumerate(DNB_list_NK) if  "SVDNB_npp_" + str(YEAR) in v] # all files names of defined year
    
    # load in data of previous years
    if (YEAR != 2012):    
        df_NK = pd.read_csv(base_dir + 'processed_locations/timeseries_NorthKorea_'+str(YEAR-1)+'.csv',  index_col=0)
        df_NK_corr = pd.read_csv(base_dir + 'processed_locations/timeseries_NorthKorea_corr_'+str(YEAR-1)+'.csv',  index_col=0)
        
    for i in idx_yr: # loop through rasters of specfied year
        name_i = DNB_list_NK[i] # access i-th filename
        tile = name_i[28:35] # access tile information
        year, month = name_i[10:14] , name_i[14:16] # access date information
        date_col = year + '-' + month
        percent = (i - idx_yr[0]) / (idx_yr[-1] - idx_yr[0])*100
        print('Working on', date_col, tile, '..... %.01f'% percent, '% done')
        
        # Load in VIIRS DNB tile
        if LOCAL_PC:
            DNB, gt  = singleTifToArray(DNB_dir + name_i) # LOCAL PC
            cloud, gt = singleTifToArray(DNB_dir + cloud_list_NK[i])
        elif GFZ_SERVER:
            DNB, gt  = singleTifToArray(DNB_dir + year + month +'/'+ name_i) # GFZ SERVER
            cloud, gt = singleTifToArray(DNB_dir + year + month +'/'+ cloud_list_NK[i])
        
        # Flatten rasters
        DNB, cloud = DNB.flatten(), cloud.flatten()
        
        # Correct DNB file
        correction = DNB_correction(corr_dir, name_i).flatten()
        DNB_corr = DNB - correction
    
        # Subset of only North Korean pixels
        DNB_NK = DNB[countries == country_ID]
        DNB_corr_NK = DNB_corr[countries == country_ID]
        cloud_NK = cloud[countries == country_ID]
    
        # Only use pixels with at least 2 cloud free observations
        DNB_NK_cloud = DNB_NK[cloud_NK > 1]
        DNB_corr_NK_cloud = DNB_corr_NK[cloud_NK > 1]
    
        df_NK[date_col][0] = np.nanmean(DNB_NK_cloud) # calculate mean radiance of cloud-free north Korean pixels
        df_NK[date_col][1] = np.nansum(DNB_NK_cloud) # calculate radiance sum of cloud-free north Korean pixels
        df_NK[date_col][2] = len(DNB_NK_cloud) # store number of cloud free pixels to df
        
        df_NK_corr[date_col][0] = np.nanmean(DNB_corr_NK_cloud) # calculate mean radiance of corrected cloud-free north Korean pixels
        df_NK_corr[date_col][1] = np.nansum(DNB_corr_NK_cloud) # calculate radiance sum of corrected cloud-free north Korean pixels
        df_NK_corr[date_col][2] = len(DNB_corr_NK_cloud[~np.isnan(DNB_corr_NK_cloud)]) # store number of corrected cloud free pixels to df
    
    # Export df to csv files
    df_NK.to_csv(base_dir + 'processed_locations/timeseries_NorthKorea_'+str(YEAR)+'.csv')
    df_NK_corr.to_csv(base_dir + 'processed_locations/timeseries_NorthKorea_corr_'+str(YEAR)+'.csv')
    

#%%###################################################################################

# VISUALIZATION OF TIME SERIES FOR SELECTED LOCATIONS

# Define datetime array of utilzed months
dates = []
for year in range(2012,2020): # create list of subfolders
    for month in range(1,13):
         date = datetime.datetime(year, month, 1)
         dates.append(date)
dates = dates[3:] # crop off first three months of apirl (no data for these months)

# Define plot properties
colors = plt.cm.viridis_r(np.linspace(0., 1, int(30)))
cols = [colors[4], colors[14], colors[29]]
markers = ["o", "^" , 'X']
abc_labels = ["a", "b", "c", "d", "e", "f"]

# Define data input properties
YEAR = 2019
types = ["pixel", "area_mean", "area_median"]

# SELECTED LOCATIONS
for i in range(3):
    
    df1 = pd.read_csv(base_dir + "processed_locations/timeseries_" + types[i] + "_%01d.csv" % YEAR,  index_col=0)
    df2 = pd.read_csv(base_dir + "processed_locations/timeseries_" + types[i] + "_corr_%01d.csv" % YEAR,  index_col=0)
    
    rad1 = df1.iloc[:,9:]
    rad2 = df2.iloc[:,9:]

    rad1[np.isnan(rad2)] = np.nan # remove raw locations where no correction is defined
    
    fig, axs = plt.subplots(2,3, figsize=(12,6.5))#, sharex=True)#, sharey=True)
    axs = axs.ravel()
           
    for j,k in enumerate((5,7,9,8,16,13)):
        axs[j].plot(dates, rad1.iloc[k], 'o', marker=markers[0],color= cols[1], label="Raw DNB")
        axs[j].plot(dates, rad2.iloc[k], 'o', marker=markers[1], color= cols[2], label="Corrected DNB")

        axs[j].set_title(df1['name'][k])
        axs[j].set_ylabel("DNB radiance [nW/cm²sr]") 
        
        # Calculate standard deviation for all values before 2017
        std1 = np.nanstd(rad1.iloc[k][0:57])
        std2 = np.nanstd(rad2.iloc[k][0:57])             
        axs[j].text(0.05,0.91, r'$\sigma$ = %0.2f' %std1 ,transform=axs[j].transAxes, color= cols[1])
        axs[j].text(0.05,0.83, r'$\sigma$ = %0.2f' %std2 ,transform=axs[j].transAxes, color= cols[2])
        
        if j < 4:
            axs[j].set_ylim([-0.2, 0.7])
        
        axs[j].set_xlim([datetime.datetime(2012, 1, 1)  , datetime.datetime(YEAR+1, 3, 1)])
        axs[j].set_xlabel("Year")
        axs[j].xaxis.set_major_locator(mdates.YearLocator(1))
        axs[j].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axs[j].text(-0.18,0.97,abc_labels[j]+')',transform=axs[j].transAxes,va="center", fontsize=12, weight="bold")
    
    axs[2].legend(edgecolor="none")
    
    fig.tight_layout()
    plt.savefig(base_dir + 'plots/final_result_selected_locations_'+types[i]+'.png', dpi=300, bbox_inches='tight') # save image
    plt.show()

#%%###################################################################################

# VERLUST DER NACHT SITES
df1 = pd.read_csv(base_dir + "processed_locations/timeseries_" + types[0] + "_%01d.csv" % YEAR,  index_col=0)
df2 = pd.read_csv(base_dir + "processed_locations/timeseries_" + types[0] + "_corr_%01d.csv" % YEAR,  index_col=0)
rad1 = df1.iloc[:,9:]
rad2 = df2.iloc[:,9:]

rad1[np.isnan(rad2)] = np.nan # remove raw locations where no correction is defined

fig, axs = plt.subplots(1,2, figsize=(10,3.5))#, sharex=True)#, sharey=True)
axs = axs.ravel()
for j in range(2):
    axs[j].plot(dates, rad1.iloc[j], 'o', marker=markers[0],color= cols[1], label="Raw DNB")
    axs[j].plot(dates, rad2.iloc[j], 'o', marker=markers[1], color= cols[2], label="Corrected DNB")

    axs[j].set_title(df1['name'][j])
    axs[j].set_ylabel("DNB radiance [nW/cm²sr]")    
    axs[j].set_ylim([-0.3,1.7])
    
    axs[j].set_xlim([datetime.datetime(2012, 1, 1)  , datetime.datetime(YEAR+1, 3, 1)])
    axs[j].set_xlabel("Year")
    axs[j].xaxis.set_major_locator(mdates.YearLocator())
    axs[j].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[j].text(-0.16,0.97,abc_labels[j]+')',transform=axs[j].transAxes,va="center", fontsize=12, weight="bold")
axs[1].legend(edgecolor="none")

fig.tight_layout()
plt.savefig(base_dir + 'plots/final_result_Verlust_der_Nacht_sites.png', dpi=300, bbox_inches='tight') # save image
plt.show()

#%%###################################################################################

# NORTH KOREA PLOT

# open North Korean data
df1 = pd.read_csv(base_dir + "processed_locations/timeseries_NorthKorea_2019.csv",  index_col=0)
df2 = pd.read_csv(base_dir + "processed_locations/timeseries_NorthKorea_corr_2019.csv",  index_col=0)

rad1 = df1.iloc[:,5:]
rad2 = df2.iloc[:,5:]
types_NK = ["mean", "sum of"]

# Remove months with too few pixels
rad1.iloc[1][rad1.iloc[2] < 1000] = np.nan
rad2.iloc[1][rad2.iloc[2] < 1000] = np.nan
rad1.iloc[1][np.isnan(rad1.iloc[0])] = np.nan
rad2.iloc[1][np.isnan(rad2.iloc[0])] = np.nan
rad1.iloc[1][np.isnan(rad2.iloc[1])] = np.nan

fig, axs = plt.subplots(1, figsize=(10,3.5))

j=1
axs.plot(dates, rad1.iloc[j]/1000, 'o', marker=markers[0],color= cols[1], label="Raw DNB")
axs.plot(dates, rad2.iloc[j]/1000, 'o', marker=markers[1], color= cols[2], label="Corrected DNB")
axs.set_xlabel("Year")
axs.set_ylabel("Sum of lights (arbitrary units)")
axs.set_title("North Korea")

axs.set_xlim([datetime.datetime(2012, 1, 1)  , datetime.datetime(YEAR+1, 3, 1)])
axs.xaxis.set_major_locator(mdates.YearLocator())
axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs.set_xlabel("Year")
    
axs.legend(edgecolor="none")
fig.tight_layout()
plt.savefig(base_dir + 'plots/final_result_NorthKorea.png', dpi=300, bbox_inches='tight') # save image
plt.show()

#%%###################################################################################

# INTRO PLOT
i = 0
df1 = pd.read_csv(base_dir + "processed_locations/timeseries_" + types[i] + "_%01d.csv" % YEAR,  index_col=0)
df2 = pd.read_csv(base_dir + "processed_locations/timeseries_" + types[i] + "_corr_%01d.csv" % YEAR,  index_col=0)

rad1 = df1.iloc[:,9:]
rad2 = df2.iloc[:,9:]

fig, axs = plt.subplots(1, figsize=(10,5))
for i,j  in enumerate((2,3,4)):
    axs.plot(dates, rad1.iloc[j],'o',marker=markers[i], label=df1['name'][j], color=cols[i])
axs.legend(loc="upper left",edgecolor="none")
axs.set_xlabel("Year")
axs.set_ylabel("DNB radiance [nW/cm²sr]")
plt.savefig(base_dir + 'plots/intro_timeline.png', dpi=300, bbox_inches='tight') # save image
plt.show()

xs = [2,2,3] # list of columns to use on x axes of subplots
ys = [3,4,4] # list of columns to use on y axes of subplots
fig, axs = plt.subplots(1,3, figsize=(10,3.5))
axs = axs.ravel()

for i in range(3):
    x1, x2 = rad1.iloc[xs[i]][:57] , rad1.iloc[xs[i]][57:] - 0.15
    y1, y2 = rad1.iloc[ys[i]][:57] , rad1.iloc[ys[i]][57:] - 0.15
   
    # Calculate Pearson correlation coefficient
    mask1 = ~np.isnan(x1) & ~np.isnan(y1) # remove nans in both datasets
    corr1 = stats.pearsonr(x1[mask1],y1[mask1])
    mask2 = ~np.isnan(x2) & ~np.isnan(y2) # remove nans in both datasets
    corr2 = stats.pearsonr(x2[mask2],y2[mask2])    
  
    axs[i].plot(x1, y1, 'o', marker=markers[0], label="Before 2017", color=cols[1])
    axs[i].plot(x2, y2, 'o', marker=markers[1], label="2017 and later", color=cols[2])
    axs[i].set_xlabel('DNB '+ df1['name'][xs[i]]+' [nW/cm²sr]')
    axs[i].set_ylabel('DNB '+df1['name'][ys[i]]+' [nW/cm²sr]')
    
    axs[i].set_xlim([-0.25, 0.3])
    axs[i].set_ylim([-0.25, 0.3])
        
    axs[i].text(0.96,0.08, r'$\rho$ = %0.2f' % corr1[0], transform=axs[i].transAxes,va="bottom",ha="right", color=cols[1])
    axs[i].text(0.96,0.02, r'$\rho$ = %0.2f' % corr2[0], transform=axs[i].transAxes,va="bottom",ha="right", color=cols[2])
    axs[i].text(-0.23,0.97,abc_labels[i]+')',transform=axs[i].transAxes,va="center", fontsize=12, weight="bold")
 
axs[0].legend(loc="upper left", edgecolor="none")
fig.tight_layout()
plt.savefig(base_dir + 'plots/intro_scatter.png', dpi=300, bbox_inches='tight') # save image
plt.show()
