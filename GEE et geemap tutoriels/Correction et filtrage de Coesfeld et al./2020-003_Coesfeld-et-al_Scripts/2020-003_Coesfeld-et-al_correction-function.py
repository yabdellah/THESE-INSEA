# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:44:32 2020

@author: Jacqueline Coesfeld, GFZ German Research Centre for Geosciences.

This script is used to correct the zero point of DNB monthly composite files. It defines a function 
to automatically apply the correction routine to any DNB input for which the corresponding 
correction image exists, and illustrates the usage with a simple example.
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
    
    # Define height and width according to utilzed tile
    if tile == "75N180W" or tile == "75N060W" or tile == "75N060E":
        dim = (28800, 18000)
    elif tile == "00N180W" or tile == "00N060W" or tile == "00N060E":
        dim = (28800, 15600)
        
    # Bring correction to correct size
    correction = cv2.resize(corr_tile, dim, interpolation = cv2.INTER_LINEAR)
        
    return correction

#%%################################################################################################ 

# EXAMPLE

# Define directories
base_dir = 'C:/Users/jacqu/Documents/GFZ/DNB_correction/' # base directory
DNB_dir = "C:/Users/jacqu/Documents/GFZ/DNB_correction/VIIRS DNB/" # directory where DNB files are stored
corr_dir = base_dir + 'zero_correction_csv/' # directory where correction csvs are stored

# DNB file name
fname = "SVDNB_npp_20121001-20121031_75N060W_vcmcfg_v10_c201602051401.avg_rade9.tif"

# Load in original DNB file
DNB, gt = singleTifToArray(DNB_dir + fname)

# Call corrcetion function
correction = DNB_correction(corr_dir, fname)

# Subtract correction from original DNB raster
DNB_corr = DNB - correction
