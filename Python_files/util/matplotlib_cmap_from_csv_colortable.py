# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:40:22 2019

@author: bramv
"""
import numpy as np
import csv
from matplotlib.colors import LinearSegmentedColormap



def cbar_interpolate(cbarlist, res, log=False): # product values must increase from bottom to top
    productvalues=cbarlist[:,0];
    colors1=cbarlist[:,1:4];
    colors2=cbarlist[:,4:7];
    colorsint=np.zeros((0,3));
    if not log: resolution=res; #Number of interpolation points in a reflectivity interval of 1 dBZ.
    else: resolution=50*res;
    for i in range(0,len(cbarlist)-1):
        if not log:
            productvalues_diff=productvalues[i+1]-productvalues[i];
        else: 
            productvalues_diff=np.log10(productvalues[i+1]/productvalues[i]);
        maxj=int(productvalues_diff*resolution); 
        for j in range(0,maxj):
            if int(colors2[i,0])==-1:
                colorsint=np.concatenate((colorsint,[colors1[i]+(colors1[i+1]-colors1[i])*j/(resolution*productvalues_diff)]));
            else:
                colorsint=np.concatenate((colorsint,[colors1[i]+(colors2[i]-colors1[i])*j/(resolution*productvalues_diff)]));                                 
    return colorsint

def read_and_process_csv_colortable(filename): #Is not ready yet. See return statement in nlradar_background.
    with open(filename, 'r') as f:
        datareader = csv.reader(f)
        datareader_text=[row for row in datareader]
                                        
        data = np.zeros((0,7))
        for row in datareader_text:
            if (len(row)==4 or len(row)==7):
                if len(row)==4: appendarray=[-1,-1,-1]
                else: appendarray=[]   
                data=np.concatenate((data,[np.concatenate([row,appendarray])])) 
    return np.array(data, dtype = 'float64')

def make_cmap_from_csv_colortable(filename, log = False, cmap_name = 'None'):
    #log specifies whether the color map should be logarithmic
    colortable_data = read_and_process_csv_colortable(filename)[::-1]
    clim = [colortable_data[0, 0], colortable_data[-1, 0]]
    colors_interpolated = cbar_interpolate(colortable_data, res = 5, log = log) / 255
    cm = LinearSegmentedColormap.from_list(cmap_name, colors_interpolated, N = len(colors_interpolated))
    return cm, clim


def crop_colorrange_cbar(full_colors, full_clim, new_clim):
    new_clim_rel = (new_clim - full_clim[0]) / (full_clim[1] - full_clim[0])
    color_indices = np.round(new_clim_rel * len(full_colors)).astype('int')
    new_colors = full_colors[color_indices[0]: color_indices[1]]
    new_cm = LinearSegmentedColormap.from_list('None', new_colors, N = len(new_colors))
    return new_cm



if __name__ == '__main__':
    cmap_Z = make_cmap_from_csv_colortable('D:/NLradar/Input_files/Color_tables/Default/colortable_Z_default.csv')
    cmap_RI = make_cmap_from_csv_colortable('D:/NLradar/Input_files/Color_tables/Default/colortable_RI_default.csv', log = True)
    print(cmap_Z, cmap_RI)