# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:36:09 2018
@author: bramv
"""

import cv2
import numpy as np
from scipy.misc import imresize
import os
opa = os.path.abspath
import time as pytime
import nlr_functions as ft

t = pytime.time()

directory = 'D:/NLradar/Input_files/'
mapname = 'Benelux_map2'
if True:
    array = cv2.imread(directory+mapname+'.jpg')[:,:,::-1]
    
#resized_array = imresize(array, (5000,5000))
#cv2.imwrite(directory+'/EU_map_resized.jpg', resized_array[:,:,::-1])
#
    
file_jgw = opa(os.path.join(directory,mapname+'.jgw'))
with open(file_jgw, 'r') as f:
    data = f.read()
    data = np.array(ft.string_to_list(data, '\n'), dtype='float')
    lon_0 = data[4]
    lat_1 = data[5]
    x_scale = data[0]
    y_scale = data[3]
    lon_1 = lon_0+x_scale*array.shape[1]
    lat_0 = lat_1+y_scale*array.shape[0]
    
resize_fac = (4,2,1) #In degree
tile_size = (500,500)
if True:
    layer_start = 5
    for k in range(len(resize_fac)):
        subdir = 'Layer_'+str(layer_start+k)+'/'
        os.makedirs(directory+subdir, exist_ok = True)
            
        arr_tile_size = (np.array(tile_size)*resize_fac[k]).astype(int)
        ni, nj = (np.array(array.shape[:2])/arr_tile_size).astype('int')
        for i in range(ni):
            for j in range(nj):
                print(i, j)
                tile = array[i*arr_tile_size[0]:(i+1)*arr_tile_size[0],j*arr_tile_size[1]:(j+1)*arr_tile_size[1]]
                tile = imresize(tile, tile_size)
                lat_range = lat_1+np.array([i+1, i])*y_scale*arr_tile_size[0]
                lon_range = lon_0+np.array([j, j+1])*x_scale*arr_tile_size[1]
                cv2.imwrite(directory+subdir+str(lat_range[0])+'_'+str(lat_range[1])+'_'+str(lon_range[0])+'_'+str(lon_range[1])+'.jpg', tile[:,:,::-1]) #Save as BGR, because that is the format used by opencv

if False:
    read_tiles = []
    ni, nj = 20, 10
    tiles_combined = np.zeros((ni*tile_size[0],nj*tile_size[1],3), dtype='uint8')
    print(pytime.time()-t)
    for i in range(ni):
        for j in range(nj):
            tiles_combined[i*tile_size[0]:(i+1)*tile_size[0], j*tile_size[1]:(j+1)*tile_size[1]] = cv2.imread('D:/NLradar/Input_files/'+str(i)+str(j)+'.jpg')
    tiles_combined = tiles_combined[:,:,::-1] #To RGB
    print(tiles_combined.shape)
    print(pytime.time()-t)