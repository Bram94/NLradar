# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:45:51 2020

@author: bramv
"""

import numpy as np
import matplotlib.pyplot as plt
import time as pytime

data = np.random.rand(360, 1000)
n = 1000
data_outliers = np.random.randint(10, 11, n)
rows = np.random.randint(0, data.shape[0], n)
cols = np.random.randint(0, data.shape[1], n)
i = np.transpose([rows, cols])

def get_window_indices(i, window, shape, periodic = 'rows'):
    """For each index (row, column format) in i, this function returns indices for all elements within a 2D array with shape 'shape'
    that are located within a window 'window' centered at the grid cell with the given index.
    
    This is done in a specific order, where the function determines for each grid cell c in 'window' consecutively which indices
    it needs to select. This means that when len(i)=n, that the first n elements in the returned rows_window and cols_window refer
    to the first grid cell in 'window'.
    So when the returned indices are used to update values within an array 'array', then this should be coded as
    array[(rows_window, cols_window)] = np.tile(updated_values, window_size), where window_size = sum([j*2+1 for j in window]).
    
    'periodic' specifies whether periodic boundaries should be used. It can be either None, 'rows', 'cols' or 'both'. Default is 'rows',
    as should be used for radar data provided on a polar grid. If the grid is not periodic along a certain axis, then indices extending
    beyond the axis are put equal to zero or axis_length-1.
    """
    
    window = np.array(window)
    rows, cols = i[:, 0], i[:, 1]
    rows_window = [rows]
    cols_window = [cols]
    
    n_azi = len(window)
    for i in range(1, int((n_azi - 1)/2)+1):
        rows_window += [rows-i]+[rows+i]
        cols_window += [cols]*2
        
    for j in range(1, max(window)+1):
        n_azi_j = np.count_nonzero(window>=j)

        cols_window += [cols-j]*n_azi_j + [cols+j]*n_azi_j
        for n in range(2):
            rows_window += [rows]
            for i in range(1, int((n_azi_j - 1)/2)+1):
                rows_window += [rows-i] + [rows+i]

    rows_window, cols_window = np.concatenate(rows_window), np.concatenate(cols_window)
    if periodic in ('rows', 'both'):
        rows_window = np.mod(rows_window, shape[0])
    elif periodic in ('cols', 'both'):
        cols_window = np.mod(cols_window, shape[1])
    
    if periodic in (None, 'rows'):
        cols_window[cols_window >= shape[1]] = shape[1]-1
    if periodic in (None, 'cols'):
        rows_window[rows_window >= shape[0]] = shape[0]-1
    return tuple(rows_window), tuple(cols_window)
                

t = pytime.time()
window = [1,4,7,4,1]
rows, cols = get_window_indices(i, window, data.shape)

window_size = sum([j*2+1 for j in window])
print(len(data[(rows, cols)]))
# plt.imshow(data)
print(pytime.time()-t)