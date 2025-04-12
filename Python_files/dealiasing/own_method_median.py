# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:02:06 2018

@author: bramv
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import time as pytime

def roll(arr, axis, shift):
    #For 2D arrays
    if axis==0:
        return np.concatenate([arr[-shift:],arr[:-shift]],axis)
    else:
        return np.concatenate([arr[:,-shift:],arr[:,:-shift]],axis)
    
def add_0(arr, axis, n): #n negative for appending on left side
    #For 2D arrays
    if axis==0:
        if n<0:
            return np.concatenate([np.zeros((abs(n),arr.shape[1])),arr],axis)
        else:
            return np.concatenate([arr,np.zeros((abs(n),arr.shape[1]))],axis)
    else:
        if n<0:
            return np.concatenate([np.zeros((arr.shape[0],abs(n))),arr],axis)
        else:
            return np.concatenate([arr,np.zeros((arr.shape[0],abs(n)))],axis)

def get_window_sum(arr, n_azi, n_rad): #the window size is given by (2*n_azi+1) * (2*n_rad+1)
    azi_sum = arr.copy()
    for i in range(1, n_azi+1):
        azi_sum += roll(arr, 0, -i) + roll(arr, 0, i)
    
    window_sum = azi_sum.copy()
    for j in range(1, n_rad+1):
        window_sum += roll(azi_sum,1,-j) + roll(azi_sum,1,j)
    
    return window_sum

def get_window_median(arr, n_azi, n_rad):
    window_arrays = []
    for i in range(-n_azi, n_azi+1):
        azi_rolled_arr = roll(arr, 0, i) if i!=0 else arr
        for j in range(-n_rad, n_rad+1):
            window_arrays += [roll(azi_rolled_arr, 1, j)] if j!=0 else [azi_rolled_arr]
            
    combi_array = np.stack(window_arrays, axis = -1)  
    return np.ma.median(combi_array, axis=-1)
    

x = np.array([np.arange(0,5),np.arange(5,10),np.arange(10,15),np.arange(15,20),np.arange(20,25)])
print(x)
print(roll(x, 0, -2))

print(get_window_median(x, 1, 1))

t=pytime.time()
with h5py.File('D:/radar_data_NLradar/KNMI/RAD62_OPER_O___TARVOL__L2__20180103T000000_20180104T000000_0001/RAD_NL62_VOL_NA_201801030020.h5') as f:
    scan = f['scan15']
    prf_h = scan.attrs['scan_high_PRF'][0]
    prf_l = scan.attrs['scan_low_PRF'][0]
    
    calibration_formula=eval(str(scan['calibration'].attrs['calibration_V_formulas'])).decode('utf-8')
    calibration_a=float(calibration_formula[calibration_formula.index('=')+1:calibration_formula.index('*')])
    calibration_b=float(calibration_formula[calibration_formula.index('+')+1:])
    data = np.array(scan['scan_V_data'])
    n_azi, n_rad = data.shape
    data_mask = data==0.
    v_e = np.ma.masked_array(data*calibration_a + calibration_b, data_mask)

    v_ae = calibration_a + abs(calibration_b) #Extended Nyquist velocity
    #calibration_a is added, because 0 bits corresponds to no data, so 1 bit denotes the first possible value.
    r_lambda = 5.3e-2 #Found at the internet somewhere for De Bilt. Might not be exact.
    v_al = r_lambda*prf_l/4.
    v_ah = r_lambda*prf_h/4.
    print(v_al,v_ah)
    N = int(round(v_ae/v_ah)) # Values are not exactly integers, so there are some slight errors in the values above.
    
    va_radials = np.tile([v_ah,v_al],180)
    va_array = np.transpose(np.tile(va_radials,(n_rad,1)))
    
    a = np.pi*v_e/v_ae
    
    window_ve_median = get_window_median(v_e, 2, 2)
    window_a_median = window_ve_median * np.pi/v_ae
    a_diff = np.abs(a - window_a_median)
    corr = a_diff > np.pi
    a_diff[corr] = 2*np.pi - a_diff[corr]
    v_diff = a_diff * v_ae/np.pi
    
    outliers = np.abs(v_diff)>va_array
    outliers[data_mask] = False
        
    print(pytime.time()-t)
    
    plt.figure(figsize=(30,30))
    radial_res = scan.attrs['scan_range_bin'][0]
    radius = radial_res*np.arange(0,n_rad)
    azimuth = np.linspace(0,2*np.pi,360)
    r, phi = np.meshgrid(radius,azimuth)
    x = r*np.sin(phi); y = r*np.cos(phi)
    P = plt.pcolormesh(x,y,outliers)
    plt.colorbar(P)
    plt.axes().set_aspect('equal')
    plt.show()

    
    
    