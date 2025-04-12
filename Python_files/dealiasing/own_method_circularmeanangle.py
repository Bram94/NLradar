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

x = np.array([np.arange(0,5),np.arange(5,10),np.arange(10,15),np.arange(15,20),np.arange(20,25)])
print(x)
print(roll(x, 0, -2))

t=pytime.time()
with h5py.File('D:/radar_data_NLradar/KNMI/RAD62_OPER_O___TARVOL__L2__20160904T000000_20160905T000000_0001/RAD_NL62_VOL_NA_201609040700.h5') as f:
    scan = f['scan15']
    prf_h = scan.attrs['scan_high_PRF'][0]
    prf_l = scan.attrs['scan_low_PRF'][0]
    
    formula = eval(str(scan['calibration'].attrs['calibration_V_formulas']))
    calibration_formula=(formula[0] if isinstance(formula,list) else formula).decode('utf-8')
    calibration_a=float(calibration_formula[calibration_formula.index('=')+1:calibration_formula.index('*')])
    calibration_b=float(calibration_formula[calibration_formula.index('+')+1:])
    data = np.array(scan['scan_V_data'])
    n_azi, n_rad = data.shape
    data_mask = data==0.
    v_e = data*calibration_a + calibration_b
    v_e[data_mask] = np.nan

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
    
    """Outlier detection step"""
    sin_a = np.sin(a)
    cos_a = np.cos(a)
    sin_a[data_mask] = 0.0; cos_a[data_mask] = 0.0
    
    sina, cosa = get_window_sum(sin_a, 2, 4), get_window_sum(cos_a, 2, 4)
    
    avg_a = np.arctan2(sina,cosa)
    a_diff = np.abs(a - avg_a)
    corr = a_diff > np.pi
    a_diff[corr] = 2*np.pi - a_diff[corr]
    v_diff = a_diff * v_ae/np.pi
    
    outliers = np.abs(v_diff)>va_array
    outliers[data_mask] = False

    """Correction step: Add 2n*v_h or 2n*v_l to the velocity of the outliers, with n such that the difference between the resulting phase and 
    avg_a is minimised."""
    #Calculate again the mean phase, but exclude the outliers from the averaging
    sin_a[outliers] = 0.0; cos_a[outliers] = 0.0
    sina, cosa = get_window_sum(sin_a, 2, 2), get_window_sum(cos_a, 2, 2)
    
    avg_a = np.arctan2(sina,cosa)
    v_diff = (a - avg_a)*v_ae/np.pi
    
    ve_corr = v_e.copy()
    correction_ints = np.round(v_diff[outliers]/(2*va_array[outliers]))
    ve_corr[outliers] -= 2*correction_ints*va_array[outliers]
    outside_extended_nyquist_interval = np.abs(ve_corr) > v_ae
    ve_corr[outside_extended_nyquist_interval] = -np.sign(ve_corr[outside_extended_nyquist_interval])*2*v_ae\
    + ve_corr[outside_extended_nyquist_interval] 

    plt.figure()
    counts, v = np.histogram(v_diff[1:360:2], bins = 30, range = (-v_ae, v_ae))
    plt.plot((v[:-1]+v[1:])/2., counts)
    counts, v = np.histogram(v_diff[0:360:2], bins = 30, range = (-v_ae, v_ae))
    plt.plot((v[:-1]+v[1:])/2., counts)
    #plt.ylim([0,100])
    plt.show()
    print('vn',2*v_ah, 2*v_al)
    1/0
        
    print(pytime.time()-t)
    
    plt.figure(figsize=(15,15))
    radial_res = scan.attrs['scan_range_bin'][0]
    radius = radial_res*np.arange(0,n_rad)
    azimuth = np.linspace(0,2*np.pi,360)
    r, phi = np.meshgrid(radius,azimuth)
    x = r*np.sin(phi); y = r*np.cos(phi)
    P = plt.pcolormesh(x,y,ve_corr)
    plt.colorbar(P)
    plt.axes().set_aspect('equal')
    plt.show()

    
    
    