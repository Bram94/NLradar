# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:37:55 2018

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

def get_window_sums(arr):
    azi_sum2 = roll(arr,0,-1) + roll(arr,0,1)
    azi_sum3 = roll(arr,0,-2) + arr + roll(arr,0,2)
    
    rad_sum2 = roll(azi_sum2,1,2)+roll(azi_sum2,1,1)+azi_sum2+roll(azi_sum2,1,-1)+roll(azi_sum2,1,-2)
    rad_sum3 = roll(azi_sum3,1,2)+roll(azi_sum3,1,1)+azi_sum3+roll(azi_sum3,1,-1)+roll(azi_sum3,1,-2)
    return rad_sum2, rad_sum3

def fold_circular(data, mod):
    """
    Values outside the specified interval are folded back into the interval.

    Parameters
    ----------
    data_ma : numpy masked array
        Data
    mod: float
        Interval (module)

    Returns
    -------
    ma_fold :  numpy masked array
        Folded data
    """

    scl = np.ones(data.shape)*mod  # array with modules
    scl[np.where(data < 0)] *= -1  # negative module when value is negative, (-pi, +pi) interval
    scl[np.where(np.ma.abs(data) > mod)] *= -1

    fold = np.mod(data, scl)

    return fold



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
    N_radials = np.reshape(np.tile([N, N+1],180), (360,1))
    
    a = np.pi*v_e/v_ae
    
    sin_ap = np.sin(N_radials * a)
    cos_ap = np.cos(N_radials * a)
    sin_ap[data_mask] = 0.0; cos_ap[data_mask] = 0.0
    
    radsum2_sinap, radsum3_sinap = get_window_sums(sin_ap)
    radsum2_cosap, radsum3_cosap = get_window_sums(cos_ap)
    radsum2_mask, radsum3_mask = get_window_sums((data_mask==False).astype('int'))
    
    b_l = np.arctan2(radsum2_sinap/radsum2_mask,radsum2_cosap/radsum2_mask)
    b_h = np.arctan2(radsum3_sinap/radsum3_mask,radsum3_cosap/radsum3_mask)
        
    sign_arr = np.ones(n_azi)
    sign_arr[va_radials == v_al] = -1
    a_ref = sign_arr[:,np.newaxis] * (b_l-b_h)
    
    v_ref = a_ref*v_ae/np.pi
    v_ref = fold_circular(v_ref, mod=v_ae)
    
    v_diff = v_ref - v_e
    
    outliers = np.abs(v_diff)>va_array
    outliers[data_mask] = False
    
    print('TEST!!!!!!!!!!!!!!!', v_e[180,320], b_h[180,320], b_l[180,320], v_ref[180,320], outliers[180,320])
        
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

    
    
    