# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:09:34 2020

@author: bramv
"""
import h5py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np

import nlr_functions as ft


remove_datasets = ['scan_CPA_data', 'scan_CPAv_data', 'scan_CCOR_data', 'scan_CCORv_data', 'scan_SQI_data', 'scan_SQIv_data']
# Zv and uZv are not needed anymore after calculating ZDR
remove_datasets = ['scan_Zv_data', 'scan_uZv_data', 'scan_CPA_data', 'scan_CPAv_data', 'scan_CCOR_data', 'scan_CCORv_data']
datasets_16bit_to_8bit = ['scan_Z_data', 'scan_uZ_data', 'scan_V_data', 'scan_Vv_data', 'scan_KDP_data', 'scan_SQI_data', 'scan_SQIv_data', 'scan_W_data', 'scan_Wv_data']
# remove_datasets = ['scan_SQI_data', 'scan_SQIv_data']
            
def compress_file(filename):
    try:
        file_changed = False
        with h5py.File(filename, 'r+') as f:
            scans = [i for i in f if i.startswith('scan')]
            for scan in scans:
                scangroup = f[scan]                            
                calibrationgroup = scangroup['calibration']
                    
                # Remove Zv in favour of ZDR. Zv could be retrieved as Zv = Zh-ZDR
                # This is done mainly because calculating ZDR after converting Zh and Zv from 16-bit to 8-bit greatly reduces ZDR resolution. 
                for j in ('', 'u'):
                    if not f'scan_{j}Zv_data' in scangroup:
                        continue

                    calibration_formula = ft.from_list_or_nolist(calibrationgroup.attrs[f'calibration_{j}Z_formulas']).decode('utf-8')
                    gain = float(calibration_formula[calibration_formula.index('=')+1:calibration_formula.index('*')])
                    offset = float(calibration_formula[calibration_formula.index('+')+1:])
                    
                    Zh = np.array(scangroup[f'scan_{j}Z_data'], dtype='float64')*gain+offset
                    Zv = np.array(scangroup[f'scan_{j}Zv_data'], dtype='float64')*gain+offset
                    ZDR = Zh-Zv
                    pmin, pmax = -10, 20
                    ZDR[ZDR < pmin] = pmin
                    ZDR[ZDR > pmax] = pmax
                    data_mask = (Zh == Zh.min()) | (Zv == Zv.min())
                    ZDR = 1+(2**8-2)*(ZDR-pmin)/(pmax-pmin)
                    ZDR = (ZDR+0.5).astype('uint8') # data+0.5 insures correct rounding when flooring during conversion to uint8
                    ZDR[data_mask] = 0
                    scangroup.create_dataset(f'scan_{j}ZDR_data', data=ZDR, dtype='uint8', compression='gzip')
                    
                    name = f'calibration_{j}ZDR_formulas'
                    gain = (pmax-pmin)/(2**8-2) # A value of zero is used for masked elements, so therefore -2 vs -1
                    offset = pmin-gain # -gain, since pmin corresponds to PV=1
                    calibration_formula = f'GEO={gain}*PV+{offset}'
                    calibrationgroup.attrs[name] = calibration_formula
                    file_changed = True
                    
                for dataset in remove_datasets:
                    if not dataset in scangroup:
                        continue
                    del scangroup[dataset]
                    file_changed = True
                    
                for dataset in datasets_16bit_to_8bit:
                    if not dataset in scangroup:
                        continue
                    if scangroup[dataset].dtype != 'uint16':
                        continue
                    
                    data = np.array(scangroup[dataset], dtype='float64')
                    data_mask = data == data.min()
                    
                    productname = dataset.split('_')[1]
                    name = 'calibration_'+productname+'_formulas'
                    calibration_formula = ft.from_list_or_nolist(calibrationgroup.attrs[name]).decode('utf-8')
                    gain = float(calibration_formula[calibration_formula.index('=')+1:calibration_formula.index('*')])
                    offset = float(calibration_formula[calibration_formula.index('+')+1:])
                    data = data*gain+offset
                    
                    pmin = offset+gain # +gain, since pmin corresponds to PV=1
                    pmax = gain*(2**16-1)+offset
                    new_gain = (pmax-pmin)/(2**8-2) # A value of zero is used for masked elements, so therefore -2 vs -1
                    new_offset = pmin-new_gain # -gain, since pmin corresponds to PV=1
                    data = (data-new_offset)/new_gain
                    data = (data+0.5).astype('uint8') # data+0.5 insures correct rounding when flooring during conversion to uint8
                    # Make sure that values that are originally not masked, also don't get masked after conversion
                    data[data_mask] = 0
                    
                    del scangroup[dataset]
                    scangroup.create_dataset(dataset, data=data, dtype='uint8', compression='gzip')
                                              
                    new_calibration_formula = f'GEO={new_gain}*PV+{new_offset}'
                    calibrationgroup.attrs[name] = new_calibration_formula
                    file_changed = True
                    
        if file_changed:
            new_filename = filename[:-3]+'_new.h5'
            os.system('h5repack '+filename+' '+new_filename)
            os.remove(filename)
            os.rename(new_filename, filename)
    except Exception as e:
        print(e)
            
         
            
# directory = 'H:/radar_data_NLradar/Current/KNMI'
# # directory = 'D:/Test_16bit_to_8bit'
# radars = ['Den Helder', 'Herwijnen']
# radars = ['Herwijnen']
# for radar in radars:
#     dates = os.listdir(directory+'/'+radar.replace(' ', ''))
#     for date in dates:
#         if 20080101 <= int(date) < 20230918:
#             continue
    
#         files = os.listdir(directory+'/'+radar.replace(' ', '')+'/'+date)
#         for file in files:
#             print(file)
#             filename = directory+'/'+radar.replace(' ', '')+'/'+date+'/'+file
#             compress_file(filename)            
            
directory = 'H:/radar_data_NLradar/KNMI'
subdirs = [j for j in os.listdir(directory) if j[:5] in ('RAD61', 'RAD62')]
for subdir in subdirs:
    print(subdir)
    date = int(subdir[27:35])
    if 20080101 <= date < 20160827:
        continue
        
    files = os.listdir(directory+'/'+subdir)
    for file in files:
        print(file)
        filename = directory+'/'+subdir+'/'+file
        compress_file(filename)