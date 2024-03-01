# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:09:34 2020

@author: bramv
"""
import h5py
import os


remove_datasets = ['scan_CPA_data', 'scan_CPAv_data', 'scan_CCOR_data', 'scan_CCORv_data', 'scan_SQI_data', 'scan_SQIv_data']
remove_datasets = ['scan_CPA_data', 'scan_CPAv_data', 'scan_CCOR_data', 'scan_CCORv_data']
# remove_datasets = ['scan_SQI_data', 'scan_SQIv_data']

# directory = 'H:/radar_data_NLradar/KNMI'
# subdirs = os.listdir(directory)
# for subdir in subdirs:
#     print(subdir)
#     try:
#         radar = int(subdir[3:5])
#         date = int(subdir[27:35])
        
#         if 20080101 <= date < 20200801:
#             continue
            
#         files = os.listdir(directory+'/'+subdir)
#         for file in files:
#             print(file)
#             try:
#                 filename = directory+'/'+subdir+'/'+file
#                 file_changed = False
#                 with h5py.File(filename, 'r+') as f:
#                     scans = [i for i in f if i.startswith('scan')]
#                     for scan in scans:
#                         scangroup = f[scan]
#                         for dataset in remove_datasets:
#                             try:
#                                 del scangroup[dataset]
#                                 file_changed = True
#                             except Exception:
#                                 continue
                            
#                 if file_changed:
#                     new_filename = filename[:-3]+'_new.h5'
#                     os.system('h5repack '+filename+' '+new_filename)
#                     os.remove(filename)
#                     os.rename(new_filename, filename)
#             except Exception as e:
#                 print(e)
#                 continue
#     except Exception as e:
#         print(e)
#         continue
            
            

directory = 'H:/radar_data_NLradar/Current/KNMI'
radars = ['Den Helder', 'Herwijnen']
for radar in radars:
    dates = os.listdir(directory+'/'+radar.replace(' ', ''))
    for date in dates:
        if 20080101 <= int(date) < 20230825:
            continue
    
        files = os.listdir(directory+'/'+radar.replace(' ', '')+'/'+date)
        for file in files:
            print(file)
            try:
                filename = directory+'/'+radar.replace(' ', '')+'/'+date+'/'+file
                file_changed = False
                with h5py.File(filename, 'r+') as f:
                    scans = [i for i in f if i.startswith('scan')]
                    for scan in scans:
                        scangroup = f[scan]
                        for dataset in remove_datasets:
                            try:
                                del scangroup[dataset]
                                file_changed = True
                            except Exception:
                                continue
                            
                if file_changed:
                    new_filename = filename[:-3]+'_new.h5'
                    os.system('h5repack '+filename+' '+new_filename)
                    os.remove(filename)
                    os.rename(new_filename, filename)
            except Exception as e:
                print(e)
                continue