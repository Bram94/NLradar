# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:35:03 2025

@author: bramv
"""
import os
import zipfile
import tarfile
import re

os.chdir('D:/NLradar/NLradar_private/Python_files')
import nlr_globalvars as gv

1/0
#%%
radars = gv.radars['Météo-France']

files = ['C:/Users/bramv/Downloads/francetransfert-548773515.zip',
         'C:/Users/bramv/Downloads/francetransfert-3002687994.zip']
for file in files:
    with zipfile.ZipFile(file) as f:
        entries = f.namelist()
        for entry in entries:
            try:
                date = entry.split('_')[-2]
                radar = entry.split('_')[-1].split('.')[0]
                radar = [j for j in radars if any(i in radar for i in re.split('\W', j) if len(i) > 2)][0]
                
                extract_dir = f'H:/radar_data_NLradar/Meteo France/{date}/{radar.replace(" ", "")}'
                extract_path = extract_dir+'/'+entry
                print(extract_path)
                os.makedirs(extract_dir, exist_ok=True)
                
                if not os.path.exists(extract_path):
                    f.extract(entry, extract_dir)
                    
                with tarfile.TarFile(extract_path) as f2:
                    f2.extractall(extract_dir)
                os.remove(extract_path)
            except Exception as e:
                print(e, 'error')
            
#%%
radars = {i:r for r,i in gv.radar_ids.items() if r in gv.radars['ARSO']}

_dir = 'C:/Users/bramv/Downloads/'
zips = [_dir+j for j in os.listdir(_dir) if j.startswith('wetransfer_') and j.endswith('.zip')]
for z in zips:
    with zipfile.ZipFile(z, 'r') as f:
        entries = f.namelist()
        for i, entry in enumerate(entries):
            if i > 0 and not 'DP' in entry and 'DP' in entries[i-1] and entry == entries[i-1].replace('_DP', ''):
                continue
            radar_id = entry[6:8]
            radar = radars[radar_id]
            date = entry.split('_')[4]
            
            extract_dir = f'H:/radar_data_NLradar/ARSO/{date}/{radar.replace(" ", "")}'
            extract_path = extract_dir+'/'+entry
            print(entry, extract_path)
            os.makedirs(extract_dir, exist_ok=True)
        
            if not os.path.exists(extract_path):
                f.extract(entry, extract_dir)
                
            with zipfile.ZipFile(extract_path) as f2:
                f2.extractall(extract_dir)
            os.remove(extract_path)