# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:42:20 2025

@author: bramv
"""
import os
import tarfile
import bz2

directory = 'E:/radar_data_NLradar/Austro Control'
tars = [j for j in os.listdir(directory) if '.tar' in j]

radar_map = {'ACGF':'Feldkirchen', 'ACGR':'Rauchenwarth', 'ACGZ':'Zirbitzkogel', 'ACGP':'Patscherkofel'}
for tar in tars:
    out_dir = directory+'/'+tar[:8]
    with tarfile.open(directory+'/'+tar, "r:bz2") as f:
        f.extractall(out_dir)
        
    files = [j for j in os.listdir(out_dir) if 'PARA' in j]
    for file in files:
        radar_id = file.split('_')[1]
        radar = radar_map[radar_id]
        new_dir = out_dir+'/'+radar+'_Z'
        os.makedirs(new_dir, exist_ok=True)
        if not os.path.exists(new_dir+'/'+file):
            os.rename(out_dir+'/'+file, new_dir+'/'+file)
        else:
            os.remove(out_dir+'/'+file)
        
print(tars)