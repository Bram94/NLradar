# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:09:30 2024

@author: bramv
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.chdir('D:/NLradar/NLradar_private/Python_files')
import nlr_globalvars as gv
import nlr_functions as ft

date = '20230723'
times = [None, '1700']
if times == locals().get('times_before', []):
    raise Exception('Same time')
    
radars_keep = ('Bourges', 'Arcis-sur-Aube', 'Blaisy-Haut', 'Trappes', 'Nancy')
radars_keep = ('Arcis-sur-Aube', 'Blaisy-Haut', 'St. Nizier', 'Nancy')
radars_keep = ('Trappes', 'Avesnes', 'Arcis-sur-Aube', 'Bourges', 'St. Nizier', 'St-Remy', 'Sembadel',
               'Brive Grezes', 'Montclar', 'Toulouse', 'Blaisy-Haut', 'Nancy', 'Bordeaux')
radars_keep = ('Bordeaux', 'Cherves', 'Bourges', 'Brive Grezes', 'St-Remy', 'Sembadel', 'Toulouse', 'St. Nizier', 
               'Bollene', 'Montancy', 'Montclar', 'Nimes', 'Opoul', 'Trappes', 'Arcis-sur-Aube',
               'Blaisy-Haut')
radars_keep = ('Arcis-sur-Aube', 'Nancy', 'Avesnes', 'Montancy')
radars_keep = []
if any(j not in gv.radars_all for j in radars_keep):
    print([j for j in radars_keep if j not in gv.radars_all])
    raise Exception('Incorrect radar')
    
coords_all = np.array([gv.radarcoords[j] for j in gv.radars['Météo-France']])
coords_selected = np.array([gv.radarcoords[j] for j in radars_keep])

plt.scatter(coords_all[:,1], coords_all[:,0])
if len(coords_selected):
    plt.scatter(coords_selected[:,1], coords_selected[:,0], c='r')
plt.title(f'Times: {times}')
plt.show()

answer = input('Do you want to continue?')

if answer.strip().lower() not in ('y', 'yes'):
    raise Exception
    
radars_keep = [j.replace(' ','').lower() for j in radars_keep]
directory = f'H:/radar_data_NLradar/Meteo France/{date}/'
temp_dir_removal = f'H:/Temp_dir_removal_french_radar/{date}/'
for subdir in os.listdir(directory):
    dir_removal = temp_dir_removal+subdir
    if subdir.lower() in radars_keep:
        continue
    else:
        files = os.listdir(directory+subdir)
        for file in files.copy():
            datetime = file[16:28] if file[:2] == 'T_' else date+file[12:16]
            datetime = int(ft.next_datetime(datetime, -5))
            if (times[0] is None or datetime >= int(date+times[0])) and (times[1] is None or datetime < int(date+times[1])):
                print(subdir, datetime)
                os.makedirs(dir_removal, exist_ok=True)
                os.rename(directory+subdir+'/'+file, dir_removal+'/'+file)
                files.remove(file)
        if len(files) == 0:
            os.rmdir(directory+subdir)
            
times_before = times.copy()
1/0            
#%%
directory = 'H:/radar_data_NLradar/Meteo France/'
dates = os.listdir(directory)
for date in dates:
    radars = os.listdir(directory+date)
    for radar in radars:
        subdir = directory+date+'/'+radar+'/'
        files = os.listdir(subdir)
        for file in files:
            if file.startswith('T_IPS'):
                print(file)
                os.remove(subdir+'/'+file)