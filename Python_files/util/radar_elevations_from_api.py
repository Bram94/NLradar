# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:59:08 2023

@author: -
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import requests
import time as pytime

os.chdir('C:/Users/bramv/NLradar/NLradar_private/Python_files')
import nlr_globalvars as gv
import nlr_functions as ft
radars = gv.radars['Austro Control']
radarcoords = {i:j for i,j in gv.radarcoords.items() if i in radars}
# 1/0
#%%
elevs = {}
for i in radars:
    print(i)
    lat, lon = radarcoords[i]
    
    output = requests.get(f'https://api.opentopodata.org/v1/eudem25m?locations={lat}%2C{lon}').content
    output = eval(output.decode('UTF-8'))
    elevs[i] = output['results'][0]['elevation']
    pytime.sleep(2)
    # print(i, output, "\n")
print(elevs)
1/0  
#%%
with open('util/radar_elevs_eu.txt', 'w', encoding="utf-8") as f:
    for i in elevs:
        f.write(i+f'\t{int(round(elevs[i]))}\n')


#%%
with open('util/radars_grlevelx.txt', 'r') as f:
    data = [ft.string_to_list(j[0]) if ',' in j[0] else j for j in ft.list_data(f.read(), '|')]
    radars = [j[0].upper() for j in data]
    radarcoords = {j[0].upper():[j[2], j[3]] for j in data}
    # print(data)
    # 1/0

elevs = {}
for i in radars:
    print(i)
    lat, lon = radarcoords[i]
    
    try:
        output = requests.get(f'https://api.opentopodata.org/v1/ned10m?locations={lat}%2C{lon}').content
        output = eval(output.decode('UTF-8'))
        elevs[i] = output['results'][0]['elevation']
    except Exception as e:
        print(i, e)
    pytime.sleep(2)
    # print(i, output, "\n")
  
#%%
# with open('util/radar_elevs_us.txt', 'w', encoding="utf-8") as f:
#     for i in elevs:
#         f.write(i+f'\t{elevs[i]}\n')