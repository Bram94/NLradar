# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:59:08 2023

@author: -
"""
import os
import sys
sys.path.insert(0, '/'.join(__file__.split(os.sep)[:-2]))
import requests
import time as pytime

os.chdir('D:/NLradar/Python_files')
import nlr_globalvars as gv
radars = gv.radars['NWS']
radarcoords = {i:j for i,j in gv.radarcoords.items() if i in radars}

api_key = ""

#%%
elevs1 = {}
elevs2 = {}
for i in radars:
    print(i)
    lat, lon = radarcoords[i]
    # print(lat, lon)
    output = requests.get(f'https://maps.googleapis.com/maps/api/elevation/json?locations={lat}%2C{lon}&key={api_key}').content
    output = eval(output.decode('UTF-8'))
    elevs1[i] = output['results'][0]['elevation']
    # print(i, output, "\n")
    
    output = requests.get(f'https://api.opentopodata.org/v1/eudem25m?locations={lat}%2C{lon}').content
    output = eval(output.decode('UTF-8'))
    elevs2[i] = output['results'][0]['elevation']
    pytime.sleep(2)
    # print(i, output, "\n")
  
#%%
with open(gv.programdir+'/util/radar_elevs_eu.txt', 'w', encoding="utf-8") as f:
    for i in elevs2:
        f.write(i+f'\t{elevs1[i]}\t{elevs2[i]}\n')


#%%
with open('D:/NLradar/Python_files/util/radars_grlevelx.txt', 'r') as f:
    data = [ft.string_to_list(j[0]) if ',' in j[0] else j for j in ft.list_data(f.read(), '|')]
    radars = [j[0].upper() for j in data]
    radarcoords = {j[0].upper():[j[2], j[3]] for j in data}
    # print(data)
    # 1/0

elevs1 = {}
elevs2 = {}
for i in radars:
    print(i)
    lat, lon = radarcoords[i]
    # print(lat, lon)
    output = requests.get(f'https://maps.googleapis.com/maps/api/elevation/json?locations={lat}%2C{lon}&key={api_key}').content
    output = eval(output.decode('UTF-8'))
    elevs1[i] = output['results'][0]['elevation']
    # print(i, output, "\n")
    
    try:
        output = requests.get(f'https://api.opentopodata.org/v1/ned10m?locations={lat}%2C{lon}').content
        output = eval(output.decode('UTF-8'))
        elevs2[i] = output['results'][0]['elevation']
    except Exception as e:
        print(i, e)
    pytime.sleep(2)
    # print(i, output, "\n")
  
#%%
with open('D:/NLradar/Python_files/util/radar_elevs_us.txt', 'w', encoding="utf-8") as f:
    for i in elevs2:
        f.write(i+f'\t{elevs1[i]}\t{elevs2[i]}\n')

