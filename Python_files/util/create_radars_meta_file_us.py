# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:33:34 2023

@author: -
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np

import nlr_functions as ft





if __name__ == '__main__':
    with open('radar_elevs_us.txt', 'r', encoding='utf-8') as f:
        data = ft.list_data(f.read(), '\t')
        elevs = {i[0]:int(round(float(i[1]))) for i in data}
        radars = sorted(list(elevs))
    with open('radars_grlevelx.txt', 'r') as f:
        data = [ft.string_to_list(j[0]) if ',' in j[0] else j for j in ft.list_data(f.read(), '|')]
        print(data)
        # 1/0
        lats = {i[0].upper():float(i[2]) for i in data}
        lons = {i[0].upper():float(i[3]) for i in data}
        radar_elevations = {i[0].upper():int(round(float(i[4]))) for i in data}
        rplaces_to_ridentifiers = {i:'' for i in radar_elevations}
        data_sources = {i:'NWS' for i in radar_elevations}
        radar_bands = {i[0].upper():'C' if i[0][0] == 't' and i[-2] != 'PR' else 'S' for i in data}
        for j in ('FWLX', 'MZZU', 'WILU'):
            radar_bands[j] = 'X'
        
    radars_exclude = ['MZZU'] # MZZU is in SIGMET format, not NEXRAD L2
    radars = [j for j in radars if not j in radars_exclude]
    
    attrs = [radars, rplaces_to_ridentifiers, lats, lons, radar_elevations, elevs, radar_bands]
    attrs = [i if isinstance(i, list) else [i[j] for j in radars] for i in attrs]
    lengths = [max(len(str(i)) for i in j) for j in attrs]
    
    source = None
    with open('D:/NLradar/NLradar/Input_files/radars_us.txt', 'w', encoding='utf-8') as f:
        for k,r in enumerate(radars):
            if data_sources[r] != source:
                if source:
                    f.write('\n')
                source = data_sources[r]
                f.write(source+'\n')
            for i,j in enumerate(attrs):
                s = str(j[k]) if j[k] != '' else '.'
                tabs = '\t'*int(np.ceil((np.ceil((lengths[i]+1)/8)*8-len(s))/8))
                f.write(s+tabs*(i+1 < len(attrs)))
            if not k == len(radars)-1:
                f.write('\n')