# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:33:34 2023

@author: -
"""
import os
import sys
sys.path.insert(0, '/'.join(__file__.split(os.sep)[:-2]))
import numpy as np

import nlr_functions as ft





if __name__ == '__main__':
    with open('radar_elevs_us.txt', 'r', encoding='utf-8') as f:
        data = ft.list_data(f.read(), '\t')
        elevs = {i[0]:int(round(float(i[2]))) for i in data}
        radars = sorted(list(elevs))
    with open('D:/NLradar/Python_files/util/radars_grlevelx.txt', 'r') as f:
        data = [ft.string_to_list(j[0]) if ',' in j[0] else j for j in ft.list_data(f.read(), '|')]
        print(data)
        # 1/0
        lats = {i[0].upper():float(i[2]) for i in data}
        lons = {i[0].upper():float(i[3]) for i in data}
        radar_elevations = {i[0].upper():int(round(float(i[4]))) for i in data}
        rplaces_to_ridentifiers = {i:'' for i in radar_elevations}
        data_sources = {i:'NWS' for i in radar_elevations}
        
    # 1 S-band, 2 C-band, 3 X-band
    radar_bands = {j:'C' if j[0] == 'T' else 'S' for j in radars}
    
    attrs = [radars, rplaces_to_ridentifiers, lats, lons, radar_elevations, elevs, radar_bands]
    attrs = [i if isinstance(i, list) else [i[j] for j in radars] for i in attrs]
    lengths = [max(len(str(i)) for i in j) for j in attrs]
    
    source = None
    with open('D:/NLradar/Input_files/radars_us.txt', 'w', encoding='utf-8') as f:
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