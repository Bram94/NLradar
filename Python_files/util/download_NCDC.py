# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:09:45 2023

@author: -
"""

import requests
import os
from urllib.request import urlretrieve

base_url = 'https://www1.ncdc.noaa.gov/pub/has/HAS012485804/'

filelist_url = base_url+'fileList.txt'
files = requests.get(filelist_url).content.decode('utf-8').split('\n')[:-1]

date = files[0][-12:-4]
date = os.path.basename(files[0])[4:12]
radar = os.path.basename(files[0])[:4]
directory = f'D:/radar_data_NLradar/NWS/{date}/{radar}/'
os.makedirs(directory, exist_ok=True)
for file in files:
    print(file)
    urlretrieve(base_url+file, directory+os.path.basename(file))


