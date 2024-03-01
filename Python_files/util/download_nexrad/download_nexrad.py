#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:26:31 2019

@author: veen
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) #Add the Code directory to the path, to enable relative imports
import datetime as dt
try:
    import pyproj
except Exception:
    os.environ['PROJ_LIB'] = '/usr/share/proj'
    import pyproj #Is used by nexradaws
import nexradaws



home = os.path.expanduser("~")
dir_NEXRAD_data = home+'/Downloads'
dir_NEXRAD_data = 'D:/radar_data_NLradar/NWS'

conn = nexradaws.NexradAwsInterface()

dt_end = '090211023'
timespan = 180
radars = ['NOP3']
    
dt_end = ('20' if int(dt_end[:2]) < 50 else '19')+dt_end+'0'
dt_end = dt.datetime(int(dt_end[:4]),int(dt_end[4:6]), int(dt_end[6:8]), int(dt_end[8:10]), int(dt_end[-2:]), tzinfo = dt.timezone.utc)
dt_start = dt_end-dt.timedelta(minutes=timespan)
for radar in radars:
    print(radar)
    try:
        scans = conn.get_avail_scans_in_range(dt_start, dt_end, radar)
        print(scans)
        scans = [scan for scan in scans if not scan.filename.endswith('MDM')] #Not sure what MDM files contain, but at least not complete volumes.
        
        for scan in scans:
            date = scan.filename[4:12]
            download_directory = dir_NEXRAD_data+'/Download/'
            directory = dir_NEXRAD_data+'/'+date+'/'+radar
            if not os.path.exists(directory+'/'+scan.filename):
                conn.download(scan, download_directory)
                try:
                    os.makedirs(directory, exist_ok=True)
                    os.rename(download_directory+'/'+scan.filename, directory+'/'+scan.filename)
                except Exception:
                    pass
    except Exception as e:
        print(e)