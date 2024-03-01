# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:22:17 2023

@author: -
"""
import os
import sys
sys.path.append('/'.join(__file__.split('\\')[:-3]))
import pickle
import numpy as np
import tensorflow as tf
import time as pytime

from decoders.nexrad_l2 import NEXRADLevel2File
from unet_vda import Unet_VDA
from unet_vda_old import Unet_VDA as Unet_VDA_Old
vda = Unet_VDA()
vda_old = Unet_VDA_Old()


def angle_diff(angle1, angle2=None, between_0_360=False):
    diff = np.diff(angle1) if angle2 is None else angle2-angle1
    angle_diff = (diff+180) % 360 - 180
    return angle_diff % 360 if between_0_360 else angle_diff

def process_azis_array(azis, da, calc_azi_offset, azi_pos='center'):
    # Assumes that azimuth increases in clockwise direction, without any exception
    diffs = angle_diff(-azis[::-1], between_0_360=True)
    # print(list(diffs))
    csum = np.cumsum(diffs)
    # Ensure that azis spans less than 360°, to prevent issues with unsampled radials in self.map_onto_uniform_grid
    n_azi = len(azis) if csum[-1] < 360 else 1+np.where(csum >= 360)[0][0]
    azis = azis[-n_azi:]
    diffs = diffs[:n_azi-1][::-1]
    
    azi_offset = 0
    # Panel is None implies calling from get_data_multiple_scans. In this case the azimuthal offset is kept at 0, and da is set such
    # that 360/da gives an integer multiple of 360. This is convenient when calculating derived momentss.
    if calc_azi_offset:
        # Exclude large differences due to missing radial sectors. Don't make the maximum deviation too small, since deviations
        # up to a bit more than 2*da have been observed outside of missing radials.
        da = diffs[diffs < 3*da].mean()
        offset = 0.5*(azi_pos == 'left')
        azi_offset = np.mean([angle_diff(0.5*da, (azis[i]+offset) % da) for i in range(n_azi)])
    return azis, n_azi, da, azi_offset, diffs


#%%
from cProfile import Profile

t_tot = {'new':[], 'old':[]}
for i in range(20):
    filename = "D:/radar_data_NLradar/NWS/20190520/KFDR/KFDR20190520_222622_V06"
    filename = "D:/radar_data_NLradar/NWS/20211211/KHPX/KHPX20211211_052828_V06"
    # filename = "D:/radar_data_NLradar/NWS/20200810/KDMX/KDMX20200810_172431_V06"
    # filename = "D:/radar_data_NLradar/NWS/19990503/KTLX/KTLX19990503_230052.gz"
    # filename = "D:/radar_data_NLradar/NWS/20040530/KTLX/KTLX20040530_004507.gz"
    filename = "D:/radar_data_NLradar/NWS/20240109/KEVX/KEVX20240109_1234"
    filename = "D:/radar_data_NLradar/NWS/20180320/KFFC/KFFC20180320_014119_V06"
    filename = "D:/radar_data_NLradar/NWS/20240212/TBNA/TBNA20240212_2128"
    test = NEXRADLevel2File(filename, read_mode="min-meta")
    
    t = pytime.time()
    t_dealias, t_dealias_old = {}, {}
    for j in range(1, 20):
        file = NEXRADLevel2File(filename, read_mode=test.scan_startend_pos[j], moments='VEL')
        
        msg = file.radial_records[file.scan_msgs[0][0]]
        
        azis = file.get_azimuth_angles(scans=[0])
        da = msg['msg_header'].get('azimuth_resolution', 2)/2
        azis, n_azi, da, azi_offset, diffs = process_azis_array(azis, da, calc_azi_offset=True)
                
        moment = 'VEL'
        if not moment in msg:
            continue
        dr = msg[moment]['gate_spacing']
        first_gate = (msg[moment]['first_gate']-0.5*dr)/dr
        n_rad = int(first_gate+msg[moment]['ngates'])
        data = file.get_data(moment, n_rad, scans=[0], raw_data=False)
        data = data.filled(np.nan)
        vn = file.get_nyquist_vel(scans=[0])[-n_azi:]
        
        # filenames = os.listdir('data')
        # print(filenames)
        # with open('data/'+filenames[0], 'rb') as f:
        #     data, azis, vn = pickle.load(f).values()
        
        diffs = -angle_diff(azis[::-1])
        csum = np.cumsum(diffs)
        # Ensure that azis spans less than 360°, to prevent issues with unsampled radials in self.map_onto_uniform_grid
        n_azi = len(azis) if csum[-1] < 360 else 1+np.where(csum >= 360)[0][0]
        data, azis, vn = data[-n_azi:], azis[-n_azi:], vn[-n_azi:]
        
        diffs = diffs[:n_azi-1][::-1]
        da_median = np.median(diffs)
        da = diffs[diffs < 3*da_median].mean()
        
        tt = pytime.time()
        # profiler = Profile()
        # profiler.enable() 
        data_new = vda(data, vn, azis, da)
        # print(data_new)
        t_dealias[j] = pytime.time()-tt
        print(t_dealias[j])
        # data_new = vda_old(data, vn, azis, da, limit_shape_changes=True)
        # t_dealias_old[j] = pytime.time()-tt-t_dealias[j]
        # profiler.disable()
        # import pstats
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(15)  

    # print(pytime.time()-t)
    # for j in t_dealias:
    #     print(j, t_dealias[j], t_dealias_old[j])
    # t_tot['new'].append(sum(t_dealias.values(), 0))
    # t_tot['old'].append(sum(t_dealias_old.values(), 0))
    # print(i, t_tot['new'][-1], t_tot['old'][-1])

#%%
# vda.vda.save('D:/NLradar/Python_files/dealiasing/unet_vda/models/test')
# print('t_save', pytime.time()-t)