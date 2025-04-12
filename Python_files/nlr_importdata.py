# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import os
opa=os.path.abspath
import re
import h5py
import xmltodict
import zlib
import numpy as np
from collections import OrderedDict
import copy #Important: Is used with exec, and therefore listed as unused!
import time as pytime
import datetime as dtime
import netCDF4 as nc
from scipy.interpolate import interp1d
import warnings

from numpy_bufr import decode_bufr

import nlr_background as bg
import nlr_functions as ft
import nlr_globalvars as gv
from dealiasing import nlr_dealiasing as da
from derived import nlr_derived_tilts as dt
from decoders.nexrad_l2 import NEXRADLevel2File
from decoders.nexrad_l3 import NEXRADLevel3File
from decoders.dorade import DORADEFile
from decoders.ukmo_polar import UKMOPolarFile



"""Remark: Always save numpy arrays as type 'float32', because that works a little faster and does not lead to severe errors for this kind of data.
"""

"""The attributes that contain information about the radar volume structure (like self.dsg.scannumbers_all etc.) are dictionaries, with for each possible
import product a dictionary that maps a scannumber to the info contained in the attribute. 
If for a particular radar always all products are available for all scans, then all dictionaries in the volume attributes are the same.
If not, then it is assumed that at least reflectivity is available for all scans. In the case of products that are not available for all scans, 
the dictionaries for these product keys must still contain the same keys (scannumbers) as for 'z', but they now link to other values.
This is for example the case for the radar in Zaventem, and the function get_scans_information in the class ODIM_hdf5 contains information about how
to handle this.

The attribute dictionaries must contain at least keys for all products that are available for a particular source. It is not necessary to
have keys for all products in gv.products_with_tilts.
"""


def process_azis_array(azis, da, calc_azi_offset, azi_pos='center'):
    # Assumes that azimuth increases in clockwise direction, without any exception
    diffs = ft.angle_diff(-azis[::-1], between_0_360=True)
    # print(list(diffs))
    csum = np.cumsum(diffs)
    # Ensure that azis spans less than 360°, otherwise some issues can occur in self.map_onto_uniform_grid.
    # In fact, a threshold of 359.9° is used, since it has been observed that due to floating point errors 360 sometimes doesn't work. 
    n_azi = len(azis) if csum[-1] < 359.9 else 1+np.where(csum >= 359.9)[0][0]
    azis = azis[-n_azi:]
    diffs = diffs[:n_azi-1][::-1]
    
    azi_offset = 0
    # Panel is None implies calling from get_data_multiple_scans. In this case the azimuthal offset is kept at 0, and da is set such
    # that 360/da gives an integer multiple of 360. This is convenient when calculating derived products.
    if calc_azi_offset:
        # Exclude large differences due to missing radial sectors. Don't make the maximum deviation too small, since deviations
        # up to a bit more than 2*da have been observed outside of missing radials.
        da = diffs[diffs < 3*da].mean()
        offset = 0.5*(azi_pos == 'left')
        azi_offset = np.mean([ft.angle_diff(0.5*da, (azis[i]+offset) % da) for i in range(n_azi)])
    return azis, n_azi, da, azi_offset, diffs

ij_map = ref_azi_offset = None
input_before = {j:None for j in ('n_azi', 'azis', 'diffs', 'da', 'azi_offset', 'azi_pos')}
def map_onto_regular_grid(data, n_azi, azis, diffs, da, azi_offset, azi_pos='center'):
    global ij_map, ref_azi_offset, input_before
    args = locals()
    _input = {k:(args[k].tolist() if type(args[k]) is np.ndarray else args[k]) for k in input_before}
    if _input == input_before:
        data = data[ij_map]
        data[ij_map == -1] = np.nan
        return data, ref_azi_offset
    
    # azi_pos ('center'/'left') specifies whether the azimuth values are given for the center of a radial, or for the left edge
    """Map the data onto a regular grid with constant azimuthal spacing. The azimuthal resolution of the grid is set such that
    the positional error (defined as the difference in actual position of an edge of a radial, and the position of that edge on 
    the regular grid) does not exceed a maximum value.
    
    This remapping might however result in substantial variations of the width of the remapped radials, as in edge cases it's
    possible that you get sequences like n=3,1,3,1 etc, where n denotes the number of times a certain radial is repeated
    (proportional to its width) in the remapped array. These width variations are reduced by removing them if that doesn't
    increase the positional error by too much.
    """
    azis %= 360 # It can apparently happen for old NEXRAD L3 data that values above 360 exist
    
    i_min, i_max = azis.argmin(), azis.argmax()
    x = np.append(azis, [azis[i_min]+360, azis[i_max]-360])
    y = np.append(np.arange(n_azi, dtype='float64'), [i_min, i_max])
    f = interp1d(x, y, kind='nearest')
    max_error = 0.05
    max_error_width_correction = 4/3*max_error
    
    diffs = np.append(diffs, ft.angle_diff(azis[-1], azis[0]))
    k_max = max(int(np.ceil(0.5*da/max_error)), 2)
    for k in range(1, k_max+1):
        ref_n_azi = int(round(360*k/da))
        ref_da = 360/ref_n_azi
        # Decrease azi_offset with decreasing ref_da, to prevent that its magnitude can become larger than ref_da
        ref_azi_offset = azi_offset % ref_da
        ref_azis = (ref_azi_offset + ref_da*np.arange(0.5, ref_n_azi)) % 360
        
        diff_offset_fac = 1. if azi_pos == 'left' else 0.5

        ij_map = np.array([int(j) for j in f(ref_azis)], dtype='int16')
        ref_diffs = ft.angle_diff(azis[ij_map], ref_azis)
        if azi_pos == 'left':
            # For azi_pos == 'left' the nearest neighbor interpolation above results in an average azimuth offset of -0.5*da. 
            # Fixing this requires that j is reduced by 1 when ref_azis[i] is actually contained in the previous radial.
            ij_map = (ij_map-(ref_diffs < 0)) % n_azi
            ref_diffs = ft.angle_diff(azis[ij_map], ref_azis)
        
        # In the case of missing radial sectors use da instead of the actual diffs
        diff = diffs[ij_map]
        if azi_pos == 'center':
            select = ref_diffs <= 0
            diff[select] = diffs[ij_map[select]-1]
        diff[np.abs(diff) > 3*da] = da
        max_allowed_diff = diff_offset_fac*diff
        # Prevent filling up missing radial sectors
        ij_map[np.abs(ref_diffs) > max_allowed_diff] = -1
        
        azis_left = azis[ij_map]
        if azi_pos == 'center':
            azis_left -= 0.5*diffs[ij_map-1]
        azis_right = azis[ij_map]+diff_offset_fac*diffs[ij_map]
        max_error_exceeded = ft.angle_diff(azis_left, ref_azis-0.5*ref_da).min() < -max_error or\
                             ft.angle_diff(azis_right, ref_azis+0.5*ref_da).max() > max_error
                             
        if not max_error_exceeded:
            break
    
    if k > 1:
        j_repeats = np.bincount(ij_map[ij_map != -1])
        j_repeats_notk = {j:n for j,n in enumerate(j_repeats) if n not in (k, 0)} # Also not 0, otherwise calling np.where below doesn't give a hit
        keys, vals = list(j_repeats_notk), list(j_repeats_notk.values())
        """These are pairs of indices for radials that are not repeated the expected # of k times in the remapped array. The radials contained
        within the azimuthal range spanned by each pair of radials will be checked for whether their positional error won't increase by too much
        in case that the width variation is removed (smoothed away). 
        Also couple the first and last radial, hence loop over the full range of len(keys).
        Only include radial pairs with different # of repetitions, since in the case of an equal number the width variation can't be smoothed away.
        """
        pairs = {keys[i-1]:keys[i] for i in range(len(keys)) if vals[i-1] != vals[i]}
        
        angle_diff_left = ft.angle_diff(azis_left, ref_azis-0.5*ref_da)
        angle_diff_right = ft.angle_diff(azis_right, ref_azis+0.5*ref_da)
        remove = []
        for j1,j2 in (_ for _ in pairs.items() if not _[0] in remove):
            sign = np.sign(j_repeats_notk[j2]-j_repeats_notk[j1])
            r = np.arange(j1, j2+(j2 < j1)*n_azi) % n_azi
            if sign == -1:
                max_error_exceeded = np.min(angle_diff_left[r] - ref_da) < -max_error_width_correction
            else:
                max_error_exceeded = np.max(angle_diff_right[r] + ref_da) > max_error_width_correction
            if not max_error_exceeded:
                i1 = np.where(ij_map == j1)[0][-1]
                i2 = np.where(ij_map == j2)[0][0]
                idx1 = np.arange(i1, i2+(i2 < i1)*ref_n_azi) + (sign == 1)
                idx2 = idx1-sign
                ij_map[idx1 % ref_n_azi] = ij_map[idx2 % ref_n_azi]
                remove.append(j2)
    
    data = data[ij_map]
    data[ij_map == -1] = np.nan
    input_before = _input
    return data, ref_azi_offset





class Leonardo_vol_rainbow3():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
        
        self.data_minmax_indices={'z':0,'v':1,'w':2} #Gives per product the index for the lists that contain minimum and maximum values for all products.
        
        
        
    def get_length_header(self,volreader):
        return volreader.index(b'\x03')+1
        
    def convert_header_to_dict(self,header):
        dic=OrderedDict()
        while len(header)>0:
            colon_index=header.index(b':')
            return_index=header.index(b'\r')
            
            new_string = header[colon_index+2:return_index].decode('utf-8')
            new_list = [s for s in new_string.split(' ') if s]
            dic[header[:colon_index-1].decode('utf-8')]=new_list
    
            header=header[return_index+1:]
        return dic
    
        
    def get_scans_information(self,filepath,product):
        with open(filepath,'rb') as vol:
            volreader=vol.read(2000)
            
            length_header=self.get_length_header(volreader)
            header=volreader[1:length_header-1]
            dic=self.convert_header_to_dict(header)
                        
            n_scans=int(dic['A9'][0])
            for j in range(1,n_scans+1):
                self.dsg.radial_res_all['z'][j]=float(dic['P5'][0])
                self.dsg.radial_bins_all['z'][j]=int(float(dic['P4'][0])/self.dsg.radial_res_all['z'][j])
                self.dsg.scanangles_all['z'][j]=float(dic['W'+ft.halftimestring(j)][0])
                self.dsg.nyquist_velocities_all_mps[j]=float(dic['P2'][1])
                prf_l = float(dic['W'+ft.halftimestring(j)][7])
                radar_wavelength = 5.3 * 1e-2
                self.dsg.low_nyquist_velocities_all_mps[j] = self.dsg.high_nyquist_velocities_all_mps[j] = None if prf_l == 0.0 else radar_wavelength*prf_l/4.
                self.dsg.scannumbers_all['z'][j]=[j]
                                
            for i in self.dsg.scannumbers_all:
                for j in gv.volume_attributes_p: 
                    self.dsg.__dict__[j][i] = copy.deepcopy(self.dsg.__dict__[j]['z'])
            
            
    def get_data(self,filepath, j): #j is the panel
        product, i_p = self.crd.products[j], gv.i_p[self.crd.products[j]]
        scan=self.dsg.scannumbers_all[i_p][self.crd.scans[j]][self.dsg.scannumbers_forduplicates[self.crd.scans[j]]]
    
        with open(filepath,'rb') as vol:
            volreader=vol.read(2000)
            length_header=self.get_length_header(volreader)
            header=volreader[1:length_header-1]
            dic=self.convert_header_to_dict(header)
            
            #Only the start and end time are given, so the scantime becomes a time range.
            self.dsg.scantimes[j]=dic['F6'][2]+':'+dic['F6'][1]+':'+dic['F6'][0]+'-'+dic['H5'][2]+':'+dic['H5'][1]+':'+dic['H5'][0]
            
            index=self.data_minmax_indices[i_p]
            data_min=float(dic['P1'][index])
            data_max=float(dic['P2'][index])
            
            self.dsg.data[j]=self.extract_data(vol,length_header,i_p,scan,data_min,data_max)
            
        data_mask = self.dsg.data[j]<=self.dsg.data[j].min()
        if i_p == 'v':
            if self.crd.apply_dealiasing[j] and not self.dsg.low_nyquist_velocities_all_mps[self.crd.scans[j]] is None:
                self.dsg.data[j] = self.dealias_velocity(self.dsg.data[j], data_mask, self.crd.scans[j])
                
        self.dsg.data[j][data_mask]=self.pb.mask_values[product]
            
            
    def dealias_velocity(self, data, data_mask, scan):
        """Important: The KMI seems to use two PRFs per radial, seems to correct velocities using the lower Nyquist velocity. Both vn_l and vn_h in apply_dual_prf_dealiasing are therefore set to vn_l.
        Further, for Jabbeke the dual PRF errors can be all multiples of the low Nyquist velocity, and not only even multiples. Hence multiplication of vn_l by 0.5.
        """
        vn_l = self.dsg.low_nyquist_velocities_all_mps[scan]
        radial_res = self.dsg.radial_res_all['v'][scan]
        window_size = [2, 2, 2]
        n_it = 1 if self.gui.vwp.updating_vwp else self.gui.dealiasing_dualprf_n_it
        return da.apply_dual_prf_dealiasing(data, data_mask, radial_res, self.dsg.nyquist_velocities_all_mps[scan],\
        vn_l, vn_l, None, window_detection = window_size, window_correction = window_size, n_it = n_it) 
        #The first azimuth is scanned with a high PRF


    def get_data_multiple_scans(self,filepath,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        if isinstance(apply_dealiasing, bool):
            apply_dealiasing = {j: apply_dealiasing for j in scans}
        
        data={}; scantimes={}        
        with open(filepath,'rb') as vol:
            volreader=vol.read(2000)
            length_header=self.get_length_header(volreader)
            header=volreader[1:length_header-1]
            dic=self.convert_header_to_dict(header)
            
            for j in scans: 
                scantimes[j] = [dic['F6'][2]+':'+dic['F6'][1]+':'+dic['F6'][0]+'-'+dic['H5'][2]+':'+dic['H5'][1]+':'+dic['H5'][0]]
                    
                index=self.data_minmax_indices[i_p]
                data_min=float(dic['P1'][index])
                data_max=float(dic['P2'][index])

                s = np.s_[:] if max_range is None else np.s_[:, :int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangles_all[i_p][j], 'gr+theta->sr') / self.dsg.radial_res_all[i_p][j]))]
                data[j]=self.extract_data(vol,length_header,i_p,j,data_min,data_max)[s]
                
                data_mask = data[j]<=data[j].min()
                if i_p == 'v':
                    if apply_dealiasing[j] and not self.dsg.low_nyquist_velocities_all_mps[j] is None:
                        data[j] = self.dealias_velocity(data[j], data_mask, j)
                data[j][data_mask]=np.nan
                data[j] = [data[j]]
        
        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes([i[0] for i in scantimes.values()])                
        # No meta information with using_unfilteredproduct and using_verticalpolarization is returned here, because these should be determined 
        # in nlr_datasourcespecific.py when determining which file to use
        return data, scantimes, volume_starttime, volume_endtime

                
    def extract_data(self,vol,length_header,product,scan,data_min,data_max):
        dims=[360,self.dsg.radial_bins_all['z'][scan]]
        vol.seek(length_header+(scan-1)*dims[0]*dims[1])
        
        datadepth = 8
        data_uint = np.reshape(ft.bytes_to_array(vol.read(dims[0]*dims[1]), datadepth),(dims[0],dims[1]))
        return (data_uint.astype('float32')*((data_max-data_min)/(2**datadepth-1))+data_min)





class Leonardo_vol_rainbow5():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
        
        self.filepath = None #Is only used in self.get_parameter_dicts to determine whether the parameter dicts need to be updated
        self.parameter_dicts={}
        
        

    def get_parameter_dicts(self,vol,product,productunfiltered):
        """Returns a dictionary with the relevant parameters like productdate, blobid, datadepth, product minimum, product maximum, the number of azimuths
        and some information required for obtaining the starting azimuth.
        """        
        #Determine different parameter dicts for filtered and unfiltered product variants, because I've observed that some of the 
        #important parameters can vary between the filtered and unfiltered product, for example the data depth for phidp for Helchteren.
        if productunfiltered:
            product = 'u'+product
            
        if product in self.parameter_dicts and self.parameter_dicts[product]['filepath']==self.filepath:
            parameter_dicts=self.parameter_dicts[product]['dict']
        else:  
            endXMLmarker = b"<!-- END XML -->"
            header = b""
            line = b""
            while not line.startswith(endXMLmarker):
                header += line[:-1]
                line = vol.readline()
                if len(line) == 0:
                    break
            xml_dict=xmltodict.parse(header,dict_constructor=dict)
            parameter_dicts=xml_dict['volume']['scan']['slice']
            self.parameter_dicts[product]={}
            self.parameter_dicts[product]['dict']=parameter_dicts
            self.parameter_dicts[product]['filepath']=self.filepath
            if product == 'v':
                # This parameter is used in dealias_velocity. Older versions of the vol format don't provide the 'multitripprfmode' attribute,
                # is e.g. the case for the Wideumont radar. In that case assume a default value of 'single', which at least works for Wideumont.         
                self.parameter_dicts[product]['multitripprfmode'] = xml_dict['volume']['scan'].get('pargroup', {}).get('multitripprfmode', 'single')               

        return parameter_dicts


    def get_scans_information(self,filepath,product):
        i_p = gv.i_p[product]
        self.filepath = filepath
        
        with open(filepath,'rb') as vol:
            parameter_dicts=self.get_parameter_dicts(vol,i_p,productunfiltered=False)
            n_dicts=len(parameter_dicts)
            
            #There is no need to check whether it is necessary to update the attributes, because for these .vol files obtaining them takes only
            #a very small amount of time.
            for j in range(1,n_dicts+1):
                subdict=parameter_dicts[j-1]
                self.dsg.scanangles_all['z'][j]=float(subdict['posangle'])
                try:
                    self.dsg.nyquist_velocities_all_mps[j]=float(subdict['dynv']['@max']) #Is available for all products
                except Exception:
                    self.dsg.nyquist_velocities_all_mps[j] = None
                    if product == 'v':
                        #For data from Wideumont for some older days before 2015 the Nyquist velocities are located under the key 
                        #'rawdata' instead of 'dynv'.
                        self.dsg.nyquist_velocities_all_mps[j]=float(subdict['slicedata']['rawdata']['@max']) #Is available for all products
                for i in range(j - 1, -1, -1):
                    try:
                        prf_l = float(parameter_dicts[i]['lowprf'])
                        break
                    except Exception:
                        continue
                radar_wavelength = 5.3 * 1e-2
                self.dsg.low_nyquist_velocities_all_mps[j] = self.dsg.high_nyquist_velocities_all_mps[j] = None if prf_l == 0.0 else radar_wavelength*prf_l/4.
                    
                self.dsg.radial_bins_all['z'][j]=int(subdict['slicedata']['rawdata']['@bins'])
                try:
                    self.dsg.radial_res_all['z'][j]=float(subdict['rangestep'])
                except Exception:
                    #Also because of a change in the data format for data from Wideumont. The range step is only listed for the 
                    #first scan.
                    self.dsg.radial_res_all['z'][j]=self.dsg.radial_res_all['z'][j-1]
                self.dsg.scannumbers_all['z'][j]=[j]
                
        if self.crd.radar == 'Zaventem':
            # See ODIM_hdf5.get_scans_information for explanation
            self.dsg.scanangles_all['z'] = {j:0.5 if a <= 0.6 else a for j,a in self.dsg.scanangles_all['z'].items()}
                
        extra_attrs = [self.dsg.nyquist_velocities_all_mps, self.dsg.high_nyquist_velocities_all_mps, self.dsg.low_nyquist_velocities_all_mps]
        self.dsg.scannumbers_all['z'] = bg.sort_volume_attributes(self.dsg.scanangles_all['z'], self.dsg.radial_bins_all['z'], self.dsg.radial_res_all['z'], extra_attrs)
                                                                  
        for i in self.dsg.scannumbers_all:
            for j in gv.volume_attributes_p: 
                self.dsg.__dict__[j][i] = copy.deepcopy(self.dsg.__dict__[j]['z'])
               
                                                   
    def get_data(self,filepath, j): #j is the panel 
        self.filepath = filepath
        product, i_p = self.crd.products[j], gv.i_p[self.crd.products[j]]
        scan=self.dsg.scannumbers_all[i_p][self.crd.scans[j]][self.dsg.scannumbers_forduplicates[self.crd.scans[j]]]
        with open(filepath,'rb') as vol:
            volreader=vol.read()
            vol.seek(0)
            parameter_dicts=self.get_parameter_dicts(vol,i_p,productunfiltered=self.crd.using_unfilteredproduct[j])
            subdict=parameter_dicts[scan-1]
            self.dsg.scantimes[j] = self.get_scan_timerange(parameter_dicts,scan)
    
            blobid=int(subdict['slicedata']['rawdata']['@blobid'])
            datadepth=int(subdict['slicedata']['rawdata']['@depth'])
            data_min=float(subdict['slicedata']['rawdata']['@min'])
            data_max=float(subdict['slicedata']['rawdata']['@max'])
            blobid_startangle=int(ft.from_list_or_nolist(subdict['slicedata']['rayinfo'])['@blobid'])
            datadepth_startangle=int(ft.from_list_or_nolist(subdict['slicedata']['rayinfo'])['@depth'])
            start_azimuth=self.extract_data(volreader,blobid_startangle,datadepth_startangle)[0]*(360/(2**datadepth_startangle-1)) 
            #The azimuth at which the radars starts the scan. The data is shifted by int(np.floor(start_azimuth)), to put it at the correct position
            #with a maximum deviation of 0.5 degrees.
            dims = (int(subdict['slicedata']['rawdata']['@rays']),self.dsg.radial_bins_all['z'][scan])
            
            self.dsg.data[j]=(np.reshape(self.extract_data(volreader,blobid,datadepth),dims)*((data_max-data_min)/(2**datadepth-1))+data_min)
            if product == 'c': self.dsg.data[j] *= 100.
            self.dsg.data[j] = np.roll(self.dsg.data[j][:360],int(np.round(start_azimuth)),axis=0)
                
            data_mask = self.dsg.data[j]<=self.dsg.data[j].min()
            if i_p == 'v':
                if self.crd.apply_dealiasing[j] and not self.dsg.low_nyquist_velocities_all_mps[self.crd.scans[j]] is None:
                    productunfiltered = self.crd.using_unfilteredproduct[j]
                    polarization = 'V' if self.crd.using_verticalpolarization[j] == 'V' else 'H'
                    self.dsg.data[j] = self.dealias_velocity(self.dsg.data[j], data_mask, productunfiltered, polarization, self.crd.scans[j])
                    
            self.dsg.data[j][data_mask]=self.pb.mask_values[product]
            
            
    def dealias_velocity(self, data, data_mask, productunfiltered, polarization, scan, max_range = None):
        vn_l = self.dsg.low_nyquist_velocities_all_mps[scan]
        """Important: The KMI seems to use two PRFs per radial, seems to correct velocities using the lower Nyquist velocity. Both vn_l and vn_h in apply_dual_prf_dealiasing are therefore set to vn_l.        
        Further, for Jabbeke the dual PRF errors can be all multiples of the low Nyquist velocity, and not only even multiples. Hence multiplication of vn_l by 0.5.
        Whether this is the case is determined by the parameter subdict['v']['multitripprfmode'], which can be either 'single' or 'Off'
        (and yes, the inconsistency in capitalisation of the 1st letter is correct)
        """
        deviation_factor = 1.
        if self.parameter_dicts['v']['multitripprfmode'] == 'Off':
            vn_l *= 0.5
            deviation_factor = 1.33 #The maximum allowed velocity deviation is increased slightly, to reduce the smoothing of the velocity field a little.
        radial_res = self.dsg.radial_res_all['v'][scan]
        
        source = self.dsg.source_Leonardo
        # Specify source function, because in that case the product for which the filepath is valid is returned.
        self.filepath, p = source.filepath('c', productunfiltered, polarization, source_function = source.get_scans_information)
        
        if p != 'c':
            c_array = None
        else:
            #Import the correlation coefficients
            with open(self.filepath,'rb') as vol:
                volreader=vol.read()
                vol.seek(0)
                parameter_dicts=self.get_parameter_dicts(vol,'v',productunfiltered=False)
                import_scan = self.dsg.scannumbers_all['v'][scan][self.dsg.scannumbers_forduplicates[scan]]
                subdict=parameter_dicts[import_scan-1]
    
                blobid=int(subdict['slicedata']['rawdata']['@blobid'])
                datadepth=int(subdict['slicedata']['rawdata']['@depth'])
                data_min=float(subdict['slicedata']['rawdata']['@min'])
                data_max=float(subdict['slicedata']['rawdata']['@max'])
                blobid_startangle=int(ft.from_list_or_nolist(subdict['slicedata']['rayinfo'])['@blobid'])
                datadepth_startangle=int(ft.from_list_or_nolist(subdict['slicedata']['rayinfo'])['@depth'])
                                
                start_azimuth=self.extract_data(volreader,blobid_startangle,datadepth_startangle)[0]*(360/(2**datadepth_startangle-1)) 
                #The azimuth at which the radars starts the scan. The data is shifted by int(np.floor(start_azimuth)), to put it at the correct position
                #with a maximum deviation of 0.5 degrees.
                dims = (int(subdict['slicedata']['rawdata']['@rays']),self.dsg.radial_bins_all['z'][scan])
                c_array = 100. * (np.reshape(self.extract_data(volreader,blobid,datadepth),dims)*((data_max-data_min)/(2**datadepth-1))+data_min)
                s = np.s_[:] if max_range is None else np.s_[:, :int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangles_all['v'][scan], 'gr+theta->sr') / self.dsg.radial_res_all['v'][scan]))]
                c_array = np.roll(c_array[:360][s],int(np.round(start_azimuth)),axis=0)
        
        window_size = [2, 2, 2] if self.crd.radar != 'Zaventem' else None
        n_it = 1 if self.gui.vwp.updating_vwp else self.gui.dealiasing_dualprf_n_it
        return da.apply_dual_prf_dealiasing(data, data_mask, radial_res, self.dsg.nyquist_velocities_all_mps[scan],\
        vn_l, vn_l, None, window_detection = window_size, window_correction = window_size, deviation_factor = deviation_factor, n_it = n_it,\
        c_array = c_array) #The first azimuth is scanned with a high PRF
                
        
    def get_scan_timerange(self,parameter_dicts,scan, include_dates=False):
        """Determination of the product time range. The start time is given, and also the antenna speed, and from this the time range can be determined.
        The antenna speed isn't given in all parameter dictionaries however, and if it is not given, then it is the same as in the last of the previous
        dictionaries that contains a value for the antenna speed. For this reason, the while loop below is used to obtain the antenna speed.
        """      
        startdate=ft.format_date(parameter_dicts[scan-1]['slicedata']['@date'],'YYYY-MM-DD->YYYYMMDD')
        starttime = parameter_dicts[scan-1]['slicedata']['@time']
        antspeed = None
        n = scan-1
        while n>=0:
            if 'antspeed' in parameter_dicts[n]:
                antspeed = float(parameter_dicts[n]['antspeed'])
                break
            n-=1
        if not antspeed is None:
            timerange = ft.get_timerange_from_starttime_and_antspeed(startdate,starttime,antspeed,include_dates)
            if include_dates:
                timerange, dates = timerange
                return timerange, dates
        else:
            timerange = starttime
            if include_dates:
                return timerange, [startdate]
            
        return timerange
        
    
    def get_data_multiple_scans(self,filepath,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        self.filepath = filepath
        if isinstance(apply_dealiasing, bool):
            apply_dealiasing = {j: apply_dealiasing for j in scans}
        
        data={}; scantimes={}       
        with open(filepath,'rb') as vol:
            volreader=vol.read()                                                                                
            vol.seek(0)
            parameter_dicts=self.get_parameter_dicts(vol,i_p,productunfiltered)
            for j in scans: 
                subdict=parameter_dicts[j-1]
                scantimes[j] = [self.get_scan_timerange(parameter_dicts,j)]

                blobid=int(subdict['slicedata']['rawdata']['@blobid'])
                datawidth=int(subdict['slicedata']['rawdata']['@depth'])
                data_min=float(subdict['slicedata']['rawdata']['@min'])
                data_max=float(subdict['slicedata']['rawdata']['@max'])
                blobid_startangle=int(ft.from_list_or_nolist(subdict['slicedata']['rayinfo'])['@blobid'])
                datadepth_startangle=int(ft.from_list_or_nolist(subdict['slicedata']['rayinfo'])['@depth'])
                n_radialbins=int(ft.from_list_or_nolist(subdict['slicedata']['rawdata'])['@bins'])
                                
                extracted_data=self.extract_data(volreader,blobid,datawidth)
                n_azimuthalbins=int(len(extracted_data)/n_radialbins)
                start_azimuth=self.extract_data(volreader,blobid_startangle,datadepth_startangle)[0]*(360/(2**datadepth_startangle-1))                
                data[j]=((np.reshape(extracted_data,(n_azimuthalbins,n_radialbins))*(data_max-data_min)/(2**datawidth-1)+data_min)[:360]).astype('float32')
                s = np.s_[:] if max_range is None else np.s_[:, :int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangles_all[i_p][j], 'gr+theta->sr') / self.dsg.radial_res_all[i_p][j]))]
                data[j]=np.roll(data[j][s],int(np.floor(start_azimuth)),axis=0)
                
                data_mask = data[j] <= data[j].min()
                if i_p == 'v':
                    if apply_dealiasing[j] and not self.dsg.low_nyquist_velocities_all_mps[j] is None:
                        data[j] = self.dealias_velocity(data[j], data_mask, productunfiltered, polarization, j, max_range)
                data[j][data_mask]=np.nan
                data[j] = [data[j]]

        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes([i[0] for i in scantimes.values()])
        # No meta information with using_unfilteredproduct and using_verticalpolarization is returned here, because these should be determined 
        # in nlr_datasourcespecific.py when determining which file to use
        return data, scantimes, volume_starttime, volume_endtime
                
    
    def extract_data(self,volreader,blobid,datadepth):
        start=0
        searchString = '<BLOB blobid="{0}"'.format(blobid)
        start = volreader.find(searchString.encode(), start)
        if start == -1:
            raise EOFError('Blob ID {0} not found!'.format(blobid))
        end = volreader.find(b'>', start)
        xmlstring = volreader[start:end + 1]
    
        # cheat the xml parser by making xml well-known
        xmldict = xmltodict.parse(xmlstring.decode() + '</BLOB>',dict_constructor=dict)
        cmpr = xmldict['BLOB']['@compression']
        size = int(xmldict['BLOB']['@size'])
        data = volreader[end + 2:end + 2 + size] # read blob data to string
        # if b'\\n' in data:
        #     data = data[:data.index(b'\\n')]
    
        # decompress if necessary
        # the first 4 bytes are neglected for an unknown reason
        if cmpr == "qt":
            data = zlib.decompress(data[4:])
                    
        data_uint = ft.bytes_to_array(data, datadepth).astype('float32')
        return data_uint





class KNMI_hdf5():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
    
        self.refscans=(7,9) #Scans for which the attributes are checked in order to determine whether the attributes for this radar volume are
        #different from those of the previous volume.
        
    
    
    def process_attr(self, attr):
        attr = ft.from_list_or_nolist(attr)
        return attr.decode('utf-8') if type(attr) in (bytes, np.bytes_) else attr
    
    def get_scans_information(self,filepath):
        with h5py.File(filepath,'r') as hf:
            n_datasets=len([0 for j in hf if j.startswith('scan')])
            
            """It is first checked whether it is necessary to obtain all attributes, because for the new radars of the KNMI doing so can 
            take up to 0.2 seconds. This is done by checking whether the scans in self.refscans have the same attributes as before. 
            self.refscans=(1,9) is chosen, because for scan 9 it is observed that for the old radars of the KNMI the number of radial bins 
            changed from 299 to 300 over time. Scan 7 is also included, because in August 2017 the KNMI changed the attibutes of the lowest 
            6 scans slightly (and unsorted scan 7 corresponds to sorted scan 1).
            When the attributes appear to be the same as before, they are not further obtained. If this is not the case, then they are 
            obtained for all scans.
            """
                                            
            for i in (1,2):
                if i==1: 
                    #Check whether the attributes for self.refscan are the same as for the previous volume.
                    #Also check whether the number of datasets that is present in the file has changed.
                    #self.dsg.savevalues_refscans_unsorted is defined in nlr_datasourcespecific.py
                    if self.crd.radar in self.dsg.determined_volume_attributes_radars:
                        scannumbers_all_before=self.dsg.determined_volume_attributes_radars[self.crd.radar]['scannumbers_all']['z']
                        n_datasets_before=sum([len(scannumbers_all_before[j]) for j in scannumbers_all_before if not j in gv.plain_products])
                    else: n_datasets_before=0
                    
                    if self.dsg.savevalues_refscans_unsorted[self.crd.radar]==[] or n_datasets!=n_datasets_before: 
                        continue #Continue with i=2, because attributes for all scans must be obtained.
                    else: scans=self.refscans
                else: 
                    scans=range(1,20)
                    for j in gv.volume_attributes_p: #Reset the volume attributes, because otherwise the refscans are listed first
                        #in the dictionaries that are updated for i==1, such that their order for i=2 gets incompatible 
                        #with the order of the dictionaries that are only updated for i=2.
                        self.dsg.__dict__[j]['z'] = {}
                for j in scans:
                    try:
                        attrs=hf['scan'+str(j)].attrs
                    except Exception: continue
                    self.dsg.scanangles_all['z'][j]=float(self.process_attr(attrs['scan_elevation']))
                    self.dsg.radial_res_all['z'][j]=float(self.process_attr(attrs['scan_range_bin']))
                    self.dsg.radial_bins_all['z'][j]=int(ft.r1dec(int(self.process_attr(attrs['scan_number_range']))))
                    if i==2:
                        #These parameters are not used to determine whether the parameters must be updated, because no change in them has been
                        #found during the period for which data is available from the KNMI, except for the transition to the new radars, which
                        #is also apparent in changes in other parameters.
                        calibrationgroup=hf['scan'+str(j)]['calibration']
                        calibration_formula=self.process_attr(calibrationgroup.attrs['calibration_V_formulas'])
                        self.dsg.nyquist_velocities_all_mps[j]=float(ft.r1dec(abs(float(calibration_formula[calibration_formula.index('+')+1:]))))
                        
                        prf_l = attrs['scan_low_PRF'][0]
                        radar_wavelength = 5.3e-2
                        self.dsg.low_nyquist_velocities_all_mps[j] = None if prf_l == 0.0 else radar_wavelength*prf_l/4.
                        self.dsg.high_nyquist_velocities_all_mps[j] = None if prf_l == 0.0 else radar_wavelength*attrs['scan_high_PRF'][0]/4.
                        
                if i==1:
                    values_refscans_now=[[self.dsg.scanangles_all['z'][j],self.dsg.radial_res_all['z'][j],self.dsg.radial_bins_all['z'][j]] for j in self.refscans]
                    if self.dsg.savevalues_refscans_unsorted[self.crd.radar]==values_refscans_now: 
                        #Use the saved values
                        self.dsg.restore_previous_attributes_radar()
                        return #No further evaluation of this function is required, so return
                    
            try:
                self.dsg.savevalues_refscans_unsorted[self.crd.radar]=[[self.dsg.scanangles_all['z'][j],self.dsg.radial_res_all['z'][j],self.dsg.radial_bins_all['z'][j]] for j in self.refscans]
            except Exception: self.dsg.savevalues_refscans_unsorted[self.crd.radar]=[]
            
            #Check whether the volume can be divided up into 2 parts.
            double_volume=True if len(self.dsg.scanangles_all['z'])==16 else False
            if double_volume:
                self.dsg.scans_doublevolume=[]
                for j in self.dsg.scanangles_all['z']:
                    if j>1 and not self.dsg.scanangles_all['z'][j]==90. and self.dsg.scanangles_all['z'][j]>self.dsg.scanangles_all['z'][j-1]:
                        #This condition is only satisfied for the new radars of the KNMI
                        self.dsg.scans_doublevolume.append([i for i in self.dsg.scanangles_all['z'] if i<=j and not self.dsg.scanangles_all['z'][i]==90.])
                        #Exclude scan 15, because for the new KNMI radars there are 3 scans with a scanangle of 0.3 degrees, of which 2 have
                        #the same high range, and 1 has a lower range. Only 2 of them are needed, and therefore the latter scan, scan 15, is excluded.
                        self.dsg.scans_doublevolume.append([i for i in self.dsg.scanangles_all['z'] if i>j and not i==15])

            extra_attrs = [self.dsg.nyquist_velocities_all_mps, self.dsg.high_nyquist_velocities_all_mps, self.dsg.low_nyquist_velocities_all_mps]
            self.dsg.scannumbers_all['z'] = bg.sort_volume_attributes(self.dsg.scanangles_all['z'], self.dsg.radial_bins_all['z'], self.dsg.radial_res_all['z'], extra_attrs)
            
            for i in range(len(self.dsg.scans_doublevolume)):
                for j in range(len(self.dsg.scans_doublevolume[i])):
                    self.dsg.scans_doublevolume[i][j] = [k for k,l in self.dsg.scannumbers_all['z'].items() if self.dsg.scans_doublevolume[i][j] in l][0]
                self.dsg.scans_doublevolume[i] = sorted(self.dsg.scans_doublevolume[i])
            
            for i in self.dsg.scannumbers_all:
                for j in gv.volume_attributes_p: 
                    self.dsg.__dict__[j][i] = copy.deepcopy(self.dsg.__dict__[j]['z'])
                                                                                                                        
        
    def try_various_combis_of_filter_and_polarization(self,product,scangroup,productunfiltered,polarization):
        """This function first checks whether the product with the desired properties (filtering and polarization) is available. If not
        then it tries various other combinations of filtering and polarization:
        """
        success=False
        u_prefix = 'u'*productunfiltered
        if product == 'd' and not 'scan_ZDR_data' in scangroup:
            polarization = 'H' # Prevent using_verticalpolarization from becoming True
            productname = u_prefix+'Zv'
        else:
            productname = u_prefix+gv.productnames_KNMI[product]+'v'*(polarization == 'V')
        dataset = 'scan_'+productname+'_data'

        using_unfilteredproduct=False; using_verticalpolarization=False
        if dataset in scangroup:
            success=True
            if productunfiltered: using_unfilteredproduct=True
            if polarization=='V': using_verticalpolarization=True
        if not success and productunfiltered and polarization=='V': #Try the unfiltered product with horizontal polarization
            productname=productname[:-1]
            dataset='scan_'+productname+'_data'
            if dataset in scangroup: success=True; using_unfilteredproduct=True
        if not success and productunfiltered: #Try the filtered product with the selected polarization if the unfiltered one is unavailable
            productname=productname[1:]+('v' if polarization=='V' else '')
            dataset='scan_'+productname+'_data'
            if dataset in scangroup: 
                success=True
                if polarization=='V': using_verticalpolarization=True
        if not success and polarization=='V': #Try the filtered product with horizontal polarization
            productname=productname[:-1]
            dataset='scan_'+productname+'_data'
            if dataset in scangroup: success=True
        return productname,dataset,success,using_unfilteredproduct,using_verticalpolarization
    
    def get_data(self,filepath, j): #j is the panel
        with h5py.File(filepath,'r') as hf:
            #Use the import product instead of the product itself, because self.dsg.scannumbers_all etc. contain only keys for the import products.
            product, i_p = self.crd.products[j], gv.i_p[self.crd.products[j]]
            scan=self.dsg.scannumbers_all[i_p][self.crd.scans[j]][self.dsg.scannumbers_forduplicates[self.crd.scans[j]]]
            scangroup=hf['scan'+str(scan)]
            
            productname, dataset, success, self.crd.using_unfilteredproduct[j], self.crd.using_verticalpolarization[j] =\
                self.try_various_combis_of_filter_and_polarization(i_p,scangroup,self.crd.productunfiltered[j],self.crd.polarization[j])
            if not success:
                raise Exception
                    
            self.dsg.scantimes[j]=self.get_scan_timerange(scangroup)
            calibrationgroup=scangroup['calibration']
            calibration_formula=self.process_attr(calibrationgroup.attrs['calibration_'+productname+'_formulas'])
            gain=float(calibration_formula[calibration_formula.index('=')+1:calibration_formula.index('*')])
            offset=float(calibration_formula[calibration_formula.index('+')+1:])
            
            if product == 'd' and not 'ZDR' in dataset:
                data_Zv=np.array(scangroup[dataset],dtype='float32')*gain+offset
                data_Zh=np.array(scangroup['scan_'+'u'*self.crd.using_unfilteredproduct[j]+'Z_data'],dtype='float32')*gain+offset
                
                data_maskvalue=data_Zh.min()
                data_mask = (data_Zh == data_maskvalue) | (data_Zv == data_maskvalue)
                self.dsg.data[j]=dt.calculate_zdr_array(data_Zh,data_Zv)
            else:
                self.dsg.data[j] = np.array(scangroup[dataset],dtype='float32')*gain+offset
                if product == 'c': self.dsg.data[j] *= 100.
                data_mask = self.dsg.data[j] == self.dsg.data[j].min()
                        
            if i_p == 'v': 
                if not self.crd.productunfiltered[j] and int(self.crd.date) >= 20200514:
                    #At this date KNMI removed their filter on the velocity, meaning that the V product became unfiltered velocity.
                    #Since this is not indicated by a change of name, I manually assign the unfiltered tag to this version of the velocity.
                    #I also add a filtered version of the velocity by applying the old filter myself.
                    try:
                        #Obtain array with Signal Quality Index
                        productname = gv.productnames_KNMI['q']
                        scangroup=hf['scan'+str(scan)]
                        calibrationgroup=scangroup['calibration']
                        calibration_formula=self.process_attr(calibrationgroup.attrs['calibration_'+productname+'_formulas'])
                        gain=float(calibration_formula[calibration_formula.index('=')+1:calibration_formula.index('*')])
                        offset=float(calibration_formula[calibration_formula.index('+')+1:])
                        sqi_array = np.array(scangroup['scan_'+productname+'_data'],dtype='float32')*gain+offset
                        
                        productname = gv.productnames_KNMI['z']
                        scangroup=hf['scan'+str(scan)]
                        calibrationgroup=scangroup['calibration']
                        calibration_formula=self.process_attr(calibrationgroup.attrs['calibration_'+productname+'_formulas'])
                        gain=float(calibration_formula[calibration_formula.index('=')+1:calibration_formula.index('*')])
                        offset=float(calibration_formula[calibration_formula.index('+')+1:])
                        z_array = np.array(scangroup['scan_'+productname+'_data'],dtype='float32')*gain+offset

                        # sqi_array = ft.get_window_mean(sqi_array, data_mask, [2,4,2])
                        z_crit = z_array < 30.
                        sqi_filter = (sqi_array < 0.3) & z_crit
                        if not self.dsg.low_nyquist_velocities_all_mps[self.crd.scans[j]] is None:
                            #Only for the dual PRF scans: Filter also the radar bins to the right (in the clockwise direction), since velocity
                            #estimates for these bins are effected by the low quality of the already selected bins (a characteristic of the dual
                            #PRF technique).
                            sqi_filter |= np.roll(sqi_filter, 1, axis=0) & z_crit
                        data_mask |= sqi_filter
                        self.crd.using_unfilteredproduct[j] = False
                    except Exception:
                        self.crd.using_unfilteredproduct[j] = True
                elif int(self.crd.date) >= 20200514:
                    self.crd.using_unfilteredproduct[j] = True

                if self.crd.apply_dealiasing[j] and not self.dsg.low_nyquist_velocities_all_mps[self.crd.scans[j]] is None:
                    self.dsg.data[j] = self.dealias_velocity(hf, self.dsg.data[j], data_mask, self.crd.scans[j])                    
                    
            self.dsg.data[j][data_mask]=self.pb.mask_values[product]
                
                
    def dealias_velocity(self, hf, data, data_mask, scan, max_range = None): #hf is the hdf5 object
        vn_l = self.dsg.low_nyquist_velocities_all_mps[scan]
        """Important: For the old radars the KNMI used two PRFs per radial, and corrects velocities using the lower Nyquist velocity. Both vn_l and vn_h in 
        apply_dual_prf_dealiasing are therefore set to vn_l.
        Further, for some years (not all!) the dual PRF errors can be all multiples of the low Nyquist velocity, and not only even multiples. Hence multiplication of vn_l by 0.5.
        Unfortunately it is not stated in the files for which files this is the case, so this is determined empirically, and is found to be the case at
        least between 2011-09-10 and 2014-01-03. As halving the Nyquist velocities leads to much more smoothing of the velocity field, it should only
        be done when really necessary.
        """
        deviation_factor = 1.
        c_array = None #Array with correlation coefficients. None if unavailable
        mask_all_nearzero_velocities = False
        if len(self.dsg.scanangles_all['z']) == 15: #For the new radars
            window_detection = [0,3,6,9,3,6,0]
            window_correction = [0,3,6,3,0]
            vn_h = self.dsg.high_nyquist_velocities_all_mps[scan]
            
            #Obtain array with correlation coefficients
            productname = gv.productnames_KNMI['c']
            import_scan = self.dsg.scannumbers_all['v'][scan][self.dsg.scannumbers_forduplicates[scan]]
            scangroup=hf['scan'+str(import_scan)]
            calibrationgroup=scangroup['calibration']
            calibration_formula=self.process_attr(calibrationgroup.attrs['calibration_'+productname+'_formulas'])
            gain=float(calibration_formula[calibration_formula.index('=')+1:calibration_formula.index('*')])
            offset=float(calibration_formula[calibration_formula.index('+')+1:])
            s = s = np.s_[:] if max_range is None else np.s_[:, :int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangles_all['v'][scan], 'gr+theta->sr') / self.dsg.radial_res_all['v'][scan]))]
            c_array = 100. * (np.array(scangroup['scan_'+productname+'_data'][s],dtype='float32')*gain+offset)
        else:
            # For the old radars there are less aliasing errors, except in the case that the dual PRF
            # errors can be all multiples of the low Nyquist velocity. If that is not the case, then the window size can be pretty small.
            window_detection = window_correction = [1,1,1] 
            if int(self.crd.date) >= 20110910 and int(self.crd.date) <= 20140103: 
                # Take the window size somewhat larger, because there are much more aliasing errors.
                vn_l *= 0.5
                window_detection = window_correction = [2,2,2]
                deviation_factor = 1.33 #The maximum allowed velocity deviation is increased slightly, to reduce the smoothing of the velocity field a little.
            vn_h = vn_l
            mask_all_nearzero_velocities = True #For the old radars there are usually fairly many bins with near-zero velocities in regions of low reflectivity.
            #By masking them they do not hinder the dealiasing of the surrounding bins.
        
        radial_res = self.dsg.radial_res_all['v'][scan]
        n_it = 1 if self.gui.vwp.updating_vwp else self.gui.dealiasing_dualprf_n_it
        return da.apply_dual_prf_dealiasing(data, data_mask, radial_res, self.dsg.nyquist_velocities_all_mps[scan],\
        vn_l, vn_h, 'h', window_detection, window_correction, deviation_factor = deviation_factor, n_it = n_it,\
        c_array = c_array, mask_all_nearzero_velocities = mask_all_nearzero_velocities) #The first azimuth is scanned with a high PRF

                
    def get_scan_timerange(self, scangroup):
        datetimestring=self.process_attr(scangroup.attrs['scan_datetime'])
        startdate = ft.format_date(datetimestring[-24:-13], 'DD-MMMl-YYYY->YYYYMMDD')
        starttime = datetimestring[-12:-4]
        antspeed = float(ft.from_list_or_nolist(scangroup.attrs['scan_antenna_velocity']))
        return ft.get_timerange_from_starttime_and_antspeed(startdate,starttime,antspeed)
                    
                
    def get_data_multiple_scans(self,filepath,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        if isinstance(apply_dealiasing, bool):
            apply_dealiasing = {j: apply_dealiasing for j in (scans[0] if isinstance(scans[0], list) else scans)}
        
        data={}; scantimes={}; meta={}
        with h5py.File(filepath,'r') as hf:
            for j in scans: 
                data[j]=[]; scantimes[j]=[]
                for scan in self.dsg.scannumbers_all[i_p][j]:
                    scangroup=hf['scan'+str(scan)]
                    scantimes[j] += [self.get_scan_timerange(scangroup)]
                    
                    productname, dataset, success, meta['using_unfilteredproduct'], meta['using_verticalpolarization'] =\
                        self.try_various_combis_of_filter_and_polarization(i_p,scangroup,productunfiltered,polarization)
                                        
                    calibrationgroup=scangroup['calibration']
                    calibration_formula=self.process_attr(calibrationgroup.attrs['calibration_'+i_p.upper()+'_formulas'])
                    gain=float(calibration_formula[calibration_formula.index('=')+1:calibration_formula.index('*')])
                    offset=float(calibration_formula[calibration_formula.index('+')+1:])
                        
                    s = np.s_[:] if max_range is None else np.s_[:, :int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangles_all[i_p][j], 'gr+theta->sr') / self.dsg.radial_res_all[i_p][j]))]
                    if product == 'd' and not 'ZDR' in dataset:
                        data_Zv=np.array(scangroup[dataset][s],dtype='float32')*gain+offset
                        data_Zh=np.array(scangroup['scan_'+productname[:-1]+'_data'][s],dtype='float32')*gain+offset
                        data_maskvalue=data_Zh.min()
                        data_mask = (data_Zh == data_maskvalue) | (data_Zv == data_maskvalue)
                        data[j]+=[dt.calculate_zdr_array(data_Zh,data_Zv)]
                    else:
                        data[j]+=[np.array(scangroup[dataset][s],dtype='float32')*gain+offset]
                        if product == 'c': data[j][-1] *= 100.
                        data_mask = data[j][-1] <= data[j][-1].min()
                        # data[j][-1] += 3.
                        if i_p == 'v':
                            if self.crd.radar == 'Herwijnen' and self.crd.date[:4] == '2019' and polarization == 'H' and 'scan_Vv_data' in scangroup:
                                """IMPORTANT: Under these conditions empty velocity bins for the horizontally polarized channel are filled with velocity bins
                                from the vertically polarized channel, when available. This is done because there was an issue with the horizontal channel
                                of Herwijnen in 2019, that reduces data availability for the VVP retrieval.
                                """
                                data_Vv = np.array(scangroup['scan_Vv_data'][s],dtype='float32')*gain+offset
                                data[j][-1][data_mask] = data_Vv[data_mask]
                                data_mask = data[j][-1] <= data[j][-1].min()
                            
                            if apply_dealiasing[j] and not self.dsg.low_nyquist_velocities_all_mps[j] is None:
                                data[j][-1] = self.dealias_velocity(hf, data[j][-1], data_mask, j, max_range)    
                    data[j][-1][data_mask] = np.nan
        
        scantimes_list = []
        for i in scantimes:
            scantimes_list += scantimes[i]
        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes(scantimes_list)
        return data, scantimes, volume_starttime, volume_endtime, meta
        
        



class ODIM_hdf5():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
                
        self.product_names = {'DBZH':'z','TH':'uz','UZ':'uz','VRAD':'v','VRADH':'v','UV':'uv','WRAD':'w','WRADH':'w','UW':'uw',
                              'ZDR':'d','AttCorrZDRCorr':'d','RHOHV':'c','URHOHV':'c','PHIDP':'p','UPHIDP':'p',
                              'KDPCorr':'k','KDP':'k','SQI':'q','SQIH':'q','SNR':'i','STAT2':'STAT2'}
        
    
    def find_product_dataset(self, scangroup, product):
        datasets = [key for key in scangroup if key.startswith('data')]
        for d in datasets:
            dataset = scangroup[d]
            p_name = dataset['what'].attrs['quantity'].decode('utf-8')
            if self.product_names.get(p_name, '') == product:
                return dataset
            elif not p_name in self.product_names:
                if p_name == 'SCAN' and len(datasets) == 1:
                    # In this case the product name is not correctly specified, and with only 1 dataset present it is assumed
                    # that it contains the requested product
                    return dataset
            
    def get_scans_information(self,filepath,product='z'):
        with h5py.File(filepath,'r') as hf: 
            scans = [int(j[7:]) for j in hf if j.startswith('dataset')]
            for j in scans:
                try:
                    scangroup=hf['dataset'+str(j)]; attrs=scangroup['where'].attrs
                except Exception: 
                    continue
                if 'how' in scangroup and scangroup['how'].attrs.get('task', '')[:3] == b'RHI':
                    # At least ESTEA files include some RHI scans
                    continue
                self.dsg.scanangles_all['z'][j]=ft.rndec(float(attrs['elangle']), 2)
                self.dsg.radial_bins_all['z'][j]=int(attrs['nbins'])
                self.dsg.radial_res_all['z'][j]=float(attrs['rscale']/1000.)
                
                self.dsg.nyquist_velocities_all_mps[j] = None
                if product == 'v':
                    try:
                        self.dsg.nyquist_velocities_all_mps[j] = abs(float(scangroup['how'].attrs['NI']))
                    except Exception:
                        dataset = self.find_product_dataset(scangroup, product)
                        self.dsg.nyquist_velocities_all_mps[j] = abs(float(dataset['what'].attrs['offset']))
                
                try:
                    radar_wavelength = float(hf['how'].attrs['wavelength']) * 1e-2
                    if not self.crd.radar in ('Jabbeke', 'Wideumont'):
                        if 'lowprf' in scangroup['how'].attrs:
                            prf_l = float(scangroup['how'].attrs['lowprf'])
                            prf_h = float(scangroup['how'].attrs['highprf'])
                        elif 'lowprf' in hf['how'].attrs:
                            prf_l = float(hf['how'].attrs['lowprf'])
                            prf_h = float(hf['how'].attrs['highprf'])
                        else:
                            prf_h = float(hf['how'].attrs['prf'])
                            prf_fac = float(hf['how'].attrs['prffac'])
                            if prf_fac == 1:
                                prf_l = prf_h # In this case the scan is mono-PRF, and setting prf_l = prf_h is used below
                            else:
                                # Seems like the PRF factor provided in file is not correct, so calculate correct one as vn_e/vn_h
                                prf_fac = int(round(self.dsg.nyquist_velocities_all_mps[j]/(radar_wavelength*prf_h/4.)))
                                prf_l = prf_h*prf_fac/(prf_fac+1)
                    else:
                        #The prfs are listed at different locations for Zaventem compared to Jabbeke and Wideumont, but more importantly: The values are wrong!
                        #They are therefore provided manually. Further, only the low prf is needed.
                        prf_l = 800 if self.crd.radar == 'Jabbeke' else 960.
                        prf_h = None
                    vn_l = radar_wavelength*prf_l/4
                    vn_h = radar_wavelength*prf_h/4
                    if gv.data_sources[self.crd.radar] == 'Austro Control':
                        vn_l = vn_h
                    elif self.crd.radar in ('Jabbeke', 'Wideumont'):
                        vn_h = vn_l
                    if prf_l == prf_h and 0.9 < vn_l/self.dsg.nyquist_velocities_all_mps[j] < 1.1:
                        raise Exception # In this case the scan is mono-PRF
                    self.dsg.low_nyquist_velocities_all_mps[j], self.dsg.high_nyquist_velocities_all_mps[j] = vn_l, vn_h
                except Exception:
                    self.dsg.high_nyquist_velocities_all_mps[j] = self.dsg.low_nyquist_velocities_all_mps[j] = None
        
        
        if self.crd.radar == 'Zaventem':
            """If the mode 'Hazardous' is used for the scanning strategy, then there are per volume 3 scans with scanangle 0.6, and one with 
            scanangle 0.51, which with respect to time fits in the temporal gap that is present in the sequence of 0.6-degree scans. For this reason
            it is assumed that they all have the same scanangle of 0.5 degrees, which is distinguished from the long-range scan with a scanangle of
            0.5 degrees by setting the scanangle to 0.51 degrees.
            Later in this function when the need to distinguish it by means of scanangle from the long-range scan, the scanangle is set to 0.5 degrees.
            """
            self.dsg.scanangles_all['z'] = {j:0.51 if a == 0.6 else a for j,a in self.dsg.scanangles_all['z'].items()}
        
        extra_attrs = [self.dsg.nyquist_velocities_all_mps, self.dsg.high_nyquist_velocities_all_mps, self.dsg.low_nyquist_velocities_all_mps]
        self.dsg.scannumbers_all['z'] = bg.sort_volume_attributes(self.dsg.scanangles_all['z'], self.dsg.radial_bins_all['z'], self.dsg.radial_res_all['z'], extra_attrs)

        for p in self.dsg.scannumbers_all:
            for j in gv.volume_attributes_p: 
                self.dsg.__dict__[j][p] = copy.deepcopy(self.dsg.__dict__[j]['z'])
                    
                
        if self.crd.radar == 'Zaventem':                
            """For this radar it is possible that not all scans are available for each product. It is therefore necessary to determine the volume 
            attributes for all products. 
            In creating the dictionaries with volume attributes, it is always assumed that reflectivity is available for all scans. The dictionaries 
            with volume attributes must for each product contain the same keys (scans). If for some scan other products are missing, then the scan 
            whose scanangle is closed to the z-scanangle should be displayed instead. 
                        
            In order to do this, first a list with the scanangles of the datasets in the p-file is created, where p refers to the product for which
            volume attributes are determined.
            Case 1: If for a scan the z-scanangle is also in the p-scanangles, then it is assumed that all other parameters are the same as for 'z', 
            and the scan information is simply copied. The exception is self.dsg.scannumbers_all, because the map from scan to dataset can still be 
            different. 
            Case 2: If for that scan the z-scanangle is not in the p-scanangles, then the p-scanangle that is closest to the z-scanangle is chosen
            instead, and information for the z-scan corresponding to that p-scanangle is copied (again except for self.dsg.scannumbers_all).
            """
            products = ['v','w']
            for p in products:
                try:
                    filepath_p = self.crd.directory+'/'+self.crd.date+self.crd.time+'00.rad.'+gv.radar_ids[self.crd.radar]+'.pvol.'+p+'rad.scan_abc.hdf'
                    with h5py.File(filepath_p, 'r') as hf:
                        # Sorting is necessary, because otherwise the scans are sorted in order of increasing first digit.                        
                        scans_p = sorted([int(j[7:]) for j in list(hf) if j[:7]=='dataset'])                        
                        scanangles_p = {j:ft.rndec(float(hf['dataset'+str(j)]['where'].attrs['elangle']), 2) for j in scans_p}
                        # For same reason as mentioned above
                        scanangles_p = {j:0.51 if a == 0.6 else a for j,a in scanangles_p.items()}
                        
                        if p == 'v':
                            nyquist_velocities_all_mps = {j:abs(float(hf['dataset'+str(j)]['data1']['what'].attrs['offset'])) for j in scans_p}
                            scansz_low_nyquist_velocities_all_mps = self.dsg.low_nyquist_velocities_all_mps
                            scansz_high_nyquist_velocities_all_mps = self.dsg.high_nyquist_velocities_all_mps
                            
                        scanangles_p_arr = np.array(list(scanangles_p.values()))
                        scanangles_p_inverse = {i:[j for j in scanangles_p if scanangles_p[j] == i] for i in scanangles_p_arr}
                        # scanangles_p_inverse includes possibly duplicates, scanangles_z_inverse does not.
                        scanangles_z_inverse = {j:i for i,j in self.dsg.scanangles_all['z'].items()}
                        for j in self.dsg.scanangles_all['z']:
                            if self.dsg.scanangles_all['z'][j] in scanangles_p_arr:
                                zscan = j
                                self.dsg.scannumbers_all[p][j] = scanangles_p_inverse[self.dsg.scanangles_all['z'][j]]
                            else:
                                # Choose the closest p-scanangle.
                                closest_scananglep = scanangles_p_arr[np.abs(scanangles_p_arr-self.dsg.scanangles_all['z'][j]).argmin()]
                                zscan = scanangles_z_inverse[closest_scananglep]
                                self.dsg.scannumbers_all[p][j] = scanangles_p_inverse[closest_scananglep]
                            
                            # If in the case of duplicate scans the p-file contains less duplicates than the z-file, then for the duplicates missing
                            # in the p-file the last duplicate in the p-file is displayed again instead.
                            diff = len(self.dsg.scannumbers_all['z'][j])-len(self.dsg.scannumbers_all[p][j])
                            self.dsg.scannumbers_all[p][j] += [self.dsg.scannumbers_all[p][j][-1] for i in range(diff)]
                            # It is also possible that the opposite is the case, and then self.dsg.scannumbers_all[p][j] must be shortened.
                            self.dsg.scannumbers_all[p][j] = self.dsg.scannumbers_all[p][j][:len(self.dsg.scannumbers_all['z'][j])]
        
                            for a in (_ for _ in gv.volume_attributes_p if _ != 'scannumbers_all'):
                                self.dsg.__dict__[a][p][j] = self.dsg.__dict__[a]['z'][zscan]   
                            if p == 'v':
                                self.dsg.nyquist_velocities_all_mps[j] = nyquist_velocities_all_mps[self.dsg.scannumbers_all['v'][j][0]]
                                self.dsg.low_nyquist_velocities_all_mps[j] = scansz_low_nyquist_velocities_all_mps[zscan]
                                self.dsg.high_nyquist_velocities_all_mps[j] = scansz_high_nyquist_velocities_all_mps[zscan]
                                
                except Exception:
                    # If a product is unavailable, then simply continue with the next one.
                    continue
                            
            for p in self.dsg.scanangles_all:
                self.dsg.scanangles_all[p] = {j:0.5 if a == 0.51 else a for j,a in self.dsg.scanangles_all[p].items()}
                
                                
    def read_data(self, filepath, product, scan, apply_dealiasing=True, productunfiltered=False, panel=None, data_mask=False, check_azis=True):
        # When a product is not included in gv.i_p, one can provide a productname instead.
        # This requires adding the map productname:productname to self.product_names
        i_p = gv.i_p.get(product, product) # In case of productname
        with h5py.File(filepath,'r') as hf:
            _i_p = i_p if i_p in self.dsg.scannumbers_all else 'z' # In case of productname
            dataset_n = self.dsg.scannumbers_all[_i_p][scan][self.dsg.scannumbers_forduplicates[scan]]
            scangroup = hf[f'dataset{dataset_n}']
                
            unfiltered = productunfiltered or gv.data_sources[self.crd.radar] == 'ARPAV'
            p = 'u'*unfiltered+i_p
            dataset = self.find_product_dataset(scangroup, p)
            if not dataset:
                p = i_p
                dataset = self.find_product_dataset(scangroup, p)
                if not dataset:
                    raise Exception(p+' not present in file')
            elif panel != None:
                self.crd.using_unfilteredproduct[panel] = productunfiltered
            
            gain = float(dataset['what'].attrs['gain'])
            offset = float(dataset['what'].attrs['offset'])
            nodata = int(dataset['what'].attrs['nodata'])
            undetect = int(dataset['what'].attrs['undetect'])
            data = dataset['data'][:]
            data_mask = (data == nodata) | (data == undetect)
            data = data.astype('float32')*gain+offset
            
            if i_p == 'c': 
                data *= 100./(254*gain if self.crd.radar in gv.radars['DMI'] else 1.)
            elif i_p == 'k' and gv.data_sources[self.crd.radar] == 'DHMZ':
                data_mask |= (data == 0.)
            elif i_p == 'p':
                data = np.rad2deg(data) if data.max() < 10 else data
                data %= 360.
            elif p[-1] == 'v' and data.max() < 2.:
                data *= self.dsg.nyquist_velocities_all_mps[scan]
                
            if gv.data_sources[self.crd.radar] == 'Austro Control' and product != 'z':
                if not productunfiltered:
                    z_filepath = self.dsg.source_AustroControl.filepath('z')
                    z_data, z_data_mask = self.read_data(z_filepath, 'z', scan, data_mask=data_mask, check_azis=False)[:2]
                    data_mask |= z_data_mask
                    
                    if i_p == 'v':
                        phi = data * np.pi/self.dsg.nyquist_velocities_all_mps[scan]
                        window = [1,2,3,2,1]
                        v_std = ft.get_window_phase_stdev(phi, data_mask, window)*self.dsg.nyquist_velocities_all_mps[scan]/np.pi
                        
                        z_std = ft.get_window_stdev(z_data, data_mask, window)
                        z_thres = (z_data < 10 ) | ((z_data >= 10) & (z_data < 25) & (z_std < 3))
                        data_mask[z_thres & (v_std > 10)] = True

                elif panel != None:
                    self.crd.using_unfilteredproduct[panel] = True  
                    
            elif gv.data_sources[self.crd.radar] == 'ESTEA' and i_p != 'z':
                if not productunfiltered:
                    kwargs = {'panel':None, 'check_azis':False}
                    SQI_data = data.copy() if i_p == 'q' else self.read_data(filepath, 'q', scan, productunfiltered=True, **kwargs)[0]
                    Z_data = self.read_data(filepath, 'z', scan, **kwargs)[0] # Use filtered Z
                    # For the dual-PRF scans use a higher threshold, because it often has issues with 2nd-trip echoes
                    # that can have fairly high dBZ values
                    threshold = 10. if self.dsg.low_nyquist_velocities_all_mps[scan] is None else 30.
                    data_mask |= (SQI_data < 0.25) & (np.isnan(Z_data) | (Z_data < threshold))
                elif panel != None:
                    self.crd.using_unfilteredproduct[panel] = True
                
            elif gv.data_sources[self.crd.radar] in ('ARPA FVG', 'ARPAV'):
                if not productunfiltered:
                    key = filepath+str(scan)
                    if hasattr(self, 'extra_mask') and self.extra_mask['key'] == key:
                        extra_mask = self.extra_mask['mask']
                    else:
                        kwargs = {'apply_dealiasing':False, 'productunfiltered':True, 'panel':None, 'check_azis':False}
                        # copy is used for storing in self.p_data, since without copy the arrays would be altered in subsequent
                        # operations on the data array
                        if gv.data_sources[self.crd.radar] == 'ARPAV':
                            Z, Z_mask = (data.copy(), data_mask) if i_p == 'z' else self.read_data(filepath, 'z', scan, **kwargs)[:2]
                            STAT2, STAT2_mask = self.read_data(filepath, 'STAT2', scan, **kwargs)[:2]
                            SQI, SQI_mask = (data.copy(), data_mask) if i_p == 'q' else self.read_data(filepath, 'q', scan, **kwargs)[:2]
                            W, W_mask = (data.copy(), data_mask) if i_p == 'w' else self.read_data(filepath, 'w', scan, **kwargs)[:2]
                            V, V_mask = (data.copy(), data_mask) if i_p == 'v' else self.read_data(filepath, 'v', scan, **kwargs)[:2]
                            ground_clutter_mask = (STAT2_mask | (STAT2 < 2)) & (np.abs(V) < 0.5) & ((W < 2.5) | W_mask)
                            if self.dsg.scanangles_all[i_p][scan] < 3:
                                extra_mask = Z_mask | ground_clutter_mask | ((SQI < 0.05) & (Z < 25.)) | SQI_mask
                            else:
                                extra_mask = Z_mask | ground_clutter_mask | (((SQI < 0.05) | SQI_mask) & (Z < 25.))
                        elif gv.data_sources[self.crd.radar] == 'ARPA FVG':
                            SNR_data = data.copy() if i_p == 'i' else self.read_data(filepath, 'i', scan, **kwargs)[0]
                            W_data = data.copy() if i_p == 'w' else self.read_data(filepath, 'w', scan, **kwargs)[0]
                            V_data, V_data_mask = (data.copy(), data_mask) if i_p == 'v' else self.read_data(filepath, 'v', scan, **kwargs)[:2]
                            ground_clutter_mask = ((np.abs(V_data) < 0.5) & (W_data < 0.75))
                            extra_mask = np.isnan(SNR_data) | (SNR_data < 3.) | ground_clutter_mask# | (SQI_data < 0.1)                            
                        
                        self.extra_mask = {'mask':extra_mask, 'key':key}
                    data_mask |= extra_mask
                    
                    if i_p == 'v' and gv.data_sources[self.crd.radar] == 'ARPA FVG':
                        phi = data * np.pi/self.dsg.nyquist_velocities_all_mps[scan]
                        window = [1,2,3,2,1]
                        v_std = ft.get_window_phase_stdev(phi, data_mask, window)*self.dsg.nyquist_velocities_all_mps[scan]/np.pi
                        
                        Z, Z_mask = self.read_data(filepath, 'z', scan, panel=None, check_azis=False)[:2]
                        Z_std = ft.get_window_stdev(Z, Z_mask, window)
                        z_thres = Z_mask | (Z < 10 ) | ((Z >= 10) & (Z < 25) & (Z_std < 3))
                        
                        data_mask[z_thres & (v_std > 10)] = True
                elif panel != None:
                    self.crd.using_unfilteredproduct[panel] = True
            
            data[data_mask] = np.nan
            if i_p == 'v':
                if apply_dealiasing and self.dsg.low_nyquist_velocities_all_mps[scan] != None:
                    data = self.dealias_velocity(data, data_mask, scan)
                    
                if panel != None and apply_dealiasing and 'Unet VDA' in self.gui.dealiasing_setting:
                    # Performing mono PRF dealiasing must be done before the regridding that's done below
                    data = self.dsg.perform_mono_prf_dealiasing(panel, data)
            
            if check_azis and 'how' in scangroup and 'startazA' in scangroup['how'].attrs:
                da = int(round(360/ft.from_list_or_nolist(scangroup['where'].attrs['nrays'])))
                azis = scangroup['how'].attrs['startazA']
                stop_azis = scangroup['how'].attrs['stopazA']
                # For at least 1 radar (Fossalon, ARPA FVG) it has been observed that radials are 'skipped', in which
                # case empty radials need to be introduced in the data array. This is done here.
                select = np.append(0., ft.angle_diff(stop_azis[:-1], azis[1:])) > 0.25*da
                indices = np.nonzero(select)[0]
                for i in indices:
                    azis[i] = (stop_azis[i-1]+0.5*ft.angle_diff(stop_azis[i-1], azis[i])) % 360.
                
                azis, n_azi, da, azi_offset, diffs = process_azis_array(azis, da, calc_azi_offset=panel!=None, azi_pos='left')
                data, azi_offset = map_onto_regular_grid(data, n_azi, azis, diffs, da, azi_offset, azi_pos='left')
                data_mask = np.isnan(data)
                if panel != None:
                    self.dsg.data_azimuth_offset[panel] = azi_offset
                                            
            scantime = self.get_scan_timerange(hf, scangroup)
            
        return data, data_mask, scantime
        
    def dealias_velocity(self, data, data_mask, scan):
        vn_l = self.dsg.low_nyquist_velocities_all_mps[scan]
        vn_h = self.dsg.high_nyquist_velocities_all_mps[scan]
        dr = self.dsg.radial_res_all['v'][scan]
        """Important: The KMI seems to use two PRFs per radial, seems to correct velocities using the lower Nyquist velocity. Both vn_l and vn_h in apply_dual_prf_dealiasing are therefore set to vn_l.
        Further, for Jabbeke the dual PRF errors can be all multiples of the low Nyquist velocity, and not only even multiples. Hence multiplication of vn_l by 0.5.
        """
        window_detection = window_correction = None
        deviation_factor = 1.
        if self.crd.radar == 'Jabbeke' or (self.crd.radar == 'Wideumont' and int(self.crd.date) > 20220501): 
            vn_l *= 0.5; vn_h *= 0.5
            deviation_factor = 1.#33 #The maximum allowed velocity deviation is increased slightly, to reduce the smoothing of the velocity field a little.
            window_detection = window_correction = [2, 2, 2]
        elif gv.data_sources[self.crd.radar] == 'Austro Control':
            vn_l = vn_h
            window_detection = [0,3,6,9,3,6,0] if dr == 0.25 else [1,2,3,4,3,2,1]
            window_correction = [0,3,6,3,0] if dr == 0.25 else [0,1,2,3,2,1,0]
        n_it = 1 if self.gui.vwp.updating_vwp else self.gui.dealiasing_dualprf_n_it
        return da.apply_dual_prf_dealiasing(data, data_mask, dr, self.dsg.nyquist_velocities_all_mps[scan],\
        vn_l, vn_h, None, window_detection = window_detection, window_correction = window_correction, deviation_factor = deviation_factor,\
        n_it = n_it, mask_all_nearzero_velocities = self.crd.radar == 'Wideumont') #The first azimuth is scanned with a high PRF

    def get_scan_timerange(self, hf, scangroup):
        starttime = ft.format_time(scangroup['what'].attrs['starttime'].decode('utf-8'))
        endtime = ft.format_time(scangroup['what'].attrs['endtime'].decode('utf-8'))
        if not starttime == endtime:
            return starttime+'-'+endtime
        else:
            startdate = scangroup['what'].attrs['startdate'].decode('utf-8')
            antspeed = float(hf['how'].attrs['rpm'])*360/60 # To °/s
            return ft.get_timerange_from_starttime_and_antspeed(startdate,starttime,antspeed)


    def get_data(self,filepath, j): #j is the panel
        product, scan = self.crd.products[j], self.crd.scans[j]
        self.dsg.data[j], data_mask, self.dsg.scantimes[j] = self.read_data(filepath, product, scan, self.crd.apply_dealiasing[j], 
                                                                            self.crd.productunfiltered[j], j)        
        self.dsg.data[j][data_mask] = self.pb.mask_values[product]    
    
    def get_data_multiple_scans(self,filepath,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        if self.crd.radar == 'Zaventem' and i_p == 'v':
            z_data, _, _, _, _ = self.dsg.get_data_multiple_scans('z',scans,productunfiltered,polarization,apply_dealiasing,max_range)
        
        if isinstance(apply_dealiasing, bool):
            apply_dealiasing = {j: apply_dealiasing for j in scans}
        
        data, scantimes = {}, {}
        meta = {'using_unfilteredproduct':False, 'using_verticalpolarization':False}
        for j in scans: 
            data[j], scantimes[j] = [], []
            for scan in self.dsg.scannumbers_all[i_p][j]:
                data_mask = False
                if i_p == 'v' and self.crd.radar == 'Zaventem':
                    """There is an issue with velocities for Zaventem for slant ranges between about 3 and 7 km. For low reflectivities
                    these slant ranges show clearly erroneous velocities, that need to be filtered away for a correct VVP retrieval.
                    That is done here.
                    """
                    i = self.dsg.scannumbers_all[i_p][j].index(scan)
                    data_range = np.tile(np.arange(z_data[j][i].shape[1]) * self.dsg.radial_res_all['z'][j], (360, 1))
                    data_mask = ((z_data[j][i] < -15.) & (data_range >= 3) & (data_range <= 7))
                _data, data_mask, _scantime = self.read_data(filepath, product, j, apply_dealiasing, productunfiltered, data_mask=data_mask)
                _data[data_mask] = np.nan
                
                data[j].append(_data)
                scantimes[j].append(_scantime)
                
        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes([i[0] for i in scantimes.values()])                
        return data, scantimes, volume_starttime, volume_endtime, meta
    
    
    
    
    
class skeyes_hdf5():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
        
        self.available_products=('z','v','w')
        
        """The scans are for this data format distributed over 3 files, making it necessary to determine per scan in which file it is located. This is 
        done by storing the index of the file, which is the index of the position in the list filepath. These indices are stored in
        self.dsg.scannumbers_all, which has now the format {'z':{1:[[0,2],[0,5]]}}, implying that self.dsg.scannumbers_all[product][scan] is a list
        of lists, instead of a list of integers (scannumbers, as for other data formats). The inner lists have the format [file_index,scannumber]. 
        
        The radar volume has one of 2 different formats, which are called 'Monitor' and 'Hazardous'. 'Monitor' is the default format, and 'Hazardous'
        is the format that is used when an area of at least 5 km² has VIL>6 kg/m2, at a distance of at maximum 50 km from the radar.
        
        The files contain for each scan 2 or 3 datasets with different products. If there are 2 datasets per scan (as is the case for file 1), then 
        they are for the products 'z' and 'v'. If there are 3 of them, then 'w' is also present. The first dataset in a group of datasets for 
        one scan contains 'z', the second 'v' and the third if present 'w'.
        """           
           

    def get_scans_information(self,filepaths):
        #For each volume there are usually 3 files  
        
        scans_only_reflectivity=[]          
        for k in range(0,len(filepaths)):
            try:
                with h5py.File(filepaths[k],'r') as hf:                                    
                    scans=range(2,37,3)
                    """It is attempted to import data for the velocity, because the datasets for this product contain the Nyquist velocity. When this is
                    not possible, then it is attempted to get data for the reflectivity, as explained below.
                    """
                    
                    attrs=hf['where'].attrs
                    for j in scans:
                        try:
                            dataset=hf['scan'+str(j)]
                        except Exception: 
                            #In this case there is apparently no dataset for the velocity, but there might still be one for reflectivity, as is the case
                            #in the first file when the volume is of type 'Hazardous'. 
                            try:
                                dataset=hf['scan'+str(j-1)]
                                scans_only_reflectivity.append(j-1)
                            except Exception:
                                continue
                        scan=j-1 #The scannumber for the reflectivity is equal to j-1.
                        
                        key=k*100+scan #Add k*100 to the scannumber, to ensure that all scannumbers are different. This is not the case when not
                        #doing this, because scans are distributed over 3 files.
                        self.dsg.scanangles_all['z'][key]=ft.rndec(float(dataset['where'].attrs['angle']),2)
                        if self.dsg.scanangles_all['z'][key]==0.6: 
                            #See reason given in function self.get_scans_information of class ODIM_hdf5. 
                            self.dsg.scanangles_all['z'][key]=0.51 
                            
                        self.dsg.radial_bins_all['z'][key]=int(attrs['xsize'])
                        self.dsg.radial_res_all['z'][key]=float(attrs['xscale']/1000.)
                        self.dsg.nyquist_velocities_all_mps[key]=np.abs(float(dataset['what'].attrs['offset']))
                        
                        prf_l = float(hf['how'].attrs['lowprf'])
                        prf_h = float(hf['how'].attrs['highprf'])
                        radar_wavelength = float(hf['how'].attrs['wavelength']) * 1e-2
                        self.dsg.low_nyquist_velocities_all_mps[key] = None if prf_l == prf_h else radar_wavelength*prf_l/4.
                        self.dsg.high_nyquist_velocities_all_mps[key] = None if prf_l == prf_h else radar_wavelength*prf_h/4.
            except Exception:
                continue
        if len(self.dsg.scanangles_all)==0:
            raise Exception
                    
        extra_attrs = [self.dsg.nyquist_velocities_all_mps, self.dsg.high_nyquist_velocities_all_mps, self.dsg.low_nyquist_velocities_all_mps]
        self.dsg.scannumbers_all['z'] = bg.sort_volume_attributes(self.dsg.scanangles_all['z'], self.dsg.radial_bins_all['z'], self.dsg.radial_res_all['z'], extra_attrs)
        for i in self.dsg.scannumbers_all['z']:
            for j in range(0,len(self.dsg.scannumbers_all['z'][i])):
                #Add the filenumber to self.dsg.scannumbers_all['z'][j], and correct for the addition of k*100 to the scannumber.
                k=self.dsg.scannumbers_all['z'][i][j]
                self.dsg.scannumbers_all['z'][i][j]=[int(k/100),int(np.mod(k,100))]
        
            if self.dsg.scanangles_all['z'][i]==0.51:
                self.dsg.scanangles_all['z'][i]=0.5
                
        for i in self.dsg.scannumbers_all:
            for j in gv.volume_attributes_p: 
                self.dsg.__dict__[j][i] = copy.deepcopy(self.dsg.__dict__[j]['z'])
             
        #Give products 'v' and 'w' the correct scannumbers.
        for i in self.dsg.scannumbers_all['z']:
            for j in range(0,len(self.dsg.scannumbers_all['z'][i])):
                self.dsg.scannumbers_all['v'][i][j][1]+=1
                self.dsg.scannumbers_all['w'][i][j][1]+=2
            
        """The scan with a scanangle of 0.5 degrees (located in file 1) does not contain the spectrum width, so the second scan is shown when selecting
        scan 1 for product 'w'.
        When the radar volume is of type 'Hazardous', then there also is no data for the velocity for this scan.
        """
        if self.dsg.scanangles_all['z'][1]==0.5 and len(self.dsg.scanangles_all['z'])>1:
            for j in gv.volume_attributes_p:
                self.dsg.__dict__[j]['w'][1] = self.dsg.__dict__[j]['w'][2]
                if j=='scannumbers_all' and len(self.dsg.scannumbers_all['w'][1])>len(self.dsg.scannumbers_all['z'][1]):
                    #Ensure that self.dsg.scannumbers_all['w'][1] has the same length as self.dsg.scannumbers_all['z'][1]
                    self.dsg.scannumbers_all['w'][1]=self.dsg.scannumbers_all['w'][1][:len(self.dsg.scannumbers_all['z'][1])]
                    
                if self.dsg.scannumbers_all['z'][1][0][1] in scans_only_reflectivity:
                    self.dsg.__dict__[j]['v'][1] = self.dsg.__dict__[j]['v'][2]
                    if j=='scannumbers_all' and len(self.dsg.scannumbers_all['v'][1])>len(self.dsg.scannumbers_all['z'][1]):
                        self.dsg.scannumbers_all['v'][1]=self.dsg.scannumbers_all['v'][1][:len(self.dsg.scannumbers_all['z'][1])]
                        
            if self.dsg.scannumbers_all['z'][1][0][1] in scans_only_reflectivity:
                for j in ('nyquist_velocities_all_mps','low_nyquist_velocities_all_mps','high_nyquist_velocities_all_mps'):
                    self.dsg.__dict__[j][1] = self.dsg.__dict__[j][2]
        
            
    def get_data(self,filepath, j): #j is the panel
        with h5py.File(filepath,'r') as hf:
            product, i_p = self.crd.products[j], gv.i_p[self.crd.products[j]]
            if not i_p in self.available_products:
                raise Exception
            scan=self.dsg.scannumbers_all[i_p][self.crd.scans[j]][self.dsg.scannumbers_forduplicates[self.crd.scans[j]]][1]

            scangroup = hf['scan'+str(scan)]
            self.dsg.scantimes[j]=ft.format_time(scangroup['what'].attrs['starttime'].decode('utf-8'))
            self.dsg.scantimes[j]+='-'+ft.format_time(scangroup['what'].attrs['endtime'].decode('utf-8'))
            calibrationgroup=scangroup['what']
            gain=float(calibrationgroup.attrs['gain'])
            offset=float(calibrationgroup.attrs['offset'])
            self.dsg.data[j]=np.array(scangroup['data'],dtype='float32')*gain+offset
        
            data_maskvalue=self.dsg.data[j].min()
            int255_datavalue=np.float32(255*gain)+offset 
            data_mask = (self.dsg.data[j]<=data_maskvalue) | (self.dsg.data[j]==int255_datavalue)
            if i_p == 'v':
                if self.crd.apply_dealiasing[j] and not self.dsg.low_nyquist_velocities_all_mps[self.crd.scans[j]] is None:
                    self.dsg.data[j] = self.dealias_velocity(self.dsg.data[j], data_mask, self.crd.scans[j])
                    
            self.dsg.data[j][data_mask]=self.pb.mask_values[product]
                    
                    
    def dealias_velocity(self, data, data_mask, scan):
        vn_l = self.dsg.low_nyquist_velocities_all_mps[scan]
        vn_h = self.dsg.high_nyquist_velocities_all_mps[scan]
        radial_res = self.dsg.radial_res_all['v'][scan]
        #The window is chosen to be slightly larger than the default window in nlr_dealiasing.py, because the low and high Nyquist velocities are larger
        #than the 30/40 kts on which the window sizes in nlr_dealiasing.py are based.
        n_it = 1 if self.gui.vwp.updating_vwp else self.gui.dealiasing_dualprf_n_it
        return da.apply_dual_prf_dealiasing(data, data_mask, radial_res, self.dsg.nyquist_velocities_all_mps[scan],\
        vn_l, vn_h, None, n_it = n_it, window_detection = [3,5,7,5,3], window_correction = [5,7,5]) #The first azimuth is scanned with a high PRF
                    
        
    def get_data_multiple_scans(self,filepaths,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        if i_p == 'v':
            z_data, _, _, _, _ = self.dsg.get_data_multiple_scans('z',scans,productunfiltered,polarization,apply_dealiasing,max_range)
        
        if isinstance(apply_dealiasing, bool):
            apply_dealiasing = {j: apply_dealiasing for j in scans}
        
        data={}; scantimes={}
        for j in scans: 
            data[j] = []; scantimes[j] = []
            for i in range(len(self.dsg.scannumbers_all[i_p][j])):
                scan=self.dsg.scannumbers_all[i_p][j][i][1]
                file_index=self.dsg.scannumbers_all[i_p][j][i][0]
                with h5py.File(filepaths[file_index],'r') as hf:
                    scangroup = hf['scan'+str(scan)]
                    scantimes[j] += [ft.format_time(scangroup['what'].attrs['starttime'].decode('utf-8'))+\
                                       '-'+ft.format_time(scangroup['what'].attrs['endtime'].decode('utf-8'))]
                        
                    calibrationgroup=scangroup['what']
                    gain=float(calibrationgroup.attrs['gain'])
                    offset=float(calibrationgroup.attrs['offset'])
                    s = np.s_[:] if max_range is None else np.s_[:, :int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangles_all[i_p][j], 'gr+theta->sr') / self.dsg.radial_res_all[i_p][j]))]
                    data[j] += [np.array(scangroup['data'][s],dtype='float32')*gain+offset]
    
                    data_mask = (data[j][-1] <= data[j][-1].min()) | (data[j][-1] == data[j][-1].max())
                    if i_p == 'v':
                        if int(self.crd.date[:4]) > 2014 and z_data[j][i].shape == data_mask.shape:
                            # Last check is included because it sometimes happens that no Z scan is available for the lowest dual-PRF scan, 
                            # in which case the mono-PRF scan with other dimensions is used
                            """There is an issue with velocities for Zaventem for slant ranges between about 3 and 7 km. For low reflectivities
                            these slant ranges show clearly erroneous velocities, that need to be filtered away for a correct VVP retrieval.
                            That is done here.
                            """
                            data_range = np.tile(np.arange(z_data[j][i].shape[1]) * self.dsg.radial_res_all['z'][j], (360, 1))
                            data_mask |= ((z_data[j][i] < -17.) & (data_range >= 3) & (data_range <= 7))
                        
                        if apply_dealiasing[j] and not self.dsg.low_nyquist_velocities_all_mps[j] is None:
                            data[j][-1] = self.dealias_velocity(data[j][-1], data_mask, j)
                    data[j][-1][data_mask] = np.nan
                
        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes([i[0] for i in scantimes.values()])                
        meta = {'using_unfilteredproduct': False, 'using_verticalpolarization': False}
        return data, scantimes, volume_starttime, volume_endtime, meta
    
    
    
    
class DWD_odimh5():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
           
           

    def get_scans_information(self, filepaths, products, fileids_per_product): #These should be the paths to the files that contain the velocity,
        #because otherwise it is not possible to determine the Nyquist velocity.
        scanangles_all, radial_bins_all, radial_res_all = {}, {}, {}
        for j in filepaths:
            product = products[j]
            try:
                with h5py.File(filepaths[j],'r') as hf:    
                    dataset=hf['dataset1']
                    attrs=dataset['where'].attrs
                    #One value for a scanangle is given in attrs['elangle'], but it appears that this value isn't the average scanangle, but 
                    #usually differs by 0-0.2 degrees from the actual average. In one case however this value differed by even 0.5 degrees
                    #from the actual average. Because of this difference, the scanangle is determined by averaging over all azimuths.
                    # Also this average scanangle can very between volumes, hence 'scanangles_all' is set to be a variable attribute below. 
                    scanangles_all[j]=np.mean(dataset['how'].attrs['startelA'])
                    radial_bins_all[j]=int(attrs['nbins'])
                    radial_res_all[j]=float(attrs['rscale']/1000.)
                    self.dsg.nyquist_velocities_all_mps[j]=np.abs(float(dataset['how'].attrs['NI'])) if product=='v' else None
                        
                    prf_l = float(dataset['how'].attrs['lowprf'])
                    prf_h = float(dataset['how'].attrs['highprf'])
                    radar_wavelength = float(hf['how'].attrs['wavelength']) * 1e-2
                    self.dsg.low_nyquist_velocities_all_mps[j] = None if prf_l == prf_h else radar_wavelength*prf_l/4.
                    self.dsg.high_nyquist_velocities_all_mps[j] = None if prf_l == prf_h else radar_wavelength*prf_h/4.
            except Exception:
                #It could happen that a certain file is corrupt. In that case continue with next file
                continue                                       
            
        self.dsg.variable_attributes = ['scanangles_all']
               
        extra_attrs = [self.dsg.nyquist_velocities_all_mps, self.dsg.high_nyquist_velocities_all_mps, self.dsg.low_nyquist_velocities_all_mps]
        scannumbers_all = bg.sort_volume_attributes(scanangles_all, radial_bins_all, radial_res_all, extra_attrs)
        #scannumbers_all now maps each scan to the number of the file in which it is contained, in the list filepaths.
                    
        """Because each product and each scan is placed in a different file, it is well possible that a particular scan is available for only
        one product. If this is the case, then for the other products the attributes for that scan should map to the values for the nearest scan that is available
        for those products, with nearest in terms of scanangle.
        """
        for p in set(['z']+list(fileids_per_product)):
            #Volume attributes for 'z' must always be available, so if 'z' is in fact not available for the volume, then the volume attributes for 'z'
            #should contain the values that are obtained for other products.
            
            scans_p = [j for j in scanangles_all if scannumbers_all[j][0] in fileids_per_product.get(p, [])] #Scans available for product p
            scanangles_p = [scanangles_all[j] for j in scans_p] #Scanangles available for product p
            for j in scanangles_all:
                fileid = scannumbers_all[j][0]
                if (p == 'z' and not p in fileids_per_product) or fileid in fileids_per_product[p]:
                    closest_scan_product = j
                else:
                    closest_scan_product = scans_p[np.abs(scanangles_p-scanangles_all[j]).argmin()]
                    
                for a in gv.volume_attributes_p:
                    self.dsg.__dict__[a][p][j] = locals()[a][closest_scan_product]

            
    def read_data(self, filepaths, product, scan, apply_dealiasing):
        i_p = gv.i_p[product]
        data, data_mask = [], []
        for i, filepath in enumerate(filepaths):
            with h5py.File(filepath, 'r') as hf:                 
                dataset=hf['dataset1']           
                calibrationgroup=dataset['data1']['what']
                gain=float(calibrationgroup.attrs['gain'])
                offset=float(calibrationgroup.attrs['offset'])
                undetect = calibrationgroup.attrs['undetect']
                nodata = calibrationgroup.attrs['nodata']
                if i == 0:
                    # These attributes are not expected to differ among the product versions
                    scantime = ft.format_time(dataset['what'].attrs['starttime'].decode('utf-8'))+\
                                  '-'+ft.format_time(dataset['what'].attrs['endtime'].decode('utf-8'))
                    if i_p == 'v':
                        prfs = dataset['how'].attrs['prf'][:2]
                        vn_first_azimuth = 'l' if prfs[0] < prfs[1] else 'h'
                    
                data += [np.array(dataset['data1']['data'])]
                if len(data[-1]) > 360:
                    # It is observed that sometimes an extra azimuthal bin is added, e.g. because 1 radial is 'repeated' in the dataset. This extra radial
                    # is removed here
                    azis = np.array(dataset['how'].attrs['startazA'])
                    diff = np.diff(azis)
                    data[-1] = np.delete(data[-1], diff.argmin(), axis=0)
                data_mask += [(data[-1]==undetect) | (data[-1]==nodata)]
                data[-1] = data[-1].astype('float32')*gain+offset
        
        if len(filepaths) > 1:
            # Combine data for the product versions
            stqual_index = 0 if 'stqual' in filepaths[0] else 1
            if stqual_index == 1:
                data.reverse()
                data_mask.reverse()
            # Use attenuation-corrected data as base for reflectivity, and use undealiased data as base for velocity
            i1 = 1 if i_p == 'z' else 0
            i2 = not i1
            data_combi = data[i1]
            if i_p == 'v':
                # Velocities < 1.5 m/s are not dealiased since they might represent clutter. So check whether other (dealiased)
                # velocities are available in the attenuation-corrected (and clutter-filtered) files
                data_mask[i1] |= (np.abs(data[i1]) < 1.5) & ~data_mask[i2]
            data_combi[data_mask[i1]] = data[i2][data_mask[i1]]
            
            data = data_combi
            data_mask = data_mask[0] & data_mask[1]
        else:
            data, data_mask = data[0], data_mask[0]
        
        if i_p == 'c':
            data *= 100.
            data_mask = data < 20.
        elif i_p == 'p':
            data %= 360.
        elif i_p == 'v':
            if apply_dealiasing and not self.dsg.low_nyquist_velocities_all_mps[scan] is None:
                data = self.dealias_velocity(data, data_mask, scan, vn_first_azimuth, combi=len(filepaths) > 0)
        
        return data, data_mask, scantime
    
    def dealias_velocity(self, data, data_mask, scan, vn_first_azimuth, combi=False):
        vn_l = self.dsg.low_nyquist_velocities_all_mps[scan]
        vn_h = self.dsg.high_nyquist_velocities_all_mps[scan]
        radial_res = self.dsg.radial_res_all['v'][scan]
        n_it = 1 if self.gui.vwp.updating_vwp else self.gui.dealiasing_dualprf_n_it
        window_detection = [1,4,7,4,1] if combi else None
        window_correction = [2,4,2] if combi else None
        return da.apply_dual_prf_dealiasing(data, data_mask, radial_res, self.dsg.nyquist_velocities_all_mps[scan],\
        vn_l, vn_h, vn_first_azimuth, window_detection, window_correction, n_it = n_it)
            

    def get_data(self,filepaths, j): #j is the panel
        product, i_p = self.crd.products[j], gv.i_p[self.crd.products[j]]
        scan = self.crd.scans[j]
                                    
        self.dsg.data[j], data_mask, self.dsg.scantimes[j] = self.read_data(filepaths, product, scan, self.crd.apply_dealiasing[j])
        self.dsg.data[j][data_mask]=self.pb.mask_values[product] 

    def get_data_multiple_scans(self,filepaths,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        if isinstance(apply_dealiasing, bool):
            apply_dealiasing = {j: apply_dealiasing for j in scans}
        
        data = {}; scantimes = {}
        for j in scans: 
            fileid = self.dsg.scannumbers_all[i_p][j][0]
            filepaths_j = filepaths[fileid]
            data[j], data_mask, scantimes[j] = self.read_data(filepaths_j, product, j, apply_dealiasing[j])
            scantimes[j] = [scantimes[j]]
            data[j][data_mask] = np.nan
            data[j] = [data[j]]
                                            
        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes([i[0] for i in scantimes.values()])                
        meta = {'using_unfilteredproduct': False, 'using_verticalpolarization': False}
        return data, scantimes, volume_starttime, volume_endtime, meta


    
    
class DWD_BUFR():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
        
        table_type = 'libdwd'
        table_path = os.path.join('Tables',table_type)
        self.bufr_decoder = decode_bufr.DecodeBUFR(table_path, table_type)
        self.DWD_bufr_productcodes = {'z':'021001','v':'021014'}
        """Important: The DWD puts data for each scan in a different file, such that each file contains only one scan. 
        For this reason self.dsg.scannumbers_all maps each scan to the file in which it is contained, instead of the scannumber that this scan
        has in the data file (as is the case for most other sources).
        """

    
    
    def get_scans_information(self, filepaths, products, fileids_per_product): #These should be the paths to the files that contain the velocity,
        #because otherwise it is not possible to determine the Nyquist velocity.
        scanangles_all, radial_bins_all, radial_res_all = {}, {}, {}   
        for j in filepaths:
            product = products[j]
            try:
                _, _, data_info, data_loops = self.bufr_decoder(filepaths[j], read_mode=['002135'])                
                data_info, data_loops = data_info[0], data_loops[0]
            except Exception:
                continue
            
            #One value for a scanangle is given in data_info['002135'][0], but it appears that this value isn't the average scanangle, but usually differs by
            #0-0.2 degrees from the actual average. In one case however this value differed by even 0.5 degrees from the actual average. 
            #Because of this difference, the scanangle is determined by averaging over all azimuths.
            # Also this average scanangle can very between volumes, hence 'scanangles_all' is set to be a variable attribute below. 
            scanangles_all[j] = np.mean(data_loops[1]['002135'])
            
            radial_bins_all[j] = int(data_info['030194'][0])
            radial_res_all[j] = data_info['021201'][0]/1000.
            
            self.dsg.nyquist_velocities_all_mps[j] = data_info['021236'][0] if product=='v' else None
            dual_prf_ratio = data_info['002194'][0]
            if not self.dsg.nyquist_velocities_all_mps[j] is None and dual_prf_ratio != 0.0:
                """Important: The dual prf ratio of 2 that is indicated in the data description for the lowest 6 scans is not correct!
                It should be 3! The calculation below gives the correct ratio in a different way, by dividing the extended and high
                Nyquist velocity.
                """
                dual_prf_ratio = int(round(self.dsg.nyquist_velocities_all_mps[j] / data_info['021237'][0]))
            self.dsg.high_nyquist_velocities_all_mps[j] = None if dual_prf_ratio == 0.0 else data_info['021237'][0]
            self.dsg.low_nyquist_velocities_all_mps[j] = None if dual_prf_ratio == 0.0 else self.dsg.high_nyquist_velocities_all_mps[j] * dual_prf_ratio / (dual_prf_ratio + 1)
            
        self.dsg.variable_attributes = ['scanangles_all']
            
        extra_attrs = [self.dsg.nyquist_velocities_all_mps, self.dsg.high_nyquist_velocities_all_mps, self.dsg.low_nyquist_velocities_all_mps]
        scannumbers_all = bg.sort_volume_attributes(scanangles_all, radial_bins_all, radial_res_all, extra_attrs)
        #scannumbers_all now maps each scan to the number of the file in which it is contained, in the list filepaths.
                  
        """Because each product and each scan is placed in a different file, it is well possible that a particular scan is available for only
        one product. If this is the case, then for the other products the attributes for that scan should map to the values for the nearest scan that is available
        for those products, with nearest in terms of scanangle.
        """
        for p in set(['z']+list(fileids_per_product)):
            #Volume attributes for 'z' must always be available, so if 'z' is in fact not available for the volume, then the volume attributes for 'z'
            #should contain the values that are obtained for other products.
            
            scans_p = [j for j in scanangles_all if scannumbers_all[j][0] in fileids_per_product.get(p, [])] #Scans available for product p
            scanangles_p = [scanangles_all[j] for j in scans_p] #Scanangles available for product p
            for j in scanangles_all:
                fileid = scannumbers_all[j][0]
                if (p == 'z' and not p in fileids_per_product) or fileid in fileids_per_product[p]:
                    closest_scan_product = j
                else:
                    closest_scan_product = scans_p[np.abs(scanangles_p-scanangles_all[j]).argmin()]
                    
                for a in gv.volume_attributes_p:
                    self.dsg.__dict__[a][p][j] = locals()[a][closest_scan_product]
                    
            
    def get_data(self,filepath, j): #j is the panel        
        product, i_p = self.crd.products[j], gv.i_p[self.crd.products[j]]
        scan = self.crd.scans[j]
                                
        _, _, data_info, data_loops = self.bufr_decoder(filepath, read_mode='all')
        data_info, data_loops = data_info[0], data_loops[0]
        # print(data_info, data_loops)
        
        self.dsg.data[j] = data_loops[1][self.DWD_bufr_productcodes[i_p]]
        
        n_azi = 360
        if self.dsg.data[j].shape[0]>n_azi:
            #It is observed that sometimes an extra azimuthal bin is. This bin must be removed, because it causes the data array to 
            # include data for an azimuthal range of more than 360 degrees (and it also causes errors in other parts of the code).
            self.dsg.data[j] = self.dsg.data[j][:n_azi]
        elif self.dsg.data[j].shape[0] < n_azi:
            #It is also possible that n_azi = 360, while the true number of available radials is 359. This has been observed for
            # old (2009) data. In that case append a radial with zeros.
            self.dsg.data[j] = np.concatenate([self.dsg.data[j], self.pb.mask_values[product]+np.zeros((n_azi-self.dsg.data[j].shape[0], self.dsg.data[j].shape[1]))], axis=0)
        
        start_azimuth = data_info['002134']
        self.dsg.data[j]=np.roll(self.dsg.data[j],int(np.floor(start_azimuth)),axis=0).astype('float32')
        
        data_mask = (self.dsg.data[j]==self.dsg.data[j].min()) | (self.dsg.data[j]==self.dsg.data[j].max())
        
        if i_p == 'v' and self.dsg.data[j][~data_mask].max() > self.dsg.nyquist_velocities_all_mps[scan]:
            # For some of the historical BUFR files the velocity for mono-PRF scans is incorrectly scaled. This corrects for that
            self.dsg.data[j][~data_mask] *= self.dsg.nyquist_velocities_all_mps[scan]/32.
                 
        if i_p == 'v':
            if self.crd.apply_dealiasing[j] and not self.dsg.low_nyquist_velocities_all_mps[self.crd.scans[j]] is None:
                self.dsg.data[j] = self.dealias_velocity(self.dsg.data[j], data_mask, self.crd.scans[j])
                
        self.dsg.data[j][data_mask] = self.pb.mask_values[product]
            
        #The end datetime is given in the filename, and the start datetime in the file itself
        datetime = ''.join([format(int(round(float(data_info['00400'+str(j)][-1]))),'02') for j in (1,2,3,4,5,7)])
        datetime = ft.next_datetime(datetime[:10], 1)+'00' if datetime[-2:] == '60' else datetime
        starttime = ft.format_time(datetime[-6:])
        end_datetime = ft.get_datetimes_from_absolutetimes(ft.get_absolutetimes_from_datetimes(datetime[:-2]+datetime[-2:])+\
                                                           round(data_loops[1]['004026'][-1]), include_seconds=True)
        endtime = ft.format_time(end_datetime[-6:])
        self.dsg.scantimes[j] = starttime+('-'+endtime if not endtime == starttime else '')
            
            
    def dealias_velocity(self, data, data_mask, scan):
        vn_l = self.dsg.low_nyquist_velocities_all_mps[scan]
        vn_h = self.dsg.high_nyquist_velocities_all_mps[scan]
        radial_res = self.dsg.radial_res_all['v'][scan]
        n_it = 1 if self.gui.vwp.updating_vwp else self.gui.dealiasing_dualprf_n_it
        return da.apply_dual_prf_dealiasing(data, data_mask, radial_res, self.dsg.nyquist_velocities_all_mps[scan],\
        vn_l, vn_h, None, n_it = n_it) #The prf of the first azimuth is unknown, hence vn_first_azimuth = None


    def get_data_multiple_scans(self,filepaths,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        if isinstance(apply_dealiasing, bool):
            apply_dealiasing = {j: apply_dealiasing for j in scans}
        
        data = {}; scantimes = {}
        for j in scans: 
            fileid = self.dsg.scannumbers_all[i_p][j][0]
            
            _, _, data_info , data_loops = self.bufr_decoder(filepaths[fileid], read_mode='all')
            data_info, data_loops = data_info[0], data_loops[0]
            
            datetime = ''.join([format(int(round(float(data_info['00400'+str(j)][-1]))),'02') for j in (1,2,3,4,5,7)])
            datetime = ft.next_datetime(datetime[:10], 1)+'00' if datetime[-2:] == '60' else datetime
            starttime = ft.format_time(datetime[-6:])
            end_datetime = ft.get_datetimes_from_absolutetimes(ft.get_absolutetimes_from_datetimes(datetime[:-2]+datetime[-2:])+\
                                                               round(data_loops[1]['004026'][-1]), include_seconds=True)
            endtime = ft.format_time(end_datetime[-6:])
            scantimes[j] = [starttime+('-'+endtime if not endtime == starttime else '')]
                        
            data[j] = data_loops[1][self.DWD_bufr_productcodes[i_p]]
            n_azi = 360
            if data[j].shape[0]>n_azi:
                #It is observed that sometimes an extra azimuthal bin is. This bin must be removed, because it causes the data array to 
                # include data for an azimuthal range of more than 360 degrees (and it also causes errors in other parts of the code).
                data[j] = data[j][:n_azi]
            elif data[j].shape[0] < n_azi:
                #It is also possible that n_azi = 360, while the true number of available radials is 359. This has been observed for
                # old (2009) data. In that case append a radial with zeros.
                data[j] = np.concatenate([data[j], np.full((n_azi-data[j].shape[0], data[j].shape[1]), data[j].min(), 'float32')], axis=0)
            s = np.s_[:] if max_range is None else np.s_[:, :int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangles_all[i_p][j], 'gr+theta->sr') / self.dsg.radial_res_all[i_p][j]))]
            data[j] = data[j][s]
            
            # With the historical DWD BUFR data it is possible that the Z and V products for some scans have a different range. So in order
            # to prevent errors when calculating derived products, it is necessary to update self.dsg.radial_bins_all[i_p][j]
            self.dsg.radial_bins_all[i_p][j] = data[j].shape[1]
                
            start_azimuth = data_info['002134']
            data[j]=np.roll(data[j],int(np.floor(start_azimuth)),axis=0).astype('float32')
                        
            data_mask = (data[j]==data[j].min()) | (data[j]==data[j].max())            
                        
            if i_p == 'v' and data[j][~data_mask].max() > self.dsg.nyquist_velocities_all_mps[j]:
                # For some of the historical BUFR files the velocity for mono-PRF scans is incorrectly scaled. This corrects for that
                data[j] *= self.dsg.nyquist_velocities_all_mps[j]/32.
            
            if i_p == 'v':
                if apply_dealiasing[j] and not self.dsg.low_nyquist_velocities_all_mps[j] is None:
                    data[j] = self.dealias_velocity(data[j], data_mask, j)
            data[j][data_mask] = np.nan
            data[j] = [data[j]]
        
        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes([i[0] for i in scantimes.values()])                
        meta = {'using_unfilteredproduct': False, 'using_verticalpolarization': False}
        return data, scantimes, volume_starttime, volume_endtime, meta
    
    
    
    

class TUDelft_nc():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
    
        self.file = None
        self.filename = None
        self.data_selected_startazimuth=self.gui.data_selected_startazimuth
    
    
    
    def open_file_if_needed(self, filepath):
        if self.filename != filepath:
            self.file = nc.Dataset(filepath, 'r')
        if self.filename != filepath or self.gui.data_selected_startazimuth != self.data_selected_startazimuth:
            self.obtain_parameters_and_split_data(file_updated = self.filename != filepath)
        self.filename = filepath
        
            
    def obtain_parameters_and_split_data(self, file_updated):
        """
        A single netcdf file contains data for a whole day. This function obtains parameters that should be obtained only
        once per file and determines how to split the dataset into different scans.
        
        self.datetimes contains the datetimes for which data is actually available (in contrast to what happens in the function self.dsg.get_datetimes_from_files, where 
        determining the actual datetimes costs too much time).
        """
        if file_updated:
            self.azimuths = self.file['azimuth'][:]
            self.ranges = self.file['range'][:]
            self.hours = self.file['time'][:]
            self.azi_nonzero = self.azimuths != 0. #azimuths are equal to zero when no data is present for that radial!
            self.i_nonzero = np.where(self.azi_nonzero)[0]
            self.azimuths = self.azimuths[self.azi_nonzero]
            if 201905080000<=int(self.crd.date+self.crd.time)<201906200000:
                #Data is rotated by 180 degrees for at least between these two dates!
                self.azimuths = np.mod(self.azimuths+np.pi, 2*np.pi)
            self.hours = self.hours[self.azi_nonzero]
            self.data_selected_startazimuth = 0
        
        da = np.deg2rad(self.gui.data_selected_startazimuth-self.data_selected_startazimuth)
        if not da == 0.:
            self.azimuths = np.mod(self.azimuths-da, 2*np.pi)
        self.data_selected_startazimuth = self.gui.data_selected_startazimuth
        
        a_diff = self.azimuths[1:] - self.azimuths[:-1]
        h_diff = self.hours[1:] - self.hours[:-1]
        select = (a_diff < 0.) | (h_diff > (15/3600)) #A scan is also terminated when the next available radial differs
        #in time by more than 15 seconds from the previous radial.
        self.i_revolutions = np.concatenate(([0], np.where(select)[0]+1, [len(self.azimuths)]))
        
        hours_scanaverages = self.hours[np.minimum(self.i_revolutions, len(self.hours)-1)]
        self.datetimes = np.array([self.crd.date+format(j, '04d') for j in np.round((hours_scanaverages//1)*100 + np.mod(hours_scanaverages, 1)*60).astype(int)])
        for i in range(len(self.datetimes)):
            #Here above minutes can get rounded to 60, which is fixed here
            if self.datetimes[i][-2:] == '60': self.datetimes[i] = ft.next_datetime(self.datetimes[i][:10]+'59', 1)
        dt = gv.volume_timestep_radars[self.crd.radar]
        self.datetimes = np.array([int(np.floor(int(j)/dt)*dt) for j in self.datetimes], dtype=str)

    
    def get_scans_information(self, filepath):
        self.open_file_if_needed(filepath)
        
        j = 1 #There is only 1 scan performed per volume
        self.dsg.radial_res_all['z'][j] = self.file['range_resolution'][:][0] / 1e3
        self.dsg.radial_bins_all['z'][j]=len(self.ranges)
        self.dsg.scanangles_all['z'][j]=np.rad2deg(self.file['elevation_angle'][:][0])
        prf = 1/(self.file['sweep_time'][:][0]/1e6) #Convert time from ms to s
        radar_wavelength = self.file['radiation_wavelength'][:][0]
        self.dsg.nyquist_velocities_all_mps[j]=radar_wavelength*prf/4
        self.dsg.low_nyquist_velocities_all_mps[j] = self.dsg.high_nyquist_velocities_all_mps[j] = None #Data is mono PRF
        
        datetime = self.crd.date+self.crd.time
        indices_datetime = np.where(self.datetimes == datetime)[0]
        self.dsg.scannumbers_all['z'][j]=[j]*len(indices_datetime) #There might be more than once scan for the same datetime!
        
        for i in self.dsg.scannumbers_all:
            for j in gv.volume_attributes_p: 
                self.dsg.__dict__[j][i] = copy.deepcopy(self.dsg.__dict__[j]['z'])
        
    
    def get_ibounds_scan(self, j): #j is the panel
        datetime = self.crd.date+self.crd.time
        indices_datetime = np.where(self.datetimes == datetime)[0]
        i = indices_datetime[self.dsg.scannumbers_forduplicates[self.crd.scans[j]]] #scannumbers_forduplicates is used since there
        #might be more than once scan for the same datetime!
        i_start = self.i_revolutions[i]
        if i_start > 0 and not self.hours[i_start] - self.hours[i_start-1] > 15/3600:
            #Add one earlier radial to the scan, since otherwise the data will often not span the full 360 degrees in azimuth
            i_start -= 1
        i_endplus1 = self.i_revolutions[i+1]
        return i_start, i_endplus1
    
            
    def get_data(self, filepath, j): #j is the panel
        product, i_p = self.crd.products[j], gv.i_p[self.crd.products[j]]
        scan = self.crd.scans[j]
        self.open_file_if_needed(filepath)
        
        i1, i2 = self.get_ibounds_scan(j)
        data = np.array(self.file[gv.productnames_TUDelft[i_p]][self.i_nonzero[i1]:self.i_nonzero[i2]], dtype = 'float32')
        data = data[self.i_nonzero[i1:i2]-self.i_nonzero[i1]]
        if data.shape[0] < 5:
            #Don't plot arrays that contain only a few radials with data
            raise Exception
            
        mask_values = {'z': data.min(), 'v': [0., data.min()], 'w': data.min(), 'd': data.min(), 'p': 0., 'x': 0.}
        if not isinstance(mask_values[i_p], list):
            data_mask = data == mask_values[i_p]
        else:
            data_mask = np.logical_or.reduce([data == i for i in mask_values[i_p]])
            
        if i_p == 'v':
            if 201808250000<=int(self.crd.date+self.crd.time)<201908280000 and self.crd.apply_dealiasing[j]:
                negative = data < 0.
                positive = negative == False
                data *= -1
                data[negative] -= self.dsg.nyquist_velocities_all_mps[scan]
                data[positive] += self.dsg.nyquist_velocities_all_mps[scan]
            else:
                data *= -1. #Velocities towards the radar are positive for IDRA while negative in this application
        elif product == 'p':
            data *= 180./np.pi
                    
        data[data_mask] = self.pb.mask_values[product]

            
        n_azi = 1500
        """Map the radials in data onto the polar grid used for self.dsg.data[j], with n_azi radials.
        Every radial in data spans an azimuthal range given by [a-da, a+da], where a is the central azimuth, and da is the azimuthal width
        of a radial. 
        """
        self.dsg.data[j] = np.full((n_azi, data.shape[1]), self.pb.mask_values[product], dtype='float32')
        
        azimuths = self.azimuths[i1:i2]
        #i2-1 since i2 represents the first radial of the next scan. i1+1, since i1 represents also the last radial of the 
        #previous scan. This combination of addition and subtraction ensures that da is calculated from azimuths that are
        #within the same azimuthal cycle from 0 to 360 degrees.
        da = (self.azimuths[i2-1]-self.azimuths[i1+1])/(self.i_nonzero[i2-1]-self.i_nonzero[i1+1])
        
        for i in range(len(azimuths)):
            if i > 0:
                azi1_previous, azi2_previous = azi1, azi2
                previous_radial1, previous_radial2 = radial1, radial2

            a = np.mod(azimuths[i]+np.deg2rad(self.gui.data_selected_startazimuth), 2*np.pi)
            azi1, azi2 = a, a+da #a represents the start azimuth of a radial, so its full azimuthal extent is given by [a, a+da] instead of [a-da/2, a+da/2].
            radial1, radial2 = int(round(azi1/(2*np.pi)*n_azi)), int(round(azi2/(2*np.pi)*n_azi))
            if radial1 < 0:
                #The first radial might extend further back than azimuth=0 degrees
                self.dsg.data[j][radial1:] = data[i]
            self.dsg.data[j][max(0, radial1): min(radial2, n_azi)] = data[i]       
            if radial2 > n_azi:
                #The last radial might extend beyond azimuth=360 degreees
                self.dsg.data[j][:radial2-n_azi] = data[i]
                
            if i > 0 and np.mod(radial1-1, n_azi) == np.mod(previous_radial2, n_azi):
                #In this case one of the radials in self.dsg.data[j] in between azimuth i and azimuth i-1 is not filled with data.
                #That is not desired, so that radial is filled here.
                empty_radial = radial1-1
                azimuth = (empty_radial+0.5)*2*np.pi/n_azi
                diff1, diff2 = abs(azimuths[i-1]-azimuth), abs(azimuths[i]-azimuth)
                self.dsg.data[j][empty_radial] = data[i-1] if diff1 < diff2 else data[i]

        self.dsg.scantimes[j] = ft.format_time(format(int((self.hours[i1]//1)*10000 + (np.mod(self.hours[i1], 1)//(1/60))*100 + np.mod(self.hours[i1], 1/60)*3600), '06d'))+\
                                 '-'+ft.format_time(format(int((self.hours[i2-1]//1)*10000 + (np.mod(self.hours[i2-1], 1)//(1/60))*100 + np.mod(self.hours[i2-1], 1/60)*3600), '06d'))

    
    def get_data_multiple_scans(self,filepath,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return #Not implemented since there is only one scan
    
    
    

    
class MeteoFrance_BUFR():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
        
        table_type = 'libdwd'
        table_path = os.path.join('Tables',table_type)
        self.bufr_decoder = decode_bufr.DecodeBUFR(table_path, table_type)
        
        # sigma is containt in both PAM (cartesian 1*1 km res) and PAG (polar 1 deg*1 km res) files. Preferentially use PAG, 
        # since it is of higher resolution near the radar, where clutter filtering matters most.
        self.product_filetypes = {'z':['PAM','PAG'], 'v':['PAG'], 'c':['PAM'], 'd':['PAM'], 'p':['PAM'], 'sigma':['PAG','PAM']}
        
        # Sigma denotes the standard deviation of reflectivity
        self.product_order = lambda date: ({'z':0, 'v':2, 'c':1, 'd':3, 'p':2, 'sigma':[1,4]} if int(date) < 20180818 else
                                           {'z':0, 'v':2, 'c':1, 'd':2, 'p':3, 'sigma':[1,4]})
        # Offsets for 'v' and 'sigma' include a correction for the fact that integer data values represent the start of an interval,
        # with the average being 0.5 higher
        self.product_offset = {'z':-10.5, 'v':60, 'c':30.5, 'd':-9.85, 'p':0.5, 'sigma':0.125}
        self.product_scale = {'z':1., 'v':-0.5, 'c':1., 'd':0.1, 'p':1., 'sigma':0.25}
        self.product_nbits = {'z':8, 'v':8, 'c':8, 'd':8, 'p':16, 'sigma':8}
        self.product_maskvals = {'z':'minmax', 'v':'min', 'c':'max', 'd':'max', 'p':'max', 'sigma':'max'}
            

    def get_file_content(self, filepath, product):
        with open(filepath, 'rb') as f:
            content = f.read()
            indices = [i.start() for i in re.finditer(b'\x1f\x8b\x08\x00', content)]
            if 'PAM' in os.path.basename(filepath):
                # In this case the last message in the file has LZW compression instead of gzip compression, and therefore needs
                # to be excluded to prevent errors in the gzip library
                indices += [content.rindex(b'\x1f\x9d\x90B')]
            index = self.product_order(self.crd.date)[product]
            if product == 'sigma':
                index = index[1] if index[1]+1 < len(indices) else index[0]
            i1, i2 = indices[index], indices[index+1] if index+1 < len(indices) else None
            return content[i1:i2]

    def get_scans_information(self, filepaths, filetypes_per_fileid):
        for attr in gv.volume_attributes_p:
            # Add 'sigma' to volume attributes, since that is easiest in remainder of code.
            self.dsg.__dict__[attr]['sigma'] = {}
            
        for fileid, filepath in filepaths.items():
            try:
                # Only one type of file is opened here, either PAM or PAG. This means that filetype-dependent parameters below
                # can't be determined from data_info.
                content = self.get_file_content(filepath, 'z')
                _, _, data_info, data_loops = self.bufr_decoder(content, read_mode=['002125'])
                data_info, data_loops = data_info[0], data_loops[0]
            except Exception as e:
                print(e)
                continue
            
            for p,filetypes in self.product_filetypes.items():
                filetype_p = filetypes[0]
                if not filetype_p in filetypes_per_fileid[fileid]:
                    # 'None' is just used to indicate that this product is not yet available. And it is important to not keep filetype_p unchanged in
                    # case of unavailability, since it becomes part of the scannumber. And this scannumber is used to check whether new content has
                    # come available for a certain product and scan, and thus whether any data array already stored in memory has to be updated.
                    filetype_p = filetypes[1] if len(filetypes) == 2 else 'None'
                # j will become the scannumber. It contains both filetype and fileid, in order to get a new Z array requested when its filetype changes
                # (Z is contained in both PAG and PAM files, with preferential use of high-res PAM file, but in real-time use PAG might come available first).
                j = filetype_p+','+fileid
                
                if p == 'v':                    
                    prfs = np.array(data_loops[1]['002125'])
                    prfs = np.unique(prfs)[::-1]
                    # if len(prfs) != len(np.unique(prfs)):
                    #     # It has been observed that 2 PRFs are repeated, which was associated with very weird, erroneous velocities.
                    #     # Such a scan is therefore excluded. Also, 2 equal PRFs can screw up the calculation of the Nyquist velocity below,
                    #     # giving a value of infinity.
                    #     continue
                    wavelength = 299792458/data_info['002121'][0]
                    vns = prfs*wavelength/4
                    # MF always aims at an extended Nyquist velocity around 60 m/s. The method to calculate it is described in https://journals.ametsoc.org/view/journals/atot/23/12/jtech1923_1.xml?tab_body=pdf.
                    # But in practice it can simply be calculated as n*vns[0], where n is the integer that gives the extended Nyquist velocity
                    # closest to 60 m/s.
                    n = round(60/vns[0])
                    self.dsg.nyquist_velocities_all_mps[j] = (n+1)*vns[0]
                    self.dsg.high_nyquist_velocities_all_mps[j] = self.dsg.low_nyquist_velocities_all_mps[j] = vns[0]
                
                self.dsg.scanangles_all[p][j] = data_info['007021'][0]
                if round(self.dsg.scanangles_all[p][j]) == 90.:
                    # Sometimes the value is a bit less than 90., which is changed below, in order to easily check for vertical scans in other code
                    self.dsg.scanangles_all[p][j] = 90.
                # These attributes are set manually, since they differ per filetype and only one filetype gets opened.
                # Otherwise it would be codes 030021 for nr, 055233 for dr
                self.dsg.radial_bins_all[p][j] = 1066 if filetype_p == 'PAM' else 256
                self.dsg.radial_res_all[p][j] = 0.24 if filetype_p == 'PAM' else 1.
                if p == 'sigma' and filetype_p == 'PAM':
                    self.dsg.radial_bins_all[p][j] = 512
                    self.dsg.radial_res_all[p][j] = 1.
                                    
        # In the case of duplicate scans (same scanangle) it becomes problematic when some duplicates are taken from PAM and others from PAG files
        # (since the program expects each duplicate to have same azimuthal and radial resolution). So in this case only duplicates from PAM files
        # are retained:
        for p in (_p for _p,filetypes in self.product_filetypes.items() if filetypes[0] == 'PAM'):
            scanangles_p = set(self.dsg.scanangles_all[p].values())
            for a in scanangles_p:
                scans_angle_a = [j for j,_a in self.dsg.scanangles_all[p].items() if _a == a]
                max_nrad = max(self.dsg.radial_bins_all[p][j] for j in scans_angle_a)
                for j in scans_angle_a:
                    if self.dsg.radial_bins_all[p][j] != max_nrad:
                        for attr in (_ for _ in gv.volume_attributes_p if _ != 'scannumbers_all'):
                            del self.dsg.__dict__[attr][p][j]
                        
        for p in self.product_filetypes:
            extra_attrs = [] if p != 'v' else [self.dsg.nyquist_velocities_all_mps, self.dsg.high_nyquist_velocities_all_mps, self.dsg.low_nyquist_velocities_all_mps]
            self.dsg.scannumbers_all[p] = bg.sort_volume_attributes(self.dsg.scanangles_all[p], self.dsg.radial_bins_all[p], self.dsg.radial_res_all[p], extra_attrs)
            # print(p, self.dsg.scanangles_all[p], self.dsg.scannumbers_all[p])
                    
        
    def read_extra_product(self, product, scan, data_shape, prefer_lowres=False):
        filetype, fileid = self.dsg.scannumbers_all[product][scan][self.dsg.scannumbers_forduplicates[scan]].split(',')[:2]
        filename = self.dsg.source_MeteoFrance.file_per_filetype_per_fileid[filetype][fileid]
        if product == 'z' and prefer_lowres:
            try:
                filename = self.dsg.source_MeteoFrance.file_per_filetype_per_fileid['PAG'][fileid]
                filetype = 'PAG'
            except Exception:
                pass
        filepath = self.crd.directory+'/'+filename
        p_data, p_data_mask = self.read_data(filepath, product, scan)[:2]
        
        na, nr = data_shape
        na_p, nr_p = p_data.shape
        # p_data might have a different shape than data, in which case regridding is necessary
        if product == 'sigma' and filetype == 'PAM':
            # In this case sigma is delivered on a 512*512 cartesian grid, centered at the radar
            a, r = 2*np.pi/na*(0.5+np.arange(na)), 256/nr*(0.5+np.arange(nr))
            a, r = np.meshgrid(a, r, indexing='ij')
            x, y = r*np.sin(a), r*np.cos(a)
            ix, iy = 256+np.floor(x).astype('int16'), 256-np.ceil(y).astype('int16')
            return p_data[(iy, ix)]
        elif na_p*nr_p > na*nr:
            a_ratio, r_ratio = na_p//na, nr_p//nr
            n = nr*r_ratio
            i = np.floor(np.linspace(0.5, nr_p-0.5, n, dtype='float32')).astype('uint16')                
            p_data[p_data_mask] = np.nan
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                return np.nanmean(np.reshape(p_data[:,i], (360, a_ratio, nr, r_ratio)), axis=(1,3))
        elif na_p*nr_p < na*nr:
            a, r = 0.5+np.arange(na), 0.5+np.arange(nr)
            a_ratio, r_ratio = na/na_p, nr/nr_p
            ia, ir = np.floor(a/a_ratio).astype('uint16'), np.floor(r/r_ratio).astype('uint16')
            return p_data[ia][:,ir]
        return p_data

    def read_data(self, filepath, product, scan, apply_dealiasing=False, productunfiltered=False):
        i_p = gv.i_p.get(product, product) # product='sigma' is not included in gv.i_p
        content = self.get_file_content(filepath, i_p)
        _, _, data_info, data_loops = self.bufr_decoder(content, read_mode='all')
        data_info, data_loops = data_info[0], data_loops[0]
        
        # Obtain n_rad from data_info and not from self.dsg.radial_bins_all, since for reflectivity it's possible that
        # the low-res version is requested, which has a different number of radial bins.
        n_rad = int(data_info['030021'][0])
        offset, scale = self.product_offset[i_p], self.product_scale[i_p]
        loop_id = list(data_loops)[-2]
        data = offset+scale*data_loops[loop_id]['030001'].astype('float32')
        n_azi = int(len(data)/n_rad)
        data = data.reshape((n_azi, n_rad))
        
        bounds = [offset, offset+scale*(2**self.product_nbits[i_p]-1)]
        # For velocity offset represents the maximum product value
        if self.product_maskvals[i_p] == 'minmax':
            data_mask = (data == min(bounds)) | (data == max(bounds))
        else:
            data_mask = data == (min(bounds) if self.product_maskvals[i_p] == 'min' else max(bounds))
            
        if not productunfiltered and product != 'sigma':
            try:
                sigma_data = self.read_extra_product('sigma', scan, data.shape)
                data_mask[sigma_data < 2.5] = True
            except Exception:
                print('Product filtering not possible')
                
        if i_p == 'v':
            data[data == -59.5] = -(59.5+self.dsg.nyquist_velocities_all_mps[scan])/2.
            data[data == 60.] = (60.+self.dsg.nyquist_velocities_all_mps[scan])/2.
            if not productunfiltered or apply_dealiasing:
                z_data = None
                try:
                    z_data = self.read_extra_product('z', scan, data.shape, prefer_lowres=True)
                except Exception:
                    print('Z array not available')
                c_data = None
                try:
                    c_data = self.read_extra_product('c', scan, data.shape)
                except Exception:
                    print('C array not available')
            
            if not productunfiltered:                 
                data_mask[np.abs(data) > self.dsg.nyquist_velocities_all_mps[scan]] = True
                phi = data * np.pi/self.dsg.nyquist_velocities_all_mps[scan]
                window = [1,2,3,2,1]
                v_std = ft.get_window_phase_stdev(phi, data_mask, window)*self.dsg.nyquist_velocities_all_mps[scan]/np.pi
                
                low_thres = False
                if not z_data is None:
                    z_std = ft.get_window_stdev(z_data, data_mask, window)
                    if not c_data is None:
                        low_thres = (z_data < 10 ) | ((z_data >= 10) & (z_data < 25) & (z_std < 3) & (c_data < 85))
                    else:
                        low_thres = (z_data < 10 ) | ((z_data >= 10) & (z_data < 25) & (z_std < 3))
                data_mask[low_thres & (v_std > 20)] = True
                # data_mask[~low_thres & (v_std > 30)] = True
            
            if apply_dealiasing and not self.dsg.low_nyquist_velocities_all_mps[scan] is None:
                data = self.dealias_velocity(data, data_mask, scan, z_data, c_data)
                
        #The end datetime is given here
        datetime = ''.join([format(int(float(data_info['00400'+str(j)][0])), '02') for j in range(1, 7)])
        enddate, endtime = datetime[:8], ft.format_time(datetime[-6:])
        antspeed = data_info['002109'][0]
        # End values are given here, not start values, so correction is needed afterwards
        timerange = ft.get_timerange_from_starttime_and_antspeed(enddate, endtime, -antspeed)
        i = timerange.index('-')
        scantime = timerange[i+1:]+'-'+timerange[:i]
        return data, data_mask, scantime


    def get_data(self, filepath, j): #j is the panel
        product = self.crd.products[j]
        scan = self.crd.scans[j]
        
        self.dsg.data[j], data_mask, self.dsg.scantimes[j] = self.read_data(filepath, product, scan, self.crd.apply_dealiasing[j], self.crd.productunfiltered[j])
        self.dsg.data[j][data_mask] = self.pb.mask_values[product]
        
        self.crd.using_unfilteredproduct[j] = self.crd.productunfiltered[j]
        
        
    def dealias_velocity(self, data, data_mask, scan, z_array=None, c_array=None):
        vn_l = self.dsg.low_nyquist_velocities_all_mps[scan]
        vn_h = self.dsg.high_nyquist_velocities_all_mps[scan]
        radial_res = self.dsg.radial_res_all['v'][scan]
        n_it = 1 if self.gui.vwp.updating_vwp else self.gui.dealiasing_dualprf_n_it
        return da.apply_dual_prf_dealiasing(data, data_mask, radial_res, self.dsg.nyquist_velocities_all_mps[scan], vn_l, vn_h, None, 
                                            window_detection = [1,2,1], window_correction = [1,2,1], deviation_factor = 1., n_it = n_it,
                                            z_array=z_array, c_array=c_array)
            
            
    def get_data_multiple_scans(self,filepaths,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        if isinstance(apply_dealiasing, bool):
            apply_dealiasing = {j: apply_dealiasing for j in scans}
        
        data = {}; scantimes = {}
        for j in scans:
            data[j], data_mask, scantimes[j] = self.read_data(filepaths[j], product, j, apply_dealiasing, productunfiltered)
            data[j][data_mask] = np.nan
            s = np.s_[:] if max_range is None else np.s_[:, :int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangles_all[i_p][j], 'gr+theta->sr') / self.dsg.radial_res_all[i_p][j]))]
            data[j] = data[j][s]
            data[j], scantimes[j] = [data[j]], [scantimes[j]]
        
        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes([i[0] for i in scantimes.values()])                
        meta = {'using_unfilteredproduct': productunfiltered, 'using_verticalpolarization': False}
        return data, scantimes, volume_starttime, volume_endtime, meta  





class MeteoFrance_NetCDF():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb   
        
        self.productnames = {'z':'TH', 'v':'VRAD'}
        
    
    def get_scans_information(self, filepath):
        with nc.Dataset(filepath) as f:
            scans = f.variables['sweep_number'][:]
            for j in scans:
                i_start = f.variables['sweep_start_ray_index'][j]
                self.dsg.scanangles_all['z'][j] = f.variables['elevation'][i_start]
                self.dsg.radial_bins_all['z'][j] = f.variables['TH'].shape[1]
                self.dsg.radial_res_all['z'][j] = f.variables['range'].meters_between_gates/1000
                
                self.dsg.nyquist_velocities_all_mps[j] = None
                self.dsg.high_nyquist_velocities_all_mps[j] = self.dsg.low_nyquist_velocities_all_mps[j] = None
                
        extra_attrs = [self.dsg.nyquist_velocities_all_mps, self.dsg.high_nyquist_velocities_all_mps, self.dsg.low_nyquist_velocities_all_mps]
        self.dsg.scannumbers_all['z'] = bg.sort_volume_attributes(self.dsg.scanangles_all['z'], self.dsg.radial_bins_all['z'], self.dsg.radial_res_all['z'], extra_attrs) 

        for i in self.dsg.scannumbers_all:
            for j in gv.volume_attributes_p: 
                self.dsg.__dict__[j][i] = copy.deepcopy(self.dsg.__dict__[j]['z'])
                
    def get_data(self, filepath, j):
        product, scan = self.crd.products[j], self.crd.scans[j]
        duplicate = self.dsg.scannumbers_forduplicates[scan]
        pname = self.productnames[product]
        
        with nc.Dataset(filepath) as f:
            idx = f.variables['sweep_start_ray_index'][self.dsg.scannumbers_all[product][scan][duplicate]]
            self.dsg.data[j] = f.variables[pname][idx:idx+720].astype('float32').filled(self.pb.mask_values[product])
            data_mask = self.dsg.data[j] == self.dsg.data[j].min()
            if product == 'v':
                self.dsg.data[j] *= -1
            
            self.dsg.data[j][data_mask] = self.pb.mask_values[product]

            volume_start_time = b''.join(f.variables['time_coverage_start'][:]).decode('utf-8').replace('\x00', '')
            volume_start_time = dtime.datetime.strptime(volume_start_time, '%Y-%m-%dT%H:%M:%SZ')
            print(idx)
            ray_times = f.variables['real_time'][idx:idx+720]
            if ray_times[0]:
                start_time = (volume_start_time+dtime.timedelta(seconds=round(ray_times.min()))).strftime('%H:%M:%S')
                end_time = (volume_start_time+dtime.timedelta(seconds=round(ray_times.max()))).strftime('%H:%M:%S')
                self.dsg.scantimes[j] = start_time+'-'+end_time
            else:
                self.dsg.scantimes[j] = ''
            






class UKMO_Polar():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
        
        self.reader = UKMOPolarFile()
        


    def get_scans_information(self, filepaths):
        # For at least one date+radar (20150701, Hameldon Hill) it has been observed that the listed average elevations are incorrect.
        # Or well, the angles themselves are in fact present in the radar volume, but they are associated with the wrong scan/file.
        # In this case the order of requested elevations was however correct, so it's therefore decided to first collect a list of
        # both avg and req elevations, and then use the observation that req elevations are correctly ordered to determine the correct
        # avg elevation.
        avg_elevations, req_elevations = [], []
        for j, filepath in enumerate(filepaths):
            volume_header, scan_header = self.reader(filepath, products=[])
            self.dsg.radial_res_all['z'][j] = volume_header['processed range bin length']/1e3
            self.dsg.radial_bins_all['z'][j] = volume_header['number of bins per ray']
            avg_elevations += [scan_header['scan average elevation']/10.]
            req_elevations += [scan_header['scan requested elevation']/10.]
            if avg_elevations[j] == 0.:
                avg_elevations[j] = req_elevations[j]
            elif abs(avg_elevations[j] - 90) < 1:
                avg_elevations[j] = 90
                
            self.dsg.nyquist_velocities_all_mps[j] = volume_header['unambiguous velocity']/100.
            prf_l, prf_h = volume_header['primary PRF'], volume_header['secondary PRF']
            radar_wavelength = volume_header['channel 1 wavelength']/1e3
            self.dsg.low_nyquist_velocities_all_mps[j] = radar_wavelength*prf_l/4. if prf_h != 0. else None
            self.dsg.high_nyquist_velocities_all_mps[j] = radar_wavelength*prf_h/4. if prf_h != 0. else None        
        sorted_avg = sorted(avg_elevations)
        sorted_req = sorted(req_elevations)
        self.dsg.scanangles_all['z'] = {j:sorted_avg[sorted_req.index(req_elevations[j])] for j in self.dsg.radial_res_all['z']}

        extra_attrs = [self.dsg.nyquist_velocities_all_mps, self.dsg.high_nyquist_velocities_all_mps, self.dsg.low_nyquist_velocities_all_mps]
        self.dsg.scannumbers_all['z'] = bg.sort_volume_attributes(self.dsg.scanangles_all['z'], self.dsg.radial_bins_all['z'], self.dsg.radial_res_all['z'], extra_attrs)
                    
        for i in self.dsg.scannumbers_all:
            for j in gv.volume_attributes_p: 
                self.dsg.__dict__[j][i] = copy.deepcopy(self.dsg.__dict__[j]['z'])

    def read_data(self, filepath, product, scan, apply_dealiasing=False, productunfiltered=False):
        i_p = gv.i_p[product]
        p_name = gv.productnames_UKMO[i_p]
        apply_dealiasing = i_p == 'v' and apply_dealiasing and not self.dsg.low_nyquist_velocities_all_mps[scan] is None

        products = set([p_name] + ['CI', 'CPA', 'SQI', 'REF']*(not productunfiltered) + ['RHO']*apply_dealiasing)
        data_products, volume_header, scan_header, ray_headers = self.reader(filepath, products)
        # Make a copy, since the product array gets cached in UKMOPolarFile and subsequent actions here should not alter that array
        data = data_products[p_name].copy()
        if product == 'p':
            data %= 360.
        
        if productunfiltered:
            data_mask = np.zeros_like(data, bool)
        else:
            if 'CI' in data_products:
                data_mask = (data_products['CI'] < 3.5) |\
                            ((data_products['CI'] < 4.25) & (data_products['REF'] < 30.)) |\
                            ((data_products['CI'] < 5.) & (data_products['REF'] < 15.))
            elif 'CPA' in data_products:
                data_mask = (data_products['CPA'] > 0.5)
            if 'SQI' in data_products:
                data_mask |= (data_products['SQI'] < 0.25) & (data_products['REF'] < 15.)
        
        if apply_dealiasing:
            data = self.dealias_velocity(data, data_mask, scan, c_array=data_products.get('RHO', None))
        
        start_time = ''.join([format(j, '02d') for j in volume_header['volume start time and date'][3:]])
        end_time = ''.join([format(j, '02d') for j in volume_header['volume stop time and date'][3:]])
        scantime = ft.format_time(start_time)+'-'+ft.format_time(end_time)
        return data, data_mask, scantime
                
    def get_data(self, filepath, j):
        product, scan = self.crd.products[j], self.crd.scans[j]
        
        self.dsg.data[j], data_mask, self.dsg.scantimes[j] = self.read_data(filepath, product, scan, self.crd.apply_dealiasing[j], self.crd.productunfiltered[j])
        self.dsg.data[j][data_mask] = self.pb.mask_values[product]
        
        self.crd.using_unfilteredproduct[j] = self.crd.productunfiltered[j]

    def dealias_velocity(self, data, data_mask, scan, z_array=None, c_array=None):
        vn_l = self.dsg.low_nyquist_velocities_all_mps[scan]
        vn_h = self.dsg.high_nyquist_velocities_all_mps[scan]
        # The staggered-PRF aliasing errors can be any linear combination of the form n1*vn_l+n2*vn_h, with n1 and n2 integers.
        # Considering all these options leads however to too much smoothing during dealiasing, so it's decided to only consider
        # the most prevalent aliasing errors, which deviate by (vn_l+vn_h)/2.
        vn_l = vn_h = (vn_l+vn_h)/2.
        radial_res = self.dsg.radial_res_all['v'][scan]
        n_it = 1 if self.gui.vwp.updating_vwp else self.gui.dealiasing_dualprf_n_it
        return da.apply_dual_prf_dealiasing(data, data_mask, radial_res, self.dsg.nyquist_velocities_all_mps[scan], vn_l, vn_h, None, 
                                            window_detection = [1,2,1], window_correction = [1,2,1], deviation_factor = 1.0, n_it = n_it,
                                            z_array=z_array, c_array=c_array)

    def get_data_multiple_scans(self,filepaths,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        if isinstance(apply_dealiasing, bool):
            apply_dealiasing = {j: apply_dealiasing for j in scans}
        
        data = {}; scantimes = {}
        for j in scans:
            data[j], data_mask, scantimes[j] = self.read_data(filepaths[j], product, j, apply_dealiasing, productunfiltered)
            data[j][data_mask] = np.nan
            s = np.s_[:] if max_range is None else np.s_[:, :int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangles_all[i_p][j], 'gr+theta->sr') / self.dsg.radial_res_all[i_p][j]))]
            data[j] = data[j][s]
            data[j], scantimes[j] = [data[j]], [scantimes[j]]
        
        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes([i[0] for i in scantimes.values()])                
        meta = {'using_unfilteredproduct': productunfiltered, 'using_verticalpolarization': False}
        return data, scantimes, volume_starttime, volume_endtime, meta  

    
    
    
    
class NEXRAD_L2():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
                
        self.filepath = None
        self.read_mode = None
        self.moments = None
    
    
    def get_scans_information(self, filepath):    
        self.read_file(filepath, 'min-meta')
        n_scans = self.file.nscans
        scans_record_indices = self.file.scan_msgs
        scans_startend_pos = self.file.scan_startend_pos
        radial_records = self.file.radial_records
        
        scanangles = self.file.get_target_angles()
        actual_scanangles = [self.file.get_elevation_angles()[indices].mean() for indices in scans_record_indices]
        max_diff = np.abs(actual_scanangles-scanangles).max()
        for i in range(len(scanangles)):
            diff = np.abs(actual_scanangles[i]-scanangles[:i]) if i else np.array([0])
            if scanangles[i] == 0. or max_diff > 1:
                # Sometimes no target scanangles are available, or target scanangles differ substantially from actual scanangles, meaning
                # that something is probably wrong. In that case use actual scanangles, while setting scanangles of scans
                # that are expected to be duplicates equal to each other, to ensure that they are also identified as such.
                i_nearest_angle = diff.argmin()
                i_select = i_nearest_angle if diff[i_nearest_angle] < 0.15 else i
                scanangles[i] = actual_scanangles[i_select]
            elif filepath.endswith('gz') and any((scanangles[:i] == scanangles[i]) & (diff > 0.15)):
                # For older files it has been observed that sometimes target angles of 2 scans are the same, while they clearly represent
                # different elevations. This step ensures that they will be treated as such.
                # Is not done for bzip2 compression, since (1) this is not used for old files and (2) the actual scanangle is based on 
                # only 1 radial for 'min-meta', which might well not give a good sample of the azimuthal mean scanangle. 
                scanangles[i] = actual_scanangles[i]
        
        unambig_ranges = [self.file.get_unambigous_range()[indices].mean() for indices in scans_record_indices]
        i_scans_z = []
        i_scans_exclude = [0] if gv.radar_bands[self.crd.radar] == 'C' else []
        if gv.radar_bands[self.crd.radar] == 'S':
            if n_scans > 1 and unambig_ranges[0]/unambig_ranges[1] < 1:
                # In this case the first z-scan/v-scan pair is incomplete, with only velocity available. Remove this v-scan, as in case of duplicates
                # not doing so leads to showing z-scans and v-scans from different pairs. 
                i_scans_exclude.append(0)
            for i in (j for j in range(n_scans-1) if abs(scanangles[j+1]-scanangles[j]) <= 0.4):
                ratio = unambig_ranges[i]/unambig_ranges[i+1]
                if ratio > 1.25:
                    i_scans_z.append(i)
                elif ratio > 0.5:
                    # Some radar volumes contain multiple repetitions of a velocity scan, each time with a slightly different PRF 
                    # (and thus unambiguous range). Only the 1st of these scans is retained, the others are excluded.
                    i_scans_exclude.append(i+1)
            if i_scans_z and n_scans-2 not in i_scans_z and scanangles[-1] < scanangles[-2]:
                # It's possible that the last z-scan/v-scan pair is incomplete, meaning that only the z-scan is available.
                # This can happen in an incomplete volume. 
                # It's not needed to do this when there's only 1 scan in the full volume, even though that could be a z-scan. The remainder
                # of the code handles this case sufficiently. Missing z-scans only becomes problematic once there are duplicates of a scan.
                i_scans_z.append(n_scans-1)
            
        i_scans_include = [i for i in range(n_scans) if not i in i_scans_exclude]
        i_scans_v = [i for i in i_scans_include[1:] if i-1 in i_scans_z]
        
        v_scan_present = 'v_scan' in str(self.dsg.product_versions_datetime) # Can be None
        products_z = ['z z_scan'] if self.dsg.product_versions_in1file else ['z']
        products_v = ['z v_scan', 'v'] if v_scan_present else ['v']
        products_all = products_z+products_v
        for j in gv.volume_attributes_p:
            ft.init_dict_entries_if_absent(self.dsg.__dict__[j], products_all, dict)
        for i in i_scans_include:
            j = i+1
            scan_record_indices = scans_record_indices[i]
            rcs = [radial_records[k] for k in scan_record_indices]
            rc = rcs[0]
                 
            products = (products_z if i in i_scans_z else products_v) if i in i_scans_z+i_scans_v else products_all
            for p in products:
                # Due to reading only a small part of each compressed data block for bz2-compressed data, metadata for only one product gets obtained
                # in this case. If available that product is 'REF', otherwise it's 'VEL'.
                # Since for multiple scans there is a difference between the number of radial bins for reflectivity and other products
                # (which have the same number of bins), this means that the correct number of radial bins for these other products needs to be 
                # determined in a different way. That is done further down
                moment = gv.productnames_NEXRAD[p[0]] if gv.productnames_NEXRAD[p[0]] in rc else 'REF'
                if not moment in rc:
                    continue
                
                if p == 'v':
                    # The Nyquist velocity can in fact change during a volume for a certain scan, meaning that different duplicates 
                    # have different values. Also, its value can even be azimuth-dependent for certain scans. Here I set its value
                    # equal to that of the first azimuth of a scan. And in the case of duplicates, the value for the first duplicate
                    # gets selected (in bg.sort_volume_attributes). Keep this in mind for certain operations!
                    vn = self.file.get_nyquist_vel(scans=[i])[0]
                    if vn == 0.:
                        # This is the case for TDWR radars where no Nyquist velocity is given. In this case it is set to 999, to prevent 
                        # issues in VWP creation where scans with low Nyquist velocity are excluded, while indicating that it is not a real
                        # Nyquist velocity.
                        vn = 999.
                    if len(self.dsg.nyquist_velocities_all_mps) and vn < 0.5*min(self.dsg.nyquist_velocities_all_mps.values()):
                        # It's possible that some z-only scans are missed, despite the attemps to detect them above. This check aims to remove
                        # those that have been missed.
                        continue
                    self.dsg.nyquist_velocities_all_mps[j] = vn
                
                dr = rc[moment]['gate_spacing']
                first_gate = (rc[moment]['first_gate']-0.5*dr)/dr
                self.dsg.radial_bins_all[p][j] = int(first_gate+rc[moment]['ngates'])
                self.dsg.radial_res_all[p][j] = dr/1e3
                self.dsg.scanangles_all[p][j] = scanangles[i]
                                
        for p in products_all:
            extra_attrs = [] if p[0] == 'z' else [self.dsg.nyquist_velocities_all_mps]
            self.dsg.scannumbers_all[p] = bg.sort_volume_attributes(self.dsg.scanangles_all[p], self.dsg.radial_bins_all[p], self.dsg.radial_res_all[p], extra_attrs)
            for i, j in self.dsg.scannumbers_all[p].items():
                self.dsg.scannumbers_all[p][i] = [scans_startend_pos[k-1] for k in j]
                
        if v_scan_present:
            # Add scannumbers for 'z' for the combi_scan. These are a combination of those for z_scan and v_scan. 
            # Since this is only done for 'z' and not for other products, a consequence is that 'z' gets twice as many duplicates.
            # And since changes to self.dsg.scannumbers_forduplicates are based on self.dsg.scannumbers_all['z'], this becomes
            # an issue for other products without some other measures. The remedy for this is a new function self.dsg.duplicate,
            # which halves the duplicate index (i.e. self.dsg.scannumbers_forduplicates[scan]) for products other than 'z'.
            self.dsg.scannumbers_all['z combi_scan'] = {}
            for j,s1 in self.dsg.scannumbers_all['z z_scan'].items():
                s2 = self.dsg.scannumbers_all['z v_scan'].get(j, s1)
                if s1 != s2:
                    self.dsg.scannumbers_all['z combi_scan'][j] = []
                    l1, l2 = len(s1), len(s2)
                    for k in range(max(l1, l2)):
                        self.dsg.scannumbers_all['z combi_scan'][j] += ([s1[k]] if k < l1 else []) + ([s2[k]] if k < l2 else [])
                else:
                    self.dsg.scannumbers_all['z combi_scan'][j] = s1
            for j in (a for a in gv.volume_attributes_p if a != 'scannumbers_all'):
                self.dsg.__dict__[j]['z combi_scan'] = self.dsg.__dict__[j]['z z_scan'].copy()
                
        self.dsg.high_nyquist_velocities_all_mps = self.dsg.low_nyquist_velocities_all_mps = {i:None for i in self.dsg.scannumbers_all['v']}
        
        for j in gv.volume_attributes_p:
            for p in ('w',):
                self.dsg.__dict__[j][p] = self.dsg.__dict__[j]['v']
            for p in ('c', 'p', 'd'):
                # Copy needed for at least 'radial_bins_all' due to steps below
                self.dsg.__dict__[j][p] = self.dsg.__dict__[j][products_z[0]].copy()
                
        if 1 in self.dsg.radial_bins_all['v']:
            # Corrects for incorrect number of radial bins as determined above for some scans for products other than reflectivity
            n_rad_max = self.dsg.radial_bins_all['v'][1]
            for p in ('v', 'w', 'c', 'p', 'd'):
                for i in self.dsg.radial_bins_all[p]:
                    self.dsg.radial_bins_all[p][i] = min(self.dsg.radial_bins_all[p][i], n_rad_max)
                                    
        # For bzip2-compressed files the start and end positions of the data for each scan are expected to vary from volume to volume
        self.dsg.variable_attributes = ['scanangles_all']+['scannumbers_all']*self.file._bzip2_compression
        
        if self.file._bzip2_compression:
            self.file.close()
                
        
    def read_data(self, product, scan, scan_index=0, n_rad=None, panel=None):
        i_p = gv.i_p[product]
        msg = self.file.radial_records[self.file.scan_msgs[scan_index][0]]
        if n_rad is None:
            n_rad = self.dsg.radial_bins_all[i_p][scan]
                    
        azis = self.file.get_azimuth_angles(scans=[scan_index])
        da = msg['msg_header'].get('azimuth_resolution', 2)/2
        azis, n_azi, da, azi_offset, diffs = process_azis_array(azis, da, calc_azi_offset=panel != None)
                
        moment = gv.productnames_NEXRAD[i_p]
        dr = msg[moment]['gate_spacing']/1e3
        r_first_gate = msg[moment]['first_gate']/1e3-0.5*dr
        first_gate = int(np.ceil(r_first_gate/dr))
        rad_offset = r_first_gate - first_gate*dr
        if panel != None:
            self.dsg.data_radius_offset[panel] = rad_offset
        
        raw_data = self.file.get_data(moment, n_rad-first_gate, scans=[scan_index], raw_data=True)
        data = np.zeros((n_azi, n_rad), raw_data.dtype)
        # first_gate can be negative, which I interpret as a sign that the first radial bin(s) of the raw data should be excluded
        data[:, max(0, first_gate):] = raw_data[-n_azi:, max(0, -first_gate):]
        data_mask = data <= 1
        
        offset, scale = np.float32(msg[moment]['offset']), np.float32(msg[moment]['scale'])
        data = (data - offset) / scale
        if product == 'c':
            data *= 100.
        data[data_mask] = np.nan
        
        # Sometimes more than one Nyquist velocity is used for different sectors of a scan.
        vn_azis = self.file.get_nyquist_vel(scans=[scan_index])[-n_azi:]
        # Exclude Nyquist velocities of 0 as is the case for TDWR radars
        if i_p == 'v' and vn_azis[0] != 0. and panel != None and self.crd.apply_dealiasing[panel] and 'Unet VDA' in self.gui.dealiasing_setting:
            data = self.dsg.perform_mono_prf_dealiasing(panel, data, vn_azis, azis, da)
        
        data, azi_offset = map_onto_regular_grid(data, n_azi, azis, diffs, da, azi_offset)
        if panel != None:
            self.dsg.data_azimuth_offset[panel] = azi_offset
        data_mask = np.isnan(data)
        
        start_datetime, time_offsets = self.file.get_times(scans=[scan_index])
        time_offsets = time_offsets[-n_azi:]
        start_time = (start_datetime+dtime.timedelta(seconds=round(time_offsets[0]))).strftime('%H:%M:%S')
        end_time = (start_datetime+dtime.timedelta(seconds=round(time_offsets[-1]))).strftime('%H:%M:%S')
        scantime = start_time+'-'+end_time
        
        # The scanangles determined in get_scans_information are only target scanangles, not actual (azimuthally averaged) scanangles.
        # Here these values are updated to actual values, in the case of duplicate scans one for each duplicate. In principle calculating
        # the average actual scanangle could be done in get_scans_information itself, but the problem with that is that for
        # bzip2-compressed files data for only a few radials is obtained, which makes calculating the actual average impossible.
        scanangle = self.file.get_elevation_angles(scans=[scan_index]).mean()
        duplicate = self.dsg.duplicate(product, scan)
        scanangle_in_dict = isinstance(self.dsg.scanangles_all[i_p][scan], dict) and\
                            duplicate in self.dsg.scanangles_all[i_p][scan]
        if self.dsg.scanangles_all[i_p][scan] != scanangle and not scanangle_in_dict:
            # For 'z' multiple product versions (pvs) are available, with combi_scan combining z_scan and v_scan.
            # This means that each scan for 'z' is contained in 2 pvs, while it gets imported only once.
            # It's therefore necessary to also update the scanangle for the other pv below.
            s = self.dsg.scannumbers_all[i_p][scan][duplicate]
            keys = ['z '+pv for pv in self.dsg.product_versions_datetime] if i_p == 'z' else [i_p]
            for k in keys:
                if s in self.dsg.scannumbers_all[k][scan]:
                    _duplicate = self.dsg.scannumbers_all[k][scan].index(s)
                    if len(self.dsg.scannumbers_all[k][scan]) == 1:
                        self.dsg.scanangles_all[k][scan] = scanangle
                    else:
                        if not isinstance(self.dsg.scanangles_all[k][scan], dict):
                            self.dsg.scanangles_all[k][scan] = {}
                        self.dsg.scanangles_all[k][scan][_duplicate] = scanangle
                
            self.dsg.update_volume_attributes = True
            # For bzip2-compressed files the start and end positions of the data for each scan are expected to vary from volume to volume
            self.dsg.variable_attributes = ['scanangles_all']+['scannumbers_all']*(not self.filepath.endswith('.gz'))
        
        if panel is None:
            return data, data_mask, scantime, rad_offset 
        else:
            return data, data_mask, scantime
        
    
    def read_file(self, filepath, read_mode, products=None, j=None): # j is the panel
        # products and j should either be not specified at all, or 1 of them should be specified. Don't specify both
        moments = None
        if not j is None:
            # Check whether subsequent panels display other products of the same scan. If true, then obtain also these products in the 
            # same function call below. This is more efficient than obtaining these products in separate calls.
            i_ps = [gv.i_p[self.crd.products[i]] for i,s in self.dsg.scannumbers_panels.items() if s == self.dsg.scannumbers_panels[j]]
            moments = [gv.productnames_NEXRAD[p] for p in i_ps]
        elif products:
            moments = [gv.productnames_NEXRAD[p] for p in products]
            
        if self.filepath == filepath:
            if type(read_mode) is list and read_mode == self.read_mode and moments:
                # Exclude moments that have already been obtained for this read_mode
                moments = [m for m in moments if not m in self.moments]
            if moments is None or len(moments) > 0:
                self.file(read_mode, moments)
        else:
            if hasattr(self, 'file'):
                # Close any previously opened file if that has not yet happened before (i.e. with gzipped files)
                self.file.close()
            self.file = NEXRADLevel2File(filepath, read_mode, moments)
        self.filepath, self.read_mode, self.moments = filepath, read_mode, moments
                
    def get_data(self, filepath, j): #j is the panel        
        product, i_p = self.crd.products[j], gv.i_p[self.crd.products[j]]
        scan = self.crd.scans[j]
        
        scan_index = self.dsg.scannumbers_all[i_p][scan][self.dsg.duplicate(product, scan)]
        self.read_file(filepath, scan_index, j=j)
        self.dsg.data[j], data_mask, self.dsg.scantimes[j] = self.read_data(product, scan, panel=j)
        self.dsg.data[j][data_mask] = self.pb.mask_values[product]
        
        if self.file._bzip2_compression:
            self.file.close()
        
    def get_data_multiple_scans(self,filepath,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        
        # In case of duplicates always take the first scan for velocity, since for other duplicates the Nyquist velocity might not be valid 
        # (see get_scans_information)
        duplicates_select = slice(0, 1) if i_p == 'v' else slice(None)
        startend_pos = sum([self.dsg.scannumbers_all[i_p][j][duplicates_select] for j in scans], [])
        self.read_file(filepath, startend_pos, product)
        
        scan_indices = [i for i,j in enumerate(self.file.scan_msgs) if len(j)]
        # scan_indices as defined above will refer to scans sorted according to elevation number, and not according to the order in
        # self.dsg.scannumbers_all. The following procedure changes that
        sort_indices = np.argsort([i[0] for i in startend_pos])
        sort_indices_inverse = [j[0] for j in sorted(enumerate(sort_indices), key=lambda x: x[1])]
        scan_indices = [scan_indices[i] for i in sort_indices_inverse]
        
        data, scantimes, radius_offsets = {}, {}, {}
        i = 0
        for j in scans:
            data[j], scantimes[j] = [], []
            n_duplicates_scan = len(self.dsg.scannumbers_all[i_p][j])
            for k in list(range(n_duplicates_scan))[duplicates_select]:
                n_rad = None if max_range is None else int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangle(i_p, j, k), 'gr+theta->sr') / self.dsg.radial_res_all[i_p][j]))
                scan_data, _, scan_scantime, rad_offset = self.read_data(product, j, scan_indices[i], n_rad)
                data[j].append(scan_data)
                scantimes[j].append(scan_scantime)
                radius_offsets[j] = rad_offset
                i += 1
        
        scantimes_all = sum(list(scantimes.values()), [])
        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes(scantimes_all)
        meta = {'using_unfilteredproduct': False, 'using_verticalpolarization': False, 'radius_offsets': radius_offsets}
        
        if self.file._bzip2_compression:
            self.file.close()
        return data, scantimes, volume_starttime, volume_endtime, meta





class NEXRAD_L3():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
            
    def get_scans_information(self, filepaths):
        j = 0
        for i, filepaths_per_product in filepaths.items():
            j += 1
            start_dts = {p:[] for p in filepaths_per_product}
            for p, fpaths in filepaths_per_product.items():
                for fpath in fpaths:
                    file = NEXRADLevel3File(fpath)
                    start_dts[p].append(file.get_volume_start_datetime())
                ranges = file.get_range()
                self.dsg.radial_res_all[p][j] = (ranges[1]-ranges[0])/1e3
                self.dsg.radial_bins_all[p][j] = len(ranges)
                self.dsg.scanangles_all[p][j] = file.get_elevation()
                self.dsg.nyquist_velocities_all_mps[j] = 999
                self.dsg.low_nyquist_velocities_all_mps[j] = self.dsg.high_nyquist_velocities_all_mps[j] = None
                file.close()
            """It can happen that the same file index in fpaths refers to a different scan for each product (e.g. index 0 refers to a 
            reflectivity scan from 00:00Z and a velocity scan from 00:01Z). These scans should be treated as 2 duplicates, even though
            each product has only 1 of the pair available. The following lines ensure that the number of duplicates is based on the 
            total number of scans available for all products. A potential consequence is that the same duplicate index does not correspond
            anymore to the same file index in fpaths. For this reason, scannumbers_all[p][j] contains a list with for each duplicate
            a sublist of 2 elements. The first element contains the fileid i, and the second the file index in fpaths to which the
            duplicate index corresponds for product p. If the duplicate is not available for product p, then this index is set to None
            (which throws an error).
            """
            start_dts_scan = np.unique(sum([list(l) for l in start_dts.values()], []))
            n_duplicates = len(start_dts_scan)
            for p in start_dts:
                self.dsg.scannumbers_all[p][j] = [[i] for k in range(n_duplicates)]
                for k, start_dt in enumerate(start_dts_scan):
                    result = np.where(np.array(start_dts[p]) == start_dt)[0]
                    index = result[0] if len(result) else None
                    self.dsg.scannumbers_all[p][j][k].append(index)

        products = np.unique(sum([list(i) for i in filepaths.values()], []))
        scans = np.unique(sum([list(self.dsg.scannumbers_all[p]) for p in products], []))
        scanangles_products = {p:np.array(list(self.dsg.scanangles_all[p].values())) for p in products}
        # If a certain scan is only available for 1 product, then for the other product the nearest available (in terms of scanangle)
        # other scan is chosen. The following lines determine the nearest scan and set the corresponding volume attributes.
        for j in scans:
            scanangles_j = {p:self.dsg.scanangles_all[p][j] for p in products if j in self.dsg.scanangles_all[p]}
            for p in products:
                if not j in self.dsg.scannumbers_all[p]:
                    other_product = [i for i in products if not i == p][0]
                    i_nearest_scanangle = np.abs(scanangles_products[p]-scanangles_j[other_product]).argmin()
                    nearest_scan = list(self.dsg.scanangles_all[p])[i_nearest_scanangle]
                    for attr in gv.volume_attributes_p:
                        self.dsg.__dict__[attr][p][j] = self.dsg.__dict__[attr][p][nearest_scan]
                            
    def read_data(self, filepath, product, scan, panel=None):
        i_p = gv.i_p[product]
        file = NEXRADLevel3File(filepath)
                
        rad_offset = file.packet_header['first_bin']/1e3
        if panel != None:
            self.dsg.data_radius_offset[panel] = rad_offset
        
        azis = file.get_azimuth()
        da = 1
        azis, n_azi, da, azi_offset, diffs = process_azis_array(azis, da, calc_azi_offset=panel != None)
        
        file_data = file.get_data().filled(np.nan)[-n_azi:]
        # Sometimes different duplicates of a scan have a slightly different number of radial bins, which can lead
        # to issues with derived product calculation. The number of radial bins is therefore explicitly set to 
        # self.dsg.radial_bins_all[i_p][scan], which can require either excluding the last few radial bins of file_data, 
        # or adding a few empty radial bins to data.
        n_rad1, n_rad2 = self.dsg.radial_bins_all[i_p][scan], file_data.shape[1]
        data = np.full((n_azi, n_rad1), np.nan, 'float32')
        data[:, :n_rad2] = file_data[:, :n_rad1]
        
        data, azi_offset = map_onto_regular_grid(data, n_azi, azis, diffs, da, azi_offset)
        if panel != None:
            self.dsg.data_azimuth_offset[panel] = azi_offset
        data_mask = np.isnan(data)
        
        if i_p == 'v' and self.crd.radar[0] == 'K':
            data /= 1.9426
        
        # The scanangle can vary among duplicate scans, so self.dsg.scanangles_all[i_p][scan] should become a dictionary with an angle
        # for each duplicate index. In principle that could already be done in get_scans_information, but since it requires opening files
        # for each duplicate scan, which consumes time, it is decided to do it here (only for the current duplicate scan, for which the file
        # is open here).
        scanangle = file.get_elevation()
        scanangle_in_dict = isinstance(self.dsg.scanangles_all[i_p][scan], dict) and\
                            self.dsg.scannumbers_forduplicates[scan] in self.dsg.scanangles_all[i_p][scan]
        if self.dsg.scanangles_all[i_p][scan] != scanangle and not scanangle_in_dict:
            if len(self.dsg.scannumbers_all[i_p][scan]) == 1:
                self.dsg.scanangles_all[i_p][scan] = scanangle
            else:
                if not isinstance(self.dsg.scanangles_all[i_p][scan], dict):
                    self.dsg.scanangles_all[i_p][scan] = {}
                self.dsg.scanangles_all[i_p][scan][self.dsg.scannumbers_forduplicates[scan]] = scanangle        
            self.dsg.update_volume_attributes = True
            self.dsg.variable_attributes = ['scanangles_all']
                
        if self.crd.radar[0] == 'T':
            scantime = file.get_volume_start_datetime().strftime('%H:%M:%S')+'-'+file.get_product_datetime().strftime('%H:%M:%S')
        else:
            scantime = file.get_product_datetime().strftime('%H:%M:%S')
            
        file.close()
        return data, data_mask, scantime
        
    def get_data(self, filepath, j): #j is the panel 
        product, scan = self.crd.products[j], self.crd.scans[j]
        
        self.dsg.data[j], data_mask, self.dsg.scantimes[j] = self.read_data(filepath, product, scan, j)
        self.dsg.data[j][data_mask] = self.pb.mask_values[product]

    def get_data_multiple_scans(self,filepaths,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """apply_dealiasing can be either a bool or a dictionary that specifies per scan whether dealiasing should be applied.
        max_range specifies a possible maximum range (in km) up to which data should be obtained.
        """
        i_p = gv.i_p[product]
        data, scantimes = {}, {}
        duplicates_select = slice(0, 1) if i_p == 'v' else slice(None)
        for j in scans:
            data[j], scantimes[j] = [], []
            n_duplicates_scan = len(self.dsg.scannumbers_all[i_p][j])
            for k in list(range(n_duplicates_scan))[duplicates_select]:
                fileid = self.dsg.scannumbers_all[i_p][j][k][0]
                filepath = filepaths[fileid][k]
                n_rad = None if max_range is None else int(np.ceil(ft.var1_to_var2(max_range, self.dsg.scanangle(i_p, j, k), 'gr+theta->sr') / self.dsg.radial_res_all[i_p][j]))
                scan_data, _, scan_scantime = self.read_data(filepath, product, j)
                data[j].append(scan_data[:, :n_rad])
                scantimes[j].append(scan_scantime)
                
        scantimes_all = sum(list(scantimes.values()), [])
        volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes(scantimes_all)
        meta = {'using_unfilteredproduct': False, 'using_verticalpolarization': False}
        return data, scantimes, volume_starttime, volume_endtime, meta





class CFRadial():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
        
    def get_scans_information(self, filepaths):
        j = 0
        for i, fpaths in filepaths.items():
            j += 1
            with nc.Dataset(fpaths[0], 'r') as f:        
                self.dsg.radial_res_all['z'][j] = float(f.variables['range'].getncattr('meters_between_gates'))/1e3
                self.dsg.radial_bins_all['z'][j] = len(f.variables['range'])
                self.dsg.scanangles_all['z'][j] = f.variables['elevation'][:].mean()
                self.dsg.nyquist_velocities_all_mps[j] = f.variables['nyquist_velocity'][0]
                self.dsg.low_nyquist_velocities_all_mps[j] = self.dsg.high_nyquist_velocities_all_mps[j] = None #Data is mono PRF   
            self.dsg.scannumbers_all['z'][j] = [i]*len(fpaths)
                    
        for i in self.dsg.scannumbers_all:
            for j in gv.volume_attributes_p: 
                self.dsg.__dict__[j][i] = copy.deepcopy(self.dsg.__dict__[j]['z'])
    
    def get_data(self, filepath, j): #j is the panel 
        i_p, product = gv.i_p[self.crd.products[j]], self.crd.products[j]
        scan = self.crd.scans[j]
        with nc.Dataset(filepath, 'r') as f:
            gv.radar_ids[self.crd.radar] = f.getncattr('instrument_name')
            lat, lon, height = [float(f[i][0]) for i in ('latitude', 'longitude', 'altitude')]
            gv.radarcoords[self.crd.radar] = [lat, lon]
            gv.radar_elevations[self.crd.radar] = int(round(height))
            
            azis = f.variables['azimuth'][:] % 360.
            clockwise = ft.angle_diff(azis[0], azis[1]) > 0.
            s = np.s_[:] if clockwise else np.s_[::-1]
            
            azis = azis[s]            
            da = 1
            azis, n_azi, da, azi_offset, diffs = process_azis_array(azis, da, calc_azi_offset=not j is None, azi_pos='left')
            
            var_name = gv.productnames_ARRC[product]
            print(var_name, product, f.variables)
            data = f.variables[var_name][s].astype('float32').filled(np.nan)[-n_azi:]
            if product == 'c':
                data *= 100.
            
            if i_p == 'v' and not j is None and self.crd.apply_dealiasing[j]:
                vn = self.dsg.nyquist_velocities_all_mps[scan]
                # vmax = np.abs(data).max()
                # if vmax > vn:
                #     vn = 26.2
                data = self.dsg.perform_mono_prf_dealiasing(j, data, vn, azis, da)
            
            self.dsg.data[j], azi_offset = map_onto_regular_grid(data, n_azi, azis, diffs, da, azi_offset, azi_pos='left')
            if not j is None:
                self.dsg.data_azimuth_offset[j] = azi_offset
            data_mask = np.isnan(self.dsg.data[j])
            self.dsg.data[j][data_mask] = self.pb.mask_values[product]
            
            self.dsg.scantimes[j] = f.getncattr('start_datetime')[11:19]+'-'+f.getncattr('end_datetime')[11:19]
            
            
            
            
            
class DORADE():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.pb = self.gui.pb
        
        self.product_names = {'z':['CR', 'DBZ', 'ZH', 'DZ'], 'v':['VE', 'VF', 'VEL'], 'w':['VW', 'SW'],
                              'c':['RHOHV', 'RH'], 'd':['ZDR', 'ZD'], 'p':['PHIDP', 'PH']}
        # self.c = 299792458
        # self.radar_wavelengths = {'MWR_05XP':self.c/9.9e9}
        
    def get_scans_information(self, filepaths):
        j = 0
        for i, fpaths in filepaths.items():
            j += 1
            file = DORADEFile(fpaths[0])
            meta = file.get_meta()
            self.dsg.radial_res_all['z'][j] = meta['dr']/1e3
            self.dsg.radial_bins_all['z'][j] = meta['n_rad']
            self.dsg.scanangles_all['z'][j] = meta['elevations'].mean()
            # rmax = meta['n_rad']*meta['dr']
            # prf = self.c/(2*rmax)
            # radar = file.radar_name
            # vn = self.radar_wavelengths[radar]*prf/4 if radar in self.radar_wavelengths else meta['v_nyquist']
            # print(rmax, vn)
            self.dsg.nyquist_velocities_all_mps[j] = meta['v_nyquist']
            self.dsg.low_nyquist_velocities_all_mps[j] = self.dsg.high_nyquist_velocities_all_mps[j] = None #Data is mono PRF
            self.dsg.scannumbers_all['z'][j] = [i]*len(fpaths)
                    
        for i in self.dsg.scannumbers_all:
            for j in gv.volume_attributes_p: 
                self.dsg.__dict__[j][i] = copy.deepcopy(self.dsg.__dict__[j]['z'])
    
    def get_data(self, filepath, j): #j is the panel 
        i_p, product = gv.i_p[self.crd.products[j]], self.crd.products[j]
        scan = self.crd.scans[j]
        
        file = DORADEFile(filepath)
        data = file.get_data(self.product_names[i_p])
        if product == 'c':
            data *= 100.
        meta = file.get_meta()      
        times = meta['times']
        azis = meta['azimuths']
        
        gv.radar_ids[self.crd.radar] = file.radar_name
        lon, lat, height = [meta[j] for j in ('lon', 'lat', 'altitude')]
        gv.radarcoords[self.crd.radar] = [lat, lon]
        gv.radar_elevations[self.crd.radar] = int(round(height))
        
        diffs = ft.angle_diff(azis)
        clockwise = np.median(diffs[diffs != 0.]) > 0.
        s = np.s_[:] if clockwise else np.s_[::-1]        
        azis, diffs, times, data = azis[s], diffs[s], times[s], data[s]
        if not clockwise:
            diffs *= -1
        
        # for i,a in enumerate(azis):
        #     print(i, a)
        #     if i < len(diffs):
        #         print(i, diffs[i], 'diffs')
        azis_doubtful = azis == 0.
        azis_doubtful[1:] |= diffs < 0.1
        i_azis_doubtful = np.nonzero(azis_doubtful)[0]
        if len(i_azis_doubtful):
            data_nan = np.isnan(data)
            i_remove = []
            for i in i_azis_doubtful:
                if azis[i] != 0. or data_nan[i].all():
                    i_remove.append(i)
            azis = np.delete(azis, i_remove)
            times = np.delete(times, i_remove)
            data = np.delete(data, i_remove, axis=0)
        
        da = 1
        azis, n_azi, da, azi_offset, diffs = process_azis_array(azis, da, calc_azi_offset=not j is None, azi_pos='center')
        times = times[-n_azi:]
        data = data[-n_azi:]
                
        if i_p == 'v' and not j is None and self.crd.apply_dealiasing[j]:
            vn = self.dsg.nyquist_velocities_all_mps[scan]
            vmax = np.abs(data).max()
            if vmax > vn:
                vn = vmax
            data = self.dsg.perform_mono_prf_dealiasing(j, data, vn, azis, da)
        
        self.dsg.data[j], azi_offset = map_onto_regular_grid(data, n_azi, azis, diffs, da, azi_offset, azi_pos='center')
        if not j is None:
            self.dsg.data_azimuth_offset[j] = azi_offset
        data_mask = np.isnan(self.dsg.data[j])
        self.dsg.data[j][data_mask] = self.pb.mask_values[product]
        
        t_start, t_end = times[0 if clockwise else -1], times[-1 if clockwise else 0]
        self.dsg.scantimes[j] = t_start+('-'+t_end)*(t_start != t_end)