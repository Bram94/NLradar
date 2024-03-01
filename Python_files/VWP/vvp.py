# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

from scipy.optimize import lsq_linear
import numpy as np
import os
import h5py
import time as pytime

import nlr_functions as ft
import nlr_globalvars as gv



"""Implementation of VVP technique as described in 'Doppler Radar Wind Profiles', Iwan Holleman, Scientific Report, KNMI WR-2003-02, 2003:
http://bibliotheek.knmi.nl/knmipubWR/WR2003-02.pdf
"""

class VVP():
    def __init__(self, gui_class, range_limits = [2., 25.], v_min = 2., height_limits = [0.1, 11.9], dh = 0.2, n_sectors = 36, min_sector_pairs_filled = 9, min_area_sector = 7., dv_outliers = 10.):
        """Parameters used within the algorithm"""
        self.gui = gui_class
        self.crd = self.gui.crd
        self.dsg = self.gui.dsg
        
        self.v_min = v_min #Minimum radial velocity that is taken into account in the VVP retrieval. It should be greater than zero,
        #to prevent contamination by ground clutter.
        
        self.range_limits = range_limits #Minimum and maximum range in km between which velocities are sought to which the VVP technique is applied
        self.height_limits = height_limits #Maximum and minimum height in km for which the velocity is determined
        self.dh = dh #Thickness of height layer in km
        
        """The azimuthal domain is divided into n_sectors sectors of 360/n_sectors degrees. These sectors are grouped into pairs of two that
        are located opposite to each other. The velocity bins in at least one sector within such a pair should cover a horizontal area of at least
        min_area_sector, and this condition should hold for at least min_sector_pairs_filled sector pairs. If not, then the height layer is not
        further processed. 
        See the function self.test_data_availability_height_layers for more info, as well as another condition posed on data availability.
        360/self.n_sectors should be an integer!!!
        """
        self.n_sectors = n_sectors
        self.min_sector_pairs_filled = min_sector_pairs_filled
        self.min_area_sector = min_area_sector
        
        self.dv_outliers = dv_outliers #After applying the first fit, outliers are identified as values that differ by more than this value from
        #the value that follows from the fit. In m/s
        
        # Should change when either the structure of the hdf5 file changes, or when the VVP calculation method changes
        self.file_content_version = 1
        
    
    def __call__(self, range_limits = None, height_limits = None, v_min = None, h0=0, V0=None):
        if not range_limits is None:
            self.range_limits = range_limits
        if not height_limits is None:
            self.height_limits = height_limits
        if not v_min is None:
            self.v_min = v_min
        self.h0 = h0 # Surface elevation relative to radar elevation, in km. Is used when surface observations are used at the bottom of the wind profile.
        self.V0 = V0 # Surface wind observation if available, is used for dealiasing velocities
        
        dataset_at_disk = self.check_if_dataset_at_disk()
        if dataset_at_disk:
            return self.h_layers, self.V, self.w, self.sigma, self.filled_sectors, self.volume_starttime, self.volume_endtime
        
        self.antenna_height = gv.radar_towerheights[self.crd.radar]/1e3 #Height of the radar antenna above ground level, in km. The heights at which the velocity is
        #determined are relative to ground level, and this antenna height is used to take into account that the radar antenna is located a bit higher.
        
        scans = []; scannumbers = []
        for j in self.dsg.nyquist_velocities_all_mps:
            if not self.dsg.scannumbers_all['v'][j][-1] in scannumbers:
                # Prevent that one scan is included more than once, which could happen when there is a different availability of
                # reflectivity and velocity scans (in this case a nearby scan for the unavailable product is repeated)
                scans.append(j)
                scannumbers.append(self.dsg.scannumbers_all['v'][j][-1])            
        scans = [j for j in scans if self.dsg.nyquist_velocities_all_mps[j]>10. and self.dsg.scanangles_all_m['v'][j] < 90.] #Exclude the mono-PRF and vertical scans
        if len(scans) == 0: 
            raise Exception('No data (yet) available')
            
        self.data, _, self.volume_starttime, self.volume_endtime, _ = self.dsg.get_data_multiple_scans('v', scans, apply_dealiasing = True, max_range = self.range_limits[1])
        self.data = {j: self.data[j][0] for j in self.data} #self.data[j] contains a list, since some scans might be present twice in the radar volume. But
        #for making a VVP it doesn't matter much which one is taken, and here it's taken to be the 1st.
        self.scanangles = {j: self.dsg.scanangle('v', j, 0) for j in self.data}
        self.radial_bins = {j: self.data[j].shape[1] for j in self.data}
        self.radial_res = self.dsg.radial_res_all['v']
        
        self.h_layers, self.V, self.w, self.sigma = np.array([]), np.array([]), np.array([]), np.array([])
        
        """Step 1: Divide the vertical dimension into height layers with a thickness of 200 meters, from 0 to 6 km. Next, list all velocities
        that are available in this layer, as a function of the azimuth phi."""
        self.get_V_height_layers()
        
        """Step 2: Divide the azimuthal dimension into sectors, and check whether enough data is available within these sectors.
        If not, then remove these height layers."""
        self.test_data_availability_height_layers()
        """Step 3: Fit the formula "Vr = w*sin(theta)+u_0*cos(theta)*sin(phi)+v_0*cos(theta)*cos(phi)" to the self.data, 
        where w = w_0 + W_f, the sum of the vertical velocity and the terminal velocity of hydrometeors."""
        self.retrieve_velocities_through_fit(determine_uncertainty=False)
        """Step 4: Remove outliers, by requiring that the radial velocities deviate by less than self.dv_outliers from the value that follows
        from the determined velocities. After removal of outliers, repeat the fit."""
        self.remove_outliers_and_repeat_fit()
        
        self.h_layers = np.array(list(self.h_layers.values()))
        
        self.write_data_to_file()
        
        return self.h_layers, self.V, self.w, self.sigma, self.filled_sectors, self.volume_starttime, self.volume_endtime
    
    
    def get_V_height_layers(self):
        """Divides the vertical dimension into height layers with a thickness of self.dh km, from self.height_limits[0] to self.height_limits[1] km.
        Next, all available velocities in these layers are listed, as a function of the azimuth phi and the scanangle theta.        
        At last the total number of radar bins available within the height layer is determined (which also includes radar bins for
        which no velocity is available).
        """
        hlayer_centers = np.arange(self.height_limits[0], self.height_limits[1]+1e-3, self.dh)
        if self.h0 > 0:
            hlayer_centers += self.h0
            
        attrs = ('h_layers', 'v_layers', 'vn_layers', 'azi_layers', 'theta_layers')
        for attr in attrs:
            self.__dict__[attr] = {h:[] for h in hlayer_centers}
        self.totalbins_layers_scans = {h: {j:0 for j in self.data} for h in hlayer_centers}
        h_before = None
        for h in hlayer_centers:
            for j in self.data:
                height_range = np.array([np.max([0, h-self.dh/2.-self.antenna_height]), h+self.dh/2.-self.antenna_height])
                ground_range = ft.var1_to_var2(height_range, self.scanangles[j], 'h+theta->gr')  
                ground_range[0] = max([ground_range[0], self.range_limits[0]])
                ground_range[1] = min([ground_range[1], self.range_limits[1]])
                if ground_range[1] <= ground_range[0]:
                    continue #This can occur when both ground_range[0] and ground_range[1] lie outside self.range_limits,
                    #and is caused by the fact that only one of the values gets corrected.
                
                slant_range = ft.var1_to_var2(ground_range, self.scanangles[j], 'gr+theta->sr')
                #Bins are included when at least half of a bin is inside the height range.
                bins_range = np.round(slant_range/self.radial_res[j]-np.array([0, 1])).astype('int')
                bins_range[1] = min([bins_range[1], self.radial_bins[j]])
                if bins_range[0] > bins_range[1]:
                    continue #This can occur when np.round(slant_range/self.radial_res[j]) gives
                    #the same result for both elements in slant_range, such that subtracting 1 of the latter leads to this problem.
                
                data_bins_range = self.data[j][:,bins_range[0]:bins_range[1]+1]
                unmasked = ~np.isnan(data_bins_range)
                #Remove velocities below self.v_min m/s, to reduce contamination by ground clutter.
                unmasked[unmasked] = (np.abs(data_bins_range[unmasked]) >= self.v_min)
                data_unmasked = data_bins_range[unmasked]

                if data_unmasked.shape[0]>0:
                    na, nr = self.data[j].shape
                    data_heights = self.antenna_height+np.tile(ft.var1_to_var2(self.radial_res[j]*np.arange(0.5+bins_range[0], bins_range[1]+1), self.scanangles[j], 'sr+theta->h'), (na, 1))
                    self.h_layers[h].append(data_heights[unmasked])
                    self.v_layers[h].append(data_unmasked)
                    self.vn_layers[h].append(np.full(data_unmasked.shape[0], self.dsg.nyquist_velocities_all_mps[j], dtype='float32'))
                    data_azimuths = np.tile(np.arange(0.5, na, 1), (nr, 1)).T*(2*np.pi/na)
                    self.azi_layers[h].append(data_azimuths[:, bins_range[0]:bins_range[1]+1][unmasked])
                    self.theta_layers[h].append(np.full(data_unmasked.shape[0], self.scanangles[j]*np.pi/180., dtype='float32'))
                    self.totalbins_layers_scans[h][j] += np.size(data_bins_range)
                
            if len(self.h_layers[h]) == 0:
                del self.h_layers[h]
                continue
            
            for attr in attrs:
                self.__dict__[attr][h] = np.concatenate(self.__dict__[attr][h])
            self.h_layers[h] = self.h_layers[h].mean()-self.h0
            
            if h_before and self.h_layers[h]-self.h_layers[h_before] < 0.1:
                n1, n2 = len(self.v_layers[h_before]), len(self.v_layers[h])
                self.h_layers[h] = (n1*self.h_layers[h_before]+n2*self.h_layers[h])/(n1+n2)
                for attr in attrs:
                    if not attr == 'h_layers':
                        self.__dict__[attr][h] = np.concatenate([self.__dict__[attr][h], self.__dict__[attr][h_before]])
                    del self.__dict__[attr][h_before]
            h_before = h
            
    def test_data_availability_height_layers(self, count_filled_sectors = False): 
        """Divide the azimuthal dimension into self.n_sectors sectors, that each span 360/self.n_sectors degrees. 
        Sectors are grouped into pairs of two, with the members being located opposite to each other. This implies that when sector 1 spans
        the azimuthal range of 0 to 45 degrees, that sector 2 spans the range of 180 to 225 degrees. 
        
        A condition on data availibility is now that for at least one of these two sectors within a pair, the radar bins with velocities available
        must at least cover an area of self.min_area_sector. This condition is posed in order to reduce the number of incorrect retrievals
        due to a low amount of data.
        The reason for grouping sectors into these pairs is that for a sine or cosine the function values in opposite sectors of the 
        the azimuthal domain should have only a flipped sign. This implies that the availability of data in both these sectors does not provide much more
        information than when it is available for only one sector. That is the reason that it is considered sufficient when only one sector within a pair
        contains enough data to satisfy the above areal condition.     
        
        There is an issue with the above condition however, and that is that the vertical resolution of data is radar-dependent. This means that for some radars
        there will be multiple scans that cover a particular height range, while for others there might be only one. In the latter case it is much more
        difficult to satisfy the areal condition. For this reason data availability within a certain sector is also considered to be sufficient when for at least
        one scan velocity is available for at least 33% of the total number of radar bins within that sector for that scan.
        
        The conditions on data availability used here is different from and probably less strict than the one used by Iwan Holleman, who divides the azimuthal domain in 8
        sectors of 45 degrees, and requires for all sectors (<- especially this is quite a criterion) that at least 5 data points are available.
        """
        sector_width = 2*np.pi/(self.n_sectors)
        self.filled_sectors = []
        for h in self.h_layers.copy():
            theta_layers_scans = {k: self.theta_layers[h] == self.scanangles[k]*np.pi/180. for k in self.data}
            self.filled_sectors += [0]
            for j in range(self.n_sectors//2):
                azirange_sector1 = sector_width*np.array([j, j+1])
                azirange_sector2 = np.pi + sector_width*np.array([j, j+1])
                
                select1 = (self.azi_layers[h]>=azirange_sector1[0]) & (self.azi_layers[h]<azirange_sector1[1])
                select2 = (self.azi_layers[h]>=azirange_sector2[0]) & (self.azi_layers[h]<azirange_sector2[1])
                
                max_fractional_nbins_sector1 = max([np.count_nonzero(select1 & theta_layers_scans[k]) / (self.totalbins_layers_scans[h][k]/self.n_sectors) for k in self.data if not self.totalbins_layers_scans[h][k] == 0])
                """IMPORTANT: With the current condition on max_fractional_bins_sector (>= 0.33) it seems like the areal condition (>= 7) has no effect anymore,
                since every sector that satisfies the areal condition also seems to satisfy the fractional condition. It should therefore be considered to remove
                the areal condition.
                """
                if max_fractional_nbins_sector1 >= 0.33:
                    self.filled_sectors[-1] += 1
                if not max_fractional_nbins_sector1 >= 0.33 or count_filled_sectors:
                    max_fractional_nbins_sector2 = max([np.count_nonzero(select2 & theta_layers_scans[k]) / (self.totalbins_layers_scans[h][k]/self.n_sectors) for k in self.data if not self.totalbins_layers_scans[h][k] == 0])
                    if max_fractional_nbins_sector2 >= 0.33:
                        self.filled_sectors[-1] += 1
                        
                availability_condition_satisfied = self.filled_sectors[-1] >= self.min_sector_pairs_filled
                if not count_filled_sectors and (availability_condition_satisfied or
                self.min_sector_pairs_filled-self.filled_sectors[-1] > self.n_sectors//2-(j+1)):
                    #In the latter case it is impossible to get self.min_sector_pairs_filled sectors filled, despite the remaining iterations.
                    break
            
            if self.filled_sectors[-1] < self.min_sector_pairs_filled:
                self.filled_sectors.pop()
                del self.h_layers[h]
        self.filled_sectors = np.array(self.filled_sectors)
                        
    def retrieve_velocities_through_fit(self, determine_uncertainty=True):      
        """Fit the formula "Vr = w*sin(theta)+u_0*cos(theta)*sin(phi)+v_0*cos(theta)*cos(phi)" to the self.data, 
        where w = w_0 + W_f, the sum of the vertical velocity and the terminal velocity of hydrometeors.
        """    
        self.V = []; self.w = []; self.sigma = []
        for i, h in enumerate(self.h_layers):
            if not determine_uncertainty and i < len(self.h_layers):
                # Attempt extended dealiasing of the velocity for the next height using the retrieved velocity for the current height
                V_below = self.V0 if i == 0 else self.V[i-1]
                if not V_below is None:
                    V, a = np.linalg.norm(V_below), np.arctan2(V_below[0], V_below[1])
                    V_diff = self.v_layers[h]-V*np.cos(self.azi_layers[h]-a)
                    select = np.abs(V_diff) > self.vn_layers[h]
                    self.v_layers[h][select] += -np.sign(V_diff[select])*2*self.vn_layers[h][select]
                
            A = np.transpose([np.cos(self.theta_layers[h])*np.sin(self.azi_layers[h]),np.cos(self.theta_layers[h])*np.cos(self.azi_layers[h]), np.sin(self.theta_layers[h])])
            #x = [u_0,v_0,w]
            b = self.v_layers[h]
            #Require that -10<=w<=0 m/s, otherwise unrealistic estimates for w might be obtained
            output = lsq_linear(A, b, bounds = ([-np.inf, -np.inf, -10.], [np.inf, np.inf, 0.]))
            params = output.x
            self.V += [params[:2]]
            self.w += [params[2]]
            
            if determine_uncertainty:
                Vr_fit = np.linalg.norm(self.V[i])*np.cos(self.theta_layers[h])*np.cos(self.azi_layers[h]-np.arctan2(self.V[i][0],self.V[i][1]))+self.w[i]*np.sin(self.theta_layers[h])
                self.sigma += [np.sqrt(np.sum(np.power(self.v_layers[h]-Vr_fit,2.))/(self.v_layers[h].shape[0]-3))]
                #Formula (3.21) from Iwan Holleman is used to calculate the standard deviation self.sigma of the radial velocity
            
        self.V, self.w, self.sigma = np.array(self.V), np.array(self.w), np.array(self.sigma)
                
    def remove_outliers_and_repeat_fit(self):
        """Remove outliers, by requiring that the radial velocities deviate by less than self.dv_outliers from the value that follows
        from the determined velocities. After removal of outliers, repeat the fit.
        """
        for i, h in enumerate(self.h_layers):
            Vr_fit = np.linalg.norm(self.V[i])*np.cos(self.theta_layers[h])*np.cos(self.azi_layers[h]-np.arctan2(self.V[i][0],self.V[i][1]))+self.w[i]*np.sin(self.theta_layers[h])
            non_outliers = np.abs(self.v_layers[h]-Vr_fit)<self.dv_outliers
            self.v_layers[h] = self.v_layers[h][non_outliers]
            self.theta_layers[h] = self.theta_layers[h][non_outliers]
            self.azi_layers[h] = self.azi_layers[h][non_outliers]
            
        self.test_data_availability_height_layers(count_filled_sectors = True)
        self.retrieve_velocities_through_fit()
  
    
  
    def get_dir_and_filename(self):
        radar_dataset = self.dsg.get_radar_dataset(no_special_char=True)
        subdataset = self.dsg.get_subdataset(product='v')

        directory = self.gui.derivedproducts_dir+'/VWP/'+radar_dataset+'/'+subdataset+'/'+self.crd.date
        filename = directory+'/'+self.crd.date+self.crd.time+'.h5'
        return directory, filename 

    def get_data_group_name(self):
        return f'range_lim={self.range_limits}, height_lim={self.height_limits}, v_min={self.v_min}, h0={self.h0}, V0={self.V0}'
    
    def check_if_dataset_at_disk(self):
        _, filename = self.get_dir_and_filename()
        if not os.path.exists(filename):
            return False
        
        with h5py.File(filename, 'r') as f:
            version, total_volume_files_size = f.attrs['version'], f.attrs['total_volume_files_size']
            if version != self.file_content_version or total_volume_files_size != self.dsg.total_files_size:
                return False
            
            group_name = self.get_data_group_name()
            if not group_name in f:
                return False
            
            group = f[group_name]
            for name in group:
                self.__dict__[name] = group[name][:]  
            self.volume_starttime, self.volume_endtime = f.attrs['producttime'].split('-')
        return True         
        
    def write_data_to_file(self):
        directory, filename = self.get_dir_and_filename()
        os.makedirs(directory, exist_ok=True)
        try:
            with h5py.File(filename, 'r') as f:
                version, total_volume_files_size = f.attrs['version'], f.attrs['total_volume_files_size']
                new_file = version != self.file_content_version or total_volume_files_size != self.dsg.total_files_size
                action = 'w' if new_file else 'a'
        except Exception: 
            action = 'w'
        
        with h5py.File(filename, action) as f:
            f.attrs['version'] = self.file_content_version
            f.attrs['total_volume_files_size'] = self.dsg.total_files_size
            f.attrs['producttime'] = self.volume_starttime+'-'+self.volume_endtime
            
            group_name = self.get_data_group_name()
            group = f.create_group(group_name)
            for name in ('h_layers', 'V', 'w', 'sigma', 'filled_sectors'):
                data = self.__dict__[name]
                dataset = group.create_dataset(name, data.shape, dtype=data.dtype, compression='gzip', track_times=False)
                dataset[...] = data