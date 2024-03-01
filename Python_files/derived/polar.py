# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import nlr_functions as ft
import nlr_globalvars as gv

import numpy as np
import time as pytime



class Polar():
    def __init__(self, dp_class, parent = None):
        self.dp = dp_class
        self.dsg = self.dp.dsg

        self.radial_bins_all_before = {}
        self.azimuthal_bins_all_before = {}

        self.productbins_unique_all_flat = {}
        self.radarbins_unique_all_flat = {}
        
        self.data_specs = None
        
        self.product_heights = {}
        self.product_data = {}
        self.product_data_specs = {}
        


    def get_product_dimensions(self):
        # The number of azimuthal bins can vary among scans. The number used for the derived products is the maximum for all scans. So
        self.product_azimuthal_bins = max(max(arr.shape[0] for arr in self.data_all[j]) for j in self.scans)
        self.product_azimuthal_res = 360/self.product_azimuthal_bins
        
        #If the bottom scan is not removed, then still use the resolution that is equal to the coarsest of the other scans.
        scans_radial_res = [self.radial_res_all[j] for j in self.scans if not (not self.dp.bottom_scan_removed and j==self.scans[0] and len(np.unique(self.scans))>1)]
        min_data_radial_res = min(scans_radial_res)
        max_data_radial_res = max(scans_radial_res)
        self.product_radial_res = min_data_radial_res if min_data_radial_res>=0.25 else max_data_radial_res
        
        s = self.scans[0] 
        max_ground_range = ft.var1_to_var2(self.radius_offsets_all[s]+self.radial_range_all[s], self.scanangles_all[s], 'sr+theta->gr')
        self.product_radial_bins = int(np.ceil(max_ground_range/self.product_radial_res))
        s = self.scans_all[0]
        max_ground_range_all = ft.var1_to_var2(self.radius_offsets_all[s]+self.radial_range_all[s], self.scanangles_all[s], 'sr+theta->gr')
        self.product_radial_bins_all = int(np.ceil(max_ground_range_all/self.product_radial_res))
        """self.product_radial_bins_all is used when calculating quantities that can be used in multiple functions, as stated above. The radial resolution and
        azimuthal specifications are the same for each plain product.
        """
                   
    def get_product_shape(self):
        self.product_shape = (self.product_azimuthal_bins, self.product_radial_bins)
        return self.product_shape
    
    def get_product_slice(self):
        self.product_slice = np.s_[:, :self.product_radial_bins]
        return self.product_slice

    
    def get_int_dtype(self):
        length = self.product_azimuthal_bins*self.product_radial_bins_all
        return 'uint32' if length < 2**32-1 else 'uint64'
                
    def assign_radarbins_to_productbins(self):
        """Determines for each bin in the reflectivity data onto which bin in the polar grid of the derived product it is mapped, by assigning the bin 
        in the derived product array onto which a reflectivity data bin is mapped to an array with the same dimensions as the reflectivity data. 
        The fact that the algorithm for calculating the derived products use flattened arrays is taken into account by adding an appropriate number to 
        each azimuthal row.
        """        
        for j in self.scans_all:
            #These attributes are the same for all products
            slantranges = self.radius_offsets_all[j]+self.radial_res_all[j]*(0.5+np.arange(self.radial_bins_all[j], dtype='float32'))
            groundranges, _ = ft.var1_to_var2(slantranges, self.scanangles_all[j])
            productbins = np.floor(groundranges/self.product_radial_res).astype(self.get_int_dtype())
            self.productbins_unique_all_flat[j] = []
            for i in range(len(self.dsg.scannumbers_all['z'][j])):
                add_array = np.array([[self.product_radial_bins_all*j] for j in range(self.data_all[j][i].shape[0])])
                self.productbins_unique_all_flat[j] += [(productbins[np.newaxis, :]+add_array).ravel()]

    def assign_productbins_to_radarbins(self):
        """Determines the inverse map of that in assign_radarbins_to_productbins.
        """        
        self.groundranges = self.product_radial_res*(0.5+np.arange(self.product_radial_bins_all, dtype='float32'))
        for j in self.scans_all:           
            slantranges = ft.var1_to_var2(self.groundranges, self.scanangles_all[j], 'gr+theta->sr')
            radarbins = np.floor((slantranges-self.radius_offsets_all[j])/self.radial_res_all[j]).astype(self.get_int_dtype())
            radarbins = radarbins[radarbins<self.radial_bins_all[j]]               
            self.radarbins_unique_all_flat[j] = []
            for i in range(len(self.dsg.scannumbers_all['z'][j])):
                add_array = np.array([[self.radial_bins_all[j]*i] for i in range(self.data_all[j][i].shape[0])])
                self.radarbins_unique_all_flat[j] += [((radarbins[np.newaxis, :]+add_array).ravel())]
        
    def get_heights_3D(self):
        """Scanning heights for each scan in a 2D array with the shape of self.product_array, where for each bin in self.product_array the height at 
        which the radar scans is determined.
        """
        n = sum([len(self.data_all[j]) for j in self.scans_all])
        self.heights_3D_all = np.zeros((n, self.product_azimuthal_bins, self.product_radial_bins_all), dtype = 'float32')
        for j in self.scans_all:
            select = self.groundranges < ft.var1_to_var2(self.radius_offsets_all[j]+self.radial_range_all[j],
                                                        self.scanangles_all[j], 'sr+theta->gr')
            heights = ft.var1_to_var2(self.groundranges[select], self.scanangles_all[j],'gr+theta->h')
            for i in range(len(self.data_all[j])):
                index = self.dp.index_all(j, i)
                self.heights_3D_all[index, :, select] = heights[:, np.newaxis]

    def get_parameters_data_mapping(self):
        azimuthal_bins_all = {j: [arr.shape[0] for arr in self.data_all[j]] for j in self.scans}
        maxheights_all = np.array([ft.var1_to_var2(self.radial_range_all[j],self.scanangles_all[j],'sr+theta->h') for j in self.scans_all])
        # Also check whether the maximum heights of the scans have changed, particularly for DWD radars, since for these the
        # scanangles are a bit variable. The mapping is re-performed when any maximum height changes by more than 100 m
        # Finally, check whether the number of duplicate scans has changed
        perform_remapping = not (self.radial_bins_all_before==self.radial_bins_all and\
                                 self.azimuthal_bins_all_before == azimuthal_bins_all and\
                                 len(self.maxheights_all_before) == len(maxheights_all) and\
                                 np.abs(self.maxheights_all_before-maxheights_all).max()<=0.1 and\
                                 self.scannumbers_all_before == self.dsg.scannumbers_all[self.i_p])
        if perform_remapping:         
            #Is only done when the structure of the radar volume has changed, or when the function has not yet been called before for self.product,
            #for the given radar volume structure.
            self.assign_radarbins_to_productbins()
            self.assign_productbins_to_radarbins()
            self.get_heights_3D()
            
            # Make sure that product heights are recalculated for every product, by emptying self.product_heights
            self.product_heights = {}
            
            self.radial_bins_all_before = self.radial_bins_all.copy()
            self.azimuthal_bins_all_before = azimuthal_bins_all.copy()
            self.maxheights_all_before = maxheights_all.copy()
            self.scannumbers_all_before = self.dsg.scannumbers_all[self.i_p].copy()
            


    def get_data_specs(self):
        return self.dp.get_import_data_specs()
        
    def calculate_Zmax_and_Zavg_3D(self):
        t = pytime.time()
        """Calculates per scan the maximum and average reflectivity on the grid on which self.product_data is defined. 
        
        If the radial size (resolution) of a radar bin in self.data_all[scan] is greater than that of self.product_data, then this function simply
        determines which bin in self.data_all[scan] is closest to a particular bin in self.product_data, and assigns its reflectivity to it.

        If the radial size (resolution) of a radar bin in self.data_all[scan] is less than that of self.product_data, then multiple bins
        in self.data_all[scan] can be mapped onto a particular bin in self.product_data, and this function determines the maximum reflectivity of
        the bins in self.data_all[scan] that get mapped onto that particular bin in self.product_data. 
        
        For each scan, the function updates all bins in self.Zmax_3D_all[scan] for which the new reflectivities are higher than the old ones. Because 
        multiple bins in the reflectivity data can be mapped onto the same bin in self.Zmax_3D_all[scan], this process must be repeated for all bins
        in the reflectivity data. This is done by masking the bins in the reflectivity data that have already been handled, and then repeating the 
        process for the non-masked part of the remaining array. The while loop that handles this stops when there are no remaining non-masked bins.
        Finally, the flat array self.Zmax_3D_all[scan] is reshaped such that it has the correct dimensions.
        
        For self.Zavg_3D_all[scan] the calculation is done in a way comparable to that of Zmax, but with as difference that the algorithm now sums all reflectivities 
        that are mapped onto a particular bin in self.Zavg_3D_all[scan], and finally divides them by the number of reflectivities that is mapped onto 
        the particular bin, to arrive at the average reflectivity.
        The average reflectivity is here given by Zavg = 10*log10(Zavg_linear), i.e., the average linear reflectivity is calculcated, and from this the 
        logarithm is taken.
        """
        if self.get_data_specs()==self.data_specs:
            return
        
        length = self.product_azimuthal_bins*self.product_radial_bins_all
        
        Z_empty = -30.       
        n = sum([len(self.data_all[j]) for j in self.scans_all])
        self.Zmax_3D_all = np.full((n, length), Z_empty, dtype='float32')
        self.Zavg_3D_all = np.full((n, length), Z_empty, dtype='float32')
        indices_occurrences = np.ones(length, dtype=self.get_int_dtype())
                                                            
        for j in self.scans_all:
            for i in range(len(self.data_all[j])):
                index = self.dp.index_all(j, i)
                flattened = self.data_all[j][i].ravel()
                n_azi = self.data_all[j][i].shape[0]
                # The number of azimuthal bins for the scan might differ from self.product_azimuthal_bins. If that's the case only
                # the first n_azi bins of the mapped Z arrays will be filled by the mapping procedure. At the end of this loop-iteration
                # the results of the mapping procedure are then repeated by a factor self.product_azimuthal_bins/n_azi.
                
                if self.product_radial_res<=self.radial_res_all[j]:
                    indices = self.radarbins_unique_all_flat[j][i]
                    radial_bins = int(len(indices)/n_azi)
                    data_array = self.Zmax_3D_all[index].reshape((self.product_azimuthal_bins,self.product_radial_bins_all))
                    data_array[:n_azi,:radial_bins] = np.reshape(flattened[indices], (n_azi, radial_bins))
                    data_array[np.isnan(data_array)] = Z_empty
                    self.Zmax_3D_all[index] = self.Zavg_3D_all[index] = data_array.reshape(length)
                else:
                    retain = ~np.isnan(flattened)
                    flattened = flattened[retain]
                    indices = self.productbins_unique_all_flat[j][i][retain]
                    #The linear reflectivity is only calculated for a bin in self.Zavg_3D_all[j] if more than one bin in self.data_all[j][i] is mapped onto that bin
                    #in self.Zavg_3D_all[j], because calculating the logartihm to arrive back at the logarithmic reflectivity is an expensive operation.
                    
                    """indices_occurrences gives the number of occurrences for each index in indices. When the while loop is finished and each bin in 
                    self.Zavg_3D_all[j] contains the sum of all reflectivities that are mapped onto that bin, then self.Zavg_3D_all[j] is divided by
                    indices_occurrences, to get the average reflectivity.
                    """
                    indices_occurrences[:] = 1
                    indices_bincount = np.bincount(indices)
                    indices_occurrences[:len(indices_bincount)] = indices_bincount
                    
                    sort_indices = np.argsort(indices)
                    indices = indices[sort_indices]
                    flattened = flattened[sort_indices]
                    
                    indices_unique_orig, flattened_indices_orig, counts = np.unique(indices, return_index=True, return_counts=True)
                    count_max = counts.max()
    
                    k = 0
                    while k < count_max:
                        if k == 0:
                            self.Zavg_3D_all[index, indices_unique_orig] = self.Zmax_3D_all[index, indices_unique_orig] =\
                                flattened[flattened_indices_orig]
                        else:   
                            select = counts > k
                            indices_unique = indices_unique_orig[select]
                            flattened_indices = flattened_indices_orig[select] + k                                

                            if k == 1:
                                # Perform this operation only once, because it is quite expensive
                                self.Zavg_3D_all[index, indices_unique] = np.power(10., 0.1*self.Zavg_3D_all[index, indices_unique])
                                
                            self.Zavg_3D_all[index, indices_unique] += np.power(10., 0.1*flattened[flattened_indices])
                            update = self.Zmax_3D_all[index, indices_unique]<flattened[flattened_indices]
                            self.Zmax_3D_all[index, indices_unique[update]] = flattened[flattened_indices[update]]
                        k += 1
                        
                    Zavg_islinear = indices_occurrences > 1
                    self.Zavg_3D_all[index, Zavg_islinear] = 10.*np.log10(self.Zavg_3D_all[index, Zavg_islinear]/indices_occurrences[Zavg_islinear])
                    
                if n_azi != self.product_azimuthal_bins:
                    # Repeat the mapping results as described at the start of the loop-iteration.
                    n_repeat = int(self.product_azimuthal_bins/n_azi)
                    for Z in (self.Zmax_3D_all, self.Zavg_3D_all):
                        data = Z[index, :int(length/n_repeat)].reshape((n_azi, self.product_radial_bins_all))
                        Z[index, :] = np.repeat(data, n_repeat, axis=0).reshape(length)
                    
        self.Zmax_3D_all.shape = (n, self.product_azimuthal_bins, self.product_radial_bins_all)                                          
        self.Zavg_3D_all.shape = (n, self.product_azimuthal_bins, self.product_radial_bins_all)                                          
                    
        self.data_specs = self.get_data_specs()
        print(pytime.time()- t, 'Zmax_and_Zavg_3D_all')
        

    def get_product_heights(self):
        s = self.dp.get_indices_scans()
        self.heights_3D = self.heights_3D_all[(s,)+self.product_slice]
        
        key = str(s)
        if any([product in self.dp.products_requiring_heightsorted_data for product in self.dp.products_per_indices[key]]):
            n = len(self.heights_3D)+1
            self.hdiffs = np.empty((n,)+self.product_shape, dtype = 'float32')
            self.hdiffs[1:-1] = 1e3*(self.heights_3D[1:]-self.heights_3D[:-1])
            
            # groundranges = self.groundranges[:self.product_radial_bins]
            # height_bottom = ft.var1_to_var2(groundranges, self.scanangles_all[s0]-0.5*gv.radar_beamwidths[self.radar], 'gr+theta->h')
            # height_top = ft.var1_to_var2(groundranges, self.scanangles_all[self.scans[-1]]+0.5*gv.radar_beamwidths[self.radar], 'gr+theta->h')
            self.hdiffs[0] = 1e3*self.heights_3D[0]
            # self.hdiffs[-1] = 1e3*(height_top-self.heights_3D[-1])
        
            self.product_heights[key] = {j:self.__dict__[j] for j in ('heights_3D', 'hdiffs')}
        else:
            self.product_heights[key] = {'heights_3D':self.heights_3D}
        
    def sort_data_and_get_hdiffs_if_necessary(self):
        s = self.dp.get_indices_scans()
        key = str(s)
        if key in self.product_heights:
            for j in self.product_heights[key]:
                self.__dict__[j] = self.product_heights[key][j]
        else:
            self.get_product_heights()
            
        if key in self.product_data_specs and self.product_data_specs[key] == self.get_data_specs():
            for j in self.product_data[key]:
                self.__dict__[j] = self.product_data[key][j]
            return
            
        self.Zmax_3D = self.Zmax_3D_all[(s,)+self.product_slice]
        self.Zavg_3D = self.Zavg_3D_all[(s,)+self.product_slice]
        
        self.product_data[key] = {j:self.__dict__[j] for j in ('Zmax_3D', 'Zavg_3D')}
        self.product_data_specs[key] = self.get_data_specs()