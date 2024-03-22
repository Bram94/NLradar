# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import nlr_functions as ft
import nlr_globalvars as gv

import numpy as np
import time as pytime



class Cartesian():
    def __init__(self, dp_class, parent = None):
        self.dp = dp_class
        self.dsg = self.dp.dsg
        self.gui = self.dp.gui
        
        self.product_res_before = None
        self.radial_bins_all_before = {}
        self.azimuthal_bins_all_before = {}

        self.productbins_unique_all_flat = {}
        self.radarbins_unique_all_flat = {}
        self.translation_vectors = {}
        
        self.data_specs = None
        
        self.sorted_heights = {}
        self.sorted_data = {}
        self.sorted_data_specs = {}
        
        
        
    def get_product_dimensions(self):
        if hasattr(self, 'product_res'):
            self.product_res_before = self.product_res
        self.product_res = self.gui.cartesian_product_res
        
        s = self.scans[0]
        max_ground_range = ft.var1_to_var2(self.radius_offsets_all[s]+self.radial_range_all[s], self.scanangles_all[s], 'sr+theta->gr')
        max_ground_range = min(self.gui.cartesian_product_maxrange, max_ground_range)
        self.product_xy_bins = 2*int(np.ceil(max_ground_range/self.product_res))
        s = self.scans_all[0]
        max_ground_range_all = ft.var1_to_var2(self.radius_offsets_all[s]+self.radial_range_all[s], self.scanangles_all[s], 'sr+theta->gr')
        max_ground_range_all = min(self.gui.cartesian_product_maxrange, max_ground_range_all)
        self.product_xy_bins_all = 2*int(np.ceil(max_ground_range_all/self.product_res))
        """self.product_xy_bins_all is used when calculating quantities that can be used in multiple functions, as stated above.
        """

    def get_product_shape(self):
        self.product_shape = (self.product_xy_bins, self.product_xy_bins)
        return self.product_shape
    
    def get_product_slice(self):
        start = (self.product_xy_bins_all-self.product_xy_bins)//2
        end = self.product_xy_bins_all-start
        self.product_slice = np.s_[start:end, start:end]
        return self.product_slice
                   

    def get_int_dtype(self):
        length = self.product_xy_bins_all**2
        return 'uint32' if length < 2**32-1 else 'uint64'
        
    def assign_radarbins_to_productbins(self):
        """Determines for each bin in the reflectivity data onto which bin in the polar grid of the derived product it is mapped, by assigning the bin 
        in the derived product array onto which a reflectivity data bin is mapped to an array with the same dimensions as the reflectivity data. 
        The fact that the algorithm for calculating the derived products use flattened arrays is taken into account by adding an appropriate number to 
        each azimuthal row.
        """        
        add = 0.5*self.product_xy_bins_all
        max_value = np.iinfo(self.get_int_dtype()).max
        for j in self.scans_all:
            #These attributes are the same for all products
            slantranges = self.radius_offsets_all[j]+self.radial_res_all[j]*(0.5+np.arange(self.radial_bins_all[j], dtype='float32'))
            _groundranges, _ = ft.var1_to_var2(slantranges,self.scanangles_all[j])
            
            self.productbins_unique_all_flat[j] = []
            for i in range(len(self.dsg.scannumbers_all['z'][j])):
                n_azimuths = self.data_all[j][i].shape[0]
                azimuths = (0.5+np.arange(n_azimuths,dtype='float32'))*2*np.pi/n_azimuths
                groundranges, azimuths = np.meshgrid(_groundranges, azimuths, copy=False)
                x, y = groundranges*np.sin(azimuths), groundranges*np.cos(azimuths)
                
                x_bins = (add+np.floor((x+self.translation_vectors[j][i][0])/self.product_res)).astype(self.get_int_dtype())
                y_bins = (add-np.ceil((y+self.translation_vectors[j][i][1])/self.product_res)).astype(self.get_int_dtype())
                outside_domain = (x_bins < 0) | (x_bins >= self.product_xy_bins_all) | (y_bins < 0) | (y_bins >= self.product_xy_bins_all)
                self.productbins_unique_all_flat[j] += [(y_bins*self.product_xy_bins_all+x_bins).ravel()]
                self.productbins_unique_all_flat[j][-1][outside_domain.ravel()] = max_value

    def get_product_coords_and_groundranges(self):
        coords = self.product_res*(0.5+np.arange(-0.5*self.product_xy_bins_all, 0.5*self.product_xy_bins_all, dtype='float32'))
        self.x, self.y = np.meshgrid(coords, coords[::-1], copy=False)
        self.groundranges = np.sqrt(self.x**2+self.y**2)
        
    def assign_productbins_to_radarbins(self):
        """Determines the inverse map of that in assign_radarbins_to_productbins.
        """        
        max_value = np.iinfo(self.get_int_dtype()).max
        for j in self.scans_all:
            self.radarbins_unique_all_flat[j] = []
            for i in range(len(self.dsg.scannumbers_all['z'][j])):
                xi, yi = self.x-self.translation_vectors[j][i][0], self.y-self.translation_vectors[j][i][1]
                groundranges = np.sqrt(xi**2+yi**2)
                select = groundranges < ft.var1_to_var2(self.radius_offsets_all[j]+self.radial_range_all[j],
                                                        self.scanangles_all[j], 'sr+theta->gr')
                groundranges = groundranges[select]
                azimuths = np.mod(np.rad2deg(np.arctan2(xi[select], yi[select])), 360)
                
                slantranges = ft.var1_to_var2(groundranges, self.scanangles_all[j], 'gr+theta->sr')
                azimuthal_bins = np.floor(azimuths*len(self.data_all[j][i])/360).astype(self.get_int_dtype())
                radial_bins = np.floor((slantranges-self.radius_offsets_all[j])/self.radial_res_all[j]).astype(self.get_int_dtype())
                
                self.radarbins_unique_all_flat[j] += [np.full(self.x.size, max_value, dtype = self.get_int_dtype())]
                self.radarbins_unique_all_flat[j][-1][select.ravel()] =\
                    azimuthal_bins*self.radial_bins_all[j]+radial_bins
        
    def get_heights_3D(self):
        """Scanning heights for each scan in a 2D array with the shape of self.product_array, where for each bin in self.product_array the height at 
        which the radar scans is determined.
        """
        n = sum([len(self.data_all[j]) for j in self.scans_all])
        max_value = np.iinfo(self.get_int_dtype()).max
        self.heights_3D_all = np.zeros((n, self.product_xy_bins_all**2), dtype='float32')
        for j in self.scans_all:
            slantranges = self.radius_offsets_all[j]+self.radial_res_all[j]*(0.5+np.arange(self.radial_bins_all[j], dtype='float32'))
            
            for i in range(len(self.dsg.scannumbers_all['z'][j])):
                index = self.dp.index_all(j, i)
                select = self.radarbins_unique_all_flat[j][i] != max_value
                heights = np.tile(ft.var1_to_var2(slantranges, self.scanangles_all[j], 'sr+theta->h'), (self.data_all[j][i].shape[0], 1))
                self.heights_3D_all[index, select] = heights.ravel()[self.radarbins_unique_all_flat[j][i][select]]
        
        self.heights_3D_all.shape = (n, self.product_xy_bins_all, self.product_xy_bins_all)
    
    def get_avg_scantime_and_translation_vectors(self):
        scantimes_all = ft.dict_sublists_to_list(self.scantimes_all)
        self.avg_scantime = ft.get_avg_scantime(scantimes_all)
        
        SM = self.gui.stormmotion
        SM = SM[1]*np.array([np.sin(np.deg2rad(SM[0])), np.cos(np.deg2rad(SM[0]))])
        # SM gives the direction from which the storm is coming, not to which it is going!
        for j in self.scans_all:
            self.translation_vectors[j] = []
            for i in range(len(self.dsg.scannumbers_all['z'][j])):
                time_offset = ft.scantimediff_s(self.avg_scantime, self.scantimes_all[j][i])
                self.translation_vectors[j] += [1e-3*time_offset*SM] # Convert m to km

    def get_parameters_data_mapping(self):
        self.get_avg_scantime_and_translation_vectors()
        
        maxheights_all = np.array([ft.var1_to_var2(self.radial_range_all[j],self.scanangles_all[j],'sr+theta->h') for j in self.scans_all])
        translation_vectors_all = np.array(ft.dict_sublists_to_list(self.translation_vectors))
        azimuthal_bins_all = {j: [arr.shape[0] for arr in self.data_all[j]] for j in self.scans}
        # Check whether the maximum heights of the scans have changed, particularly for DWD radars, since for these the
        # scanangles are a bit variable. The mapping is re-performed when any maximum height changes by more than 100 m
        # Mappings are also recalculated when any translation vector has changed by more than 100 m
        perform_remapping = not (self.product_res_before == self.product_res and\
                                 self.radial_bins_all_before==self.radial_bins_all and\
                                 self.azimuthal_bins_all_before == azimuthal_bins_all and\
                                 len(self.maxheights_all_before) == len(maxheights_all) and\
                                 np.abs(self.maxheights_all_before-maxheights_all).max()<=0.1 and\
                                 len(self.translation_vectors_all_before) == len(translation_vectors_all) and\
                                 np.abs(self.translation_vectors_all_before-translation_vectors_all).max()<=0.1)
        if perform_remapping:
            #Is only done when the structure of the radar volume has changed, or when the function has not yet been called before for self.product,
            #for the given radar volume structure.
            # t = pytime.time()
            self.assign_radarbins_to_productbins()
            # print(pytime.time() - t,'t1')
            self.get_product_coords_and_groundranges()
            # print(pytime.time() - t,'t2')
            self.assign_productbins_to_radarbins()
            # print(pytime.time() - t,'t3')
            self.get_heights_3D()
            # print(pytime.time() - t,'t4')
            
            # Make sure that sorted heights are recalculated for every product, by emptying self.sorted_heights
            self.sorted_heights = {}
            
            self.radial_bins_all_before = self.radial_bins_all.copy()
            self.azimuthal_bins_all_before = azimuthal_bins_all.copy()
            self.maxheights_all_before = maxheights_all.copy()
            self.translation_vectors_all_before = translation_vectors_all.copy()


    def get_data_specs(self):
        return self.dp.get_import_data_specs()+str(self.translation_vectors)+str(self.product_res)
        
    def calculate_Zmax_and_Zavg_3D(self):
        t = pytime.time()
        """Calculates per scan the maximum and average reflectivity on the grid on which self.product_data is defined. 
        See the same function in polar.py for more information, here only the difference with that implementation are described:
        the only difference is that we cannot know beforehand whether
        """
        if self.get_data_specs()==self.data_specs:
            return
        
        length = self.product_xy_bins_all**2
        max_value = np.iinfo(self.get_int_dtype()).max
        
        Z_empty = -30.       
        n = sum([len(self.data_all[j]) for j in self.scans_all])
        self.Zavg_3D_all = np.full((n, length), Z_empty, dtype='float32')        
        indices_occurrences = np.ones(length, dtype=self.get_int_dtype())
                                                            
        for j in self.scans_all:
            for i in range(0, len(self.data_all[j])):
                index = self.dp.index_all(j, i)
                flattened = self.data_all[j][i].ravel()
                
                indices = self.productbins_unique_all_flat[j][i]
                retain = (indices != max_value) & ~np.isnan(flattened)
                flattened = flattened[retain]
                indices = indices[retain]
                #The linear reflectivity is only calculated for a bin in self.Zavg_3D_all[j] if more than one bin in self.data_all[j][i] is mapped onto that bin
                #in self.Zavg_3D_all[j], because calculating the logartihm to arrive back at the logarithmic reflectivity is an expensive operation.
                
                kmax = max(1, int(np.ceil(40*self.product_res**2)))
                """indices_occurrences gives the number of occurrences for each index in indices. When the while loop is finished and each bin in 
                self.Zavg_3D_all[j] contains the sum of all reflectivities that are mapped onto that bin, then self.Zavg_3D_all[j] is divided by
                indices_occurrences, to get the average reflectivity.
                """
                indices_occurrences[:] = 1
                indices_bincount = np.minimum(np.bincount(indices), kmax)
                indices_occurrences[:len(indices_bincount)] = indices_bincount
                
                sort_indices = np.argsort(indices)
                indices = indices[sort_indices]
                flattened = flattened[sort_indices]
                
                indices_unique_orig, flattened_indices_orig, counts = np.unique(indices, return_index=True, return_counts=True)
                count_max = counts.max()

                k = 0
                while k < min([kmax, count_max]):
                    if k == 0:
                        self.Zavg_3D_all[index, indices_unique_orig] = flattened[flattened_indices_orig]
                    else:                                   
                        select = counts > k
                        indices_unique = indices_unique_orig[select]
                        flattened_indices = flattened_indices_orig[select] + k
                        
                        if k == 1:
                            # Perform this operation only once, because it is quite expensive
                            self.Zavg_3D_all[index, indices_unique] = np.power(10., 0.1*self.Zavg_3D_all[index, indices_unique])
                            
                        self.Zavg_3D_all[index, indices_unique] += np.power(10., 0.1*flattened[flattened_indices])
                    k += 1
                
                Zavg_islinear = indices_occurrences > 1
                self.Zavg_3D_all[index, Zavg_islinear] = 10.*np.log10(self.Zavg_3D_all[index, Zavg_islinear]/indices_occurrences[Zavg_islinear])

                flattened = self.data_all[j][i].ravel()
                Zavg_empty = (self.Zavg_3D_all[index] == Z_empty) & (self.radarbins_unique_all_flat[j][i] != max_value)
                self.Zavg_3D_all[index, Zavg_empty] = flattened[self.radarbins_unique_all_flat[j][i][Zavg_empty]]
                
        self.Zavg_3D_all[np.isnan(self.Zavg_3D_all)] = Z_empty  
        self.Zavg_3D_all.shape = (n, self.product_xy_bins_all, self.product_xy_bins_all)
        self.Zmax_3D_all = self.Zavg_3D_all

        self.data_specs = self.get_data_specs()
        print(pytime.time()- t, 'Zmax_and_Zavg_3D_all')
        
        
    def get_separate_scangroups(self):
        """Sometimes the radar range decreases with increasing scanangle, in which case there are multiple blocks of scans
        with varying range. This function divides the product scans into these blocks, and determines for each block which
        bins of the product grid it covers (up to when a new block is reached with shorter range, implying that more scans are
        involved). This information is used in determining what part of the 3D arrays needs to be sorted, with the goal of
        avoiding actions on parts of arrays that are actually empty.
        """
        self.scangroups = {}
        add_to_range = max([max([np.linalg.norm(i) for i in j]) for j in self.translation_vectors.values()])
        for i in range(len(self.scans)-1):
            scan1 = self.scans[i]
            scan2 = self.scans[i+1]
            slantrange1 = self.radius_offsets_all[scan1]+self.radial_range_all[scan1]
            slantrange2 = self.radius_offsets_all[scan2]+self.radial_range_all[scan2]
            if slantrange2 < slantrange1:
                if len(self.scangroups) == 0:
                    self.scangroups[scan1] = np.ones((self.product_xy_bins, self.product_xy_bins), dtype=bool)
                else:
                    self.scangroups[scan1] = ~np.logical_or.reduce(list(self.scangroups.values()))
                groundrange1,_ = ft.var1_to_var2(slantrange1, self.scanangles_all[scan1])
                groundrange1 += add_to_range
                max_bin = int(np.ceil(groundrange1/self.product_res))
                if max_bin < self.product_xy_bins//2:
                    s = np.s_[self.product_xy_bins//2-max_bin:self.product_xy_bins//2+max_bin]
                    self.scangroups[scan1][s] = 0
        self.scangroups[self.scans[-1]] = ~np.logical_or.reduce(list(self.scangroups.values()))
        if list(self.scangroups)[0] == 1:
            # In this case the first scan group consists of only one scan, in which case no sorting is needed
            del self.scangroups[1]
              
    def get_sorted_heights(self):
        s = self.dp.get_indices_scans()
        # Keep in mind that sorting will alter self.heights_3D, so in that case self.heights_3D should not refer to self.heights_3D_all.
        # This would be the case when s = np.s_[:] and no copy is made. s = np.s_[:] however only occurs for products not requiring height-sorting.
        self.heights_3D = self.heights_3D_all[(s,)+self.product_slice]
        
        key = str(s)
        if any([product in self.dp.products_requiring_heightsorted_data for product in self.dp.products_per_indices[key]]):
            self.get_separate_scangroups()
            self.sort_indices = {}
            for j in self.scangroups:
                i_max = self.scans.index(j)+1
                self.sort_indices[j] = np.argsort(self.heights_3D[:i_max, self.scangroups[j]], axis=0)
                self.heights_3D[:i_max, self.scangroups[j]] = np.take_along_axis(self.heights_3D[:i_max, self.scangroups[j]], self.sort_indices[j], axis=0)
        
            n = len(self.heights_3D)+1
            self.hdiffs = np.empty((n,)+self.product_shape, dtype = 'float32')
            self.hdiffs[1:-1] = 1e3*(self.heights_3D[1:]-self.heights_3D[:-1])
            
            # groundranges = self.groundranges[self.product_slice]
            # height_bottom = ft.var1_to_var2(groundranges, self.scanangles_all[self.scans[0]]-0.5*gv.radar_beamwidths[self.radar], 'gr+theta->h')
            # height_top = ft.var1_to_var2(groundranges, self.scanangles_all[self.scans[-1]]+0.5*gv.radar_beamwidths[self.radar], 'gr+theta->h')
            self.hdiffs[0] = 1e3*self.heights_3D[0]
            # self.hdiffs[-1] = 1e3*(height_top-self.heights_3D[-1])
            
            self.sorted_heights[key] = {j:self.__dict__[j] for j in ('scangroups', 'sort_indices', 'heights_3D', 'hdiffs')}
        else:
            self.sorted_heights[key] = {'heights_3D':self.heights_3D}
         
    def sort_data_and_get_hdiffs_if_necessary(self):
        s = self.dp.get_indices_scans()
        key = str(s)
        # self.sorted_heights gets emptied when the data mapping changes, which is also the only moment at which the sorted heights
        # need to be updated. So there is no need for another check than the one below to determine whether updating is needed
        if key in self.sorted_heights:
            for j in self.sorted_heights[key]:
                self.__dict__[j] = self.sorted_heights[key][j]
        else:
            self.get_sorted_heights()
            
        if key in self.sorted_data_specs and self.sorted_data_specs[key] == self.get_data_specs():
            for j in self.sorted_data[key]:
                self.__dict__[j] = self.sorted_data[key][j]
            return
        
        self.Zavg_3D = self.Zavg_3D_all[(s,)+self.product_slice]
        
        if any([product in self.dp.products_requiring_heightsorted_data for product in self.dp.products_per_indices[key]]):
            for j in self.scangroups:
                i_max = self.scans.index(j)+1
                self.Zavg_3D[:i_max, self.scangroups[j]] = np.take_along_axis(self.Zavg_3D[:i_max, self.scangroups[j]], self.sort_indices[j], axis=0)
        self.Zmax_3D = self.Zavg_3D
                    
        self.sorted_data[key] = {j:self.__dict__[j] for j in ('Zmax_3D', 'Zavg_3D')}
        self.sorted_data_specs[key] = self.get_data_specs()