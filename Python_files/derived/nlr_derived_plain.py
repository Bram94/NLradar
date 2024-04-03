# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import nlr_background as bg
import nlr_functions as ft
import nlr_globalvars as gv
from derived.polar import Polar
from derived.cartesian import Cartesian

import copy
import numpy as np
import time as pytime
import h5py
import os
import shutil
# import matplotlib.pyplot as plt



class DerivedPlain():
    def __init__(self, dsg_class, parent = None):
        self.dsg = dsg_class
        self.gui = self.dsg.gui
        self.crd = self.dsg.crd
        self.pb = self.gui.pb
        
        self.import_data_specs = None
        
        self.filename_version = 2
        # self.gui.derivedproducts_filename_version is set equal to self.filename_version in self.gui when closing the application
        if not self.gui.derivedproducts_filename_version == self.filename_version and os.path.exists(self.gui.derivedproducts_dir):
            shutil.rmtree(self.gui.derivedproducts_dir)
        
        self.hdf5_structure_version = 38
        self.product_version = {'e':16,'r':11,'a':5,'m':5,'h':8,'l':17} #Product version, must be updated when the method for calculating the product has changed.
        self.product_parameters = {'e':'min_dBZ_value_echotops','a':'PCAPPI_height','m':'Zmax_minheight','h':'cap_dBZ, VIL_threshold','l':'cap_dBZ, VIL_minheight'}
        self.product_attributes = {'e':['scans_ranges','elevations_minside','elevations_plusside'],'m':['scans_ranges','elevations_minside','elevations_plusside'],'h':['scans_ranges','elevations_minside','elevations_plusside'],'l':['scans_ranges','elevations_minside','elevations_plusside']}
        self.product_proj_attributes = {'pol': ['product_radial_bins','product_radial_res','product_azimuthal_bins','product_azimuthal_res'],
                                        'car': ['product_xy_bins', 'product_res']}
        # Defines the maximum number of datasets per product within a file (different datasets have a different product parameter)
        # Allow 2 more for the cartesian products, because here also the resolution can be varied, instead of only the product parameter
        self.product_datasets_max = {'pol':4, 'car':6}
        
        self.plain_products_functions = {'independent':{'e':self.calculate_echotops,'a':self.calculate_PCAPPI,'m':self.calculate_Zmax,'l':self.calculate_VIL},
                                         'dependent':{'r':self.calculate_R}}
        # Only base products that are used to decide which function should be called need to be specified below
        self.products_requiring_heightsorted_data = ['e','a','l']

        self.meta_PP = {j: {} for j in gv.plain_products}
        
        self.mapping_classes = {'pol': Polar(self), 'car': Cartesian(self)}
        self.mapping_parameters = ['Zmax_3D','Zavg_3D','heights_3D','hdiffs']
        
    
    
    def get_dir_and_filename(self, proj):
        radar_dataset = self.dsg.get_radar_dataset(no_special_char=True)
        subdataset = self.dsg.get_subdataset(product=self.i_p)

        directory = self.gui.derivedproducts_dir+('/regular/' if proj == 'pol' else '/SM_correction/')+\
                    radar_dataset+'/'+subdataset+'/'+self.crd.date
        filename = directory+'/'+self.crd.date+self.crd.time+'.h5'
        return directory, filename

    def check_if_product_at_disk(self, p_param, proj, check_filtered_for_unfiltered=False):
        # check_filtered_for_unfiltered should be set to True when doing a 2nd attempt at importing data, now for the filtered version of a product, 
        # after it became clear that the unfiltered version is not available.
        product, param = self.get_product_and_param(p_param)        
        double_volume_index = self.dsg.scannumbers_forduplicates[product]
        
        _, filename = self.get_dir_and_filename(proj)
        if not os.path.exists(filename):
            return False
        
        product_at_disk = False
        try:
            with h5py.File(filename,'r+') as f:
                version, total_volume_files_size = f.attrs['version'], f.attrs['total_volume_files_size']
                if version != self.hdf5_structure_version or total_volume_files_size != self.dsg.total_files_size:
                    return False
                
                # When requesting an unfiltered product while this unfiltered version is unavailable (which becomes clear when importing the volumetric
                # data; self.using_unfilteredproduct), then this function is called again for the same p_param but for the filtered version. In this 
                # case the attribute 'u'+gv.i_p[product]+'_unavailable' is also updated, because it might be that the filtered version is indeed available, 
                # in which case this attribute does not get updated in write_file.
                if not check_filtered_for_unfiltered:
                    unfiltered_unavailable = f.attrs.get('u'+gv.i_p[product]+'_unavailable', False)
                else:
                    unfiltered_unavailable = self.productunfiltered and not self.using_unfilteredproduct
                    if unfiltered_unavailable:
                        f.attrs['u'+gv.i_p[product]+'_unavailable'] = True
                
                group_name = 'u' if self.productunfiltered and not unfiltered_unavailable else ''
                if len(self.dsg.scannumbers_all['z'][product])==2:
                    group_name += gv.productnames[product]+'_'+str(double_volume_index+1)
                else: 
                    group_name += gv.productnames[product]
                if not group_name in f:
                    return False
                
                group = f[group_name]
                version = group.attrs['version']
                if version != self.product_version[product]:
                    f.__delitem__(group_name)
                    return False
                
                if proj == 'car':
                    subgroup_present = False
                    for subgroup in group.values():
                        if (subgroup.attrs['stormmotion'] == self.gui.stormmotion).all():
                            subgroup_present = True; break
                    if not subgroup_present:
                        return False
                else:
                    subgroup = group
                
                dataset_name = self.get_dataset_name(product, param, proj)
                    
                datasets = list(subgroup)
                if dataset_name in datasets:
                    dataset = subgroup[dataset_name]
                    self.product_arrays[p_param] = np.asarray(dataset, dtype='uint'+str(gv.products_data_nbits[product]))
                    self.p_param_attributes[p_param] = {}
                    for attr in self.product_attributes.get(product, [])+self.product_proj_attributes[proj]+['proj']:
                        o = dataset if attr in self.product_proj_attributes[proj] and proj == 'car' else group
                        self.p_param_attributes[p_param][attr] = o.attrs[attr]
                        
                    dataset.attrs['n_displayed'] += 1
                    dataset.attrs['last_view_time'] = pytime.time()
                    self.producttimes[p_param] = group.attrs['producttime']
                    self.using_unfilteredproduct = self.productunfiltered and not unfiltered_unavailable
                    product_at_disk = True
        except Exception as e:
            print(e, product)
            pass
        
        return product_at_disk
    
    def get_dataset_name(self, product, param, proj):
        dataset_name = 'data' if not product in self.gui.PP_parameter_values else f'data_pval{param}'
        if proj == 'car':
            dataset_name += f'_res{self.gui.cartesian_product_res}_maxrange{self.gui.cartesian_product_maxrange}'
        return dataset_name

    def write_file(self, p_param, proj):
        product, param = self.get_product_and_param(p_param)
        double_volume_index = self.dsg.scannumbers_forduplicates[product]

        directory, filename = self.get_dir_and_filename(proj)
        os.makedirs(directory, exist_ok=True)
        try:
            with h5py.File(filename, 'r') as f:
                version, total_volume_files_size = f.attrs['version'], f.attrs['total_volume_files_size']
                new_file = version != self.hdf5_structure_version or total_volume_files_size != self.dsg.total_files_size
                action = 'w' if new_file else 'a'
        except Exception: 
            action = 'w'
        
        with h5py.File(filename, action) as f:
            f.attrs['version'] = self.hdf5_structure_version
            f.attrs['total_volume_files_size'] = self.dsg.total_files_size
            if self.productunfiltered and not self.using_unfilteredproduct:
                # Indicate that the unfiltered version of the import product is unavailable, such that no time will be wasted on trying again 
                # when it is requested a next time
                f.attrs['u'+gv.i_p[product]+'_unavailable'] = True
                
            group_name = 'u' if self.using_unfilteredproduct else ''
            if len(self.dsg.scannumbers_all['z'][product])==2:
                #This is the case for the products in plain_products_affected_by_double_volume for the new radars of the KNMI.
                #Append in this case the volume number to the group_name.
                group_name += gv.productnames[product]+'_'+str(double_volume_index+1)
            else: 
                group_name += gv.productnames[product]
            group = f.create_group(group_name) if not group_name in f else f[group_name]
            group.attrs['version']=self.product_version[product]
            group.attrs['producttime']=self.producttimes[p_param]
            
            if proj == 'car':
                # In this case different subgroups are created for different cases (different storm motions), since storm motion influences the product
                subgroup_present = False
                for subgroup in group.values():
                    if (subgroup.attrs['stormmotion'] == self.gui.stormmotion).all():
                        subgroup_present = True; break
                        
                if not subgroup_present:
                    subgroup = group.create_group(f'case{len(group)+1}')
                    subgroup.attrs['stormmotion'] = self.gui.stormmotion
            else:
                subgroup = group
            
            dataset_name = self.get_dataset_name(product, param, proj)
                            
            datasets = list(subgroup)
            if len(datasets) == self.product_datasets_max[proj]:
                #Parameter values for which the plain product is currently being plotted, for which the dataset should not be overwritten.
                params_in_use = [self.gui.PP_parameter_values[product][self.gui.PP_parameters_panels[j]] for j in self.pb.panellist if self.crd.products[j] == self.product]
                n_displayed = [subgroup[j].attrs['n_displayed'] for j in datasets if not subgroup[j].attrs[self.product_parameters[product]] in params_in_use]
                min_displayed = np.min(n_displayed)
                # Multiple datasets might have been displayed by the minimum number of times. In that case delete the oldest of these
                last_view_times = np.array([[j, subgroup[dset].attrs['last_view_time']] for (j, dset) in enumerate(datasets) if n_displayed[j] == min_displayed], dtype='uint64')
                dataset_remove = datasets[last_view_times[np.argmin(last_view_times[:, 1])][0]]
                dataset = subgroup[dataset_name] = subgroup[dataset_remove]
                del subgroup[dataset_remove]
            else:
                if dataset_name in subgroup:
                    del subgroup[dataset_name]
                dataset = subgroup.create_dataset(dataset_name, self.product_arrays[p_param].shape, maxshape=(None, None) if proj == 'car' else None,
                                                  dtype='uint'+str(gv.products_data_nbits[product]), compression='gzip', track_times=False)
                
            #The -product_range/int_range corrects for the fact that the integer value 1 and not 0 corresponds to a product value of
            #gv.products_maxrange[product][0].
            #Masked elements get an integer value of 0.
            int_range = 2**gv.products_data_nbits[product]-2 #A value of zero is used for masked elements, so therefore -2 vs -1.
            product_range = gv.products_maxrange[product][1]-gv.products_maxrange[product][0]
            group.attrs['calibration_formula']=str(product_range/int_range)+'*PV+'+str(gv.products_maxrange[product][0]-product_range/int_range)
            
            self.product_arrays[p_param] = self.dsg.convert_dtype_float_to_uint(self.product_arrays[p_param], product)
            if not dataset.shape == self.product_arrays[p_param].shape:
                dataset.resize(self.product_arrays[p_param].shape)
            dataset[...]=self.product_arrays[p_param]
                
            dataset.attrs['n_displayed'] = 1
            dataset.attrs['last_view_time'] = pytime.time()
            if not param is None:
                dataset.attrs[self.product_parameters[product]] = param
            for (attr, value) in self.p_param_attributes[p_param].items():
                o = dataset if attr in self.product_proj_attributes[proj] and proj == 'car' else group
                o.attrs[attr] = value
                

                
    def calculate_plain_products(self, panellist):
        """This function returns an array of uint-type product values!!!
        """
        t = pytime.time()
        self.double_volume_index = self.dsg.scannumbers_forduplicates['a']
        
        panels_filtered = [j for j in panellist if not self.crd.productunfiltered[j]]
        panels_unfiltered = [j for j in panellist if self.crd.productunfiltered[j]]
        self._calculate_plain_products(panels_filtered)
        self._calculate_plain_products(panels_unfiltered, productunfiltered=True)
        print(pytime.time()-t, 't_derived_tot')
        
    def get_product_and_param(self, p_param):
        product = p_param[:p_param.index('_')] if '_' in p_param else p_param
        param = p_param[p_param.index('_')+1:] if '_' in p_param else None                            
        return product, param
    
    def check_p_params_at_disk(self, p_params_bases, p_params_projs, check_filtered_for_unfiltered=False):
        for p_param in p_params_bases.copy():
            proj = p_params_projs[p_param]
            if proj == 'car' and not self.gui.current_case_shown(mode='loose'):
                # Cartesian products are usually not saved, because they vary as a function of storm motion.
                # The exception is when a case is currently shown, in which case it is likely that the stored product will be of use again
                continue
            product, param = self.get_product_and_param(p_param)
            
            if self.check_if_product_at_disk(p_param, proj, check_filtered_for_unfiltered):
                del p_params_bases[p_param]
            else:
                for base_p_param in p_params_bases[p_param].copy():
                    base_product, param = self.get_product_and_param(base_p_param)
                    if not p_param == base_p_param and self.check_if_product_at_disk(base_p_param, proj, check_filtered_for_unfiltered):
                        p_params_bases[p_param].remove(base_p_param)
                        # Convert uint to float for calculations
                        self.product_arrays[base_p_param] = self.dsg.convert_dtype_float_to_uint(self.product_arrays[base_p_param], base_product, inverse=True)
        return p_params_bases
    
    def _calculate_plain_products(self, panellist, productunfiltered=False):
        self.productunfiltered = productunfiltered
        
        p_params_panels = {}
        p_params_projs = {}
        p_params_bases = {}
        for panel in panellist:
            product = self.crd.products[panel]
            p_param = product
            if product in self.gui.PP_parameter_values:
                key = self.gui.PP_parameters_panels[panel]
                param = self.gui.PP_parameter_values[product][key]
                p_param += f'_{param}'
            p_params_panels[panel] = p_param
            
            p_params_projs[p_param] = 'car' if self.gui.stormmotion[1] != 0. and product in gv.plain_products_correct_for_SM else 'pol'
            p_params_bases[p_param] = {'r':[f'a_{gv.CAPPI_height_R}']}.get(product, [p_param])
                
            for base_p_param in p_params_bases[p_param]:
                # Make sure that the projection of the base_p_params is also defined
                p_params_projs[base_p_param] = p_params_projs[p_param]
        # When p_params are already present at the disk, they will below be removed from p_params_bases. But p_params_bases_all will 
        # keep referring to the original list.
        p_params_bases_all = copy.deepcopy(p_params_bases)


        self.i_p = gv.i_p['z']

        self.product_arrays = {} # Will contain data arrays per p_param
        self.producttimes = {} # Will contain time ranges per product, not also per param because these times are independent of those
        self.p_param_attributes = {} # Will contain attributes per p_param that are needed for saving to a file and for setting some
        # product-specific parameters
        
        self.check_p_params_at_disk(p_params_bases, p_params_projs)


        if any([len(p_params_bases[p_param]) > 0 for p_param in p_params_bases]):
            self.import_data_plain_products()
            self.get_info_per_product()
            
            if productunfiltered and not self.using_unfilteredproduct:
                # Filtered products will be used instead, check whether these are maybe already available at the disk
                p_params_bases = self.check_p_params_at_disk(p_params_bases, p_params_projs, check_filtered_for_unfiltered=True)
            
            
        self.base_p_params = []
        p_params_functions = {}
        for p_param in p_params_bases:
            for base_p_param in p_params_bases[p_param]:
                base_product, param = self.get_product_and_param(base_p_param)
                self.base_p_params += [base_p_param]
                # It's possible that a product can best be calculated in the same function that is used for another product.
                # In that case put that other product in p_params_functions.
                p_param_function = {'h':'l_[True, 0]'}.get(base_product, base_p_param)
                p_params_functions[p_param_function] = p_params_functions.get(p_param_function, [])+[base_p_param]
        
                # Make sure that the projection of p_param_function is also defined
                p_params_projs[p_param_function] = p_params_projs[p_param]
        
            
        product_slices = {}
        for p_param in p_params_functions:
            product, param = self.get_product_and_param(p_param)
            self.product = product
            
            for attr in ('bottom_scan_removed', 'scans'):
                self.__dict__[attr] = self.__dict__[attr+'_products'][self.product]
            
            self.proj = p_params_projs[p_param]
            for attr in ('i_p', 'scans', 'scans_all', 'data_all', 'scanangles_all', 'radial_bins_all', 'radial_res_all', 
                         'radial_range_all', 'scantimes_all', 'radius_offsets_all'):
                self.mapping_classes[self.proj].__dict__[attr] = self.__dict__[attr]
                
            self.mapping_classes[self.proj].get_product_dimensions()
            self.product_shape = self.mapping_classes[self.proj].get_product_shape()
            self.product_slice = self.mapping_classes[self.proj].get_product_slice()
            product_slices[p_param] = self.product_slice
            
            self.mapping_classes[self.proj].get_parameters_data_mapping()
            self.mapping_classes[self.proj].calculate_Zmax_and_Zavg_3D()
            self.mapping_classes[self.proj].sort_data_and_get_hdiffs_if_necessary()
            for parameter in self.mapping_parameters:
                if hasattr(self.mapping_classes[self.proj], parameter):
                    self.__dict__[parameter] = self.mapping_classes[self.proj].__dict__[parameter]
                
            self.plain_products_functions['independent'][self.product](param)
            self.product_arrays[p_param] = self.product_array # Gets defined in the plain product function
                                   
            volume_starttime, volume_endtime = self.volume_starttime, self.volume_endtime             
            # In the case of a double volume the volume start and end time should be adjusted
            if len(self.dsg.scannumbers_all['z'][product])==2:
                i = self.double_volume_index
                volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes([self.scantimes_all[j][-i] for j in self.dsg.scans_doublevolume[i]])
            volumetime = volume_starttime+'-'+volume_endtime                
            self.producttimes[p_param] = volumetime if self.proj == 'pol' or all([i[0] == volumetime for i in self.scantimes_all.values()]) else\
                                         self.mapping_classes['car'].avg_scantime

            if self.product in gv.plain_products_show_max_elevations:
                self.get_maxelevations_plainproducts()
                
            self.p_param_attributes[p_param] = {}
            for attr in self.product_attributes.get(product, []):
                self.p_param_attributes[p_param][attr] = self.__dict__[attr]
            for attr in self.product_proj_attributes[self.proj]:
                self.p_param_attributes[p_param][attr] = self.mapping_classes[self.proj].__dict__[attr]
            self.p_param_attributes[p_param]['proj'] = self.proj
            
            for p_param2 in p_params_functions[p_param]:
                # Update volume times and attributes for p_params that were calculated within the same function call
                self.producttimes[p_param2] = self.producttimes[p_param]
                self.p_param_attributes[p_param2] = self.p_param_attributes[p_param]
            
            self.crd.increase_sleeptime_after_plotting = True
            

        for p_param in p_params_bases: #Calculate products that depend on other products
            base_p_params = p_params_bases_all[p_param] # Use all base_p_params, not just the ones that had to be recalculated
            first_base = base_p_params[0]
            if not p_param == first_base:
                product, param = self.get_product_and_param(p_param)
                bins = [self.p_param_attributes[base]['product_radial_bins' if p_params_projs[base] == 'pol' else\
                                                  'product_xy_bins'] for base in base_p_params]
                # The below way of determining self.product_slice only works when the base_p_params together have not more than 2 different dimensions
                start = (max(bins)-min(bins))//2
                end = max(bins)-start
                self.product_slice = np.s_[start:end, start:end]
                self.plain_products_functions['dependent'][product](param)
                self.product_arrays[p_param] = self.product_array # Gets defined in the plain product function 
                self.producttimes[p_param] = self.producttimes[first_base]
                
                self.p_param_attributes[p_param] = {}
                for attr in self.p_param_attributes[first_base]:
                    self.p_param_attributes[p_param][attr] = self.p_param_attributes[first_base][attr]


        for panel in panellist:                
            p_param = p_params_panels[panel]
            product, param = self.get_product_and_param(p_param)
            proj = self.p_param_attributes[p_param]['proj']

            if p_param in p_params_bases and (proj == 'pol' or self.gui.current_case_shown(mode='loose')) and self.product_arrays[p_param].dtype == 'float32':
                # First condition means that it had to be calculated, and second is included because only polar products
                # and cartesian products for cases are saved (cartesian products are storm motion dependent, and therefore quite variable).
                # Third is included to prevent that a file is created more than once when a p_param is displayed in multiple panels
                self.write_file(p_param, proj)
            elif self.product_arrays[p_param].dtype == 'float32':
                # This is done here instead of in self.dsg, because it's possible that one p_param is shown in more than
                # one panel, in which case more than one panel would use the same data array. That is normally not a problem,
                # except when values within this same array get converted multiple times to integers. And doing the conversion
                # here prevents that from happening in self.dsg, and thereby fixes an observed bug
                self.product_arrays[p_param] = self.dsg.convert_dtype_float_to_uint(self.product_arrays[p_param], product)
            
            for attr in self.p_param_attributes[p_param]:
                self.meta_PP[product][attr.replace('product_', '')] = self.p_param_attributes[p_param][attr]
                
            self.dsg.data[panel] = self.product_arrays[p_param]
            self.dsg.scantimes[panel] = self.producttimes[p_param]
            self.crd.using_unfilteredproduct[panel] = self.using_unfilteredproduct
            self.crd.using_verticalpolarization[panel] = False
               
        
    def get_import_data_specs(self):
        radar_dataset = self.dsg.get_radar_dataset()
        subdataset = self.dsg.get_subdataset(product=self.i_p)
        return radar_dataset+subdataset+str(self.dsg.total_files_size)+self.crd.date+self.crd.time+str(self.productunfiltered)

    def import_data_plain_products(self):
        """The extension _all is used for scans, scanangles and data when importing data, where all data is imported
        that is required for one of the derived products, such that it has to be done only once per volume.
        Because not all products require the use of all scans however, self.scans etc. are used in other functions, and contain only the data
        used for a particular product. The exceptions are functions that calculate things for multiple products in general, like 
        self.assign_radarbins_to_productbins, self.calculate_Zmax_3D and self.calculate_Zavg_3D, self.get_range_3D and 
        self.get_heights_3D.
        
        Further, for the new radars of the KNMI it is the case that the volume can be divided into 2 parts, giving a time resolution of 2.5 minutes
        for some products. These 2 parts do not contain the same scans however, because when going through the scans from bottom to top, then the
        the scans belong alternately to one or the other part of the volume. This has as a disadvantage that it can give a flip-flop effect when 
        viewing derived products for which the volume is divided into 2 parts. The exception is a PCAPPI at a height just above the ground, because 
        the lowest scan (0.3 degree) is included in both parts of the volume.
        This 'double volume' is handled by adding the derived products to the keys in self.dsg.scannumbers_all['z'] etc. and
        self.dsg.scannumbers_forduplicates, such that it can be handled in the same way as duplicate scans are handled. 
        """
        if self.get_import_data_specs() != self.import_data_specs:
            self.scans_all=[i for i, j in self.dsg.scanangles_all[self.i_p].items() if j != 90.]                            
            self.scanangles_all = {j:self.dsg.scanangle(self.i_p, j, 0) for j in self.scans_all}
            self.radial_bins_all = {j:self.dsg.radial_bins_all[self.i_p][j] for j in self.scans_all}
            self.radial_res_all = {j:self.dsg.radial_res_all[self.i_p][j] for j in self.scans_all}
            self.radial_range_all = {j:self.dsg.radial_range_all[self.i_p][j] for j in self.scans_all}
            
            #self.data_all contains for each scan a list of data arrays, which will usually have length 1, except for a duplicate scan in the case of
            #a double volume.
            self.data_all, self.scantimes_all, self.volume_starttime, self.volume_endtime, meta = self.dsg.get_data_multiple_scans(self.i_p, self.scans_all, productunfiltered=self.productunfiltered)
            self.using_unfilteredproduct = meta['using_unfilteredproduct']
            self.radius_offsets_all = meta.get('radius_offsets', {j:0 for j in self.data_all})
            
            self.import_data_specs = self.get_import_data_specs()

    def get_info_per_product(self):
        self.bottom_scan_removed_products = {}; self.scans_products = {}; self.scanangles_products = {}
        self.products_per_indices = {}
        
        for product in self.plain_products_functions['independent']:
            self.bottom_scan_removed_products[product] = False
            
            if len(self.scans_all) > 1 and\
            ft.rndec(self.scanangles_all[self.scans_all[1]]-self.scanangles_all[self.scans_all[0]],3)<=0.1: 
                if product in ('a','m'):
                    if len(self.dsg.scannumbers_all['z'][product])==2:
                        #This is the case for the plain products in plain_products_affected_by_double_volume when having a double volume.                                               
                        #Take the first or second series of scans, depending on the value of self.double_volume_index.
                        self.scans_products[product] = self.dsg.scans_doublevolume[self.double_volume_index]
                        self.scanangles_products[product] = {j:self.dsg.scanangles_all[self.i_p][j] for j in self.scans_products[product]}
                    else:
                        self.scans_products[product] = self.scans_all.copy()
                        self.scanangles_products[product] = self.scanangles_all.copy()
                else:
                    self.bottom_scan_removed_products[product] = True
                    #Because the bottom scan is usually rather noisy it is removed, except when the difference between the lowest two scans is greater than 
                    #0.1 degrees, too prevent throwing away too much data. The exception is 'm', where the bottom scan is used at ranges that are not 
                    #spanned by other scans.
                    self.scans_products[product] = self.scans_all[1:]
                    self.scanangles_products[product] = {j:self.scanangles_all[j] for j in self.scans_products[product]}
            else:
                self.scans_products[product] = self.scans_all.copy()
                self.scanangles_products[product] = self.scanangles_all.copy()
            
            s = self.get_indices_scans(product, self.scans_products[product])
            key = str(s)
            if not key in self.products_per_indices:
                self.products_per_indices[key] = []
            self.products_per_indices[key].append(product)
        
    
    def get_maxelevations_plainproducts(self):
        #Function expects the scan range to increase for decreasing scanangle.
        self.scans_ranges = [[self.scans[j], self.radial_bins_all[self.scans[j]]] for j in\
                             range(len(self.scans[:-1])) if self.radial_bins_all[self.scans[j]]-2 >\
                             self.radial_bins_all[self.scans[j+1]]]
        self.scans_ranges.append([self.scans[-1],self.radial_bins_all[self.scans[-1]]])
        scanangles_maxside = np.asarray([self.scanangles_all[j[0]] for j in self.scans_ranges])
        scanangles_minside = np.append(self.scanangles_all[self.scans[0]],np.asarray([self.scanangles_all[j[0]] for j in self.scans_ranges])[:-1])
        self.scans_ranges = np.asarray([j[1] for j in self.scans_ranges])
        if not self.bottom_scan_removed:
            scanangles_minside = scanangles_minside[1:]
            self.elevations_minside = ft.r1dec(ft.var1_to_var2(self.scans_ranges[1:],scanangles_minside)[1])
        else:
            scanangles_minside[0] = self.scanangles_all[self.scans_all[0]] #Use the removed scanangle as the first scanangle
            self.elevations_minside = ft.r1dec(ft.var1_to_var2(self.scans_ranges,scanangles_minside)[1])
        self.elevations_plusside = ft.r1dec(ft.var1_to_var2(self.scans_ranges,scanangles_maxside)[1])
        
    def get_true_elevations_plainproducts(self, panellist, heights_scan1, radii_scan1):
        """For the products in plain_products_show_true_elevations this function determines the 'true' elevations at which the product is displayed.
        For the PCAPPI,'a', these elevations are given by the elevations for the lowest scanangle, plus two times the PCAPPI height, where
        the ring for the first PCAPPI height is shown at the range at which the highest scan reaches this height, and the second at the range at which 
        the lowest scan reaches the PCAPPI height. Then finally, some rings are removed when they are to close together.
        
        Function is called from within self.pb.get_parameters_heightrings, because these rings must be updated after every pan or zoom action.
        """
        heights, radii = {j:heights_scan1 for j in panellist}, {j:radii_scan1 for j in panellist}
        
        x = ((self.pb.corners[0][-1,0]-self.pb.corners[0][0,0])*self.pb.ncolumns)
        dr = x/8
        for j in panellist:
            if self.crd.products[j] in gv.plain_products_show_true_elevations:
                if self.crd.products[j]=='a': CAPPI_height = self.gui.PP_parameter_values['a'][self.gui.PP_parameters_panels[j]]
                elif self.crd.products[j]=='r': CAPPI_height = gv.CAPPI_height_R
                #Range at which the highest scan reaches a height equal to CAPPI_height
                angles = np.sort([self.dsg.scanangle(gv.i_p['a'], j, self.double_volume_index) for j in self.dsg.scanangles_all[gv.i_p['a']]])
                CAPPI_height_r1 = ft.var1_to_var2(CAPPI_height, angles[angles != 90.][-1], 'h+theta->gr')
                #Range at which the lowest scan reaches a height equal to CAPPI_height
                CAPPI_height_r2 = ft.var1_to_var2(CAPPI_height, self.dsg.scanangle(gv.i_p['a'], 1, self.double_volume_index), 'h+theta->gr')
                heights_greater_than_CAPPI_height = heights[j]>CAPPI_height
                heights[j] = np.append([CAPPI_height, CAPPI_height],heights[j][heights_greater_than_CAPPI_height])
                radii[j] = np.append([CAPPI_height_r1, CAPPI_height_r2],radii[j][heights_greater_than_CAPPI_height])
                
                retain = np.ones(len(radii[j]),dtype='bool')
                if radii[j][1]-radii[j][0] < 0.2*dr:
                    retain[0] = 0
                if len(radii[j]) > 2 and radii[j][2]-radii[j][1] < 0.5*dr:
                    retain[2] = 0
                
                heights[j] = heights[j][retain]
                radii[j] = radii[j][retain]
        return heights, radii
         
               
    def get_subscan_index(self, scan):
        #For radars with a double volume, this index indicates which part of the volume is used now in the calculations. It is 0 when viewing the first
        #part, and 1 when viewing the second part. 
        #For radars without a double volume, the index is always 0.
        if self.crd.radar in gv.radars_with_double_volume:
            index = min([self.double_volume_index, len(self.dsg.scannumbers_all['z'][scan])-1])
        else:
            index = 0
        return index
    
    def index_all(self, scan, sub_scan = None):
        if sub_scan is None:
            sub_scan = self.get_subscan_index(scan)
        return sum([len(self.data_all[j]) for j in self.scans_all if j<scan])+sub_scan
    
    def get_indices_scans(self, product=None, scans=None):
        if not product:
            product = self.product
        if not scans:
            scans = self.scans
        indices = [self.index_all(j) for j in scans]
        return indices if scans != self.scans_all or product in self.products_requiring_heightsorted_data else np.s_[:]
    
    def get_highest_notempty_scan(self):        
        self.highest_notempty = np.zeros(self.product_shape, dtype = 'int8')
        empty = np.ones(self.product_shape, dtype = bool)
        Z_empty = -30.
        for i in range(len(self.scans)-1,-1,-1):
            update = empty & (self.Zavg_3D[i] != Z_empty)
            self.highest_notempty[update] = i
            empty[update] = 0
            # The lines below are needed because at the edge of the range of a group of scans it can happen
            # that due to translation out-of-range bins (that have a height of 0) are positioned below inside-range bins.
            # These lines ensure that any reflectivity value above a 0-height is not included, to prevent issues with negative hdiffs
            height_0 = self.heights_3D[i] == 0.
            empty[height_0] = 1
        
    
    def calculate_echotops(self, param): 
        min_dBZ_value_echotops = float(param)
        self.product_array = np.full(self.product_shape, -1000., dtype='float32')
        
        product_array_update_before = None
        for i in range(len(self.heights_3D)):
            Z_i = self.Zmax_3D[i]
            h_i = self.heights_3D[i]
            product_array_update = Z_i >= min_dBZ_value_echotops
            self.product_array[product_array_update] = h_i[product_array_update]
            if i > 0:
                difference = product_array_update_before & ~product_array_update & (Z_i != -30.)
                Z_imin1 = self.Zmax_3D[i-1]
                self.product_array[difference] += (h_i[difference]-self.product_array[difference])*\
                    (Z_imin1[difference]-min_dBZ_value_echotops)/(Z_imin1[difference]-Z_i[difference])
            product_array_update_before = product_array_update.copy()

        
    def calculate_Zmax(self, param):
        Zmax_minheight = float(param)
        Zmax_3D = self.Zmax_3D.copy()
        Zmax_3D[:-1][self.heights_3D[:-1]<Zmax_minheight] = -30.

        self.product_array = np.max(Zmax_3D, axis = 0)            
        empty = self.product_array==-30.
        self.product_array[empty] = self.pb.mask_values['m']
    
        
    def calculate_PCAPPI(self, param):
        """Calculate a PCAPPI, by interpolating between the two scans that surround the PCAPPI level. If PCAPPI_height is smaller than the
        minimum height for which data is available, then data for the lowest scan is shown at those positions. If PCAPPI_height is larger than
        the maximum height for which data is available, then data for the highest scan for which data is available is shown.
        """
        PCAPPI_height = float(param)                
        self.get_highest_notempty_scan()
        I, J = np.indices(self.highest_notempty.shape)
        height_highest_notempty = self.heights_3D[self.highest_notempty, I, J]
        
        self.product_array = np.full(self.product_shape, -1000., dtype='float32')
        
        Z_bottom = self.product_array.copy(); Z_top = self.product_array.copy()
        hdiff_bottom = np.zeros(self.product_array.shape, dtype='float32'); hdiff_top = hdiff_bottom.copy()
        
        for i in range(len(self.heights_3D)):
            Z_bottom_update = (self.heights_3D[i]<PCAPPI_height) & (self.heights_3D[i] > 0.)
            Z_bottom[Z_bottom_update] = self.Zavg_3D[i,Z_bottom_update]
            hdiff_bottom[Z_bottom_update] = np.abs(PCAPPI_height-self.heights_3D[i,Z_bottom_update])
            
            Z_top_update = (Z_top==-1000.) & (self.heights_3D[i]>PCAPPI_height)
            Z_top[Z_top_update] = self.Zavg_3D[i,Z_top_update]
            hdiff_top[Z_top_update] = np.abs(PCAPPI_height-self.heights_3D[i,Z_top_update])
        Zb = Z_bottom==-1000.; Zt = Z_top==-1000.

        hdiff_sum = hdiff_bottom+hdiff_top; hdiff_sum_nonzero = hdiff_sum!=0.
        #Update the bins in self.product_array for which hdiff_sum_nonzero==True
        self.product_array[hdiff_sum_nonzero] = (Z_bottom[hdiff_sum_nonzero]*hdiff_top[hdiff_sum_nonzero]+Z_top[hdiff_sum_nonzero]*hdiff_bottom[hdiff_sum_nonzero])/hdiff_sum[hdiff_sum_nonzero]
        self.product_array[height_highest_notempty < PCAPPI_height] = -1000.
        
        #Update the bins in self.product_array for which hdiff_sum_nonzero==False, but for which this occurs because Z_bottom or Z_top is available 
        #'exactly' at the CAPPI height.
        Z_bottom_exactly_at_CAPPI_height = (hdiff_sum_nonzero==False) & (Zb==False)
        Z_top_exactly_at_CAPPI_height = (hdiff_sum_nonzero==False) & (Zt==False)
        self.product_array[Z_bottom_exactly_at_CAPPI_height] = Z_bottom[Z_bottom_exactly_at_CAPPI_height]
        self.product_array[Z_top_exactly_at_CAPPI_height] = Z_bottom[Z_top_exactly_at_CAPPI_height]
        
        self.product_array[Zt] = self.Zavg_3D[-1, Zt]
        self.product_array[Zb] = self.Zavg_3D[0, Zb]           
        self.product_array[np.abs(self.product_array+35.)<0.1] = self.pb.mask_values['a']
                
        
    def calculate_VIL(self, param = None):
        """
        An important remark is that the part of the VIL below the lowest scan is calculated by assuming that the reflectivity there is the same as at the
        lowest scan. This is in a lot of cases a bad assumption, because of vertical tilting of storms with height due to the fact that different scans 
        are performed at different times, among other things. Just neglecting this part of the VIL can result however in a substantial underestimation, 
        and the current assumption then seems better.
        """            
        cap_dBZ, VIL_minheight = eval(param)
        # Multiple height p_params might be calculated in the same VIL call, as long as only the VIL threshold varies.
        p_params_h = [p_param for p_param in self.base_p_params if p_param.startswith('h_')]
        calculate_h = len(p_params_h) > 0
        self.get_highest_notempty_scan()
        
        self.product_array = np.zeros(self.product_shape, dtype = 'float32')
        if calculate_h:
            # Since reflectivity is capped in a different way for VIL versus CMH (see below), a different VIL array is needed for CMH.
            product_array_l = self.product_array.copy()
            product_array_h = self.product_array.copy()
            
        s = self.Zavg_3D != -30.
        Z_linear = np.zeros(self.Zavg_3D.shape, dtype = 'float32')
        Z_linear[s] = 10**(0.1*self.Zavg_3D[s])
        Zlinear_56dBZ = 10**5.6
        Z_linear_cap = np.minimum(Z_linear, Zlinear_56dBZ)
            
        for i in range(len(self.scans)-1):
            Z1, Z2 = Z_linear[i], Z_linear[i+1]
            Z1_cap, Z2_cap = Z_linear_cap[i], Z_linear_cap[i+1]
            s1, s2 = s[i], s[i+1]
            
            ss = (s1 | s2) & (self.highest_notempty >= i+1)
            Z = 0.5*(Z1[ss]+Z2[ss])
            if cap_dBZ:
                # Cap the average linear reflectivity, in agreement with https://vlab.noaa.gov/web/wdtd/-/vertically-integrated-liquid-vil-
                Z[Z > Zlinear_56dBZ] = Zlinear_56dBZ
            
            hdiffs = self.hdiffs[i+1, ss]
            if VIL_minheight:
                change = VIL_minheight > self.heights_3D[i, ss]
                hdiffs[change] = 1e3*np.maximum(0., self.heights_3D[i+1, ss][change]-VIL_minheight)
            self.product_array[ss] += (3.44*10**-6)*hdiffs*Z**(4./7.)
            
            if calculate_h:
                # Note that there is a discrepancy between how capping reflectivity is handled for VIL and for CMH. For VIL the average 
                # reflectivity is capped, while for CMH the individual reflectivities are capped. This is because there is no possibility 
                # here to cap the average reflectivity because of multiplication with height.
                z1, z2 = (Z1_cap, Z2_cap) if cap_dBZ else (Z1, Z2)
                hZ47 = 0.5*(self.heights_3D[i, ss]*z1[ss]**(4./7.)+
                            self.heights_3D[i+1, ss]*z2[ss]**(4./7.))
                product_array_h[ss] += (3.44*10**-6)*hdiffs*hZ47
                if cap_dBZ: # Otherwise use the above definition of Z
                    Z = 0.5*(z1[ss]+z2[ss])
                product_array_l[ss] += (3.44*10**-6)*hdiffs*Z**(4./7.)
            
            if i == 0:
                hdiffs = 1e3*np.maximum(0., self.heights_3D[0, s1]-VIL_minheight) if VIL_minheight else self.hdiffs[0, s1]
                z1 = Z1_cap if cap_dBZ else Z1
                delta_l = (3.44*10**-6)*hdiffs*z1[s1]**(4./7.)
                self.product_array[s1] += delta_l
                if calculate_h:
                    hZ47 = 0.5*self.heights_3D[0, s1]*z1[s1]**(4./7.)
                    product_array_h[s1] += (3.44*10**-6)*hdiffs*hZ47
                    product_array_l[s1] += delta_l
                    
        select = self.product_array < 1e-5
        if calculate_h:
            product_array_h /= product_array_l
            for p_param in p_params_h:
                _, param_h = self.get_product_and_param(p_param)
                VIL_threshold = eval(param_h)[1]
                select_h = self.product_array < VIL_threshold
                self.product_arrays[p_param] = product_array_h.copy()
                self.product_arrays[p_param][select_h] = self.pb.mask_values['h']
        self.product_array[select] = self.pb.mask_values['l']
        
        
    def convert_dBZ_to_mmph(self, data, inverse = False):
        if not inverse:
            data = np.power(np.divide(np.power(10.,np.divide(data, 10.)),200.),5./8.)
        else:
            data = 10.*np.log10(200.*np.power(data, 8./5.))
        return data

    def calculate_R(self, param = None):
        self.product_array = np.log10(self.convert_dBZ_to_mmph(self.product_arrays[f'a_{gv.CAPPI_height_R}']))
        #Log10 is used because the colormap for 'r' is logarithmic.
        self.product_array[self.product_array<=1e-6]=self.pb.mask_values['r']