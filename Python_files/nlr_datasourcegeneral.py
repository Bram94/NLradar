# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

from derived.nlr_derived_plain import DerivedPlain
from derived import nlr_derived_tilts as dt
import nlr_datasourcespecific as dss
import nlr_importdata as ird
import nlr_background as bg
import nlr_functions as ft
import nlr_globalvars as gv

import sys
import os
opa = os.path.abspath
import numpy as np
import time as pytime
import pickle
import copy
import traceback



"""DataSource_General contains functions that are related to the import of data, and contains functions that can in general be used for
different data sources (by calling the appropriate functions in the classes in nlr_datasourcespecific.py).
"""

class DataSource_General():
    #Base class for importing radar data
    def __init__(self, gui_class, crd_class, parent = None):        
        self.gui = gui_class
        self.crd = crd_class
        self.dp = DerivedPlain(dsg_class = self)
        self.pb = self.gui.pb
                        
        self.Gematronik_vol_rainbow3 = ird.Gematronik_vol_rainbow3(gui_class = self.gui, dsg_class = self)
        self.Gematronik_vol_rainbow5 = ird.Gematronik_vol_rainbow5(gui_class = self.gui, dsg_class = self)
        self.KNMI_hdf5 = ird.KNMI_hdf5(gui_class = self.gui, dsg_class = self)
        self.KMI_hdf5 = ird.KMI_hdf5(gui_class = self.gui, dsg_class = self)
        self.skeyes_hdf5 = ird.skeyes_hdf5(gui_class = self.gui, dsg_class = self)
        self.DWD_odimh5 = ird.DWD_odimh5(gui_class = self.gui, dsg_class = self)
        self.DWD_bufr = ird.DWD_BUFR(gui_class = self.gui, dsg_class = self)
        self.TUDelft_nc = ird.TUDelft_nc(gui_class = self.gui, dsg_class = self)
        self.NEXRAD_L2 = ird.NEXRAD_L2(gui_class = self.gui, dsg_class = self)
        self.NEXRAD_L3 = ird.NEXRAD_L3(gui_class = self.gui, dsg_class = self)
        self.CFRadial = ird.CFRadial(gui_class = self.gui, dsg_class = self)
        self.DORADE = ird.DORADE(gui_class = self.gui, dsg_class = self)
        self.MeteoFrance_BUFR = ird.MeteoFrance_BUFR(gui_class = self.gui, dsg_class = self)
        
        self.source_KNMI = dss.Source_KNMI(gui_class = self.gui, dsg_class = self)
        self.source_KMI = dss.Source_KMI(gui_class = self.gui, dsg_class = self)
        self.source_skeyes = dss.Source_skeyes(gui_class = self.gui, dsg_class = self)
        self.source_VMM = self.source_KMI # VMM data can be completely handled by the self.source_KMI class
        self.source_DWD = dss.Source_DWD(gui_class = self.gui, dsg_class = self)
        self.source_TUDelft = dss.Source_TUDelft(gui_class = self.gui, dsg_class = self)
        self.source_IMGW = dss.Source_IMGW(gui_class = self.gui, dsg_class = self)
        self.source_DMI = dss.Source_DMI(gui_class = self.gui, dsg_class = self)
        self.source_NWS = dss.Source_NWS(gui_class = self.gui, dsg_class = self)
        self.source_ARRC = dss.Source_ARRC(gui_class = self.gui, dsg_class = self)
        self.source_MeteoFrance = dss.Source_MeteoFrance(gui_class = self.gui, dsg_class = self)
        self.source_classes = {'KNMI':self.source_KNMI,'KMI':self.source_KMI,'skeyes':self.source_skeyes,'VMM':self.source_VMM,'DWD':self.source_DWD, 'TU Delft': self.source_TUDelft, 'IMGW': self.source_IMGW, 'DMI': self.source_DMI, 'NWS': self.source_NWS, 'ARRC': self.source_ARRC, 'Météo-France': self.source_MeteoFrance}
                
        self.classes = [self.source_KNMI,self.source_KMI,self.source_skeyes,self.source_DWD,self.source_TUDelft,self.source_IMGW,self.source_DMI,self.source_NWS,self.source_MeteoFrance,self.Gematronik_vol_rainbow3,self.Gematronik_vol_rainbow5,self.KNMI_hdf5,self.KMI_hdf5,self.skeyes_hdf5,self.DWD_odimh5,self.DWD_bufr,self.TUDelft_nc,self.NEXRAD_L2,self.NEXRAD_L3,self.CFRadial,self.DORADE,self.MeteoFrance_BUFR]
        
        self.radial_res_all = {}; self.radial_bins_all = {}; self.radial_range_all = {}
        # Scanangles might vary among duplicate scans, in which case self.scanangles_all for this scan will contain a dictionary
        # with a scanangle per duplicate index. self.scanangles_all_m contains the mean scanangle over all duplicates, which in
        # absence of duplicates or in case of a constant scanangle will just be equal to the corresponding entry in self.scanangles_all.
        # When only an approximate scanangle is needed, self.scanangles_all_m should be used. self.scanangles_all should be used
        # when needing the actual scanangle for a certain duplicate. In this case the function self.scanangle should be called,
        # since it can also handle the absence of duplicates or a constant scanangle per duplicate.
        self.scannumbers_all = {}; self.scanangles_all = {}; self.scanangles_all_m = {}
        self.nyquist_velocities_all_mps = {}; self.low_nyquist_velocities_all_mps = {}; self.high_nyquist_velocities_all_mps = {}
        self.scannumbers_forduplicates = {} # Specifies which scan will be requested when two or more scans have the same properties
        self.determined_volume_attributes_radars = {}
        self.scannumbers_forduplicates_radars = {}
        
        self.scans_doublevolume = []                                                                                                                                                                                    
        self.variable_attributes = []
        self.update_volume_attributes = False # Should be set to True when some volume attributes have been updated during a call
        # of get_data, like what happens with NEXRAD level II data
        
        self.savevalues_refscans_unsorted = {j:[] for j in gv.radars_all} #Is used in nlr_importdata.py
        
        self.data = {j:-1e6*np.ones((362,1)).astype('float32') for j in range(10)}
        self.scantimes = {}
        # The left edge of the first radial might not correspond to an azimuth of 0°. If that's the case, this offset value should 
        # be set for the corresponding panel. Only values between -1 and +1 are allowed, bigger offsets should be prevented by rolling
        # the data array when necessary.
        self.data_azimuth_offset = {j:0. for j in range(10)}
        # Similarly as for the azimuth, with radius offsets between -1 and +1 km supported
        self.data_radius_offset = {j:0. for j in range(10)}
        self.stored_data = {}
        
        self.range_nyquistvelocity_scanpairs_indices = {j:0 for j in range(10)}
        self.scans_radars = {} #Gets updated in self.pb.set_newdata!
        self.selected_scanangles={} #selected_scanangles is used when the radar volume structure changes during the course of a day, as can be the case
        #for the radar in Zaventem. When selecting a particular scanangle, that is not available with the next volume structure, then it is chosen 
        #automatically again when the structure changes back to something that contains the selected scanangle.
        #Should not be updated when going to the left/right.
        self.selected_scanangles_before = {}
        self.panel_center_heights = {}
        self.time_last_panzoom = 0
        self.time_last_purposefulscanchanges = 0
        self.time_last_choosenearestheight = 0
        self.time_last_choosenearestscanangle = 0
        
        self.files_datetimesdict = {}
        self.files_datetime = None # Files for the current datetime
        # It could be the case that multiple versions of a product are available for a single radar volume (and that these can't be considered
        # as a filtered-unfiltered pair). The following variables contain information about this, and are set when needed.
        self.product_versions_datetimesdict = None
        self.product_versions_directory = None # Available product versions for the whole current directory. 
        self.product_versions_datetime = None # Available product versions for the current datetime.
        self.products_version_dependent = None # Products that are version-dependent. This information is used to determine whether
        # derived products should be calculated separately for each product version.
        
        # Should be updated when volume attributes for all data sources should be reset.
        self.attributes_version = 16
        # Should be updated when the structure of volume attributes has changed only for particular data sources.
        # Updating it resets only attributes for that particular data source.
        self.attributes_version_sources = {'Météo-France':1}
                    
        self.scanangles = {}
        
        self.attributes_descriptions_filename = os.path.join(opa(gv.programdir+'/Generated_files'),'attributes_descriptions.pkl')
        self.attributes_IDs_filename = os.path.join(opa(gv.programdir+'/Generated_files'),'attributes_IDs.pkl')
        self.attributes_variable_filename = os.path.join(opa(gv.programdir+'/Generated_files'),'attributes_variable.pkl')
        try:
            if self.gui.reset_volume_attributes: raise Exception

            with open(self.attributes_descriptions_filename,'rb') as f:
                self.attributes_descriptions = pickle.load(f)

            with open(self.attributes_IDs_filename,'rb') as f:
                self.attributes_IDs = pickle.load(f)
            
            with open(self.attributes_variable_filename,'rb') as f:
                self.attributes_variable = pickle.load(f)
        except Exception:
            self.attributes_descriptions, self.attributes_IDs, self.attributes_variable = {}, {}, {}
            
        self.gui.reset_volume_attributes = False
            
        if not 'version' in self.attributes_descriptions or self.attributes_descriptions['version'] != self.attributes_version:
            self.attributes_descriptions = {'version':self.attributes_version}
            self.attributes_IDs, self.attributes_variable = {}, {}
        ft.init_dict_entries_if_absent(self.attributes_descriptions, 'version_sources', dict)
            
        for j in gv.radars_all:
            key_extensions = ('_V','_Z') if j in gv.radars_with_datasets else ('',)
            source = gv.data_sources[j]
            reset_attrs_radar = self.attributes_descriptions['version_sources'].get(source, None) != self.attributes_version_sources.get(source, None)
            for i in key_extensions:
                if reset_attrs_radar or j+i not in self.attributes_descriptions: self.attributes_descriptions[j+i] = {}
                if reset_attrs_radar or j+i not in self.attributes_IDs: self.attributes_IDs[j+i] = {}
                if reset_attrs_radar or j+i not in self.attributes_variable: self.attributes_variable[j+i] = {}
        self.attributes_descriptions['version_sources'] = self.attributes_version_sources
                
        
    def get_scans_information(self, set_data):
        """This function should deliver the attributes self.radial_res_all, self.radial_bins_all, scanangles_all, 
        nyquist_velocities_all_mps, low_nyquist_velocities_all_mps, high_nyquist_velocities_all_mps, scannumbers_all, scannumbers_forduplicates.
        
        In this application the scans are ordered from lower to higher scanangle, so a higher scannumber (starting at 1)
        corresponds to a higher scanangle. It is however possible that the scans in the imported volume are not ordered in 
        the same way (as is the case for the new KNMI volumes). Because of this possibility, scannumbers_all maps each scan
        (according to my way of ordening) to the corresponding scannumber(s) in the data file, which are put in a list.
        They are put in a list because it is possible that some scans are performed more than once during a complete volume scan.
        If a scan is performed more than once, then the list will include more than one element.
        scannumbers_forduplicates is then a list with for each scan the index of the scannumber that is taken from the list
        of scannumbers in the imported radar volume. Its elements are zero when the scans are performed only once during a 
        complete volume scan, and it therefore has the form [0/1,0,0,0,...] for the new KNMI radars. 
        
        It is possible that the attributes differ (slightly) per product, and for this reason they are determined for each product that is available
        for a particular data source (excluding products that are derived from other ones, i.e. only products in gv.i_p). The only product 
        that must always be available is 'z', whether or not it is really available for the source. This is because attribute['z'] is used in most of the
        functions that use info about particular volume attributes. If 'z' is not available, then the attributes for 'z' should be equal to those for products
        that are available.
        self.scannumbers_all etc. therefore have the form {'z':{1:...,2:...}} etc.
        
        To prevent that attributes are obtained more than once for a particular file, information about it is stored in a file. Each particular series
        of attributes gets assigned a unique ID, and this ID in stored in self.attributes_IDs[radar_dataset][dir_string]. 
        Here, dir_string is the directory string that is present in self.gui.radardata_dirs, for the
        index given in self.gui.radardata_dirs_indices. dir_string is used as key in addition to radar_dataset, because multiple directories can be
        selected for one radar_dataset, and if these directories contain different files for the same date and time, this would lead to problems.
        A disadvantage of the use of dir_string is that the attributes must be determined again when the data is moved to another directory.
        
        self.attributes_descriptions[radar_dataset] contains the series of attributes that corresponds to a particular ID. 
        When calling this function, it is first checked whether an attribute ID is available for a particular radar, date and time, and if so, the 
        corresponding attributes are obtained from self.attributes_descriptions[radar_dataset].
        If no attribute ID is available, then the attributes will be obtained, and the ID corresponding to it is stored in 
        self.attributes_IDs[radar_dataset][dir_string].
        For radars for which the data for one volume is stored in multiple files, it is however possible that the attributes get stored at a moment
        at which not all files were available yet, such that the saved attributes should be updated when more files come available. In order to
        handle this, the number of files that was present when saving the attributes is also stored, which is done in
        self.attributes_IDs[radar_dataset][dir_string]. The values stored in this dictionary are therefore lists of the form 
        [attributes_ID,saved_total_files_size].
        
        Sometimes attributes can be expected to vary from volume to volume, in which case it's not wise to add them to self.attributes_descriptions,
        since that would mean that also the other less variable attributes are stored again and again. In this case they are stored separately
        in self.attributes_variable, and the corresponding entries in self.attributes_descriptions are replaced by the string 'variable'.
        
        self.attributes_descriptions, self.attributes_IDs and self.attributes_variable are dumped into a pickle file when exiting the program,
        which occurs in the function closeEvent in nlr.py.
        """

        # (deep) copy of attributes is not needed, since any change of the current volume attributes will come either with a deep copy
        # of attributes stored in a file, or in case of newly determining attributes will start with initialization of empty dictionary.
        # Should only be updated when setting data
        self.attrs_before = {j:self.__dict__[j] for j in gv.volume_attributes_all}
        self.attrs_before['scannumbers_forduplicates'] = self.scannumbers_forduplicates
        self.scanpair_present_before = self.check_presence_large_range_large_nyquistvelocity_scanpair() if\
                                       self.attrs_before['scannumbers_all'] else False
                      
        attributes_available = self.restore_volume_attributes()            

        if not attributes_available:
            try:
                self.nyquist_velocities_all_mps = {}; self.low_nyquist_velocities_all_mps = {}; self.high_nyquist_velocities_all_mps = {}
                for attr in gv.volume_attributes_p: #Defined in nlr_globalvars.
                    self.__dict__[attr] = {}
                    for j in (gv.i_p[p] for p in gv.products_with_tilts):
                        self.__dict__[attr][j] = {}
                    
                self.scans_doublevolume = [] #scans_doublevolume is used for the new radars of the KNMI, where the volume can be divided into 2 parts.
                #It is also saved to self.attributes_descriptions[radar_dataset]. It gets defined in the functions get_scans_information in
                #nlr_importdata.py, if a double volume is present.
                self.variable_attributes = [] #variable_attributes is used for volume attributes that are expected to be different for (almost)
                # each volume. These attributes are then stored separately
                
                self.source_classes[self.data_source()].get_scans_information()
                # Check for 'z'. It can happen that for other products the length is not zero. But other parts of the code
                # use len(self.scannumbers_all['z']) for multiple operations, and a length of 0 leads to errors there. 
                # So don't continue when no z-scan is available.
                # Check for any subdict whose key starts with 'z'. This is done since for self.product_versions_in1file=True, 
                # keys will have the product version appended to them. See self.process_products_with_pvs_in_keys for more info.
                if all(len(j) == 0 for i,j in self.scannumbers_all.items() if i[0] == 'z'):
                    raise Exception

                # Make sure that all volume attribute dictionaries are sorted in order of increasing scans. In the past an issue has been noted
                # due to non-ascending scans in one of the import classes, and sorting volume attributes here guarantees that this won't occur again,
                # regardless of the implementation of get_scans_information in the import classes. The time this sorting takes is negligible.
                for attr in gv.volume_attributes_save:
                    if isinstance(self.__dict__[attr], dict):
                        if attr in gv.volume_attributes_p:
                            for p in self.__dict__[attr]:
                                # Sorting only integer keys is done because there might be string keys for derived products in self.scannumbers_all['z']
                                self.__dict__[attr][p] = dict(sorted((i,j) for i,j in self.__dict__[attr][p].items() if type(i) == int))
                        else:
                            self.__dict__[attr] = dict(sorted(self.__dict__[attr].items()))
                
                self.store_volume_attributes()
                        
                self.determined_volume_attributes_radars[self.crd.radar] = {j:eval('self.'+j, {'self': self}) for j in gv.volume_attributes_save}
                
            except Exception as e:
                print('restore attributes')
                self.restore_previous_attributes()
                # This exception should be catched in the function self.pb.set_newdata
                raise Exception('get_scans_information', e)

        self.get_derived_volume_attributes()
                
        if set_data:            
            self.scanpair_present = self.check_presence_large_range_large_nyquistvelocity_scanpair(update_scanpairs_indices=True)
            
            if not self.pb.firstplot_performed or (self.crd.scans != self.pb.scans_before and not self.gui.setting_saved_choice):
                # Initialize self.selected_scanangles for all panels when not self.pb.firstplot_performed
                self.update_selected_scanangles(update_allpanels=not self.pb.firstplot_performed)
            
            if self.selected_scanangles != self.selected_scanangles_before:
                #Is only updated in the case of purposeful scan changes, i.e. changes in scans caused by pressing UP/DOWN or a number key, or by
                #pressing F1-F12 for a saved panel choice, or when the scans change during a change in panels (see function
                #self.pb.change_panels).
                self.time_last_purposefulscanchanges = pytime.time()
                
            #Ensure that the selected scans are available for panellist. This might not be the case after a change of radar/dataset
            panellist = self.pb.panellist if self.pb.firstplot_performed else range(self.pb.max_panels)
            max_scan = len(self.scanangles_all['z'])
            if not 1 in self.scanangles_all['z']:
                print('set_max_scan', max_scan, self.scannumbers_all, self.scanangles_all)
                raise Exception('max scan should not be 0!!!!!!!!')
            for j in panellist:
                self.crd.scans[j] = min(self.crd.scans[j], max_scan)
                            
            if not self.attrs_before['scannumbers_all'] or\
            [len(j) for j in self.attrs_before['scannumbers_all']['z'].values()] != [len(j) for j in self.scannumbers_all['z'].values()]:
                #Update self.scannumbers_forduplicates and self.crd.scans if scannumbers_all has changed, because their current values might be invalid
                #for the new scannumbers_all.
                self.update_scannumbers_forduplicates()
                                                 
            if self.crd.scan_selection_mode != 'scan' or self.gui.setting_saved_choice:
                self.check_need_scans_change()
            else:
                for j in self.pb.panellist:
                    if self.scanpair_present and self.crd.scans[j] == 1:
                        self.crd.scans[j] = (1,2)[self.range_nyquistvelocity_scanpairs_indices[j]]
                    elif not self.scanpair_present and self.scanpair_present_before and self.crd.scans[j] in (1,2):
                        self.crd.scans[j] = 1
                                
            
                
    def get_derived_volume_attributes(self):            
        self.radial_range_all, self.scanangles_all_m = {}, {}
        for j in self.scannumbers_all:
            self.radial_range_all[j], self.scanangles_all_m[j] = {}, {}
            for i,a in self.scanangles_all[j].items():
                self.radial_range_all[j][i] = self.radial_bins_all[j][i]*self.radial_res_all[j][i]
                self.scanangles_all_m[j][i] = sum(self.scanangles_all[j][i].values())/len(a) if isinstance(a, dict) else a
        # Indicate for plain products whether they are affected by a double volume. This information is used in nlr_changedata.py and
        # nlr_derivedproducts.py
        for j in gv.plain_products:
            self.scannumbers_all['z'][j] = [0,1] if len(self.scans_doublevolume)>0 and\
                j in gv.plain_products_affected_by_double_volume else [0]         
                
    def scanangle(self, product, scan, duplicate):
        # Helper function for obtaining a scan's scanangle that can handle the presence of a different scanangle per duplicate
        i_p = gv.i_p[product]
        if not scan in self.scanangles_all[i_p] or not scan in self.scanangles_all_m[i_p]:
            print(i_p, self.scanangles_all, self.scanangles_all_m)
        return self.scanangles_all[i_p][scan].get(duplicate, self.scanangles_all_m[i_p][scan]) if\
               isinstance(self.scanangles_all[i_p][scan], dict) else self.scanangles_all[i_p][scan]


    def get_subdataset(self, pv=None, product=None): # pv is product version
        """Gives a string representation of the subdataset (combination of selected directory string and product version). 
        Contains only information that is considered important to distinguish different subdatasets (and thus contains e.g.
        no parts of directory string that are duplicate among all directory strings for this radar_dataset).
        """
        dir_string_list, current_dir_string, n_dirs = self.get_variables(self.crd.radar, self.crd.dataset)[1:]
            
        subdataset = ''
        if n_dirs > 1:
            subpaths = [j.split('/') for j in dir_string_list]
            min_n_subpaths = min(map(len, subpaths))
            subpaths_equal = np.array([len(set(j[i] for j in subpaths)) == 1 for i in range(min_n_subpaths)], dtype='bool')
            n = min_n_subpaths-1 if subpaths_equal.all() else np.where(~subpaths_equal)[0][0]
            _subpaths = [j[n:] for j in subpaths]
            # Remove subpaths that contain variables, unless that gives the same result for all dir_strings
            _subpaths_novar = ['-'.join(i for i in j if not '${' in i) for j in _subpaths]
            _subpaths_novar_unique = set(_subpaths_novar)
            index = dir_string_list.index(current_dir_string)
            if len(_subpaths_novar_unique) == n_dirs:
                subdataset = _subpaths_novar[index]
            else:
                subdataset = '-'.join(_subpaths[index])
            
        if pv is None:
            pv = self.gui.radardata_product_versions[self.radar_dataset]
        i_p = gv.i_p.get(product, None)
        if pv and self.product_versions_datetime and (i_p is None or i_p in self.products_version_dependent):
            _pv = (pv if pv in self.product_versions_datetime else self.product_versions_datetime[0])
            subdataset += '_'*bool(subdataset)+_pv
        return subdataset
    
    def process_products_with_pvs_in_keys(self, mode='store'): #'store' or 'restore'
        # If self.product_versions_in1file=True, then volume attributes for different product versions have been determined all at once,
        # with product keys for different versions distinguished by appending the product version to the key. This function processes 
        # these products_with_pvs_in_keys, depending on the mode. If mode='store', then remove any non-pv key for this product (that doesn't
        # contain the product version). If mode='restore', then restore the non-pv key based on the currently selected product version.
        if not self.product_versions_in1file:
            return
        
        products_with_pvs_in_keys = set([j[0] for j in self.scannumbers_all if any(j.endswith(i) for i in self.product_versions_datetime)])    
        pv = self.gui.radardata_product_versions[self.radar_dataset]
        pv = pv if pv in self.product_versions_datetime else self.product_versions_datetime[0]
        for p in products_with_pvs_in_keys:
            for j in gv.volume_attributes_p:
                if mode == 'store' and p in self.__dict__[j]:
                    del self.__dict__[j][p]
                else:
                    self.__dict__[j][p] = self.__dict__[j][p+' '+pv]

    def store_volume_attributes(self):
        for j in self.scannumbers_all['z'].copy():
            # Remove keys for plain products if they are present. They lead to errors with sorting keys, and are not needed
            # since they are added afterwards.
            if isinstance(j, str):
                del self.scannumbers_all['z'][j]
        
        self.process_products_with_pvs_in_keys('store')
        
        current_attrs = copy.deepcopy([eval('self.'+j, {'self': self}) for j in gv.volume_attributes_save])
        self.compress_volume_attributes(current_attrs)
        
        variable_attrs = []
        for i,j in enumerate(gv.volume_attributes_save):
            if j in self.variable_attributes:
                variable_attrs.append(current_attrs[i])
                current_attrs[i] = 'variable'
                        
        current_attrs_ID = None; ID = 0
        for ID in self.attributes_descriptions[self.radar_dataset]:
            attrs = self.attributes_descriptions[self.radar_dataset][ID]
            if attrs == current_attrs:
                #The series of attributes is the same as one of the series that is already stored, and the corresponding ID is used.
                current_attrs_ID = ID; break
        if not current_attrs_ID:
            #The series of attributes is new, and a new ID is used, that is 1 higher than the largest existing one.
            current_attrs_ID = ID+1
            self.attributes_descriptions[self.radar_dataset][current_attrs_ID] = current_attrs
                                
        # if self.product_versions_in1file=True, then volume attributes for all product versions have been determined at once, hence no need
        # to use a different subdataset for different product versions.
        subdataset = self.get_subdataset(pv='' if self.product_versions_in1file else None)
        #Insert the attributes ID in self.attributes_IDs
        ft.create_subdicts_if_absent(self.attributes_IDs[self.radar_dataset], [subdataset, self.crd.date])
        data_selected_startazimuth = self.gui.data_selected_startazimuth if self.crd.radar in gv.radars_with_adjustable_startazimuth else 0
        self.attributes_IDs[self.radar_dataset][subdataset][self.crd.date][self.crd.time] = [current_attrs_ID,self.total_files_size,data_selected_startazimuth]
        
        if variable_attrs:
            ft.create_subdicts_if_absent(self.attributes_variable[self.radar_dataset], [subdataset, self.crd.date])
            self.attributes_variable[self.radar_dataset][subdataset][self.crd.date][self.crd.time] = variable_attrs
            
        self.process_products_with_pvs_in_keys('restore')
                        
    def restore_volume_attributes(self):
        subdataset = self.get_subdataset(pv='' if self.product_versions_in1file else None)
        try:
            attrs_ID,saved_total_files_size,saved_data_selected_startazimuth = self.attributes_IDs[self.radar_dataset][subdataset][self.crd.date][self.crd.time]
            data_selected_startazimuth = self.gui.data_selected_startazimuth if self.crd.radar in gv.radars_with_adjustable_startazimuth else 0
            if saved_total_files_size!=self.total_files_size or saved_data_selected_startazimuth!=data_selected_startazimuth:
                return False

            attrs = copy.deepcopy(self.attributes_descriptions[self.radar_dataset][attrs_ID])
            i_variable_attrs = [i for i,attr in enumerate(attrs) if attr == 'variable']
            if i_variable_attrs:
                variable_attrs = copy.deepcopy(self.attributes_variable[self.radar_dataset][subdataset][self.crd.date][self.crd.time])
                for i,j in enumerate(i_variable_attrs):
                    attrs[j] = variable_attrs[i]
            self.decompress_volume_attributes(attrs)
                 
            scannumbers_all = attrs[gv.volume_attributes_save.index('scannumbers_all')]
            attributes_available = all(i in (j[0] for j in scannumbers_all) for i in gv.i_p.values())
            if attributes_available:
                for i,j in enumerate(gv.volume_attributes_save):
                    self.__dict__[j] = attrs[i]
                    
            self.process_products_with_pvs_in_keys('restore')
            
            return attributes_available
        except Exception:
            return False

    def _merge_repeated_values(self, dic):
        keys, vals = np.array(list(dic), dtype=object), np.array(list(map(str, dic.values())))
        _, unique_indices = np.unique(vals, return_index=True)
        unique = [i for i in keys if i in keys[unique_indices]] # Preserve original order in keys
        # Compare string-converted values (vals) instead of dic values, since when dic values have
        # different types it can be that the first comparison yields False, while the second yields True.
        # This has been observed at least once with values that are 'equal' except for their type (float32 vs float64).
        hits = {k:keys[vals[i] == vals] for i,k in enumerate(keys) if k in unique}
                    
        for i,j in hits.items():
            diff = [j[k]-j[k-1] if all(type(l) == int for l in (j[k], j[k-1])) else 0 for k in range(1, len(j))]
            key = str(j[0])
            for k in range(len(diff)):
                if diff[k] == 1:
                    key = (key[:key.index(str(j[k]))] if k and diff[k-1] == 1 else key+'-')+str(j[k+1])
                else:
                    key += ','+str(j[k+1])
            if key.isdigit():
                key = int(key)
            dic[key] = dic[i]
            if not key == i:
                for k in j:
                    del dic[k]
                    
    def _split_merged_values(self, dic):
        for i,j in dic.copy().items():
            if isinstance(i, str) and any(k in i for k in (',', '-')):
                split_keys = i.split(',')
                for k,l in enumerate(split_keys.copy()):
                    if '-' in l:
                        n1, n2 = l.split('-')
                        split_keys[k] = list(range(int(n1), int(n2)+1))
                    else:
                        split_keys[k] = [int(l) if l.isdigit() else l]
                for k in sum(split_keys, []):
                    dic[k] = j
                del dic[i]
                
    def compress_volume_attributes(self, attributes):
        for attr in attributes:
            if not isinstance(attr, dict):
                continue
            products = list(attr)
            if not products or not isinstance(products[0], str):
                # Skip the attributes that are not product-dependent, such as the Nyquist velocities
                if products:
                    self._merge_repeated_values(attr)
                continue
            
            for p1 in products:
                hits = [p for p in products[:products.index(p1)] if attr[p1] == attr[p]]
                if hits:
                    attr[p1] = hits[0]
            products = [p for p in products if isinstance(attr[p], dict)]
            
            for p1 in products:
                for i in attr[p1]:
                    hits = [p for p in products[:products.index(p1)] if i in attr[p] and attr[p1][i] == attr[p][i]]
                    if hits:
                        attr[p1][i] = hits[0]
                        
            for p1 in products:
                self._merge_repeated_values(attr[p1])
            self._merge_repeated_values(attr)
            
    def decompress_volume_attributes(self, attributes):
        for i, attr in enumerate(attributes):
            if not isinstance(attr, dict):
                continue
            self._split_merged_values(attr)
            products = list(attr)
            if not isinstance(products[0], str):
                attributes[i] = self._sort_attributes_by_key(attr)
                continue
                    
            for p1 in products:
                if isinstance(attr[p1], str):
                    p2 = attr[p1]
                    attr[p1] = attr[p2].copy()
                else:
                    self._split_merged_values(attr[p1])
                    for j in attr[p1]:
                        if attr[p1][j] in products:
                            p2 = attr[p1][j]
                            attr[p1][j] = attr[p2][j]
                attr[p1] = self._sort_attributes_by_key(attr[p1])
                
    def _sort_attributes_by_key(self, attr):
        sorted_keys = sorted(attr)
        if not sorted_keys == list(attr):
            values = [attr[i] for i in sorted_keys]
            attr = dict(zip(sorted_keys, values))
        return attr
        

    def generate_dataspecs_string(self,product,productunfiltered,polarization,apply_dealiasing,panel,proj=None):
        # proj only needs to be specified for plain products
        #Include total_files_size, because a change in the number of scans in a volume means that a particular scan can get
        #different volume attributes compared to what was previously the case.
        radar_dataset = self.get_radar_dataset()
        subdataset = self.get_subdataset(product=product)
        dataspecs_string = radar_dataset+subdataset
                
        data_selected_startazimuth = self.gui.data_selected_startazimuth if self.crd.radar in gv.radars_with_adjustable_startazimuth else 0
        # dataspecs_string += str(self.total_files_size)+'_'+self.crd.date+self.crd.time+'_'+product+'_'+str(data_selected_startazimuth)
        dataspecs_string += '_'+self.crd.date+self.crd.time+'_'+product+'_'+str(data_selected_startazimuth)
        
        # It is assumed that any change in volume/scan content is reflected in a change in scannumbers_all
        scannumbers_all = self.scannumbers_all[gv.i_p[product]]
        scan = self.crd.scans[panel]
        duplicate = self.scannumbers_forduplicates[product if product in gv.plain_products else scan]
        if product in gv.plain_products:
            if product in gv.plain_products_with_parameters:
                dataspecs_string+= '_'+str(self.gui.PP_parameter_values[product][self.gui.PP_parameters_panels[panel]])
            dataspecs_string+= '_'+str(productunfiltered)+'_'+polarization
            dataspecs_string+= '_'+str(scannumbers_all)+str(duplicate)
            dataspecs_string+= '_'+proj
            if proj == 'car':
                dataspecs_string+= '_'+str(self.gui.stormmotion)+'_'+str(self.gui.cartesian_product_res)+'_'+str(self.gui.cartesian_product_maxrange)
        else: 
            if product == 'v':
                dataspecs_string += '_'+str(apply_dealiasing) + '_' + self.gui.dealiasing_setting + '_' + str(self.gui.dealiasing_dualprf_n_it)
            dataspecs_string+= '_'+str(productunfiltered)+'_'+polarization
            dataspecs_string+= '_'+str(scannumbers_all[scan][duplicate])
        return dataspecs_string
    
    def get_dataspecs_string_panel(self, j, return_params=False): #j is the panel
        product = self.crd.products[j]
        if product in gv.products_with_tilts_derived_nosave:
            # In this case import product is saved instead of actual product, since the latter can be cheaply calculated from import product.
            product = gv.i_p[product]
        productunfiltered = self.crd.using_unfilteredproduct[j]
        polarization = {True:'V', False:'H'}[self.crd.using_verticalpolarization[j]]
        apply_dealiasing = self.crd.apply_dealiasing[j]
        proj = self.dp.meta_PP[product]['proj'] if product in gv.plain_products else None
        dataspecs_string = self.generate_dataspecs_string(product,productunfiltered,polarization,apply_dealiasing,j,proj)
        if return_params:
            return dataspecs_string, productunfiltered, polarization, apply_dealiasing, proj
        else:
            return dataspecs_string
        
    def store_data_in_memory(self, j): #j is the panel
        product = self.crd.products[j]
        dataspecs_string, productunfiltered, polarization, apply_dealiasing, proj = self.get_dataspecs_string_panel(j, True)
        # if not data changed we can still use self.crd.using_verticalpolarization[j] etc due to dataspecs_string_requested below
        if self.data_changed[j]:  
            self.stored_data[dataspecs_string] = {'last_use_time':pytime.time(),'data':self.data[j].copy(),'data_azimuth_offset':self.data_azimuth_offset[j],'data_radius_offset':self.data_radius_offset[j],'scantime':self.scantimes[j],'using_unfilteredproduct':self.crd.using_unfilteredproduct[j],'using_verticalpolarization':self.crd.using_verticalpolarization[j]}
        else:
            self.stored_data[dataspecs_string] = {'last_use_time':pytime.time(),'data':np.zeros((1,1))}
            
        if product in gv.plain_products:
            self.stored_data[dataspecs_string]['meta_PP'] = self.dp.meta_PP[product].copy()
                
        if productunfiltered != self.crd.productunfiltered[j] or polarization != self.crd.polarization[j]:
            # In this case the new data array is both stored for the actual and the requested combination of productunfiltered and 
            # polarization, but for the requested combination only the key for the actual combination is given as value.
            # This key can then be used to obtain the desired dictionary with data
            dataspecs_string_requested = self.generate_dataspecs_string(product,self.crd.productunfiltered[j],self.crd.polarization[j],apply_dealiasing,j,proj)
            self.stored_data[dataspecs_string_requested] = dataspecs_string    
            
        stored_data_size = np.sum([float(sys.getsizeof(j['data'])) for j in self.stored_data.values() if not isinstance(j, str)])
        #Convert to float, since the number might exceed the maximum value for 32-bit ints.
        while stored_data_size>1e9*self.gui.max_radardata_in_memory_GBs:
            #Remove the dataset with the most outdated last_use_time
            index = np.argmax([pytime.time()-j['last_use_time'] for j in self.stored_data.values() if not isinstance(j, str)])
            most_outdated_last_use_time_key = list(self.stored_data)[index]
            # Also remove possible other keys that map onto the key that will be removed
            keys_remove = [most_outdated_last_use_time_key]+\
                [j for j in self.stored_data if type(self.stored_data[j]) == str and self.stored_data[j] == most_outdated_last_use_time_key]
            for key in keys_remove:
                del self.stored_data[key]
            stored_data_size = np.sum([float(sys.getsizeof(j['data'])) for j in self.stored_data.values() if not isinstance(j, str)])
        
    def check_presence_data_in_memory(self,product,productunfiltered,polarization,apply_dealiasing,panel):
        if self.gui.max_radardata_in_memory_GBs <= 0:
            return
        derived_nosave = product in gv.products_with_tilts_derived_nosave
        if derived_nosave:
            # In this case import product is saved instead of actual product, since the latter can be cheaply calculated from import product.
            product = gv.i_p[product]
        
        proj = None
        if product in gv.plain_products:
            proj = 'car' if self.gui.stormmotion[1] != 0. and product in gv.plain_products_correct_for_SM else 'pol'
        try:
            dataspecs_string = self.generate_dataspecs_string(product,productunfiltered,polarization,apply_dealiasing,panel,proj)
        except Exception:
            # Can happen when requested scan or duplicate is unavailable
            return
        
        if dataspecs_string in self.stored_data:
            data_dict = self.stored_data[dataspecs_string]
            if isinstance(data_dict, str): # In this case data_dict actually is a key that maps onto another data_dict
                data_dict = self.stored_data[data_dict]
            last_use_time = data_dict['last_use_time']
            if last_use_time<self.pb.cmap_lastmodification_time[product] or last_use_time<self.gui.time_last_removal_volumeattributes:
                return False #In this case the color map has been modified in the mean time, implying that
                #self.pb.mask_values_int[product] could have been changed. If this is the case then the number of masked elements
                #will likely change, which requires an update of the data.
            
            # An empty array has been saved to memory when attempts to import data were unsuccessful. In this case don't update the data
            # array and attributes, but also don't re-import data, which requires that self.import_data[panel] is still set to False.
            if data_dict['data'].size > 1:
                self.data[panel] = data_dict['data']
                if derived_nosave:
                    # Make a copy of data, since otherwise data of import product will be altered when calculating derived product
                    self.data[panel] = self.data[panel].copy()
                self.data_azimuth_offset[panel] = data_dict['data_azimuth_offset']
                self.data_radius_offset[panel] = data_dict['data_radius_offset']
                self.scantimes[panel] = data_dict['scantime']
                self.crd.using_unfilteredproduct[panel] = data_dict['using_unfilteredproduct']
                self.crd.using_verticalpolarization[panel] = data_dict['using_verticalpolarization']
                if product in gv.plain_products:
                    self.dp.meta_PP[product] = data_dict['meta_PP'].copy()
                
                self.data_changed[panel] = True
            self.import_data[panel] = False
            data_dict['last_use_time'] = pytime.time()
            
            
    def convert_dtype_float_to_uint(self,data,product,inverse=False):
        """Convert the data to unsigned integers.
        For an explanation of the process of converting floating point data values to unsigned integers, see nlr_globalvars.py.
        """
        n_bits = gv.products_data_nbits[product]
        p_lim = gv.products_maxrange[product]
        pm_lim = gv.products_maxrange_masked[product]
            
        if not inverse:
            data_notmasked = (data!= self.pb.mask_values[product])
            #These 2 lines are necessary, to assure that no errors arise when p_lim does not capture the whole range of 
            #product values.
            data[(data<p_lim[0]) & data_notmasked] = p_lim[0]
            data[data>p_lim[1]] = p_lim[1]
            new_data = np.full(data.shape, self.pb.mask_values_int[product], f'uint{n_bits}')
            new_data[data_notmasked] = ft.convert_float_to_uint(data[data_notmasked],n_bits,pm_lim)
        else:
            data_notmasked = data != self.pb.mask_values_int[product]
            new_data = np.full(data.shape, self.pb.mask_values[product], 'float32')
            new_data[data_notmasked] = ft.convert_uint_to_float(data[data_notmasked],n_bits,pm_lim)
        return new_data
    
    def apply_binfilling(self,j): #j is the panel
        """If reflectivity is shown, then fill empty radar bins if at least 2 neighbouring bins are non-empty, in order to reduce ugly interpolation effects.
        This is only performed for bins with a reflectivity >= 20 dBZ, to prevent enlarging of areas with low reflectivity.
        """
        initial_dtype = self.data[j].dtype
        if initial_dtype != 'float32':
            self.data[j] = self.data[j].astype('float32')
        for i in range(3):
            data_mask = self.data[j] == 0. if initial_dtype != 'float32' else self.data[j] == self.pb.mask_values[self.crd.products[j]]
            neighbours = ft.get_window_sum((data_mask == False).astype('float32'), [0,1,0])
            bins_to_fill = (data_mask) & (neighbours >= 2)
            
            if initial_dtype == 'float32':
                self.data[j][data_mask] = 0
            self.data[j][bins_to_fill] = ft.get_window_sum(self.data[j], [0,1,0])[bins_to_fill] / neighbours[bins_to_fill]
            if initial_dtype == 'float32':
                self.data[j][(data_mask) & (bins_to_fill == False)] = self.pb.mask_values[self.crd.products[j]]
                
            unfill = np.zeros(self.data[j].shape, dtype='bool')
            if initial_dtype == 'float32':
                unfill[bins_to_fill] = self.data[j][bins_to_fill] < 20
            else:
                unfill[bins_to_fill] = self.data[j][bins_to_fill] < ft.convert_float_to_uint(20, gv.products_data_nbits[self.crd.products[j]], gv.products_maxrange_masked[self.crd.products[j]])
            self.data[j][unfill] = 0 if initial_dtype != 'float32' else self.pb.mask_values[self.crd.products[j]]
        if initial_dtype != 'float32':
            self.data[j] = self.data[j].astype(initial_dtype)

    def get_data(self,panellist,change_radar,change_dataset, set_data):
        # from cProfile import Profile
        # profiler = Profile()
        # profiler.enable() 
        """This function checks for each panel if the selected product is available, and if so, then it imports the data.
        When the product is in plain_products, then the import of data takes place in get_data_multiple_scans.
        
        Returns self.data_changed, which specifies for each panel whether the data has been changed.
        
        if set_data=False, then only volume attributes are determined and returned. The volume attributes that correspond to the radar volume that is 
        currently displayed are restored after retrieving the desired volume attributes.
        
        If set_data=False, then this function returns retrieved_attrs. If set_data=True, then it returns self.data_changed!
        """        
        self.changing_radar = change_radar; self.changing_dataset = change_dataset
        
        self.select_files_datetime() #This updates self.files_datetime
        # Calculate total_files_size at the start of get_data. It's important to not wait with this until it's actually needed (e.g. in 
        # self.generate_dataspecs_string), since it's possible that the total size of files increases during the actions in get_data (like
        # when downloading current data), implying that a later determined files size might be larger than that used in earlier actions. 
        # And since total_files_size is used to decide whether updates are needed, it could then happen that it is incorrectly decided that
        # no update is needed. By calculating it at the start here it can happen that total_files_size is smaller than what's used in later
        # actions, but this only has as disadvantage that an unnecessary update is performed. Which is better than e.g. not updating 
        # half-finished scans.
        # self.total_files_size should also be used elsewhere where information about total files size is needed.
        self.total_files_size = self.get_total_volume_files_size()
        
        self.radar_dataset = self.get_radar_dataset()
        
        if set_data:
            self.radar_dataset_before = self.crd.before_variables['radar']
            
            if self.crd.before_variables['radar'] in gv.radars_with_datasets:
                self.radar_dataset_before+= ' '+self.crd.before_variables['dataset'] 
        
        self.get_scans_information(set_data)
        
        if not set_data:
            retrieved_attrs = {j:self.__dict__[j] for j in gv.volume_attributes_all}
            self.restore_previous_attributes()
            return retrieved_attrs, self.total_files_size
        
        #Update self.scannumbers_forduplicates_radars[self.crd.radar], which is used in the function self.update_scannumbers_forduplicates
        self.scannumbers_forduplicates_radars[self.crd.radar] = self.scannumbers_forduplicates.copy()
            
        """It is first checked whether data is present in the memory for the selected product, productunfiltered, polarization and apply_dealiasing. 
        If not, then the function self.source_classes[self.data_source()].get_data is called.
        """
        self.data_changed = {j:False for j in panellist}
        self.import_data = {j:True for j in panellist}
        # from cProfile import Profile
        # profiler = Profile()
        # profiler.enable() 

        for j in panellist:
            #By setting it here to zero, this variable needs only be updated in nlr_importdata.py when it deviates from zero
            self.data_azimuth_offset[j] = 0.
            self.data_radius_offset[j] = 0.
            
            self.check_presence_data_in_memory(self.crd.products[j],self.crd.productunfiltered[j],self.crd.polarization[j],self.crd.apply_dealiasing[j],j)
            
            i_p, scan = gv.i_p[self.crd.products[j]], self.crd.scans[j]
            duplicate = self.scannumbers_forduplicates[scan]
            if duplicate >= len(self.scannumbers_all[i_p].get(scan, [])):
                # In this case the requested duplicate scan is not available for this product
                self.import_data[j] = False

        self.update_volume_attributes = False
        
        panellist_import = [j for j in panellist if self.import_data[j]]
        if panellist_import:
            panellist_plain = [j for j in panellist_import if self.crd.products[j] in gv.plain_products]
            panellist_notplain = [j for j in panellist_import if not self.crd.products[j] in gv.plain_products]
            
            for j in panellist_import:
                self.crd.using_unfilteredproduct[j] = False
                self.crd.using_verticalpolarization[j] = False
            
            # Order panellist_import in such a way that panels that import the same scan are processed consecutively. 
            # This allows some of the import code in nlr_importdata.py to efficiently obtain data for multiple products
            # self.scannumbers_panels is used in nlr_importdata.py, at least for NEXRAD L2 data
            self.scannumbers_panels = {}
            for j in panellist_notplain:
                i_p, scan = gv.i_p[self.crd.products[j]], self.crd.scans[j]
                if scan in self.scannumbers_all[i_p]:
                    # Convert to string, as self.scannumbers_all[i_p][scan] might contain entities that can't be sorted (e.g. None)
                    self.scannumbers_panels[j] = str(self.scannumbers_all[i_p][scan])
            panellist_notplain = sorted(self.scannumbers_panels, key=self.scannumbers_panels.get)
            for j in panellist_notplain:
                ft.create_subdicts_if_absent(self.scanangles, [self.crd.radar, self.crd.date])
                try:
                    # Mono PRF dealiasing normally is performed below, except when a special treatment is required, e.g. when more than one 
                    # Nyquist velocity is used for the scan. In that case the function self.perform_mono_prf_dealiasing is called in nlr_importdata.py,
                    # after which self.mono_prf_dealiasing_performed is set to True in this function.
                    self.mono_prf_dealiasing_performed = False
                    
                    if j in self.scantimes:
                        before = self.data[j], self.scantimes[j], self.data_azimuth_offset[j], self.data_radius_offset[j]
                    self.source_classes[self.data_source()].get_data(j)
                    
                    if self.crd.requesting_latest_data and not self.changing_radar and 'before' in locals():
                        # When plotting recent data, check whether data is available for the azimuth of the panel's center. If not,
                        # go back to previous data. This check is useful for real-time data streams that provide partial scans, e.g. for NWS
                        panel_center_xy = self.pb.screencoord_to_xy(self.pb.panel_centers[j])
                        azimuth = ft.azimuthal_angle(panel_center_xy, deg=True)
                        row = int(azimuth//1)
                        if j in self.pb.data_attr['scantime'] and self.scantimes[j] != self.pb.data_attr['scantime'][j] and\
                        np.all(self.data[j][row] == self.pb.mask_values[self.crd.products[j]]):
                            print('back to before', j)
                            self.data[j], self.scantimes[j], self.data_azimuth_offset[j], self.data_radius_offset[j] = before
                            continue
                    
                    v_nyquist = self.nyquist_velocities_all_mps.get(self.crd.scans[j], None)
                    if gv.i_p[self.crd.products[j]] == 'v' and self.crd.apply_dealiasing[j] and 'Unet VDA' in self.gui.dealiasing_setting and\
                    self.data[j].dtype == 'float32' and v_nyquist not in (None, 999.) and v_nyquist <= self.gui.dealiasing_max_nyquist_vel and\
                    not self.mono_prf_dealiasing_performed:
                        # A Nyquist velocity of 999. indicates that it could not be determined, while it is at least high enough to include
                        # the scan in operations that require a sufficiently high Nyquist velocity.
                        self.data[j] = self.perform_mono_prf_dealiasing(j, self.data[j])
                except Exception as e:
                    print(e, 'get_data, panel '+str(j))
                    traceback.print_exception(type(e), e, e.__traceback__)
                    continue

                self.data_changed[j] = True

            if panellist_plain:
                # To do: think of error handling, and data_changed
                self.dp.calculate_plain_products(panellist_plain)
                for j in panellist_plain:
                    self.data_changed[j] = True
           
        for j in panellist_import:    
            product = self.crd.products[j]
            if product in gv.products_possibly_exclude_lowest_values:
                # Hide product values that are below the minimum value that the user wants to view
                if self.data[j].dtype.name.startswith('float'):
                    min_value, mask_value = self.pb.data_values_colors[product][0], self.pb.mask_values[product]
                else:
                    min_value, mask_value = self.pb.data_values_colors_int[product][0], self.pb.mask_values_int[product]
                self.data[j][self.data[j] < min_value] = mask_value
            if self.data[j].dtype.name.startswith('float'):
                self.data[j] = self.convert_dtype_float_to_uint(self.data[j], product)
            
            # When self.data_changed[j]=False an empty array (created in self.store_data_in_memory) will be saved to memory, to indicate that no
            # data has been obtained. In the past nothing was saved to memory at all, but this had as disadvantage that new requests of the same
            # data would lead to renewed attempts to import, which were a waste of time.
            if self.gui.max_radardata_in_memory_GBs > 0:
                self.store_data_in_memory(j)

        for j in (i for i in panellist if self.data_changed[i]):
            if self.crd.products[j] in gv.products_with_tilts_derived:
                self.calculate_derived_with_tilts(j)
            
            if self.pb.use_interpolation and self.crd.products[j] in gv.products_with_interpolation_and_binfilling:
                self.apply_binfilling(j)
                    
        if self.update_volume_attributes:
            # Some volume attributes have apparently been updated, and they also have to be updated in the attribute dictionaries.
            self.store_volume_attributes()
            self.get_derived_volume_attributes()
                  
        self.selected_scanangles_before = self.selected_scanangles.copy()
        # profiler.disable()
        # import pstats
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(20)  
        return self.data_changed, self.total_files_size
    
    def perform_mono_prf_dealiasing(self, j, data, vn=None, azis=None, da=None): # j is the panel
        if not hasattr(self, 'vda'):
            from dealiasing.unet_vda.unet_vda import Unet_VDA
            self.vda = Unet_VDA()
        data[data == self.pb.mask_values['v']] = np.nan
        t = pytime.time()
        vn = self.nyquist_velocities_all_mps[self.crd.scans[j]] if vn is None else vn
        data = self.vda(data, vn, azis, da, extra_dealias='extra' in self.gui.dealiasing_setting)
        self.mono_prf_dealiasing_performed = True
        print(pytime.time()-t, 't unet vda')
        return data
    
    def calculate_derived_with_tilts(self, j): # j is the panel
        """Currently only calculates SRV.
        Also, SRV is calculated from uint velocity data (which is dtype in which velocity is available at this point), 
        since this is both well possible and clearly cheaper than first converting uint to float and then back after calculation.
        This is taken into account in the function dt.calculate_srv_array.
        When later calculating more products than just SRV, there might be the desire to store these products into memory. 
        This might then be done in this function too.
        """
        product, i_p = self.crd.products[j], gv.i_p[self.crd.products[j]]
        mask_value_ip, mask_value_p = self.pb.mask_values_int[i_p], self.pb.mask_values_int[product]
        data_mask = self.data[j] == mask_value_ip
        if product == 's':
            self.data[j] = dt.calculate_srv_array(self.data[j], self.gui.stormmotion, self.data_azimuth_offset[j])
        self.data[j][data_mask] = mask_value_p

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """Function that imports data for products that require data from more than one scan
        """
        return self.source_classes[self.data_source()].get_data_multiple_scans(product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
    
    def restore_previous_attributes(self):
        for j in self.attrs_before:
            self.__dict__[j] = self.attrs_before[j]
        
    def restore_previous_attributes_radar(self):
        for j in self.determined_volume_attributes_radars[self.crd.radar].keys():
            self.__dict__[j] = self.determined_volume_attributes_radars[self.crd.radar][j]
        #All attributes are now the same as for the last succesfull trial for self.crd.radar
    
    
    def data_source(self,radar = None):
        """Gives the data source for a given input radar, or if not given, the data source for the current radar; self.crd.radar.
        """
        datasource_radar = radar if radar!= None else self.crd.radar
        return gv.data_sources[datasource_radar]
    
    def update_scannumbers_forduplicates(self):
        """When going back to the previous plot, the previous values for scannumbers_forduplicates are restored in the function 
        self.crd.back_to_previous_plot.
        """
        if not self.crd.going_back_to_previous_plot:
            self.scannumbers_forduplicates = {i:j for i,j in self.scannumbers_forduplicates.items() if i in self.scannumbers_all['z']}
            for i,j in self.scannumbers_all['z'].items():
                if self.crd.lrstep_beingperformed:
                    if self.scannumbers_forduplicates.get(i, 0)+1 > len(j):
                        self.scannumbers_forduplicates[i] = len(j)-1
                    elif not i in self.scannumbers_forduplicates or self.crd.desired_timestep_minutes() < self.crd.volume_timestep_m:
                        self.scannumbers_forduplicates[i] = 0 if self.crd.lrstep_beingperformed == 1 else len(j)-1
                elif self.crd.requesting_latest_data:
                    self.scannumbers_forduplicates[i] = len(j)-1
                else:
                    #Use the last values for scannumbers_forduplicates that were used for this radar, if they are valid.
                    #self.scannumbers_forduplicates_radars[self.crd.radar] gets updated in self.pb.set_newdata each time that function is called.
                    previous_val_radar = self.scannumbers_forduplicates_radars.get(self.crd.radar, {self.crd.radar:{}}).get(i, 100)
                    self.scannumbers_forduplicates[i] = min(len(j)-1, previous_val_radar)  
                  
                    
    def update_selected_scanangles(self, update_allpanels=False):
        panellist = range(self.pb.max_panels) if update_allpanels else self.pb.panellist
        for j in panellist:
            #'z' is always available as key for the volume attributes
            product = gv.i_p[self.crd.products[j]] if self.scanangles_all_m[gv.i_p[self.crd.products[j]]] else 'z'
            max_scanangle = max(self.scanangles_all_m[product].values())
            lowest_scan = (1, 2)[self.range_nyquistvelocity_scanpairs_indices[j]] if self.scanpair_present else 1
            if self.crd.scans[j] == lowest_scan:
                self.selected_scanangles[j] = 0.
            else:
                self.selected_scanangles[j] = self.scanangles_all_m[product].get(self.crd.scans[j], max_scanangle)          
                    
    def get_scanangles_allproducts(self, scanangles_all):
        """This function combines the dictionaries with per product the scanangles for all scans, given in scanangles_all, into one dictionary, that contains
        all the different scanangles that are available for the scans. If all scanangles are available for all products, then the returned dictionary is equal
        to scanangles_all[product], where product is any of the availabe products.
        """
        scanangles_allproducts = {}
        products = [p for p in scanangles_all if len(scanangles_all[p])]
        for j in scanangles_all['z']: #scanangles_all['z'] should contain keys for all available scans
            if j==1:
                scanangles_allproducts[j] = np.min([scanangles_all[p][1] for p in products])
            else:
                diff_greaterthanzero = [scanangles_all[p][j]-scanangles_allproducts[j-1] for p in products if scanangles_all[p][j]-scanangles_allproducts[j-1]>0.]
                if len(diff_greaterthanzero):
                    products_diff_greaterthanzero = [p for p in products if scanangles_all[p][j]-scanangles_allproducts[j-1]>0.]
                    p = products_diff_greaterthanzero[np.argmin(diff_greaterthanzero)]
                else: 
                    p = 'z'
                scanangles_allproducts[j] = scanangles_all[p][j]
        return scanangles_allproducts
        
    def get_panel_center_heights(self, dist_to_radar_panels, scanangles, scanangles_all, scanpair_present, panellist=None):
        panellist = self.pb.panellist if panellist is None else panellist
        heights = {}
        for j in panellist:
            lowest_scan = (1, 2)[self.range_nyquistvelocity_scanpairs_indices[j]] if scanpair_present else 1
            if scanangles[j] == scanangles_all[lowest_scan]:
                # Set height equal to 0 when the lowest scan is shown, such that this scan will keep being shown
                heights[j] = 0.
            else:
                heights[j] = ft.var1_to_var2(dist_to_radar_panels[j], scanangles[j], 'gr+theta->h')
        return heights
    def manually_set_panel_center_heights(self, heights):
        # This function is called in self.gui.set_choice, in order to use selected heights from a saved choice.
        self.panel_center_heights = {'time':pytime.time(), 'heights':heights}
                                
    def check_need_scans_change(self):
        """This function checks whether it is desired to switch from scans. This could be the case when the radar volume changes, e.g. because 
        of a change of radar or dataset. Or it can be because it is desired to keep scan beam height at panel center as constant as possible, 
        which can require a change of scans when switching from radar or when translating view with storm motion.
        """
        move_view_with_storm = self.gui.use_storm_following_view and self.pb.firstplot_performed and not self.gui.setting_saved_choice and\
                               (self.crd.process_datetimeinput_running or self.crd.lrstep_beingperformed)
        choose_nearest_height = self.crd.scan_selection_mode == 'height' and (self.changing_radar or move_view_with_storm or 
                                                                              self.gui.setting_saved_choice)
        try:
            s, sb = self.scanangles_all_m, self.attrs_before['scanangles_all_m']
            scanangles_all_changed = any([any([abs(sb[p][i]-s[p][i]) > 0.1 for i in s[p]]) for p in s])
        except Exception:
            scanangles_all_changed = True
        choose_nearest_scanangle = not choose_nearest_height and (scanangles_all_changed or self.gui.setting_saved_choice)
        #If self.gui.setting_saved_choice, then it is always desired to choose the scan with the appropriate Nyquist velocity 
        #(that is not too low when viewing a velocity-related product).
        if (self.pb.firstplot_performed or self.gui.setting_saved_choice) and (choose_nearest_height or choose_nearest_scanangle) and\
        len(self.scanangles_all['z']) > 1:
            """When choose_nearest_height == True this function chooses the scans in such a way that the beam elevation at the center
            of the screen is closest to the value that it was before, for the previous radar or time. 
            If this is not the case, then this function simply chooses the scan whose scanangle is closest to the selected scanangle.
            """

            if choose_nearest_height: 
                self.pb.get_corners()
                if move_view_with_storm:
                    # When using storm-centering the translation of the view only takes place after calling this function, so self.pb.corners
                    # doesn't yet take into account this new translation. This translation is approximately performed here (the real
                    # translation depends on scan time differences, not on volume time differences).
                    dt_before = self.gui.current_case['datetime'] if self.gui.switch_to_case_running else self.crd.before_variables['datetime']
                    delta_time = ft.datetimediff_s(dt_before, self.crd.date+self.crd.time)
                    if abs(delta_time) <= self.pb.max_delta_time_move_view_with_storm:
                        for j in self.pb.panellist:
                            self.pb.corners[j] += self.pb.translation_dist_km(delta_time)
        
            #scanangles_all contains all scanangles for which data is present in the volume. It is determined in this way, because it is possible that the number
            #of scans differs per product, and therefore also the number of scanangles. If this is the case, then self.scanangles_all_m[product] refers for these 
            #scans to the nearest other scan (in terms of scanangle) for which data is present. 
            #If self.scanangles_all_m[product] contains less scanangles than scanangles_all, then using self.scanangles_all_m[product] could lead to finding a
            #suboptimal 'nearest' scan. When viewing product this would not matter, but when viewing another product for which other scanangles are available,
            #this could lead to displaying a suboptimal scan, which is not desired.
            scanangles_all = self.get_scanangles_allproducts(self.scanangles_all_m)
            scanangles_all_values = np.array(list(scanangles_all.values()))
            if self.radar_dataset_before in self.scans_radars:
                try:
                    scanangles_all_before = self.get_scanangles_allproducts(self.scans_radars[self.radar_dataset_before]['scanangles_all'])
                except Exception as e:
                    print(e, self.radar_dataset_before, self.scans_radars[self.radar_dataset_before]['scanangles_all'])
                    raise Exception(1/0)
            
            """In case of a change of radar or dataset (i.e. not when using storm-centered view with self.crd.lrstep_beingperformed):
            If the scans haven't been changed purposefully (by pressing UP/DOWN, a number key or F1-F12), then the scans that were used 
            during the last time that data for self.crd.radar was plotted, are used. If choose_nearest_height, then another condition
            is that the scans have not been changed by choosing the nearest scanangle in the mean time. If choose_nearest_scanangle,
            then this condition is that the scans have not been changed by choosing the nearest elevation in the mean time.
            Another condition is that the structure of the volume should be the same as during the last time that data for that 
            radar was plotted (at least the scanangles should be equal). 
            If these conditions are satisfied then saved data is used, because otherwise the scanangle doesn't need to go back to
            its initial value when going back to the previous radar or dataset. Further, it saves computation time.
            """
            condition1 = (self.changing_radar or self.changing_dataset) and not self.gui.switch_to_case_running and\
            self.radar_dataset in self.scans_radars and self.time_last_purposefulscanchanges < self.scans_radars[self.radar_dataset]['time'] and\
            self.scanangles_all_m == self.scans_radars[self.radar_dataset]['scanangles_all']
            
            if condition1 and choose_nearest_height and all([j < self.scans_radars[self.radar_dataset]['time'] for j in (
            self.crd.time_last_change_scan_selection_mode, self.time_last_choosenearestscanangle, self.time_last_panzoom)]):
                self.crd.scans = self.scans_radars[self.radar_dataset]['scans'].copy()
                #Only when choosing the nearest elevation. In other cases the selected scanangle should not change.
                for j in self.pb.panellist:
                    self.selected_scanangles[j] = scanangles_all[self.crd.scans[j]]
                self.time_last_choosenearestheight = pytime.time()
            elif condition1 and choose_nearest_scanangle and all([j < self.scans_radars[self.radar_dataset]['time'] for j in (
            self.crd.time_last_change_scan_selection_mode, self.time_last_choosenearestheight)]) and not self.gui.setting_saved_choice:
                self.crd.scans = self.scans_radars[self.radar_dataset]['scans'].copy()
                self.time_last_choosenearestscanangle = pytime.time()
            else:
                scanangles_before = {}
                for j in self.pb.panellist:
                    try:
                        if j in self.scans_radars[self.radar_dataset_before]['panellist']:
                            scanangles_before[j] = scanangles_all_before[self.scans_radars[self.radar_dataset_before]['scans'][j]]
                        else:
                            scanangles_before[j] = scanangles_all[self.crd.scans[j]]
                    except Exception:
                        #self.scans_radars is not yet defined in this case, so use the current values of the parameters.
                        scanangles_before[j] = scanangles_all[self.crd.scans[j]]

                if choose_nearest_height:
                    save_time = self.panel_center_heights.get('time', None)
                    need_update = not save_time or any(j > save_time for j in (self.time_last_panzoom, self.time_last_purposefulscanchanges, 
                                  self.time_last_choosenearestscanangle, self.crd.time_last_change_scan_selection_mode))
                    new_panels = None if need_update else [j for j in self.pb.panellist if not j in self.panel_center_heights['heights']]
                    if need_update or new_panels:
                        """These are only updated when zooming or panning has taken place in the mean time, or when the scans have been changed 
                        purposefully, or when the nearest scanangle is chosen in the mean time. This implies e.g. that when going to multiple radars, 
                        always the first radar in a sequence is used for determining the center heights, such that the center heights 
                        do not change in the next part of the sequence.
                        """
                        old_corners = self.scans_radars.get(self.radar_dataset_before, {}).get('corners', self.pb.corners)
                        old_distance_to_radar = {j:np.linalg.norm(old_corners[j].mean(axis=0)) for j in self.pb.panellist}
                        heights = self.get_panel_center_heights(old_distance_to_radar, scanangles_before, scanangles_all_before,
                                                                self.scanpair_present_before, panellist=new_panels)
                        if need_update:
                            self.panel_center_heights = {'time':pytime.time(), 'heights':heights}
                        else:
                            self.panel_center_heights['heights'].update(heights)
                
                for j in self.pb.panellist:
                    
                    if choose_nearest_height:
                        #Find the scanangle for which the beam elevation at the center of the screen is closest to the value it was before,
                        #for the previous radar.
                        new_distance_to_radar = np.linalg.norm(self.pb.corners[j].mean(axis=0))
                        # print(j, scanangles_all_values, new_distance_to_radar, self.panel_center_heights['heights'][j])
                        new_scanangle = ft.find_scanangle_closest_to_beamelevation(scanangles_all_values,new_distance_to_radar,self.panel_center_heights['heights'][j])
                    else:
                        """Find the scanangle that is closest to the selected scanangle. If there are 2 scanangles that are equally close,
                        then the one is chosen that is closest to the scanangle for which currently data is displayed in panel j.
                        """
                        new_scanangle, index = ft.closest_value(scanangles_all_values,self.selected_scanangles[j], return_index=True)
                        if len(scanangles_all) > 1:
                            scanangles_all_without_newscanangle = np.delete(scanangles_all_values, index)
                            next_closest_scanangle = ft.closest_value(scanangles_all_without_newscanangle, self.selected_scanangles[j])
                            if abs(self.selected_scanangles[j]-new_scanangle) == abs(self.selected_scanangles[j]-next_closest_scanangle) and\
                            abs(scanangles_before[j]-next_closest_scanangle) < abs(scanangles_before[j]-new_scanangle):
                                new_scanangle = next_closest_scanangle
                                          
                            if self.crd.plot_mode in ('Row', 'Column') or self.gui.setting_saved_choice:
                                """It might be the case that multiple selected scanangles get mapped onto the same actual new scanangle, in which case 
                                multiple panels might show the same product-scan combination. That is not desired under these conditions, so in that
                                case new_scanangle gets changed below.
                                """
                                for i in self.pb.panellist[:self.pb.panellist.index(j)]:
                                    scanangle_i = scanangles_all[self.crd.scans[i]]
                                    if new_scanangle == scanangle_i and self.crd.products[i] == self.crd.products[j] and (self.gui.setting_saved_choice or 
                                    all(var[i] == var[j] for var in (self.crd.apply_dealiasing, self.crd.productunfiltered))) and\
                                    len(scanangles_all_values)-index > 1:
                                        new_scanangle = scanangles_all_values[index+1]
                                        index += 1
                                                         
                    if self.scanpair_present and new_scanangle in [scanangles_all[i] for i in (1,2)]:
                        self.crd.scans[j] = (1,2)[self.range_nyquistvelocity_scanpairs_indices[j]]
                    else:
                        self.crd.scans[j] = [i for i,k in scanangles_all.items() if k==new_scanangle][0]
                    
                    if choose_nearest_height:
                        #Only in this case. In other cases the selected scanangle should not change.
                        self.selected_scanangles[j] = scanangles_all[self.crd.scans[j]]
                        
                if choose_nearest_height:
                    self.time_last_choosenearestheight = pytime.time()
                else:
                    self.time_last_choosenearestscanangle = pytime.time()
                    
    def check_presence_large_range_large_nyquistvelocity_scanpair(self,update_scanpairs_indices = False):
        """A range-nyquist velocity scan pair is a pair of scans of which the first has a large range but low Nyquist velocity, and the second has
        a lower range but larger Nyquist velocity, while the scanangles for both scans differ by at most 0.1 degrees. When such a pair is present,
        it might be desired to display e.g. reflectivity for the scan with the larger range, and velocity for the scan with the higher Nyquist
        velocity. In order to let this also be the case when switching from radar, self.range_nyquistvelocity_scanpairs_indices is used, which
        stores the index of the scan in the scan pair that is displayed. 
        For a correct handling of the display of 2 scans of the scan pair, it is important that going up/down updates only the scans in the scan pair,
        because otherwise you will start viewing data for scans with clearly different scanangles, which is likely not desired. This is handled in
        the function self.crd.process_keyboardinput.
        
        This function determines whether a scan pair is present, and returns True if this is the case, and False otherwise. If a pair is present, but
        the scans involved are not 1 and 2, then scanpair_present = False, as in this case handling such a scan pair is much more difficult.
        Further, if update_scanpairs_indices = True, then this function determines the index of the scan in the scan pair that
        is currently being displayed. This index is 0 for the first scan, and 1 for the second.
        """
        scanpair_present = False
        if len(self.scanangles_all['z'])>1:
            scanangle1 = self.scanangles_all_m['z'][1]
            scanangle2 = self.scanangles_all_m['z'][2]
            
            #In the case of Zaventem it is possible that the first scan misses for the velocity, in which case all volume attributes for this 
            #scan are set equal to that for the second. In this case the condition for the difference in Nyquist velocities between both scans
            #is not satisfied (because it is zero), but the scans should still be regarded as a scan pair. That is ensured by including this
            #bool v_scan1_equals_scan2.
            try:
                #Don't check for 'scannumbers_all', because it is possible that scannumbers_all['v'][1] != scannumbers_all['v'][2], while the rest
                #of the attributes is equal. This is the case when the number of duplicates differs between scan 1 and 2 (when e.g. scan 2 has duplicates,
                #and for scan 1 only one of these duplicates is shown). In this case v_scan1_equals_scan2 should be True.
                v_scan1_equals_scan2 = all([getattr(self,j)['v'][1]==getattr(self,j)['v'][2] for j in gv.volume_attributes_p if not j=='scannumbers_all'])
            except Exception:
                #Occurs when no velocity is available, or when there is only one scan.
                v_scan1_equals_scan2 = False
            
            if ft.r1dec(np.abs(scanangle1-scanangle2)) <= 0.1 and self.radial_range_all['z'][1]/self.radial_range_all['z'][2] > 1.25 and\
            not any([self.nyquist_velocities_all_mps[j] is None for j in (1,2)]) and (v_scan1_equals_scan2 or
            np.abs(self.nyquist_velocities_all_mps[2]/self.nyquist_velocities_all_mps[1]) > 1.25):
                scanpair_present = True
                
                if update_scanpairs_indices:
                    for i in (1,2):
                        for j in self.pb.panellist:
                            if self.crd.scans[j]==i and not self.crd.products[j] in gv.plain_products:
                                self.range_nyquistvelocity_scanpairs_indices[j] = i-1
        return scanpair_present
        
    def get_dir_string(self,radar,dataset=None,dir_index = None, return_dir_string_list = False):
        radar_dataset = self.get_radar_dataset(radar, dataset)
        
        dir_string_list = bg.dirstring_to_dirlist(self.gui.radardata_dirs[radar_dataset])
        index = dir_index if not dir_index is None else self.gui.radardata_dirs_indices[radar_dataset]
        if not return_dir_string_list:
            return dir_string_list[index]
        else:
            return dir_string_list[index], dir_string_list 
     
    def get_directory(self,date,time,radar,dataset = None,dir_string = None, dir_index = None):
        #Either dir_string or dataset has to be specified. dir_index is an optional argument for self.get_dir_string, 
        #which only has an effect when dir_string = None
        """Returns the directory in which the data is/should be stored for a particular combination of date and time
        """
        if dir_string is None:
            dir_string = self.get_dir_string(radar,dataset,dir_index)        
        if date=='c':
            #This should only be the case when time is also 'c'.
            return bg.get_last_directory(dir_string,radar,self.get_filenames_directory)
        else:
            return opa(bg.convert_dir_string_to_real_dir(dir_string,radar,date,time))
        
    def get_radar_dataset(self, radar=None, dataset=None, no_special_char=False):
        radar = radar if radar else self.crd.radar
        if no_special_char:
            radar = gv.radars_nospecialchar_names[radar]
        dataset = dataset if dataset else self.crd.dataset
        return radar+f'_{dataset}'*(radar in gv.radars_with_datasets)
    def split_radar_dataset(self, radar_dataset):
        index = radar_dataset.find('_')
        radar = radar_dataset if index == -1 else radar_dataset[:index]
        dataset = radar_dataset[len(radar)+1:]
        return radar, dataset
    
    def get_variables(self,radar,dataset):
        radar_dataset = self.get_radar_dataset(radar, dataset)
        
        dir_string_list = bg.dirstring_to_dirlist(self.gui.radardata_dirs[radar_dataset])
        current_dir_string = dir_string_list[self.gui.radardata_dirs_indices[radar_dataset]]
        n = len(dir_string_list)
        return radar_dataset, dir_string_list, current_dir_string, n
        
    def get_nearest_directory(self,radar,dataset,date,time):
        """Returns the absolute path to the nearest directory for which data is available, where nearest is relative to the input date and time.
        As the function self.get_next_directory below, this function takes into account that multiple directory strings can be present in
        self.gui.radardata_dirs[radar_dataset]. That is also done in the same way as in that function, so I refer to that function
        for more explanation.
        """
        radar_dataset, dir_string_list, current_dir_string, n = self.get_variables(radar,dataset)
        #This should only be the case when time is also 'c'.
        find_last_dir = date == 'c'
        if find_last_dir:
            date = ''.join(ft.get_ymdhm(pytime.time())[:3])
        
        directory = bg.get_last_directory(current_dir_string,radar,self.get_filenames_directory) if find_last_dir else\
                    bg.get_nearest_directory(current_dir_string,radar,date,time,self.get_filenames_directory,self.get_datetimes_from_files)
        if not directory:
            return None
        dir_date,_ = bg.get_date_and_time_from_dir(directory,current_dir_string,radar)
        if n == 1 or dir_date == date:
            return directory
        
        try:
            directories = []
            for j in dir_string_list:
                if j == current_dir_string:
                    directories.append(directory)
                else:
                    directories += [bg.get_last_directory(j,radar,self.get_filenames_directory) if find_last_dir else
                                    bg.get_nearest_directory(j,radar,date,time,self.get_filenames_directory,self.get_datetimes_from_files)]
            dir_dates = [bg.get_date_and_time_from_dir(directories[j],dir_string_list[j],radar)[0] for j in range(n)]
                
            datediffs = np.array([np.abs(ft.datetimediff_s(j+'0000',date+'0000')) for j in dir_dates],dtype = 'int64')
            index = np.argmin(datediffs)
            selected_date = dir_dates[index]
            if selected_date == dir_date:
                return directory
        except Exception as e:
            print(e,'self.dsg.get_nearest_directory')
            return directory
        
        #The current directory string has changed, such that the corresponding index must be updated.
        self.gui.radardata_dirs_indices[radar_dataset] = index
        return directories[index]  

            
    def get_next_directory(self,radar,dataset,date,time,direction,desired_newdate = None,desired_newtime = None): 
        """Returns the absolute path of the next directory for which data is available. If desired_newdate and desired_newtime are given,
        then this function first finds the directory for which the date and time are closest to the desired ones. If it differs from the current
        directory, then it is returned. If not, then this function finds the nearest next directory for which data is available.
        
        It is possible that multiple directory strings
        are given in self.gui.radardata_dirs[radar_dataset], and this function takes them all into account. This implies that it determines
        for each directory string the next directory, i.e. the first directory for which the date is changed in the direction given by direction.
        The function then determines the nearest next directory from all next subdirectories that have been found, and returns this one.
        If the nearest next directory has been found for a directory string that differs from the current one, then 
        self.gui.radardata_dirs_indices[radar_dataset] is updated.
        """
        radar_dataset, dir_string_list, current_dir_string, n = self.get_variables(radar,dataset)
        if desired_newdate and desired_newtime:
            current_dir = self.get_directory(date,time,radar,dir_string = current_dir_string)
            directory = self.get_nearest_directory(radar,dataset,desired_newdate,desired_newtime)
            if directory != current_dir:
                return directory
            
        """If no directory is returned, then continue with finding the nearest next directory for which there is data.
        """
        #directory is equal to the current directory if there is no next directory for the current directory string.
        directory = bg.get_next_directory(current_dir_string,direction,radar,self.get_filenames_directory,date = date,time = time)
        dir_date,dir_time = bg.get_date_and_time_from_dir(directory,current_dir_string,radar)
        #dir_time is None if there is no ${time} variable in current_dir_string.
              
        #An exception occurs for example when there is no ${date} variable in a directory string, in which case comparing dates causes errors.
        #This should normally not occur, as it is unusual that there is no date in a directory string (that would mean that all data is located in the
        #same folder). If it occurs, then simply directory is returned.
        try:
            """If n>1, first check whether there is a directory for the next date for current_dir_string, and if so, return this one.
            In the case that a ${time} variable is present in current_dir_string, it is also checked whether there is a next directory for the same
            date but for a different time.
            """
            next_date = ft.next_date(date, direction)
            if n == 1 or (not dir_time and dir_date == next_date) or (dir_time not in (time, None) and dir_date in (date, next_date)):
                return directory
            else:
                directories = []
                for j in dir_string_list:
                    if j == current_dir_string:
                        directories.append(directory)
                    else:
                        dir_j = bg.get_nearest_directory(j,radar,date,time,self.get_filenames_directory,self.get_datetimes_from_files)
                        dir_date_j, dir_time_j = bg.get_date_and_time_from_dir(dir_j,j,radar)
                        #A multiplication by direction is performed to ensure that this comparison gives the desired result also when direction==-1
                        if direction*int(dir_date_j) > direction*int(date):
                            directories.append(dir_j)
                        else:
                            directories.append(bg.get_next_directory(j,direction,radar,self.get_filenames_directory,date=dir_date_j,time=dir_time_j))
                dir_dates = [bg.get_date_and_time_from_dir(directories[j],dir_string_list[j],radar)[0] for j in range(n)]
                
                orig_dir_string_list = dir_string_list.copy()
                # Remove entries for directory strings that have no next directory in the desired direction
                i_dirs_remove = [i for i,dir_date in enumerate(dir_dates) if direction*int(dir_date) <= direction*int(date)]
                for obj in (dir_string_list, directories, dir_dates):
                    for i in i_dirs_remove[::-1]:
                        obj.pop(i)
                        
                if not dir_string_list:
                    #Return the current directory, because there is no next one
                    return opa(bg.convert_dir_string_to_real_dir(current_dir_string,radar,date,time))
                else:
                    datediffs = np.array([np.abs(ft.datetimediff_s(date+'0000',j+'0000')) for j in dir_dates],dtype = 'int64')
                    #All elements in datediffs are positive
                    index = np.argmin(datediffs)
                    selected_date = dir_dates[index]
                    if selected_date == dir_date:
                        #If dir_date equals selected_date, then always return directory, because it is in this case
                        #not desired that the directory string changes.
                        return directory
                    else:
                        #The current directory string has changed, such that the corresponding index must be updated.
                        new_dir_string = dir_string_list[index]
                        self.gui.radardata_dirs_indices[radar_dataset] = orig_dir_string_list.index(new_dir_string)
                        return directories[index]
        except Exception as e:
            print(e,'self.dsg.get_next_directory')
            return directory
                             
    
    def get_download_directory(self,radar,dataset=None):
        """Determine the directory in which files that get downloaded should be stored. This directory is determined from dir_string, and is 
        the part of dir_string before any variable (with '${') is encountered. 
        If multiple dir_strings are provided, then for downloading always the 1st (hence dir_index = 0) is used.
        """
        dir_string = self.get_dir_string(radar,dataset,dir_index = 0) 
        return bg.get_download_directory(dir_string)
        
    def get_newest_datetimes_currentdata(self,radar,dataset):
        """Returns the datetimes of the 2 newest (newest first, second-newest second) radar volumes that are available for the radar.
        It is used by the automatic download part of 
        nlr_currentdata.py to determine whether the user is viewing the most recent scans (i.e. the scans that were most recent before the
        download was finished), which is used to determine whether it is desired to plot data for the downloaded file.
        """
        dir_string = self.get_dir_string(radar,dataset)        
        lastdir = bg.get_last_directory(dir_string,radar,self.get_filenames_directory)
        lastdir_filenames = self.get_filenames_directory(radar,lastdir)
        lastdir_datetimes = self.get_datetimes_from_files(radar,lastdir_filenames,lastdir)
        n1 = len(lastdir_datetimes)
        if n1 > 1:
            return lastdir_datetimes[-2:][::-1]
        else:
            secondlastdir = bg.get_next_directory(dir_string,-1,radar,self.get_filenames_directory,current_dir = lastdir)
            secondlastdir_filenames = self.get_filenames_directory(radar,secondlastdir)
            secondlastdir_datetimes = self.get_datetimes_from_files(radar,secondlastdir_filenames,secondlastdir)
            returns = np.append(lastdir_datetimes, secondlastdir_datetimes[-(2-n1):][::-1])
            if len(returns) == 0:
                return None, None
            elif len(returns) == 1:
                return returns[0], None
            else:
                return returns

    def get_filenames_directory(self,radar,directory):
        """Returns the filenames in a list with directory entries (which could also include directories, and which get removed from the list here).
        """
        if directory is None: return [] #Calling os.listdir with argument None lists the current working directory, which is not desired.
        return self.source_classes[self.data_source(radar)].get_filenames_directory(radar,directory)
               
    def get_datetimes_from_files(self,radar,filenames,directory=None,dtype = str,return_unique_datetimes = True, mode='simple'):
        # directory is in most cases not needed to determine datetimes from filenames, but Meteo-France archived files are an exception, since
        # they contain only time in their names. Hence directory is added as argument.
        """If filenames = None this function returns the datetimes of the files that are present in the directory corresponding to the particular 
        radar and dataset. If not filenames = None, then this function returns the datetimes of the filenames that are given as input.
        Mode can be either 'simple' or 'dates ', and the latter should be used when it is desired to also return dates present in the current
        directory if they are determined (is the case for TU Delft).
        The returned object is either an array of datetimes, 1 for each filename, or it is a dictionary with 
        """
        return self.source_classes[self.data_source(radar)].get_datetimes_from_files(filenames,directory,dtype,return_unique_datetimes, mode)
    
    def get_datetimes_directory(self,radar,directory,dtype = str,return_unique_datetimes = True):
        filenames = self.get_filenames_directory(radar,directory)
        # This function gets regularly called from within self.crd.switch_to_nearby_radar, so it's important to cache results.
        # Earlier this was done using the modification time (mtime) of the directory that would also remove the need for always determining filenames.
        # But this method appeared to be not trustworthy, as some filesystems don't update it when the number of files in the directory changes.
        if not hasattr(self, 'nfiles_directory'):
            self.nfiles_directory, self.datetimes_directory = {}, {}
        if not directory in self.nfiles_directory or self.nfiles_directory[directory] != len(filenames):
            self.datetimes_directory[directory] = self.get_datetimes_from_files(radar,filenames,directory,dtype,return_unique_datetimes, mode='simple')
            self.nfiles_directory[directory] = len(filenames)
        return self.datetimes_directory[directory]
    
    def get_product_versions(self, radar, filenames, datetimes):
        self.product_versions_datetimesdict = self.product_versions_directory = self.products_version_dependent =\
        self.product_versions_in1file = None
        source_class = self.source_classes[self.data_source(radar)]
        if getattr(source_class, 'get_product_versions', None) and len(filenames):
            # self.product_versions_in1file indicates whether the different product versions are contained in 1 file. If this is the case,
            # then volume attributes are determined for all product versions at once, and attributes for different versions are distinguished
            # by appending the product version to the product key, e.g. 'z' -> 'z z_scan'.
            self.product_versions_datetimesdict, self.products_version_dependent, self.product_versions_in1file =\
                source_class.get_product_versions(filenames, datetimes)
            if self.product_versions_datetimesdict:
                self.product_versions_directory = np.unique(np.concatenate(list(self.product_versions_datetimesdict.values())))    
    
    def get_files(self,radar,directory,return_datetimes = False):
        self.files = self.get_filenames_directory(radar,directory)
            
        if radar in gv.radars_with_onefileperdate:
            #Here there is one file per date, and therefore multiple radar volumes per file. datetimes contains datetimes of all the radar volumes
            #within all the files, while dates lists just one date for each file.
            datetimes, self.dates = self.get_datetimes_from_files(radar,self.files,directory,dtype = str,return_unique_datetimes = False, mode='dates')
            #self.dates is used in the function self.get_dates_with_archived_data
            datetimes_unique = np.unique(datetimes)
            self.files_datetimesdict = {self.dates[j]: self.files[j] for j in range(len(self.dates))}
        else:
            datetimes = self.get_datetimes_from_files(radar,self.files,directory,dtype = str,return_unique_datetimes = False)
            datetimes_unique = np.unique(datetimes)
            self.files_datetimesdict = {j:self.files[datetimes==j] for j in datetimes_unique}
            
        self.get_product_versions(radar, self.files, datetimes)
            
        if return_datetimes:
            return datetimes_unique
    
    def select_files_datetime(self):
        if self.crd.radar in gv.radars_with_onefileperdate:
            self.files_datetime = self.files_datetimesdict[self.crd.date]
        else:
            self.files_datetime = self.files_datetimesdict[self.crd.date+self.crd.time]
            
        self.product_versions_datetime = None
        if self.product_versions_datetimesdict:
            self.product_versions_datetime = self.product_versions_datetimesdict[self.crd.date+self.crd.time]
    
    def get_total_volume_files_size(self, datetime=None):
        files_datetime = self.files_datetimesdict[datetime] if datetime else self.files_datetime
        return sum([os.path.getsize(self.crd.directory+'/'+j) for j in files_datetime])
    
    def get_filenames_and_datetimes_in_datetime_range(self,radar,dataset = None,dir_string = None,startdatetime = None,enddatetime = None,return_abspaths = False,return_completely_selected_directories = False):
        #Either dir_string or dataset should be given as input
        
        """Returns filenames and datetimes of all files within a particular datetime range. If return_abspaths, then filenames contains absolute paths
        to the files, and if return_completely_selected_directories, then names of directories that are completely selected get returned. This 
        information is used in nlr.py to determine whether it is allowed to move a complete directory instead of moving files individually, as the
        former method is much faster.
        """
        if dir_string is None:
            dir_string = self.get_dir_string(radar,dataset)
            
        dirs_abspaths = bg.get_abspaths_directories_in_datetime_range(dir_string,radar,startdatetime,enddatetime)[0]
        
        completely_selected_directories = []
        requested_filenames = np.array([],dtype = 'int64'); requested_datetimes = np.array([],dtype = 'int64')
        for i in dirs_abspaths:
            filenames = self.get_filenames_directory(radar,i)
            if return_abspaths:
                filenames = np.array([opa(i+'/'+j) for j in filenames])
            datetimes = self.get_datetimes_from_files(radar,filenames,i,dtype = 'int64',return_unique_datetimes = False)
        
            if not startdatetime is None and not enddatetime is None:
                requested = (datetimes>= int(startdatetime)) & (datetimes<= int(enddatetime))
            elif not startdatetime is None:
                requested = datetimes>= int(startdatetime)
            elif not enddatetime is None:
                requested = datetimes<= int(enddatetime)
            else:
                requested = np.ones(len(datetimes),dtype = 'bool')
                
            requested_filenames = np.append(requested_filenames,filenames[requested])
            requested_datetimes = np.append(requested_datetimes,datetimes[requested])
        
            if return_completely_selected_directories and np.count_nonzero(requested)==len(datetimes):
                completely_selected_directories.append(i)
            
        if return_completely_selected_directories:
            return completely_selected_directories,requested_filenames,requested_datetimes
        else:
            return requested_filenames,requested_datetimes
        
    def get_dates_with_archived_data(self,radar,dataset):
        if radar in gv.radars_with_onefileperdate:
            return self.dates
        else:
            _,dir_string_list = self.get_dir_string(radar,dataset,return_dir_string_list = True)
            dates = np.array([])
            for j in dir_string_list:
                dates = np.append(dates,bg.get_dates_with_archived_data(j,radar))
            return np.unique(dates)
    
    def get_radars_with_archived_data_for_date(self,date):   
        if date[0]=='c': date = date[1:]
        startdatetime = date+'0000'
        enddatetime = ''.join(ft.next_date_and_time(date,'0000',1440))
        
        radars_with_data = []
        for i in gv.radars_all:
            datasets = (None,) if not i in gv.radars_with_datasets else ('Z','V')
            for j in datasets:
                radar_dataset = self.get_radar_dataset(i, j)
                _,dir_string_list = self.get_dir_string(i,j,return_dir_string_list = True)
                for k in dir_string_list:
                    if '${date' in k:
                        for l in range(24):
                            # Check for each hour of the day whether a directory is available. This is a crude way to consider both directory formats
                            # with 1 folder per date and formats with folders for different times (e.g. hours).
                            directory = bg.convert_dir_string_to_real_dir(k, i, date, format(l, '04d'))
                            if os.path.exists(directory):
                                radars_with_data.append(radar_dataset)
                                break
                            elif not ('${datetime' in k or '${time' in k):
                                break
                    else:
                        dirs_abspaths,dates_filtered = bg.get_abspaths_directories_in_datetime_range(k,i,startdatetime,enddatetime)
                        if i in gv.radars_with_onefileperdate:
                            for path in dirs_abspaths:
                                files = np.sort(self.get_filenames_directory(i, path))
                                _, dates = self.get_datetimes_from_files(i, files, path, mode='dates')
                                if date in dates:
                                    radars_with_data.append(radar_dataset)
                                    break
                        elif len(dirs_abspaths)>0 and dates_filtered:
                            #If not dates_filtered, then it was not possible to determine which directories contain data for the input date, because no 
                            #${date} variable is located in dir_string.
                            radars_with_data.append(radar_dataset)
                    if radar_dataset in radars_with_data:
                        break
    
        return radars_with_data