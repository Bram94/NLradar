# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import nlr_globalvars as gv
import nlr_background as bg
import nlr_functions as ft

import os
opa=os.path.abspath
import numpy as np
import re
import time as pytime


"""This module contains the functions that are specific to a particular data source. Each data source has a particular (and of course usually different...) way to store the data,
and classes that import data for a specific data format are contained in nlr_importdata.py. This is because there might be some data sources that use the same data format.
Because also in this case there are very likely differences in the naming of the files and folders in which they are stored, these data sources can't be totally treated the same,
and that's why there are two layers for importing data, this being the first.
Each class should at least implement the functions get_scans_information, get_data, get_data_multiple_scans, get_filenames_directory and get_datetimes_from_files.
"""

class Source_KNMI():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        
        
    def filepath(self):
        filepath=opa(os.path.join(self.crd.directory,'RAD_NL'+str(gv.radar_ids[self.crd.radar])+'_VOL_NA_'+self.crd.date+self.crd.time+'.h5'))     
        return filepath
    
    def get_scans_information(self):
        self.dsg.KNMI_hdf5.get_scans_information(self.filepath())

    def get_data(self, j): #j is the panel
        self.dsg.KNMI_hdf5.get_data(self.filepath(),j)

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return self.dsg.KNMI_hdf5.get_data_multiple_scans(self.filepath(),product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                    
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort(np.array([j for j in entries if j[-2:]=='h5' and j[:15]=='RAD_NL'+gv.radar_ids[radar]+'_VOL_NA']))
        return filenames
    
    def get_datetimes_from_files(self,filenames,directory=None,dtype=str,return_unique_datetimes=True, mode='simple'):
        datetimes=np.array([j[-15:-3] for j in filenames],dtype=dtype)
        return datetimes





class Source_KMI():
    """This class handles data from the radars of the KMI in Jabbeke and Wideumont, but also from the radar of skeyes in Zaventem, in the case 
    that it is provided in the same format as one of the formats in which the KMI provides their data. This is the case for files with a .hdf extension.
    """
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        self.vol_classes={'rainbow3':self.dsg.Gematronik_vol_rainbow3,'rainbow5':self.dsg.Gematronik_vol_rainbow5}
        
                                                                            
                    
    def correct_filename(self,filename,product):
        if product is None:
            return filename[:12]==self.crd.date+self.crd.time
        
        file_extension=os.path.splitext(filename)[1][1:]
        i_p = gv.i_p[product]
        pname=gv.productnames_KMI[file_extension][i_p]
        
        if file_extension=='hdf':
            correct_name=True if '.'+pname+'.' in filename and filename[:12]==self.crd.date+self.crd.time else False
            if self.crd.radar in gv.radars_with_datasets:
                #Prevent that a file for the wrong dataset is used, in the case that it was put in the wrong directory.
                #For other file extensions there is no way to check this based on the filename only.
                dataset_str='scanv' if self.crd.dataset=='V' else 'scanz'
                correct_name=correct_name if filename[-9:-4]==dataset_str else False
        elif file_extension=='h5':
            correct_name=True if pname+'.vol' in filename and filename[:12]==self.crd.date+self.crd.time else False
        elif file_extension=='vol':
            if not isinstance(pname,list):
                correct_name=True if pname+'.vol' in filename and filename[:12]==self.crd.date+self.crd.time else False
            else:
                correct_name=any([j+'.vol' in filename and filename[:12]==self.crd.date+self.crd.time for j in pname])
                
        if correct_name and i_p in gv.productnames_KMI_possible_mistake[file_extension]:
            #Make sure that not the wrong product is selected, which could be the case when the productname of one product is contained in productnames
            #of other products.
            pname=gv.productnames_KMI_possible_mistake[file_extension][i_p]
            if not isinstance(pname,list):
                correct_name=pname+'.vol' not in filename
            else:
                correct_name=all([j+'.vol' not in filename for j in pname])
            
        return correct_name

    def filepath(self,product,source_function=None):
        try:
            #First try to obtain a filename for the correct product, date ànd time
            filename=[i for i in self.dsg.files_datetime if self.correct_filename(i,product)][0]     
        except Exception:
            if source_function==self.get_scans_information:
                try:
                    #Try to find any file with the correct date and time, and if found, then determine the product contained in it
                    filename=[i for i in self.dsg.files_datetime if self.correct_filename(i,None)][0]
                    product='z' #The default product. 'z' does not need to be present, but the only information required for determining whether it is possible to 
                    #obtain the Nyquist velocities is that the product is not equal to 'v'.
                except Exception:
                    filename=''
            else:
                filename=''
        filepath=opa(os.path.join(self.crd.directory,filename))

        return [filepath, product] if source_function==self.get_scans_information else filepath
    
    
    def get_volume_type(self,filepath):
        try:
            with open(filepath,'rb') as vol:
                line=vol.read(10).decode('utf-8')
                vol.seek(0)
                return 'rainbow5' if line[0]=='<' else 'rainbow3'
        except Exception:
            return None


    def get_scans_information(self):
        #product 'v' is tried initially, because it enables the program to obtain the Nyquist velocities
        #In the case of Zaventem it is however possible that the file that contains the reflectivity has more scans than the file that contains
        #the velocity. In this case it is therefore necessary to take 'z' as product.
        product='v' if not self.crd.radar=='Zaventem' else 'z'
        filepath, product=self.filepath(product,source_function=self.get_scans_information)
        file_extension=os.path.splitext(filepath)[1][1:]
        #The testproduct is used to determine whether it is possible to obtain the Nyquist velocities (only when testproduct=='v')
        if file_extension=='vol':
            vol_type=self.get_volume_type(filepath)
            self.vol_classes[vol_type].get_scans_information(filepath,product)
        else:
            self.dsg.ODIM_hdf5.get_scans_information(filepath,product)
            
    def try_product(self,product,try_unfiltered):
        filename=''; try_again=True
        if try_unfiltered: #If unfiltered data has been requested but is not available, then first try the filtered variant
            try:
                product='u'+product
                filename=[i for i in self.dsg.files_datetime if self.correct_filename(i,product)][0]
                try_again=False                         
            except Exception:
                pass
        if try_again:
            try:
                if try_unfiltered: product=product[1:]
                filename=[i for i in self.dsg.files_datetime if self.correct_filename(i,product)][0]
                try_again=False                         
            except Exception:
                pass
        return product,filename,try_again
        
    def get_data(self, j): #j is the panel
        file_available=True
        
        """First try to obtain a file for self.crd.products[j], which is filtered/unfiltered depending on self.crd.productunfiltered[j].
        If the unfiltered product is chosen and is not available, then first the filtered version is chosen.

        If again not available, then file_available is set to False, except when self.crd.products[j] in ('z','v'), in which case it
        is attempted to find a file for the other product (except when self.crd.products[j] is equal to that other product).
        This is done because for old data for Wideumont it is the case that the V-dataset does not contain reflectivities, such that you
        need to switch from dataset to view reflectivities. If self.crd.products[j]=='v', then without this last procedure it would
        be the case that you need to switch manually to 'z' to view reflectivity, whereas with this last procedure you will automatically show
        the reflectivity for the Z-dataset (at least when no velocity is available for that dataset, as is the case with old data for Wideumont).
        """
        product=self.crd.products[j]
        product, filename, try_again=self.try_product(product,self.crd.productunfiltered[j])
        if try_again:
            changing_radardataset=self.dsg.changing_radar or self.dsg.changing_dataset
            if changing_radardataset and self.crd.products[j]=='v':
                #It is done only when changing from radar of dataset, because otherwise it could cause the product to change while pressing 
                #LEFT/RIGHT, which is not desired. If in that case the selected product is not available, then the plot should not be updated.
                product='z'
            elif changing_radardataset and self.crd.products[j]=='z':
                product='v'
            else:
                file_available=False
                
            if file_available:
                product, filename, try_again=self.try_product(product,self.crd.productunfiltered[j])
                if try_again:
                    file_available=False
                                
        if file_available:
            self.crd.products[j] = product[1] if product[0]=='u' else product
            self.crd.using_unfilteredproduct[j] = product[0]=='u' and len(product)>1
        
        if not file_available:
            raise Exception
        else:
            filepath=opa(os.path.join(self.crd.directory,filename))
            
        file_extension=os.path.splitext(filepath)[1][1:]
        if file_extension=='vol':
            vol_type=self.get_volume_type(filepath)
            self.vol_classes[vol_type].get_data(filepath, j)
        else:
            self.dsg.ODIM_hdf5.get_data(filepath, j)
        
        
    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        product, filename, _ = self.try_product(product,productunfiltered)
        # This needs to be done here
        meta = {'using_unfilteredproduct': product[0]=='u' and len(product)>1, 'using_verticalpolarization': False}
        if meta['using_unfilteredproduct']:
            product = product[1:]
        
        filepath=opa(os.path.join(self.crd.directory,filename))
        file_extension=os.path.splitext(filepath)[1][1:]
        
        if file_extension=='vol':
            vol_type=self.get_volume_type(filepath)
            data, scantimes, volume_starttime, volume_endtime =\
                self.vol_classes[vol_type].get_data_multiple_scans(filepath,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)            
        else:
            data, scantimes, volume_starttime, volume_endtime, _ =\
                self.dsg.ODIM_hdf5.get_data_multiple_scans(filepath,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
        
        return data, scantimes, volume_starttime, volume_endtime, meta
                
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort(np.array([j for j in entries if os.path.splitext(j)[1][1:] in ('hdf','h5','vol')]))
        return filenames
    
    def get_datetimes_from_files(self,filenames,directory=None,dtype=str,return_unique_datetimes=True, mode='simple'):
        datetimes=np.array([j[:12] for j in filenames],dtype=dtype)
        return np.unique(datetimes) if return_unique_datetimes else datetimes   
    




class Source_skeyes():
    """Data from skeyes (radar in Zaventem) can be delivered both by skeyes and the KMI, and they use somewhat different formats. The KMI
    uses the same format as one of the formats that they use for their other radars, which implies that the class Source_KMI can handle the data when 
    it is provided by the KMI.
    """
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        """For this radar the scans are stored in 3 different files per volume. It is therefore necessary to store for each product and scan
        the index of the file in which that scan is located. This information is stored in self.dsg.scannumbers_all, in the manner explained in the
        class self.dsg.skeyes_hdf5.
        """
        
    
                
    def get_scans_information(self):
        if self.dsg.files_datetime[0].endswith('scan_abc.hdf'):
            self.dsg.source_classes['KMI'].get_scans_information(); return
                
        filepaths=[opa(os.path.join(self.crd.directory,j)) for j in self.dsg.files_datetime]
        self.dsg.skeyes_hdf5.get_scans_information(filepaths)                
        
    
    def get_data(self, j): #j is the panel
        if self.dsg.files_datetime[0].endswith('scan_abc.hdf'):
            self.dsg.source_classes['KMI'].get_data(j); return
        
        """Scans are stored in 3 different files per volume, and file_index contains the index of the file that contains data for the scan
        self.crd.scans[j].
        """        
        file_index=self.dsg.scannumbers_all[gv.i_p[self.crd.products[j]]][self.crd.scans[j]][self.dsg.scannumbers_forduplicates[self.crd.scans[j]]][0]
        filepath=opa(os.path.join(self.crd.directory,self.dsg.files_datetime[file_index]))
        
        self.dsg.skeyes_hdf5.get_data(filepath, j)

    
    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        if self.dsg.files_datetime[0].endswith('scan_abc.hdf'):
            return self.dsg.source_classes['KMI'].get_data_multiple_scans(product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
        
        filepaths=[opa(os.path.join(self.crd.directory,j)) for j in self.dsg.files_datetime]
        return self.dsg.skeyes_hdf5.get_data_multiple_scans(filepaths,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)


    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort(np.array([j for j in entries if os.path.splitext(j)[1][1:] in ('hdf','h5')]))
        
        if len(filenames)==0 or filenames[0].endswith('hdf'):
            return filenames
        #The code below is specifically for the h5 files delivered by skeyes, where some issues with repeated content must be addressed.
        #Cases in which hdf and h5 files are put in the same folder are not permitted!
            
        
        startdatetimes=[j[4:16] for j in filenames]
        unique_startdatetimes=np.unique(startdatetimes)
        if len(startdatetimes)==len(unique_startdatetimes): return filenames
        
        """If not, then it is necessary to filter the files, in order to prevent that the same content is shown more than once by NLradar. 
        The first thing to check, is whether some files are simply 'repeated', in which case subsequent files have the same startdatetime, but a different file ID (the number 
        after RAW). This case has been observed, and can be recognized by the fact that both files have the same size. The code below indentifies these files, and removes
        one of them from the list with filenames. 
        If the resulting list of filenames contains only unique startdatetimes, then this list is returned. If not, then another method is used to deal with another case that
        leads to repeated startdatetimes.
        """
        rm_indices=[]
        for j in range(1,len(filenames)):
            if startdatetimes[j-1]==startdatetimes[j]:
                size1=os.path.getsize(opa(directory+'/'+filenames[j-1]))
                size2=os.path.getsize(opa(directory+'/'+filenames[j]))
                if size1==size2:
                    rm_indices.append(j)
                    
        filenames_filtered=np.delete(filenames,rm_indices)
        startdatetimes_filtered=[j[4:16] for j in filenames_filtered]
        unique_startdatetimes_filtered=np.unique(startdatetimes_filtered)
        if len(startdatetimes_filtered)==len(unique_startdatetimes_filtered): return filenames_filtered  
                
        
        """For data from Zaventem from before 2013, the data is provided both in the format from after 2012 (usually 3 files per volume),
        and per scan separately. The resulting files are all put in the same folder, whereas this program needs only the files that have
        the first format. The other files are therefore removed from the list. The files that should remain on the list can be recognized
        based on their start time, which is equal to that of the next or previous file. Of 2 subsequent files with the same start time, 
        the one that should be taken is the one with the largest size. 
        In the current cases that I have seen, the file that should be taken is the first of a pair of files. Because I am not sure about
        whether this is generally true, I check which file should be taken by calculating the file sizes. This is done only for the first
        pair of files (for computational reasons), thus it is assumed that the order of volume and single scan file at least doesn't change
        during the course of 1 day.
        
        It is important to realize that the case in which only a few files have the same startdatetime (and
        thus only a few volumes), is not treated correctly in the current situation.
        """
        filenames_filtered=[]
        desired_file=None
        for j in range(1,len(filenames)):
            if startdatetimes[j-1]==startdatetimes[j]:
                if desired_file is None:
                    size1=os.path.getsize(opa(directory+'/'+filenames[j-1]))
                    size2=os.path.getsize(opa(directory+'/'+filenames[j]))
                    desired_file=-1 if size1>size2 else 0
                    #Take the largest file
                filenames_filtered.append(filenames[j+desired_file])
        filenames_filtered=np.array(filenames_filtered)
        
        return filenames_filtered
    
    def get_datetimes_from_files(self,filenames,directory=None,dtype=str,return_unique_datetimes=True, mode='simple'):
        """For the skeyes hdf5 format there are multiple files (usually 3) per volume, such that the 
        datetimes determined here (per volume) are not simply the datetimes for all files, but the datetimes
        of the first files of all volumes. 
        Further, because the filename does not contain information about which file belongs to which volume, this
        must be determined in another way. It is here done by determining the absolute time for each datetime
        given in the filename, and then grouping files into one volume when the the absolute times differ by
        less than 120 seconds from those of the previous file in the volume.
        There is no fixed timestep between volumes (at least not after 2014), making it impossible to determine
        which file belongs to which volume based on solely the datetime.
        
        If the current method does not work, then a possible other method would be to add a particular file to the
        'current' volume, if the time to the last file in the current volume is smaller than the time to the next file
        after the currently regarded file. It would be added to a new volume if this condition is not satisfied.
        """
        datetimes=[]
        datetimes_h5=[]; seconds_h5=[]
        for j in filenames:
            if j.endswith('hdf'):
                datetimes.append(j[:12])
            else:
                datetimes_h5.append('20'+j[4:14])
                seconds_h5.append(int(j[14:16]))
        datetimes=np.array(datetimes)
        datetimes_h5=np.array(datetimes_h5); seconds_h5=np.array(seconds_h5)
        
        abstimes_h5=ft.get_absolutetimes_from_datetimes(datetimes_h5)+seconds_h5
        
        n_h5=len(datetimes_h5)
        nfiles_lastindex=0
        if n_h5>0:
            indices=[0]
            last_abstime=abstimes_h5[0]
            for j in range(1,n_h5+1):
                #The iteration for j=n_h5 is only included to apply the check for nfiles_lastindex==1 also to the last volume.
                if j<n_h5 and abstimes_h5[j]<last_abstime+120:
                    nfiles_lastindex+=1
                    indices.append(indices[-1])
                else:
                    if nfiles_lastindex==1:
                        #If a volume consists of just one file according to the above division of files into volume, then this
                        #file/volume is added to the nearest volume, except when the time difference exceeds 7.5 minutes.
                        #Adding is done to prevent the presence of very small volumes (containing only a few scans).
                        diff1=np.abs(abstimes_h5[indices[-1]]-abstimes_h5[indices[-2]])
                        diff2=diff1 if j==n_h5 else np.abs(abstimes_h5[indices[-1]]-abstimes_h5[j])
                        if np.min([diff1,diff2])<450:
                            if j==n_h5 or diff1<diff2:
                                indices[-1]=indices[-2]
                            else:
                                indices[-1]=j  
                                
                    if j<n_h5:
                        nfiles_lastindex=1
                        indices.append(j)
                if j<n_h5:
                    last_abstime=abstimes_h5[j]
                    
            datetimes=np.sort(np.append(datetimes,datetimes_h5[indices]))
            
        datetimes=datetimes.astype(dtype)
            
        return np.unique(datetimes) if return_unique_datetimes else datetimes   
    
    
    
    

class Source_DWD():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui = gui_class
        self.dsg = dsg_class
        self.crd = self.dsg.crd
        self.dp = self.dsg.dp
        self.pb = self.gui.pb
        self.import_classes = {'bz2': self.dsg.DWD_bufr, 'buf': self.dsg.DWD_bufr, 'hd5': self.dsg.DWD_odimh5}
        
        
        
    def get_extension(self, filename=None):
        filename = self.dsg.files_datetime[0] if not filename else filename
        return filename[-3:]
        
    def get_file_availability_info(self):        
        if self.get_extension() == 'hd5':
            n = len(self.dsg.files_datetime)
            files_pvs, pvs = self.files_product_versions_datetimesdict[self.crd.date+self.crd.time], self.dsg.product_versions_datetime
            filenames_per_pv = {i: [self.dsg.files_datetime[j] for j in range(n) if files_pvs[j] == i] for i in pvs}
            
            desired_pv = self.gui.radardata_product_versions[self.dsg.radar_dataset]
            pv = desired_pv if desired_pv in filenames_per_pv else pvs[0]
            if pv == 'combi':
                # In this case data for the 2 different product versions should be combined in nlr_importdata.py
                filenames = self.dsg.files_datetime
            else:
                filenames = filenames_per_pv[pv]
        else:
            filenames = self.dsg.files_datetime
                
        self.products_per_fileid, self.fileids_per_product, self.files_per_product_per_fileid = {}, {}, {}
        for j in filenames:
            for i in gv.productnames_DWD:
                for product in ('z', 'v'):
                    product_str = '_'+gv.productnames_DWD[i][product]+'_'
                    if j.endswith(i) and product_str in j:
                        index = j.index(product_str)+len(product_str)
                        # Historical DWD BUFR files do not always contain a file index, but instead of that they then contain a
                        # scanangle towards the end of the filename
                        fileid = int(j[index: j.index('-20')]) if '-20' in j else float(j[index+15: -11])
                        break
                else:
                    continue  # only executed if the inner loop did NOT break
                break  # only executed if the inner loop DID break
                        
            ft.init_dict_entries_if_absent(self.fileids_per_product, product, list)
            ft.init_dict_entries_if_absent(self.products_per_fileid, fileid, list)
            ft.create_subdicts_if_absent(self.files_per_product_per_fileid, [product, fileid], type_last_entry=list)
                
            self.products_per_fileid[fileid].append(product)
            self.fileids_per_product[product].append(fileid)
            self.files_per_product_per_fileid[product][fileid].append(j)
        
    def get_scans_information(self):
        self.get_file_availability_info()
    
        filepaths = {}; products = {}
        for fileid in self.products_per_fileid:
            #If present, then use for each fileid the file that contains the velocity, because otherwise the Nyquist velocity cannot be
            #determined.
            products[fileid] = 'v' if 'v' in self.products_per_fileid[fileid] else self.products_per_fileid[fileid][0]
            filepaths[fileid] = opa(self.crd.directory+'/'+self.files_per_product_per_fileid[products[fileid]][fileid][0])
            
        self.import_classes[self.get_extension()].get_scans_information(filepaths, products, self.fileids_per_product)
                        
        
    def get_data(self, j): #j is the panel
        self.get_file_availability_info()
         
        i_p = gv.i_p[self.crd.products[j]]
        if i_p in self.fileids_per_product:
            fileid = self.dsg.scannumbers_all[i_p][self.crd.scans[j]][0]
            extension = self.get_extension()
            if extension == 'hd5':
                # When combining data for the 2 product versions, both filenames need to be supplied
                filepaths = [opa(os.path.join(self.crd.directory, i)) for i in self.files_per_product_per_fileid[i_p][fileid]]
                self.import_classes[extension].get_data(filepaths, j)
            else:
                filepath = opa(os.path.join(self.crd.directory, self.files_per_product_per_fileid[i_p][fileid][0]))
                self.import_classes[extension].get_data(filepath, j)
        else:
            raise Exception('Product not available')

    
    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        self.get_file_availability_info()
        extension = self.get_extension()
        if extension == 'hd5':
            filepaths = {i: [opa(os.path.join(self.crd.directory, k)) for k in j] for i,j in self.files_per_product_per_fileid[product].items()}
            return self.import_classes[extension].get_data_multiple_scans(filepaths,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
        else:
            filepaths = {i: opa(os.path.join(self.crd.directory, j[0])) for i,j in self.files_per_product_per_fileid[product].items()}
            return self.import_classes[extension].get_data_multiple_scans(filepaths,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)


    def get_product_versions(self, filenames, datetimes):
        if self.get_extension(filenames[0]) == 'hd5':
            files_product_versions = np.array([j[:j.index('_sweeph5onem')] for j in filenames])
    
            self.files_product_versions_datetimesdict, product_versions_datetimesdict = {}, {}
            for j in np.unique(datetimes):
                self.files_product_versions_datetimesdict[j] = files_product_versions[datetimes == j]
                product_versions_datetimesdict[j] = list(np.unique(self.files_product_versions_datetimesdict[j]))
                if len(product_versions_datetimesdict[j]) > 1:
                    # Add a product version for a combination of the 2 individual product versions
                    product_versions_datetimesdict[j].append('combi')
                    
            products_version_dependent = ['z', 'v'] # i.e. all products available for DWD
            product_versions_in1file = False
            return product_versions_datetimesdict, products_version_dependent, product_versions_in1file
        else:
            return None, None, None
 
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]        
        return np.sort([j for j in entries if any([j.endswith(i) for i in self.import_classes])])
    
    def get_datetimes_from_files(self,filenames,directory=None,dtype=str,return_unique_datetimes=True, mode='simple'):
        #Flooring to 5 minutes is performed, because the DWD puts data for each scan separately in files, with a corresponding range of datetimes, 
        #while all these files belong to the same radar volume, that starts at the datetime to which a datetime below gets floored.
        # Historical DWD BUFR files might have a different naming compared to files made available on DWD's open data server, 
        # in which case the datetime also needs to be determined in a different way. Also note that just indexing on '20'
        # doesn't work, because '20' can be a fileid
        datetimes=np.array([int(np.floor(int(j[j.index('-20')+1: j.index('-20')+13] if '-20' in j\
                                             else j[j.index('20'): j.index('20')+12])/5)*5) for j in filenames],dtype=dtype)
        return np.unique(datetimes) if return_unique_datetimes else datetimes   
    
    
    
    
class Source_TUDelft():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui = gui_class
        self.dsg = dsg_class
        self.crd = self.dsg.crd
        self.dp = self.dsg.dp
        self.pb = self.gui.pb
        
        
    
    def filepath(self):
        return os.path.join(self.crd.directory, self.dsg.files_datetime)
        
    def get_scans_information(self):
        return self.dsg.TUDelft_nc.get_scans_information(self.filepath())
        
        
    def get_data(self, j): #j is the panel
        return self.dsg.TUDelft_nc.get_data(self.filepath(), j)
        
        
    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return self.dsg.TUDelft_nc.get_data_multiple_scans(self.filepath(),product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
        
        
    def get_filenames_directory(self, radar, directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort(np.array([j for j in entries if j[-3:]=='.nc']))
        return filenames
        
        
    def get_datetimes_from_files(self,filenames,directory=None,dtype=str,return_unique_datetimes=True, mode='simple'):
        """This function actually returns both datetimes and dates instead of just datetimes. This is done because there is one file per date, and
        the dates are also used in nlr_datasourcegeneral.py.
        """
        dates = []
        for filename in filenames:
            dates += [ft.format_date(filename[5:15], 'YYYY-MM-DD->YYYYMMDD')]
        dates = np.array(dates, dtype = dtype)
        dt = gv.volume_timestep_radars[self.crd.radar]
        times = np.array([ft.time_to_minutes(j,inverse = True) for j in range(0, 1440, dt)], dtype=np.uint64)
        try:
            datetimes = np.concatenate(np.array(dates, dtype = np.uint64)[:, np.newaxis]*10000 + times).astype(dtype)
        except Exception: # happens when no dates are available
            datetimes = np.array([])
        return (datetimes, dates) if mode == 'dates' else datetimes
    
    
    
    
    
class Source_IMGW():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui = gui_class
        self.dsg = dsg_class
        self.crd = self.dsg.crd
        self.dp = self.dsg.dp
        self.pb = self.gui.pb
        
        
        
    def filepath(self,product,source_function=None):
        try:
            #First try to obtain a filename for the correct product
            filename=[i for i in self.dsg.files_datetime if i[16:-4] == gv.productnames_IMGW[product]][0] 
        except Exception:
            if source_function==self.get_scans_information:
                try:
                    #Try to find any file with the correct date and time, and if found, then determine the product contained in it
                    filename=self.dsg.files_datetime[0]
                    product='z' #The default product. 'z' does not need to be present, but the only information required for determining whether it is possible to 
                    #obtain the Nyquist velocities is that the product is not equal to 'v'.
                except Exception:
                    filename=''
            else:
                filename=''
        filepath=opa(os.path.join(self.crd.directory,filename))

        return [filepath, product] if source_function==self.get_scans_information else filepath
    
        
    def get_scans_information(self):
        return self.dsg.Gematronik_vol_rainbow5.get_scans_information(*self.filepath('v', source_function=self.get_scans_information))
        
        
    def get_data(self, j): #j is the panel        
        return self.dsg.Gematronik_vol_rainbow5.get_data(self.filepath(gv.i_p[self.crd.products[j]]), j)
        
        
    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        data, scantimes, volume_starttime, volume_endtime =\
            self.dsg.Gematronik_vol_rainbow5.get_data_multiple_scans(self.filepath(product),product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
        meta = {'using_unfilteredproduct': False, 'using_verticalpolarization': False}
        return data, scantimes, volume_starttime, volume_endtime, meta
        
        
    def get_filenames_directory(self, radar, directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort(np.array([j for j in entries if j[-4:]=='.vol']))
        return filenames
        
        
    def get_datetimes_from_files(self,filenames,directory=None,dtype=str,return_unique_datetimes=True, mode='simple'):
        datetimes=np.array([int(os.path.basename(j)[:12]) for j in filenames],dtype=dtype)
        return np.unique(datetimes) if return_unique_datetimes else datetimes   
    
    
    
    
    
class Source_DMI():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        
        
    def filepath(self):
        filepath=opa(os.path.join(self.crd.directory,gv.radar_ids[self.crd.radar]+'_'+self.crd.date+self.crd.time+'.vol.h5'))
        return filepath
    
    def get_scans_information(self):
        self.dsg.ODIM_hdf5.get_scans_information(self.filepath(),'v')

    def get_data(self, j): #j is the panel
        self.dsg.ODIM_hdf5.get_data(self.filepath(),j)
        self.crd.using_unfilteredproduct[j] = self.crd.products[j] == 'z' and self.crd.productunfiltered[j]

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return self.dsg.ODIM_hdf5.get_data_multiple_scans(self.filepath(),product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                    
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort(np.array([j for j in entries if j[-2:]=='h5' and j.startswith(gv.radar_ids[radar])]))
        return filenames
    
    def get_datetimes_from_files(self,filenames,directory=None,dtype=str,return_unique_datetimes=True, mode='simple'):
        datetimes=np.array([j[-19:-7] for j in filenames],dtype=dtype)
        return datetimes





class Source_CHMI():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        
        
    def filepath(self, product=None):
        if self.crd.radar in ('Skalky', 'Brdy-Praha'):
            filenames = [j for j in self.dsg.files_datetime if gv.productnames_CHMI.get(product, '') in j]
            return self.crd.directory+'/'+filenames[0] if filenames else None
        else:
            return self.crd.directory+'/'+self.dsg.files_datetime[0]
    
    def get_scans_information(self):
        self.dsg.ODIM_hdf5.get_scans_information(self.filepath('v'), 'v')

    def get_data(self, j): #j is the panel
        product = 'uz' if self.crd.products[j] == 'z' and self.crd.productunfiltered[j] else self.crd.products[j]
        self.dsg.ODIM_hdf5.get_data(self.filepath(product), j)
        self.crd.using_unfilteredproduct[j] = product == 'uz'

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return self.dsg.ODIM_hdf5.get_data_multiple_scans(self.filepath(product),product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                    
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames = np.sort(np.array([j for j in entries if '.h' in j[-5:]]))
        return filenames
    
    def get_datetimes_from_files(self,filenames,directory=None,dtype=str,return_unique_datetimes=True, mode='simple'):
        ext = os.path.splitext(filenames[0])[1] if filenames else ''
        floor_minutes = 5 if self.crd.radar in ('Skalky', 'Brdy-Praha') else 1
        datetimes = np.array([ft.floor_datetime(re.sub('[T_]', '', j[-len(ext)-15:-len(ext)-2]), floor_minutes) for j in filenames], dtype=dtype)
        return datetimes
        
    
    
    

class Source_MeteoFrance():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
                
        
    def get_file_availability_info(self):
        filenames = self.dsg.files_datetime
        
        self.filetypes_per_fileid, self.fileids_per_filetype, self.file_per_filetype_per_fileid = {}, {}, {}
        for j in filenames:
            filetype = 'PAG' if 'PAG' in j else 'PAM'
            fileid = j[j.index(filetype)+3]
                        
            ft.init_dict_entries_if_absent(self.fileids_per_filetype, filetype, list)
            ft.init_dict_entries_if_absent(self.filetypes_per_fileid, fileid, list)
            ft.init_dict_entries_if_absent(self.file_per_filetype_per_fileid, filetype, dict)
                
            self.filetypes_per_fileid[fileid].append(filetype)
            self.fileids_per_filetype[filetype].append(fileid)
            self.file_per_filetype_per_fileid[filetype][fileid] = j
            
    def get_scans_information(self):
        self.get_file_availability_info()
    
        filepaths = {}
        for fileid in self.filetypes_per_fileid:
            filetype = self.filetypes_per_fileid[fileid][0]
            filepaths[fileid] = opa(self.crd.directory+'/'+self.file_per_filetype_per_fileid[filetype][fileid])
            
        self.dsg.MeteoFrance_BUFR.get_scans_information(filepaths, self.filetypes_per_fileid)

    def get_data(self, j): #j is the panel
        self.get_file_availability_info()
    
        i_p, scan = gv.i_p[self.crd.products[j]], self.crd.scans[j]
        filetype, fileid = self.dsg.scannumbers_all[i_p][scan][self.dsg.scannumbers_forduplicates[self.crd.scans[j]]].split(',')[:2]
        filepath = opa(self.crd.directory+'/'+self.file_per_filetype_per_fileid[filetype][fileid])
        self.dsg.MeteoFrance_BUFR.get_data(filepath, j)

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        self.get_file_availability_info()
        
        i_p = gv.i_p[product]
        filepaths = {}
        for j in scans:
            filetype, fileid = self.dsg.scannumbers_all[i_p][j][0].split(',')[:2]
            filepaths[j] = opa(self.crd.directory+'/'+self.file_per_filetype_per_fileid[filetype][fileid])
        return self.dsg.MeteoFrance_BUFR.get_data_multiple_scans(filepaths,product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                    
    def get_filenames_directory(self, radar, directory):
        try:
            entries = os.listdir(directory)
        except Exception: entries = []
        filenames = np.sort([j for j in entries if any(i in j for i in ('PAM', 'PAG'))])
        return filenames
    
    def get_datetimes_from_files(self, filenames, directory=None, dtype=str, return_unique_datetimes=True, mode='simple'):
        if not len(filenames):
            return []
        if filenames[0].startswith('T_'):
            datetimes = [j[16:28] for j in filenames]
        else:
            _, _, current_dir_string, _ = self.dsg.get_variables(self.crd.selected_radar, self.crd.selected_dataset)
            # No date is given in the filename, it therefore has to be determined from the directory 
            date, _ = bg.get_date_and_time_from_dir(directory, current_dir_string, self.crd.selected_radar)
            datetimes = [date+j[12:16] for j in filenames]
        # -5 minutes, since end instead of start datetimes are given in the filenames
        datetimes = np.array([ft.next_datetime(j, -5) for j in datetimes], dtype=dtype)
        return np.unique(datetimes) if return_unique_datetimes else datetimes




    
class Source_NWS():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        self.import_classes = {2:self.dsg.NEXRAD_L2, 3:self.dsg.NEXRAD_L3}
        self.fileid_pos = {'N':2, 'T':3}
        # For NEXRAD 'Z' contains long-range low-res reflectivity, 'R' short-range higher-res
        # For TDWR 'Z' contains super-res reflectivity, 'R' legacy-res
        # In case of list the labels are listed in order of decreasing preference
        self.product_labels_l3 = {'z':{'N':['R','Z'], 'T':['Z','R']}, 'v':'V'}
        
        
    
    def get_level(self, filename=None):
        if not filename:
            filename = self.dsg.files_datetime[0]
        return 3 if not filename[5].isdigit() else 2
    def get_l3_type(self, filename=None):
        if not filename:
            filename = self.dsg.files_datetime[0]
        return filename[12] # Either 'N' for NEXRAD or 'T' for TDWR
    def l3_scandescr(self, l3_type, p_label, fileid):
        descr = f'_{l3_type}{p_label}'
        pos = self.fileid_pos[l3_type]
        return descr[:pos]+f'{fileid}'+descr[pos:]
    def get_product_labels(self, l3_type, product):
        labels = self.product_labels_l3[product]
        if type(labels) == dict:
            labels = labels[l3_type]
        return labels #can be either a 1-character string or a list with > 1 element. Both are iterable, so no need to put string in list
    
    def import_class(self):
        level = self.get_level()
        return self.dsg.NEXRAD_L2 if level == 2 else self.dsg.NEXRAD_L3
        
    def filepaths(self, product=None, scan=None, duplicate=None):
        level = self.get_level()
        if level == 2:
            return self.crd.directory+'/'+self.dsg.files_datetime[0]
        else:
            files = self.dsg.files_datetime
            l3_type = self.get_l3_type()
            fileids = np.array([j[j.index(f'_{l3_type}')+self.fileid_pos[l3_type]:][0] for j in files])
            filedatetimes = np.array([j[-12:] for j in files])
            indices = np.argsort([fileids[i]+filedatetimes[i] for i in range(len(files))])
            files, fileids = files[indices], fileids[indices]
            _files = {}
            for i,file in enumerate(files):
                fileid = fileids[i]
                fileproduct = [p for p,labels in self.product_labels_l3.items() if any(self.l3_scandescr(l3_type, j, fileid) in file for j in 
                                                                                       self.get_product_labels(l3_type, p))][0]
                ft.init_dict_entries_if_absent(_files, fileid, dict)
                ft.init_dict_entries_if_absent(_files[fileid], fileproduct, list)
                _files[fileid][fileproduct].append(self.crd.directory+'/'+file)
            if scan:
                # see NEXRAD_L3.get_scans_information for how scannumbers_all is set up
                fileid = self.dsg.scannumbers_all[product][scan][0][0]
                fileindex = self.dsg.scannumbers_all[product][scan][duplicate][1]
                return _files[fileid][product][fileindex]
            elif product:
                return {fnum:fdict[product] for fnum,fdict in _files.items()}
            else:
                return _files
        
    def get_scans_information(self):
        self.import_class().get_scans_information(self.filepaths())

    def get_data(self, j): #j is the panel
        i_p, scan = gv.i_p[self.crd.products[j]], self.crd.scans[j]
        filepath = self.filepaths(i_p, scan, self.dsg.scannumbers_forduplicates[scan])
        self.import_class().get_data(filepath, j)

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        i_p = gv.i_p[product]
        return self.import_class().get_data_multiple_scans(self.filepaths(i_p),product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                

    def get_product_versions(self, filenames, datetimes):
        fname = filenames[0]
        if self.get_level(fname) == 2:
            radar = fname[:4]
            if radar[0] == 'T': # No different product versions for TDWR
                return None, None, None
            i = fname.find('_V')
            file_V_version = -1 if i == -1 else int(fname[i+2:i+4])
            # In a small examination I found that volumes with reflectivity also available for V-scans, have a V version of 3 or 6,
            # hence the following choice. Real-time volumes (bzip2-compressed) from Iowa State don't have the V version in the filename however. 
            # But since currently files always have reflectivity for V-scans, it is then assumed that it's present.
            v_scan_present = file_V_version % 3 == 0 or not fname[-3:] == '.gz'
            product_versions = ['z_scan']+['v_scan']*v_scan_present
            product_versions_datetimesdict = {j:product_versions for j in datetimes} 
            products_version_dependent = ['z'] # Only 'z' has 2 versions available
            product_versions_in1file = True
            return product_versions_datetimesdict, products_version_dependent, product_versions_in1file
        else:
            return None, None, None
    
    def get_filenames_directory(self, radar, directory):
        try:
            entries = os.listdir(directory)
            level = self.get_level(entries[0])
            if level == 3:
                # There are many kinds of L3 files. Here only those that contain the desired products are kept
                entries = [j for j in entries if 'SDUS' in j]
                l3_type = self.get_l3_type(entries[0])
                _entries = []
                for p in self.product_labels_l3:
                    plabels = self.get_product_labels(l3_type, p)
                    for plabel in plabels:
                        hits = [j for j in entries if j[12] == l3_type and
                                ((j[13].isdigit() and j[14] == plabel) or (j[14].isdigit() and j[13] == plabel))]
                        if hits:
                            _entries += hits
                            break # Use only files for the first plabel for which files are available
                entries = _entries
        except Exception: 
            entries = []
        filenames = np.sort(entries)
        return filenames
    
    def get_datetimes_from_files(self, filenames, directory=None, dtype=str, return_unique_datetimes=True, mode='simple'):
        try:
            level = self.get_level(filenames[0])
            if level == 2:
                # Files downloaded from https://mesonet-nexrad.agron.iastate.edu contain an extra '_' between the radar ID and datetime
                datetimes = np.array([j[4:12]+j[13:17] if not j[4] == '_' else j[5:13]+j[14:18] for j in filenames], dtype=dtype)
            else:
                datetimes = np.array([ft.floor_datetime(j[-12:], 6) for j in filenames], dtype=dtype)
        except Exception:
            level = 0
            datetimes = np.array([], dtype=dtype)
        return np.unique(datetimes) if level == 3 and return_unique_datetimes else datetimes   
    




class Source_ARRC():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
    def import_class(self):
        return self.dsg.CFRadial if self.dsg.files_datetime[0].endswith('.nc') else self.dsg.DORADE
    
    def get_scannumbers(self, filenames):
        if filenames[0].endswith('.nc'):        
            return np.array([format(int(j[j.rindex('_s')+2:-3]), '02d') for j in filenames])
        else:
            scannumbers = [j[:j.rindex('.')+2] for j in filenames]
            return np.array([j[-4:] if j[-5] == '.' else '0'+j[-3:] for j in scannumbers])
        
    def get_filedatetimes(self, filenames):
        if filenames[0].endswith('.nc'):
            return np.array([j[6:14]+j[15:21] for j in map(os.path.basename, filenames)])
        else:
            return np.array([('20' if int(j[5:7]) < 50 else '19')+j[5:17] for j in map(os.path.basename, filenames)])

    def filepaths(self, scan=None, duplicate=None):
        files = self.dsg.files_datetime
        scannumbers = self.get_scannumbers(files)
        filedatetimes = self.get_filedatetimes(files)
        indices = np.argsort([scannumbers[i]+filedatetimes[i] for i in range(len(files))])
        files, scannumbers = files[indices], scannumbers[indices]
        _files = {}
        for i,file in enumerate(files):
            scannumber = scannumbers[i]
            ft.init_dict_entries_if_absent(_files, scannumber, list)
            _files[scannumber].append(self.crd.directory+'/'+file)
        if scan:
            scannumber = self.dsg.scannumbers_all['z'][scan][0]
            return _files[scannumber][duplicate]
        else:
            return _files
        
    def get_scans_information(self):
        self.import_class().get_scans_information(self.filepaths())

    def get_data(self, j): #j is the panel
        filepath = self.filepaths(self.crd.scans[j], self.dsg.scannumbers_forduplicates[self.crd.scans[j]])
        self.import_class().get_data(filepath, j)
        
    def get_filenames_directory(self, radar, directory):
        filenames = []
        for root,dirs,files in os.walk(directory, topdown=True):
            folder = root.replace('\\', '/')[len(directory):].lstrip('/')
            filenames += [folder+'/'*(len(folder) > 0)+file for file in files]
        filenames = np.array([j for j in filenames if os.path.basename(j).startswith('swp.') or j.endswith('.nc')])
        print(filenames)
        scannumbers = self.get_scannumbers(filenames)
        filedatetimes = self.get_filedatetimes(filenames)
        indices = np.argsort([filedatetimes[i]+num for i,num in enumerate(scannumbers)])
        # print(list(filenames[indices]))
        return filenames[indices]
    
    def get_datetimes_from_files(self, filenames, directory=None, dtype=str, return_unique_datetimes=True, mode='simple'):
        # print(filenames)
        scannumbers = self.get_scannumbers(filenames)
        datetimes = []
        for i, num in enumerate(scannumbers):
            file = os.path.basename(filenames[i])
            datetime = file[3:15] if file.endswith('.nc') else\
                       ('20' if int(file[5:7]) < 50 else '19')+file[5:15]
            if i == 0 or int(scannumbers[i-1].replace('.', ''))-int(num.replace('.', '')) > 5 or\
            ft.datetimediff_s(datetime, datetimes[-1]) > 600:
                volume_datetime = datetime
            datetimes.append(volume_datetime)
        # for i in range(len(filenames)):
        #     print(i, filenames[i], datetimes[i])
        return np.array(datetimes, dtype=dtype)