# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import sys
import os
opa=os.path.abspath
import numpy as np

import nlr_functions as ft



python_version=str(sys.version_info[0])+'.'+str(sys.version_info[1])+'.'+str(sys.version_info[2])
    
programdir=opa(os.path.dirname(os.getcwd()))
sys.path.append(opa(programdir+'/Python_files/vispy'))


radars_all, data_sources_all = [], []
radars, data_sources = {}, {}
rplaces_to_ridentifiers, radarcoords, radar_elevations, radar_towerheights, radar_bands = {}, {}, {}, {}, {}
with open(programdir+'/Input_files/radars_eu.txt', 'r', encoding='utf-8') as f:
    data = ft.list_data(f.read(), '\t')
with open(programdir+'/Input_files/radars_us.txt', 'r', encoding='utf-8') as f:
    data += ft.list_data(f.read(), '\t')
for j in data:
    if len(j) == 1:
        data_source = j[0]
        data_sources_all.append(data_source)
    else:
        radar = j[0]
        radars_all.append(radar)
        ft.init_dict_entries_if_absent(radars, data_source, list)
        radars[data_source].append(radar)
        data_sources[radar] = data_source
        rplaces_to_ridentifiers[radar] = j[1]*(j[1] != '.')
        radarcoords[radar] = [float(j[2]), float(j[3])]
        radar_elevations[radar] = int(j[5])
        radar_towerheights[radar] = int(j[4])-int(j[5])
        radar_bands[radar] = j[6]
    
def replace_characters(string):
    character_map = {'\u00F6':'oe','\u00FC':'ue','\u0144':'n','\u017c':'z','\u00F3':'o','\u015A':'S','\u0119':'e'}
    for character in character_map:
        string = string.replace(character, character_map[character])
    return string
radars_nospecialchar_names = {radar:replace_characters(radar) for radar in radars_all}    


radars_with_datasets=['Jabbeke','Wideumont']+radars['DWD']+radars['IMGW']+radars['DMI'] 
#Radars for which the data is distributed over two datasets; one with large maximum range but small Nyquist velocity, and the other vice versa.
radars_with_double_volume=('Den Helder','Herwijnen') 
#Radars for which the volume can devided into two parts, with slightly different scans. This is the case for the new radars of the KNMI.
radars_with_onefileperdate=('Cabauw',)
radars_with_adjustable_startazimuth=('Cabauw',)
        
radarsources_dirs_Default = {}
default_basedir = programdir+'/Radar_data'
for source in radars:
    datasets = np.unique(np.concatenate([['Z','V'] if j in radars_with_datasets else [''] for j in radars[source]]))
    for j in datasets:
        key = source+f'_{j}'*len(j)
        radarsources_dirs_Default[key] = default_basedir+'/'+source.replace(' ','_')+'/'
        if source == 'KNMI':
            radarsources_dirs_Default[key] += 'RAD${radarID}_OPER_O___TARVOL__L2__${date}T000000_${date+}T000000_0001'
        elif source in ('KMI', 'skeyes', 'IMGW', 'DMI', 'NWS', 'Météo-France'):
            radarsources_dirs_Default[key] += '${date}/${radar}'+f'_{j}'*len(j)
        elif source == 'DWD':
            radarsources_dirs_Default[key] += '${date}/${radar}_'+j+'/${time60}-${time60+}'
derivedproducts_dir_Default=programdir+'/Radar_data/Derived_products'

intervals_autodownload={'KNMI':300,'KMI':300,'skeyes':300,'VMM':300,'DWD':300,'TU Delft':300,'IMGW':600,'DMI':300,'NWS':300,'ARRC':300,'Météo-France':300}
timeoffsets_autodownload={'KNMI':[75,120,180,240],'KMI':[75,120,180,240],'skeyes':[75,120,180,240],'VMM':[75,120,180,240],'DWD':[45,105,165,225,285],'TU Delft': [45,105,165,225,285],'IMGW':[240,300,540,600],'DMI':[135,180,240,300],'NWS':list(range(0, 300, 30)),'ARRC':[0],'Météo-France':list(range(0, 300, 60))}
multifilevolume_autodownload={'KNMI':False,'KMI':True,'skeyes':True,'VMM':True,'DWD':True,'TU Delft':False,'IMGW':True,'DMI':False,'NWS':False,'ARRC':False,'Météo-France':True}
fileperscan_autodownload={'KNMI':False,'KMI':False,'skeyes':False,'VMM':False,'DWD':True,'TU Delft':False,'IMGW':False,'DMI':False,'NWS':False,'ARRC':False,'Météo-France':True}
api_keys = {'KNMI': ['opendata', 'sfcobs'],
            'DMI': ['radardata'],
            'Météo-France': ['radardata']}
max_download_errors_tooslow = 2 #Attempts to download are aborted when the download is too slow for more than max_download_errors_tooslow times.
max_download_errors_nottooslow = 1 #Attempts to download are aborted when an exception different from the TooSlowException is encountered more
#than max_download_errors_nottooslow times.

volume_timestep_radars={j:5 for j in radars_all} # Typical timestep between radar volumes in minutes
for j in radars['IMGW']+radars['DMI']:
    volume_timestep_radars[j]=10
# volume_timestep_radars['Cabauw'] = 1

volume_attributes_all = ('scannumbers_all','scanangles_all','radial_bins_all','radial_res_all','scans_doublevolume',
                        'nyquist_velocities_all_mps','low_nyquist_velocities_all_mps','high_nyquist_velocities_all_mps',
                        'radial_range_all','scanangles_all_m')
volume_attributes_save = [j for j in volume_attributes_all if not j in ('radial_range_all','scanangles_all_m')]
#Attributes that can be different for each import product, and are assigned values by using 'exec' in nlr_importdata.py
volume_attributes_p = ('scannumbers_all','scanangles_all','radial_bins_all','radial_res_all')
                    
products_all=('z','a','m','h','r','e','l','v','s','w','p','k','c','x','q','t','y','d')
#i_p = import_products
i_p={'z':'z','r':'z','v':'v','s':'v','w':'w','d':'d','p':'p','k':'k','c':'c','x':'x','q':'q','t':'t','y':'y','e':'z','a':'z','m':'z','h':'z','l':'z'}
#Unfiltered products have the letter 'u' prepended
productnames_KNMI={'z':'Z','v':'V','w':'W','d':'ZDR','p':'PhiDP','k':'KDP','c':'RhoHV','q':'SQI','t':'CCOR','y':'CPA','uz':'uZ','ud':'uZDR','up':'uPhiDP'}
#There are 3 different name formats for KMI data, and each name format corresponds to a different file extension
productnames_KMI={'hdf':{'z':'dbzh','v':'vrad','w':'wrad'},'h5':{'z':'dBZ','v':'V','w':'W','d':'ZDR','p':'PhiDP','k':'KDP','c':'RhoHV','uz':'dBuZ','up':'uPhiDP'},'vol':{'z':['z','Z','dBZ'],'v':['v','V'],'w':['w','W'],'d':'ZDR','p':'PhiDP','k':'KDP','c':'RhoHV','uz':'dBuZ','up':'uPhiDP'}}
productnames_KMI_possible_mistake={'hdf':{},'h5':{'v':'RhoHV'},'vol':{'v':'RhoHV'}}
#If product in productnames_KMI_possible_mistake, then the value to which the map points contains productnames that could lead to confusion when
#searching for the particular product, because the productname of the product of interest is contained in the productname(s) of (an)other product(s).
productnames_DWD={'hd5':{'z':'dbzh','v':'vradh'},'buf.bz2':{'z':'z','v':'v'},'buf':{'z':'z','v':'v'}}
productnames_TUDelft={'z':'equivalent_reflectivity_factor','v':'radial_velocity','w':'spectrum_width','d':'differential_reflectivity','p':'differential_phase','x':'linear_depolarisation_ratio'}
productnames_IMGW={'z':'dBZ','v':'V','w':'W','d':'ZDR','p':'PhiDP','k':'KDP','c':'RhoHV','up':'uPhiDP','uk':'uKDP'}
productnames_DMI={'z':'DBZH','v':'VRAD','w':'WRAD','d':'ZDR','c':'RHOHV','p':'PHIDP','x':'LDR'}
productnames_NEXRAD={'z':'REF','v':'VEL','w':'SW','d':'ZDR','c':'RHO','p':'PHI'}
productnames_ARRC={'z':'DBZ','v':'VEL','w':'WIDTH','d':'ZDR','c':'RHOHV','p':'PHIDP'}


productnames={'z':'Z','a':'PCAPPI','m':'Zmax','h':'CMH','r':'RI','v':'V','s':'SRV','w':'W','p':'PhiDP','k':'KDP','c':'CC','x':'LDR','d':'ZDR','q':'SQI','t':'CCOR','y':'CPA','e':'ETH','l':'VIL'}
productnames_cmaps={'z':'Z','a':'PCAP','m':'Zmax','h':'CMH','r':'RI','v':'V','s':'SRV','w':'W','p':'PhiDP','k':'KDP','c':'CC','x':'LDR','d':'ZDR','q':'SQI','t':'CCOR','y':'CPA','e':'ETH','l':'VIL'}
productnames_cmapstab={'z':'Reflectivity','a':'Pseudo CAPPI','m':'Maximum reflectivity','h':'Center of mass height','r':'Rain intensity','v':'Velocity','s':'Storm-relative velocity','w':'Spectrum width','p':'Differential phase','k':'Specific differential phase','c':'Correlation coefficient','x':'Linear depolarisation ratio','d':'Differential reflectivity','q':'Signal quality index','t':'Clutter correction','y':'Clutter phase alignment','e':'Echo top height','l':'Vertically integrated liquid'}
productunits_default={'z':'dBZ','a':'dBZ','m':'dBZ','h':'km','r':'mm/h','v':'kts','s':'kts','w':'kts','p':u'\u00b0','k':u'\u00b0'+'/km','c':'%','x':'','q':'','t':'','y':'','d':'dBZ','e':'km','l':'kg/m\u00B2'}
scale_factors_velocities={'m/s':1,'mph':2.23694,'kts':1.94384449,'km/h':3.6}
products_possibly_exclude_lowest_values=('z','r','a','m','e','l') #Products for which it is possible to exclude the lowest part of the product range, 
#by means of the choice of minimum product values for the colormap.

products_with_interpolation=('z','a','m','r','l')
products_with_interpolation_and_binfilling = ('z','a','m') #If interpolation is applied to these products, then empty radar bins get filled by the average product value in
#the 4 neighbouring bins, if at least there are enough non-empty neighbouring bins, and if the average at least exceeds a particular product value.

velocity_dealiasing_settings = ('dual-PRF', 'dual-PRF + Unet VDA', 'dual-PRF + Unet VDA + extra')

"""The data gets stored as 8- or 16-bits unsigned integers (uint), and the number of bits is given in products_data_nbits. 
products_maxrange gives the maximum range of data values that is supported. If a data value falls outside the supported range, 
then the lower or upper limit of this range is shown instead.
Further, to let interpolation work properly, it is necessary to fill masked data elements with a value that differs by a not too small amount
from the minimum value that is supported by the color map. If this is not done, then at the boundary of regions with data, half of the 
first neighbouring bin without data gets filled with data. It is realized by enlarging the color map range by a factor of 
interpolation_fac/(1-interpolation_fac) in the negative direction, where the lower limit of this new range becomes the masking 
value.
To assure that it is possible to use this value as the masking value (the corresponding integer value cannot be smaller than zero), I use 
cmaps_maxrange_masked in converting float values to uint. cmaps_maxrange_masked follows from cmaps_maxrange in the same way as the enlarged color map
range follows from the input color map range (as given in the color table). The lower limit of cmaps_maxrange_masked is low enough to ensure that 
the integer value corresponding to the masking value is >=0.
For products for which interpolation is not allowed, it is enough to assure that it is possible to let the integer masking value be 1 less
than the minimum data value that represent non-masked data (without letting it become negative). This is assured by taking the lower limit
of cmaps_maxrange_masked to be 1/(2**n_bits-2)*products_maxrange less than the lower limit of cmaps_maxrange.
"""
#'a' should have the same number of bits as 'r', 'because 'a' is used in the calculation of 'r'!
products_data_nbits={'z':8,'a':8,'m':8,'h':8,'e':8,'l':16,'r':16,'v':16,'s':16,'w':8,'c':16,'x':16,'q':16,'t':16,'y':16,'d':8,'p':16,'k':8}
#The values in products_maxrange are valid for the default scale factors given in scale_factors_Default.
#If values in products_maxrange are changed for any of the plain products, then it is necessary to change the corresponding product versions in nlr_derivedproducts.py,
#because the scale factors used in converting floats to uints have changed. 
products_maxrange={'z':[-35.,90.],'a':[-35.,90.],'m':[-35.,90.],'h':[0.,25.],'e':[0.,25.],'l':[0.,500.],'r':[-3.,3.],'v':[-1000.,1000.],'s':[-1000.,1000.],'w':[0.,25.],'c':[0.,100.],'x':[-50.,0.],'q':[0,1],'t':[0,1],'y':[0,1],'d':[-10.,20.],'p':[0.,360.],'k':[-10.,20.]}
cmaps_maxrange={'z':[-35.,90.],'a':[-35.,90.],'m':[-35.,90.],'h':[0.,25.],'e':[0.,25.],'l':[0.,200.],'r':[-3.,3.],'v':[-100.,100.],'s':[-100.,100.],'w':[0.,25.],'c':[0.,100.],'x':[-50.,0.],'q':[0,1],'t':[0,1],'y':[0,1],'d':[-10.,20.],'p':[0.,360.],'k':[-10.,20.]}
#cmaps_maxrange gives for each product the range of values that is supported for the color map. At maximum 256 colors can be displayed, so there is a 
#trade-off between range and resolution. This range should at least not be larger than products_maxrange.
products_maxrange={j:np.array(products_maxrange[j]) for j in products_maxrange}
cmaps_maxrange={j:np.array(cmaps_maxrange[j]) for j in cmaps_maxrange}

#The values of interpolation_fac are chosen for each product in such a way that interpolation produces reasonably looking results.
interpolation_fac={'z':0.3,'m':0.3,'a':0.3,'r':0.7,'l':0.013}
cmaps_maxrange_masked={}
for j in products_all:
    c_lim=cmaps_maxrange[j]
    c_range=c_lim[1]-c_lim[0]
    n_bits=products_data_nbits[j]
    if j in products_with_interpolation:
        if productunits_default[j] == 'dBZ':
            # Use dBZ spacing of exactly 0.5
            cmaps_maxrange_masked[j] = np.array([-37.5, 90.])
        else:
            cmaps_maxrange_masked[j]=c_lim-np.array([interpolation_fac[j]*c_range,0])
    else:
        #Make sure that the integer mask value that is used to indicate masked data can be at least 1 lower than the lowest integer data
        #value (the integer data values cannot be smaller than zero).
        #(2**n_bits-2 vs 2**n_bits-1 is used, because the range of integer values that represents non-masked data is 2**n_bits-2).
        cmaps_maxrange_masked[j]=c_lim-np.array([1/(2**n_bits-2)*c_range,0])
products_maxrange_masked={j:np.array([cmaps_maxrange_masked[j][0],products_maxrange[j][1]]) for j in products_all}

products_with_tilts=('z','v','s','w','p','d','k','c','x','q','t','y')
products_with_tilts_derived = ('s',)
# These products are not stored in memory, since they can be cheaply calculated from their import product:
products_with_tilts_derived_nosave = ('s',)
plain_products=('e','r','a','m','h','l')
plain_products_affected_by_double_volume=('a','r')
plain_products_with_parameters=('e','a','m','h','l')
plain_products_correct_for_SM = ['m','h','e','l']
plain_products_parameter_description={'e':'Minimum reflectivity echo tops (dBZ)','a':'PCAPPI height (km)','m':'Minimum height Zmax (km)','h':['Cap Z values at 56 dBZ in calculation (yes/no)', 'Mask values where VIL < threshold (kg/m\u00B2)'],'l':['Cap Z values at 56 dBZ in calculation (yes/no)', 'Minimum height/lower boundary of integration (km)']}
plain_products_show_max_elevations=('e','l','m','h')
plain_products_show_true_elevations=('a','r')

CAPPI_height_R=1.5

colortables_dirs_filenames_Default={'z':programdir+'/Input_files/Color_tables/colortable_Z.csv','a':programdir+'/Input_files/Color_tables/colortable_Z.csv','m':programdir+'/Input_files/Color_tables/colortable_Z.csv','h':programdir+'/Input_files/Color_tables/Default/colortable_ET_default.csv','r':programdir+'/Input_files/Color_tables/Default/colortable_RI_default.csv','v':programdir+'/Input_files/Color_tables/colortable_V.csv','s':programdir+'/Input_files/Color_tables/colortable_V.csv','w':programdir+'/Input_files/Color_tables/Default/colortable_W_default.csv','k':programdir+'/Input_files/Color_tables/Default/colortable_KDP_default.csv','c':programdir+'/Input_files/Color_tables/Default/colortable_CC_default.csv','x':programdir+'/Input_files/Color_tables/Default/colortable_LDR_default.csv','q':programdir+'/Input_files/Color_tables/Default/colortable_SQI_default.csv','t':programdir+'/Input_files/Color_tables/Default/colortable_SQI_default.csv','y':programdir+'/Input_files/Color_tables/Default/colortable_SQI_default.csv','p':programdir+'/Input_files/Color_tables/Default/colortable_PhiDP_default.csv','d':programdir+'/Input_files/Color_tables/Default/colortable_ZDR_default.csv','e':programdir+'/Input_files/Color_tables/Default/colortable_ET_default.csv','l':programdir+'/Input_files/Color_tables/Default/colortable_VIL_default.csv'}
colortables_dirs_filenames_NWS={'z':programdir+'/Input_files/Color_tables/NWS/colortable_Z.csv','a':programdir+'/Input_files/Color_tables/NWS/colortable_Z.csv','m':programdir+'/Input_files/Color_tables/NWS/colortable_Z.csv','v':programdir+'/Input_files/Color_tables/NWS/colortable_V.csv','s':programdir+'/Input_files/Color_tables/NWS/colortable_V.csv','w':programdir+'/Input_files/Color_tables/NWS/colortable_W.csv','c':programdir+'/Input_files/Color_tables/NWS/colortable_CC.csv','p':programdir+'/Input_files/Color_tables/NWS/colortable_PhiDP.csv','l':programdir+'/Input_files/Color_tables/NWS/colortable_VIL.csv'}
colortables_dirs_filenames_Default={i:opa(j) for i,j in colortables_dirs_filenames_Default.items()}
colortables_dirs_filenames_NWS={i:opa(j) for i,j in colortables_dirs_filenames_NWS.items()}

animation_frames_directory = opa(programdir+'/Generated_files/ani_frames')

vwp_sm_names = {'MW': '0-6 km MW', 'LM': 'Bunkers LM', 'RM': 'Bunkers RM', 'SM':'Observed SM', 'DTM':'Deviant TM'}
vwp_sm_colors = {'MW': 'orange', 'LM': [0,1,0,1], 'RM': 'cyan', 'SM':'magenta', 'DTM':'brown'}