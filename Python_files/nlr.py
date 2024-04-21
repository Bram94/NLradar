# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

from nlr_plotting import Plotting
from nlr_changedata import Change_RadarData
from nlr_currentdata import AutomaticDownload, DownloadOlderData, CurrentData
from VWP.nlr_vwp import GUI_VWP
import nlr_background as bg
import nlr_functions as ft
import nlr_globalvars as gv

from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# When setting QtCore.Qt.AA_EnableHighDpiScaling to True, the variable screen.devicePixelRatio() would be needed in the plotting code
# QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
# QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy, Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

from vispy import gloo

import sys
import numpy as np
import os
# os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1" #I haven't yet seen this doing anything
opa=os.path.abspath #opa should always be used when specifying a pathname
import io
import subprocess
import glob
import time as pytime
from PIL import Image
import av
from fractions import Fraction
import shutil
import pickle
import copy
import bisect



"""
General structure of the code:

The class GUI (gui) contains all the GUI-related code. Plotting (pb) performs all basic plotting actions (in VisPy), 
and Change_RadarData (crd) contains the part of the code that processes changes in time, radar or products. CurrentData (cd)
checks whether a download is needed in the case of current data, and downloads it when that's the case.
DataSource_Specific (dsg) is the class that handles all things that are specific to a particular data source, like the import of radar data
DerivedProducts (dp) contains the functions in which derived products are created.

There is interaction between the different classes, and in a particular class another class is referred to as 
self.[classabbreviation]. 
All variables and methods used in the code 'live' in a particular class, and when it is used in another class it is therefore
referred to as self.[classabbreviation].[variablename/methodname]. Most of the changes to a particular variable take place in its
living class, but there are some exceptions.
Variables that are predefined here below (and which are saved to the settings.pkl file) all have GUI as their living class,
with the exception of a number of variables that belong to/live in the crd class. These variables are indicated below.

nlr_functions contains small functions that can be used in general, also outside the code for this program.
nlr_background contains larger functions or functions that are more specific for use in this program.
nlr_globalvars contains global parameters that don't change within the application. 
"""



#Initialization of variables, mostly saved from a previous run of the application. 
#At first, each variable is assigned its default value (defined in nlr_globalvars), but if they are saved from a previous run this value is
#subsequently overwritten.

radarsources_dirs = {}
radardirs_additional = {}
radardata_dirs={}
radardata_dirs_indices={}
derivedproducts_dir=gv.derivedproducts_dir_Default
colortables_dirs_filenames={}
savefig_filename=gv.programdir+'/'
savefig_include_menubar = False
animation_filename=gv.programdir+'/'
ani_delay_ref = 'frame'
ani_delay = 25
ani_delay_end = 50
ani_sort_files = True
ani_group_datasets = False
ani_quality = 5

radardata_product_versions={} # Sometimes more than 1 version of a product is available (as determined in self.dsg.get_product_versions).
# This dictionary contains for each radar-dataset a string that identifies the currently selected product version.
selected_product_versions_ordered=[] # Ordered list of product versions that have been selected within the function 
# self.crd.change_product_version, with the most recently selected placed at the end. It contains each product version at most once, which means
# that earlier occurrences are removed from the list. It is used to ensure that a switch of product version persists when
# switching from radar/dataset. It is needed, because a switch of product version within self.crd.change_product_version only affects a single
# radar-dataset.
derivedproducts_filename_version=None


movefiles_parameters={'startdate':'','starttime':'','enddate':'','endtime':'','oldstructure':'','newstructure':''}
      
"""The variables in this section belong to/live in the crd class."""  
radar='Herwijnen'
scan_selection_mode='scanangle'
date='c'
time='c'
current_case_list_name=None
current_case_list=None
current_case=None
cases_offset_minutes=0
cases_looping_speed=0.5
cases_animation_window=np.array([-30, 15])
cases_use_case_zoom=True
cases_loop_subset=False
cases_loop_subset_ncases=10
dataset='V'
products=['z','z','z','z','z','v','v','v','v','v']
productunfiltered={j:False for j in range(10)}
polarization={j:'H' for j in range(10)}
apply_dealiasing = {j:True for j in range(10)}
dealiasing_setting = gv.velocity_dealiasing_settings[-1]
dealiasing_max_nyquist_vel = 100/gv.scale_factors_velocities['kts']
dealiasing_dualprf_n_it = 50
cartesian_product_res = 1.
cartesian_product_maxrange = 200.
scans=[1,2,3,4,5,1,2,3,4,5]
plot_mode='Single'

animation_duration=60
animation_speed_minpsec=90
animation_hold_lastframe=0.5
desired_timestep_minutes=0
max_timestep_minutes=180 # Value of 1e10 corresponds to empty input in menu widget
maxspeed_minpsec=1e10 # Value of 1e10 corresponds to empty input in menu widget
networktimeout=60
minimum_downloadspeed=0.01
api_keys = {} # Default values are assigned below when necessary
pos_markers_latlons = []
pos_markers_positions = []
pos_markers_latlons_save = []
stormmotion_save = {'sm':np.array([0,0]), 'radar':None}
use_storm_following_view = False
view_nearest_radar = False
radar_bands_view_nearest_radar = ['S', 'C', 'X']
data_selected_startazimuth = 0 #Selected (desired) start azimuth for the scans for radars in gv.radars_with_adjustable_startazimuth
show_vwp = False
include_sfcobs_vwp = True
vwp_manual_sfcobs = {} 
vwp_manual_axlim = [] 
vvp_range_limits = [2, 25]
vvp_height_limits = [0.1, 11.9]
vvp_vmin_mps = 3.
vwp_sigmamax_mps = None
vwp_shear_layers = {1:[0,1], 2:[0,3], 3:[0,6], 4:[1,6]}
vwp_vorticity_layers = {1:[0,0.5], 2:[0,1]}
vwp_srh_layers = {1:[0,1], 2:[0,2], 3:[0,3]}
vwp_sm_display = {j:True for j in gv.vwp_sm_names if not j == 'SM'}

base_url_obs = 'https://kachelmannwetter.com/de/messwerte/18584bea4ec779beb796d3770ed37f8e/'

cmaps_minvalues={j:'' for j in gv.products_all}
for p in gv.products_all:
    if gv.i_p[p] == 'z':
        cmaps_minvalues[p] = -20
cmaps_maxvalues={j:'' for j in gv.products_all}

PP_parameter_values={}
PP_parameter_values['e']={1:18.5,2:30.,3:40.,4:50.}
PP_parameter_values['a']={1:0.3,2:0.5,3:1.5,4:3.}
PP_parameter_values['m']={1:0.5,2:1.5,3:5.,4:10.}
PP_parameter_values['h']={1:[True, 1], 2:[True, 3], 3:[True, 5], 4:[True, 10]}
PP_parameter_values['l']={1:[True, 0], 2:[False, 0], 3:[False, 0], 4:[False, 0]}
PP_parameters_panels={j:1 for j in range(10)}

max_radardata_in_memory_GBs=2
sleeptime_after_plotting=0.01
"""All variables that represent the state of a QCheckbox should take on values 0 or 2, where a state of 2 means checked! 
"""
use_scissor=2

dimensions_main={'height':0.32,'width':0.8} #Dimensions of the areas of the screen that are occupied by the color bars and titles
fontsizes_main={'titles':8,'cbars_labels':7,'cbars_ticks':6}
bgcolor=0.92*np.array([255,255,255])
panelbdscolor=np.array([75,75,75])

bgmapcolor=np.array([0,0,0])
mapvisibility=False
mapcolorfilter=(1.0,1.0,1.0,0.975) #Color display can differ per OS
maptiles_update_time = 0.1 #In seconds
radar_markersize=7.5
radar_colors={'Default':np.array([0,255,255]),'Selected':np.array([255,0,0]),'Automatic download':np.array([255,255,0]),'Automatic download + selected':np.array([255,128,0])}
lines_names=['countries','provinces','rivers','grid','heightrings']
lines_colors={'countries':np.array([255,255,255,255]),'provinces':np.array([255,255,0,150]),'rivers':np.array([0,255,255,130]),'grid':np.array([74,74,74,170]),'heightrings':np.array([74,74,74,220])}
lines_width=1.35
lines_antialias=True
show_heightrings_derivedproducts={j:not j in gv.plain_products_show_max_elevations for j in gv.plain_products}
showgridheightrings_panzoom=False
showgridheightrings_panzoom_time=0.6
gridheightrings_fontcolor={'bottom':np.array([0,0,0]),'top':np.array([255,255,255])}
gridheightrings_fontsize=10.8
grid_showtext=True 
lines_show=[j for j in lines_names if not j in ('rivers',)]
ghtext_names=['grid','heightrings']
ghtext_show=[j for j in ghtext_names]

reset_volume_attributes = True #Gets set to False in nlr_datasourcegeneral.py



variables_names_raw=['variables_resettodefault_version','reset_volume_attributes','radarsources_dirs','radardirs_additional','radardata_dirs','radardata_dirs_indices','derivedproducts_dir','derivedproducts_filename_version','radardata_product_versions','selected_product_versions_ordered','movefiles_parameters','radar','scan_selection_mode','date','time','current_case_list_name','current_case','cases_offset_minutes','cases_looping_speed','cases_animation_window','cases_use_case_zoom','cases_loop_subset','cases_loop_subset_ncases','animation_duration','animation_speed_minpsec','animation_hold_lastframe','desired_timestep_minutes','self.max_timestep_minutes','maxspeed_minpsec','dataset','products','productunfiltered','polarization','apply_dealiasing','dealiasing_setting','dealiasing_max_nyquist_vel','dealiasing_dualprf_n_it','cartesian_product_res','cartesian_product_maxrange','scans','plot_mode','savefig_filename','savefig_include_menubar','animation_filename','ani_delay_ref','ani_delay','ani_delay_end','ani_sort_files','ani_group_datasets','ani_quality','networktimeout','minimum_downloadspeed','api_keys','stormmotion_save','pos_markers_latlons','pos_markers_latlons_save','use_storm_following_view','view_nearest_radar','radar_bands_view_nearest_radar','data_selected_startazimuth','show_vwp','include_sfcobs_vwp','vwp_manual_sfcobs','vwp_manual_axlim','vvp_range_limits','vvp_height_limits','vvp_vmin_mps','vwp_sigmamax_mps','vwp_shear_layers','vwp_vorticity_layers','vwp_srh_layers','vwp_sm_display','base_url_obs','cmaps_minvalues','cmaps_maxvalues','PP_parameter_values','PP_parameters_panels','max_radardata_in_memory_GBs','sleeptime_after_plotting','use_scissor','colortables_dirs_filenames','dimensions_main','fontsizes_main','bgcolor','panelbdscolor','bgmapcolor','mapvisibility','mapcolorfilter','maptiles_update_time','radar_markersize','radar_colors','lines_colors','lines_show','lines_width','lines_antialias','ghtext_show','grid_showtext','show_heightrings_derivedproducts','showgridheightrings_panzoom','showgridheightrings_panzoom_time','gridheightrings_fontcolor','gridheightrings_fontsize','grid_showtext']
variables_names_withclassreference=['variables_resettodefault_version','self.reset_volume_attributes','self.radarsources_dirs','self.radardirs_additional','self.radardata_dirs','self.radardata_dirs_indices','self.derivedproducts_dir','self.derivedproducts_filename_version','self.radardata_product_versions','self.selected_product_versions_ordered','self.movefiles_parameters','self.crd.radar','self.crd.scan_selection_mode','self.crd.date','self.crd.time','self.current_case_list_name','self.current_case','self.cases_offset_minutes','self.cases_looping_speed','self.cases_animation_window','self.cases_use_case_zoom','self.cases_loop_subset','self.cases_loop_subset_ncases','self.animation_duration','self.animation_speed_minpsec','self.animation_hold_lastframe','self.desired_timestep_minutes','self.max_timestep_minutes','self.maxspeed_minpsec','self.crd.dataset','self.crd.products','self.crd.productunfiltered','self.crd.polarization','self.crd.apply_dealiasing','self.dealiasing_setting','self.dealiasing_max_nyquist_vel','self.dealiasing_dualprf_n_it','self.cartesian_product_res','self.cartesian_product_maxrange','self.crd.scans','self.crd.plot_mode','self.savefig_filename','self.savefig_include_menubar','self.animation_filename','self.ani_delay_ref','self.ani_delay','self.ani_delay_end','self.ani_sort_files','self.ani_group_datasets','self.ani_quality','self.networktimeout','self.minimum_downloadspeed','self.api_keys','self.stormmotion_save','self.pos_markers_latlons','self.pos_markers_latlons_save','self.use_storm_following_view','self.view_nearest_radar','self.radar_bands_view_nearest_radar','self.data_selected_startazimuth','self.show_vwp','self.include_sfcobs_vwp','self.vwp_manual_sfcobs','self.vwp_manual_axlim','self.vvp_range_limits','self.vvp_height_limits','self.vvp_vmin_mps','self.vwp_sigmamax_mps','self.vwp_shear_layers','self.vwp_vorticity_layers','self.vwp_srh_layers','self.vwp_sm_display','self.base_url_obs','self.cmaps_minvalues','self.cmaps_maxvalues','self.PP_parameter_values','self.PP_parameters_panels','self.max_radardata_in_memory_GBs','self.sleeptime_after_plotting','self.use_scissor','self.colortables_dirs_filenames','self.dimensions_main','self.fontsizes_main','self.bgcolor','self.panelbdscolor','self.bgmapcolor','self.mapvisibility','self.mapcolorfilter','self.maptiles_update_time','self.radar_markersize','self.radar_colors','self.lines_colors','self.lines_show','self.lines_width','self.lines_antialias','self.ghtext_show','self.grid_showtext','self.show_heightrings_derivedproducts','self.showgridheightrings_panzoom','self.showgridheightrings_panzoom_time','self.gridheightrings_fontcolor','self.gridheightrings_fontsize','self.grid_showtext']

#Variables that are reset to their default for the next update. Needs to be updated before every new update, 
#and 'variables_resettodefault_version' should always be included!!!!! reset_volume_attributes maybe too.
variables_resettodefault_forupdate=['variables_resettodefault_version', 'vwp_sm_display']
variables_resettodefault_version = 9 #Version for variables_resettodefault_forupdate. Number needs to be increased by 1 before every new update!!!!!

try:
    #pickle.load appears to be incompatible with changes in pyqt version, i.e. when the file is saved while using pyqt5, then it also needs pyqt5 for loading the file.
    settings_filename=opa(os.path.join(gv.programdir+'/Generated_files','stored_settings.pkl'))
    if os.path.exists(settings_filename):
        with open(settings_filename,'rb') as f:
            settings=pickle.load(f)
    else: settings={}
    
    # Deal with some variable name changings
    if 'KNMI_apikeys' in settings:
        api_keys['KNMI'] = settings['KNMI_apikeys']
    if 'bgcolor' in settings and not 'bgmapcolor' in settings:
        bgmapcolor = settings['bgcolor']
        del settings['bgcolor']
    
    resettodefault = not 'variables_resettodefault_version' in settings or variables_resettodefault_version != settings['variables_resettodefault_version']
        
    for name in variables_names_raw:
        try:
            if resettodefault and name in variables_resettodefault_forupdate:
                continue #Don't update these variables, since they are reset to their default value            
            elif name in settings:
                exec(name+"=settings[name]")
        except Exception:
            pass
except Exception:
    pass


if radar not in gv.radars_all:
    # Can happen when a radar has been removed from the radar meta files
    radar = gv.radars_all[0]

#Some variables are treated separately, because these are dictionaries for which the number of items (products) might vary between
#different versions of the application.   

for source in gv.radars:
    for i in gv.radars[source]:
        datasets = ('Z', 'V') if i in gv.radars_with_datasets else ('',)
        for j in datasets:
            radar_dataset = i+f'_{j}'*len(j)
            source_dataset = source+f'_{j}'*len(j)
            if not source_dataset in radarsources_dirs:
                radarsources_dirs[source_dataset] = gv.radarsources_dirs_Default[source_dataset]
            if not radar_dataset in radardata_dirs:
                radardata_dirs[radar_dataset] = radarsources_dirs[source_dataset]
                radardata_dirs_indices[radar_dataset] = 0
                radardata_product_versions[radar_dataset] = None
                  
                        
for i in gv.radars_all:                              
    #Remove files that have '.crdownload' as extension, as these are likely unfinished downloads that weren't removed when the program exited.
    #It is also possibly that these are files that are currently being downloaded, but in this case a PermissionError is raised, and the file
    #won't be deleted.
    for k in ('','_Z','_V'):
        try:
            download_directory=bg.get_download_directory(radardata_dirs[i+k])
            if os.path.exists(download_directory):
                files=os.listdir(download_directory)
                for file in files:
                    try:
                        os.remove(os.path.join(download_directory,file))
                    except Exception: pass
        except Exception: pass


for datasource in gv.api_keys:
    if not datasource in api_keys:
        api_keys[datasource] = {}
    for key in gv.api_keys[datasource]:
        if not key in api_keys[datasource]:
            api_keys[datasource][key] = ''
            
            
for j in gv.colortables_dirs_filenames_Default:
    if j not in colortables_dirs_filenames or not os.path.exists(colortables_dirs_filenames[j]):
        colortables_dirs_filenames[j]=gv.colortables_dirs_filenames_Default[j]
for j in gv.products_all:
    if j not in cmaps_minvalues:
        cmaps_minvalues[j]=''
for j in gv.products_all:
    if j not in cmaps_maxvalues:
        cmaps_maxvalues[j]=''

        




cases_lists_filename = opa(os.path.join(gv.programdir+'/Generated_files','cases_lists.pkl'))
if os.path.exists(cases_lists_filename):
    with open(cases_lists_filename, 'rb') as f:
        cases_lists=pickle.load(f)
else:
    cases_lists = {}

# current_case_list_name=None
# current_case=None
# PP_parameter_values['l']={1:[True, 0], 2:[False, 0], 3:[False, 0], 4:[False, 0]}
# PP_parameter_values['h']={1:[True, 1], 2:[True, 3], 3:[True, 5], 4:[True, 10]}
# vwp_sm_display = {j: True for j in ('MW', 'LM', 'RM')}
# vwp_shear_layers = {1:[0,1], 2:[0,3], 3:[0,6], 4:[1,6]}


class ListWidgetItem(QListWidgetItem):
    def __lt__(self, other):
        listwidget = self.listWidget()
        qlabel = listwidget.itemWidget(self)
        qlabel_other = listwidget.itemWidget(other)
        descr, other_descr = qlabel.text(), qlabel_other.text()
        s = sorted([descr, other_descr])
        return s[0] == descr
    
previous_screen_DPI = None
def screen_DPI(screen):
    global previous_screen_DPI
    try:
        screen_DPI = screen.physicalDotsPerInch()
    except RuntimeError:
        # Can happen when the monitor is disconnected
        screen_DPI = previous_screen_DPI if previous_screen_DPI else 96
    previous_screen_DPI = screen_DPI
    return screen_DPI

class GUI(QWidget):
    def __init__(self, parent=None):
        super(GUI, self).__init__(parent) 
        self.changing_fullscreen=False
        self.showMaximized()
        
        self.setWindowTitle('NLradar')
        
        # self.screen_DPI is a function! This is done to let it update automatically when the DPI changes after startup of the program.
        self.screen_DPI = lambda: screen_DPI(self.screen())
        self.screen_pixel_ratio = lambda: self.screen().devicePixelRatio()
        self.screen_size = lambda: np.array([self.screen().size().width(), self.screen().size().height()])
        print('screen_size=',self.screen_size())
        print('screen_DPI=',self.screen_DPI())        
        self.screen_physicalsize = lambda: self.screen_size()/self.screen_DPI()*2.54
        print('screen_physicalsize=',self.screen_physicalsize())
        print('scale_fac=',self.logicalDpiX() / 96.0)
        self.ref_screen_size=np.array([1920.,1080])
        self.ref_screen_DPI=141.58475185806762
        self.ref_screen_physicalsize=self.ref_screen_size/self.ref_screen_DPI*2.54

        
        #Variables with gui as their living class are assigned to gui here.
        self.radarsources_dirs = radarsources_dirs
        self.radardirs_additional = radardirs_additional
        self.radardata_dirs=radardata_dirs
        self.radardata_dirs_indices=radardata_dirs_indices
        self.derivedproducts_dir=derivedproducts_dir
        self.derivedproducts_filename_version=derivedproducts_filename_version
        self.radardata_product_versions=radardata_product_versions
        self.selected_product_versions_ordered=selected_product_versions_ordered
        self.movefiles_parameters=movefiles_parameters
        self.animation_duration=animation_duration
        self.animation_speed_minpsec=animation_speed_minpsec
        self.animation_hold_lastframe=animation_hold_lastframe
        self.desired_timestep_minutes=desired_timestep_minutes
        self.max_timestep_minutes=max_timestep_minutes
        self.maxspeed_minpsec=maxspeed_minpsec
        self.savefig_filename=savefig_filename
        self.savefig_include_menubar = savefig_include_menubar
        self.animation_filename=animation_filename
        self.ani_delay_ref = ani_delay_ref
        self.ani_delay = ani_delay
        self.ani_delay_end = ani_delay_end
        self.ani_sort_files = ani_sort_files
        self.ani_group_datasets = ani_group_datasets
        self.ani_quality = ani_quality
        self.dealiasing_setting = dealiasing_setting
        self.dealiasing_max_nyquist_vel = dealiasing_max_nyquist_vel
        self.dealiasing_dualprf_n_it = dealiasing_dualprf_n_it
        self.cartesian_product_res = cartesian_product_res
        self.cartesian_product_maxrange = cartesian_product_maxrange
        self.networktimeout=networktimeout
        self.minimum_downloadspeed=minimum_downloadspeed
        self.api_keys = api_keys
        self.pos_markers_latlons = pos_markers_latlons
        self.pos_markers_latlons_save = pos_markers_latlons_save
        self.stormmotion_save = stormmotion_save
        self.stormmotion = np.array([0,0], dtype='float32') #Don't use the saved storm motion vector, always start with no storm motion.
        self.use_storm_following_view = False
        self.view_nearest_radar = view_nearest_radar
        self.radar_bands_view_nearest_radar = radar_bands_view_nearest_radar
        self.data_selected_startazimuth = data_selected_startazimuth
        self.show_vwp = show_vwp
        self.include_sfcobs_vwp = include_sfcobs_vwp
        self.vwp_manual_sfcobs = vwp_manual_sfcobs
        self.vwp_manual_axlim = vwp_manual_axlim
        self.vvp_range_limits = vvp_range_limits
        self.vvp_height_limits = vvp_height_limits
        self.vvp_vmin_mps = vvp_vmin_mps
        self.vwp_sigmamax_mps = vwp_sigmamax_mps
        self.vwp_shear_layers = vwp_shear_layers
        self.vwp_vorticity_layers = vwp_vorticity_layers
        self.vwp_srh_layers = vwp_srh_layers
        self.vwp_sm_display = vwp_sm_display
        self.base_url_obs = base_url_obs
        self.cmaps_minvalues=cmaps_minvalues
        self.cmaps_maxvalues=cmaps_maxvalues
        self.PP_parameter_values=PP_parameter_values
        self.PP_parameters_panels=PP_parameters_panels
        self.max_radardata_in_memory_GBs=max_radardata_in_memory_GBs
        self.sleeptime_after_plotting=sleeptime_after_plotting
        self.use_scissor=use_scissor
        self.colortables_dirs_filenames=colortables_dirs_filenames
        self.dimensions_main=dimensions_main
        self.fontsizes_main=fontsizes_main
        self.bgcolor=bgcolor
        self.panelbdscolor=panelbdscolor
        self.bgmapcolor=bgmapcolor
        self.mapvisibility=mapvisibility
        self.mapcolorfilter=mapcolorfilter
        self.maptiles_update_time = maptiles_update_time
        self.radar_markersize=radar_markersize
        self.radar_colors=radar_colors
        self.lines_names=lines_names
        self.lines_colors=lines_colors
        self.lines_show=lines_show    
        self.lines_width=lines_width
        self.lines_antialias=lines_antialias
        self.ghtext_names=ghtext_names
        self.ghtext_show=ghtext_show    
        self.show_heightrings_derivedproducts=show_heightrings_derivedproducts
        self.showgridheightrings_panzoom=showgridheightrings_panzoom
        self.showgridheightrings_panzoom_time=showgridheightrings_panzoom_time
        self.gridheightrings_fontcolor=gridheightrings_fontcolor
        self.gridheightrings_fontsize=gridheightrings_fontsize
        self.grid_showtext=grid_showtext
        
        self.reset_volume_attributes = reset_volume_attributes
        
        self.current_case_list_name = current_case_list_name
        self.current_case_list = cases_lists[current_case_list_name] if not current_case_list_name is None else None
        self.current_case = current_case if not self.current_case_list is None and str(current_case) in self.get_cases_as_strings() else None
        self.previous_case_list_name = None
        self.previous_case = None
        self.cases_lists = cases_lists
        self.cases_offset_minutes = cases_offset_minutes
        self.cases_looping_speed = cases_looping_speed
        self.cases_animation_window = cases_animation_window
        self.cases_use_case_zoom = cases_use_case_zoom
        self.cases_loop_subset = cases_loop_subset
        self.cases_loop_subset_ncases = cases_loop_subset_ncases
        
        
                    
        self.pos_markers_positions = []
        self.sm_marker_present = False
        self.sm_marker_position = None
        self.sm_marker_latlon = None
        self.sm_marker_scantime = None
        self.sm_marker_scandatetime = None
        
        self.time_set_textbar_new=0
        self.time_last_removal_volumeattributes = 0
        self.switch_to_case_running=False
        self.move_to_next_case_call_ID=None
        self.move_to_next_case_running=False
        self.fullscreen=False
        self.need_rightclickmenu=False
        self.setting_saved_choice=False
        self.continue_savefig=False
        self.creating_animation = False
        self.exit=False #Set to True when exitting the program
        self.radars_automatic_download=[] #Radars for which currently data is automatically being downloaded.
        self.radars_download_older_data=[] #Radars for which older data is currently being downloaded.      
        #TODO:
   
        # First perform 'empty init' of self.pb, which allows use of self.pb during subsequent initialisation of classes, without getting issues
        # due to these other classes not being defined yet when referenced in self.pb.__init__.
        self.pb=Plotting(gui_class=self, empty_init=True)
        self.crd=Change_RadarData(gui_class=self, radar=radar, scan_selection_mode=scan_selection_mode, date=date, time=time, products=products, dataset=dataset, productunfiltered=productunfiltered, polarization=polarization, apply_dealiasing = apply_dealiasing, scans=scans, plot_mode=plot_mode)  
        self.dsg=self.crd.dsg
        self.dp=self.dsg.dp
        self.ani=self.crd.ani
        self.ad={}; self.dod={}; self.cd={}
        for j in gv.radars_all:
            self.ad[j]=AutomaticDownload(gui_class=self,radar=j)
            self.dod[j]=DownloadOlderData(gui_class=self,radar=j)
            self.cd[j]=CurrentData(gui_class=self,radar=j)
            self.cds = self.cd[j].cds #is the same for all radars
        #Enable self.crd to also use self.dod
        self.crd.dod=self.dod
        
        # Perform full init of self.pb
        self.pb.__init__(gui_class=self)
        self.vwp = self.pb.vwp
        self.gui_vwp = GUI_VWP(gui_class=self)
        
        

        self.datew=QLineEdit(self.crd.selected_date)
        self.datew.setToolTip('Date (YYYYMMDD or c, if the time is also c)')
        self.timew=QLineEdit(self.crd.selected_time)
        self.timew.setToolTip('Time (HHMM or c, if the date is also c)')
        
        self.download_startstopw=QPushButton('Start', autoDefault=True)
        self.download_startstopw.setToolTip('Start/Stop download of data, for the time range specified in the widget to the right')
        self.download_timerangew=QLineEdit('Download')
        self.download_timerangew.setToolTip('Time range (minutes) for which data is downloaded when clicking Start. Download starts at input date and time, and continues backward until time range is spanned.')
                
        self.animation_settingsw=QPushButton('Ani', autoDefault=True)
        self.animation_settingsw.setToolTip('Animation settings')
        self.desired_timestep_minutesw=QLineEdit(str(self.desired_timestep_minutes))
        self.desired_timestep_minutesw.setToolTip("Desired timestep (minutes) when pressing LEFT/RIGHT, can be set to 'V' for moving by one full radar volume")
        self.max_timestep_minutesw=QLineEdit(str(ft.rifdot0(self.max_timestep_minutes))*(self.max_timestep_minutes < 1e10))
        self.max_timestep_minutesw.setToolTip('Maximum allowed timestep (minutes) when pressing LEFT/RIGHT. Can be left empty for no maximum.') 
        self.maxspeed_minpsecw=QLineEdit(str(ft.rifdot0(self.maxspeed_minpsec))*(self.maxspeed_minpsec < 1e10))
        self.maxspeed_minpsecw.setToolTip('Maximum speed (minutes/second). Can be left empty for no maximum.')
          
        self.textbar=QLineEdit()
        self.textbar.setReadOnly(True)
        self.textbar.setToolTip('Shows messages, or shows (latitude, longitude), (x, y), distance to radar or marker, beam elevation, product value for mouse cursor/min/max values for product within view.')
        
        self.hodow=QPushButton('VWP', autoDefault=True)
        self.hodow.setToolTip('Display/hide radar-derived vertical wind profile')        

        self.casesw=QPushButton('Cases', autoDefault=True)
        self.casesw.setToolTip('Switch to one of the cases stored in your list(s)')
        
        self.help_font = QFont()
        
        self.savefig_include_menubarw = QCheckBox()
        self.savefig_include_menubarw.setTristate(False)
        self.savefig_include_menubarw.setCheckState(2*self.savefig_include_menubar)
        self.savefig_include_menubarw.setToolTip('Whether to include this menu bar when saving a figure')
    
        self.extraw=QPushButton('Extra', autoDefault=True)
        self.settingsw=QPushButton('Settings', autoDefault=True)
        self.helpw=QPushButton('Help', autoDefault=True)
             
        self.widgets = ('datew', 'timew', 'casesw', 'download_timerangew', 'download_startstopw', 'animation_settingsw', 'desired_timestep_minutesw', 'max_timestep_minutesw', 'maxspeed_minpsecw', 'textbar', 'hodow', 'savefig_include_menubarw', 'extraw', 'settingsw', 'helpw')
        self.f1 = QFont('Times')
        f2 = QFont('Consolas') # Use a monospace font for the textbar
        # self.f1.setPixelSize(int(round(self.pb.scale_pixelsize(14))))
        # QApplication.instance().setFont(self.f1)
        for w in self.widgets:
            widget = getattr(self, w)
            widget.setMinimumWidth(1)
            # setFixedHeight is currently disabled because it can lead to issues when moving the app to a different screen with different size, resolution etc
            # widget.setFixedHeight(QFontMetrics(f1).height())
            # widget.setStyleSheet( "margin: 0px;" )
            # widget.setMinimumHeight(int(round(self.pb.scale_pixelsize(24))))
            widget.setFont(f2 if w == 'textbar' else self.f1)
        hbox=QHBoxLayout()
        hbox.addWidget(self.datew,8)
        hbox.addWidget(self.timew,5)
        hbox.addWidget(self.download_startstopw,5)
        hbox.addWidget(self.download_timerangew,6)
        hbox.addWidget(self.animation_settingsw,4)
        hbox.addWidget(self.desired_timestep_minutesw,4)
        hbox.addWidget(self.max_timestep_minutesw,4)
        hbox.addWidget(self.maxspeed_minpsecw,4)
        hbox.addWidget(self.textbar,75)
        hbox.addWidget(self.hodow,5)
        hbox.addWidget(self.casesw,5)
        hbox.addStretch(12)
        hbox.addWidget(self.savefig_include_menubarw,2)
        hbox.addWidget(self.extraw,5)
        hbox.addWidget(self.settingsw,6)
        hbox.addWidget(self.helpw,5)
        
        self.layout=QVBoxLayout()
        self.layout.addLayout(hbox)
        self.plotwidget=QWidget()
        self.plotwidget_layout=QHBoxLayout()
        self.plotwidget_layout.addWidget(self.pb.native)
        self.plotwidget_layout.setSpacing(0)
        self.plotwidget_layout.setContentsMargins(0,0,0,0)
        self.plotwidget.setLayout(self.plotwidget_layout)
        self.layout.addWidget(self.plotwidget)
        self.layout.setSpacing(1)
        self.layout.setContentsMargins(1,1,1,1)
        self.setLayout(self.layout)
        
        self.datew.returnPressed.connect(self.crd.process_datetimeinput)
        self.timew.returnPressed.connect(self.crd.process_datetimeinput)
        self.casesw.clicked.connect(self.cases_menu)
        self.download_startstopw.clicked.connect(self.startstop_download_oldercurrentdata)
        self.animation_settingsw.clicked.connect(self.change_animation_settings)
        self.desired_timestep_minutesw.editingFinished.connect(self.change_desired_timestep_minutes)
        self.max_timestep_minutesw.editingFinished.connect(self.change_max_timestep_minutes)
        self.maxspeed_minpsecw.editingFinished.connect(self.change_maxspeed_minpsec)
        self.hodow.clicked.connect(self.change_show_vwp)
        self.savefig_include_menubarw.stateChanged.connect(self.change_savefig_include_menubar)
        self.extraw.clicked.connect(self.extra)
        self.settingsw.clicked.connect(self.settings)
        self.helpw.clicked.connect(self.helpwidget)
        
        
                    
        
        #At the start (self.firstplot_performed=False), try to find a file with as date self.crd.selected_date.
        k = QShortcut(QKeySequence('Return'),self.pb.native,lambda: self.crd.process_datetimeinput())
        # Only enable when self.pb.native is in focus, in order to allow for clicking menu buttons by using tab and enter
        k.setContext(Qt.WidgetShortcut)
        QShortcut(QKeySequence('Backspace'),self,lambda: self.crd.back_to_previous_plot(False))
        QShortcut(QKeySequence('SHIFT+Backspace'),self,lambda: self.crd.back_to_previous_plot(True))
        QShortcut(QKeySequence('LEFT'),self,lambda: self.crd.process_keyboardinput(-1,0,0,'0',None,False))
        QShortcut(QKeySequence('RIGHT'),self,lambda: self.crd.process_keyboardinput(1,0,0,'0',None,False))
        QShortcut(QKeySequence('SHIFT+LEFT'),self,lambda: self.crd.process_keyboardinput(-12,0,0,'0',None,False))
        QShortcut(QKeySequence('SHIFT+RIGHT'),self,lambda: self.crd.process_keyboardinput(12,0,0,'0',None,False))
        QShortcut(QKeySequence('ALT+LEFT'),self,lambda: self.ani.change_continue_type('leftright',-1))
        QShortcut(QKeySequence('ALT+RIGHT'),self,lambda: self.ani.change_continue_type('leftright',1))
        QShortcut(QKeySequence('SPACE'),self,lambda: self.ani.change_continue_type('ani',0))
        QShortcut(QKeySequence('ALT+C'),self,lambda: self.crd.plot_mostrecent_data(True))
        QShortcut(QKeySequence('ALT+SHIFT+C'),self,lambda: self.crd.plot_mostrecent_data(False))
        QShortcut(QKeySequence('CTRL+Return'),self,lambda: self.move_to_next_case(None,0))
        QShortcut(QKeySequence('CTRL+LEFT'),self,lambda: self.move_to_next_case(None,-1))
        QShortcut(QKeySequence('CTRL+RIGHT'),self,lambda: self.move_to_next_case(None,1))
        QShortcut(QKeySequence('CTRL+BACKSPACE'),self,self.back_to_previous_case)
        QShortcut(QKeySequence('CTRL+SPACE'),self,lambda: self.ani.change_continue_type('cases',0))
        QShortcut(QKeySequence('SHIFT+SPACE'),self,lambda: self.ani.change_continue_type('ani_case',0))
        QShortcut(QKeySequence('CTRL+SHIFT+SPACE'),self,lambda: self.ani.change_continue_type('ani_cases',0))
        QShortcut(QKeySequence('DOWN'),self,lambda: self.crd.process_keyboardinput(0,-1,0,'0',None,False))
        QShortcut(QKeySequence('UP'),self,lambda: self.crd.process_keyboardinput(0,1,0,'0',None,False))
        QShortcut(QKeySequence('SHIFT+DOWN'),self,lambda: self.crd.process_keyboardinput(0,-1.1,0,'0',None,False))
        QShortcut(QKeySequence('SHIFT+UP'),self,lambda: self.crd.process_keyboardinput(0,1.1,0,'0',None,False))
        
        QShortcut(QKeySequence('HOME'),self,self.pb.reset_panel_view)      
        QShortcut(QKeySequence('SHIFT+HOME'),self,lambda: self.pb.reset_panel_view(False))
        QShortcut(QKeySequence('CTRL+HOME'),self,lambda: self.pb.reset_panel_view(True, False))
        QShortcut(QKeySequence('F'),self,self.change_use_storm_following_view)
        
        for product in gv.products_all:        
            QShortcut(QKeySequence(product.upper()),self,lambda product=product: self.crd.process_keyboardinput(0,0,0,product,None,False))
        
        QShortcut(QKeySequence('SHIFT+U'),self,self.crd.change_productunfiltered)
        QShortcut(QKeySequence('SHIFT+P'),self,self.crd.change_polarization)
        QShortcut(QKeySequence('SHIFT+V'),self,self.crd.change_apply_dealiasing)
        QShortcut(QKeySequence('Alt+V'),self,self.select_dealiasing_settings)
        QShortcut(QKeySequence('SHIFT+Q'),self,self.change_plainproducts_parameters)
        QShortcut(QKeySequence('SHIFT+I'),self,self.pb.change_interpolation)
        QShortcut(QKeySequence('SHIFT+Z'),self,self.pb.change_radarimage_visibility)
        
        for j in range(1, 16):
            if j < 10: k = str(j)
            elif j == 10: k = '0'
            else: k = f'SHIFT+{j-10}'
            QShortcut(QKeySequence(k),self,lambda j=j: self.crd.process_keyboardinput(0,0,j,'0',None,False))
                
        for j in (1, 2, 3, 4, 6, 8, 10):
            k = j if not j == 10 else 0
            QShortcut(QKeySequence(f'ALT+{k}'),self,lambda j=j: self.pb.change_panels(j))
                        
        QShortcut(QKeySequence('SHIFT+D'),self,self.crd.change_dataset)
        QShortcut(QKeySequence('CTRL+D'),self,self.crd.change_dir_index)
        QShortcut(QKeySequence('CTRL+P'),self,self.crd.change_product_version)
        QShortcut(QKeySequence('SHIFT+S'),self,lambda: self.crd.change_scan_selection_mode('scan'))
        QShortcut(QKeySequence('SHIFT+E'),self,lambda: self.crd.change_scan_selection_mode('scanangle'))
        QShortcut(QKeySequence('SHIFT+H'),self,lambda: self.crd.change_scan_selection_mode('height'))
        
        QShortcut(QKeySequence('SHIFT+N'),self,lambda: self.change_plot_mode('Single'))
        QShortcut(QKeySequence('SHIFT+A'),self,lambda: self.change_plot_mode('All'))
        QShortcut(QKeySequence('SHIFT+R'),self,lambda: self.change_plot_mode('Row'))
        QShortcut(QKeySequence('SHIFT+C'),self,lambda: self.change_plot_mode('Column'))
                
        for j in range(1, 13):
            QShortcut(QKeySequence(f'SHIFT+F{j}'),self,lambda j=j: self.save_choice(j))
            QShortcut(QKeySequence(f'F{j}'),self,lambda j=j: self.set_choice(j))
        
        for j in range(1, 11):
            k = j if not j == 10 else 0
            QShortcut(QKeySequence(f'CTRL+{k}'),self,lambda j=j: self.crd.switch_to_nearby_radar(j))
            QShortcut(QKeySequence(f'CTRL+ALT+{k}'),self,lambda j=j: self.crd.switch_to_nearby_radar(j, False))
        QShortcut(QKeySequence('N'),self,self.change_view_nearest_radar)
        QShortcut(QKeySequence('ALT+N'),self,self.select_radar_bands_for_view_nearest_radar)

        QShortcut(QKeySequence('ALT+F1'),self,self.view_choices)
        QShortcut(QKeySequence('ALT+A'),self,self.show_archiveddays)
        QShortcut(QKeySequence('ALT+P'),self,self.show_scans_properties)
        
        QShortcut(QKeySequence('SHIFT+F'),self,self.show_fullscreen)
        QShortcut(QKeySequence('CTRL+S'),self,self.savefig)
        QShortcut(QKeySequence('CTRL+ALT+S'),self,self.change_continue_savefig)
        QShortcut(QKeySequence('CTRL+SHIFT+S'),self,lambda: self.change_continue_savefig(True))
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showrightclickMenu)
                         
    
        
        
        
    def set_textbar(self,text=None,color=None,minimum_display_time=None):
        if pytime.time()>self.time_set_textbar_new:
            
            if text is None or color is None:
                if self.cd[self.crd.selected_radar].cd_message==None: color='black'
                else: color='green' if self.cd[self.crd.selected_radar].cd_message_type in ('Progress_info','Download_info') else 'red'
                if self.continue_savefig and not color=='red':
                    color='blue'
                self.textbar.setStyleSheet('QLineEdit {color:'+color+'}')
                textbar_text=''
                if self.pb.datareadout_text!=None and self.cd[self.crd.selected_radar].cd_message_type!='Error_info': 
                    textbar_text+=self.pb.datareadout_text
                if self.pb.radar_mouse_selected!=None and self.cd[self.crd.selected_radar].cd_message_type!='Error_info': 
                    textbar_text+=', '+self.pb.radar_mouse_selected
                if self.cd[self.crd.selected_radar].cd_message!=None:
                    textbar_text+=', '+self.cd[self.crd.selected_radar].cd_message
                if self.ad[self.crd.selected_radar].timer_info!=None and self.cd[self.crd.selected_radar].cd_message_type!='Error_info':
                    textbar_text+=', '+self.ad[self.crd.selected_radar].timer_info
                if self.current_case_shown():
                    case = self.current_case
                    label = case['label']
                    if 'extra_datetimes' in case:
                        # Also for extra datetimes a label can have been provided. In this case use the label (excluding empty strings)
                        # for which the scantime is closest to the current scantime
                        case_st = case.get('scantime', case['datetime'][8:10]+':'+case['datetime'][-2:]+':00')
                        extra_sts = [j.get('scantime', i[8:10]+':'+i[-2:]+':00') for i,j in case['extra_datetimes'].items() if j['label']]
                        sts = extra_sts+[case_st]
                        time = self.crd.time
                        ref_st = time[:2]+':'+time[-2:]+':00' if (self.pb.data_empty[0] or self.pb.data_isold[0]) else self.pb.data_attr['scantime'][0]
                        i_nearest_st = np.argmin([abs(ft.scantimediff_s(j, ref_st)) for j in sts])
                        if i_nearest_st != len(sts)-1: # Otherwise the scantime of the case is nearest, so the case label remains used
                            label = list(case['extra_datetimes'].values())[i_nearest_st]['label']
                    textbar_text += '. '+label
                        
                if textbar_text[:2] == ', ': 
                    textbar_text=textbar_text[2:]
            else:
                textbar_text=text
                self.textbar.setStyleSheet('QLineEdit {color:'+color+'}')
                
            if not minimum_display_time is None:
                #No new text will be displayed for a time of minimum_display_time seconds.
                self.time_set_textbar_new=pytime.time()+minimum_display_time
            
            self.textbar.setText(textbar_text)
            self.textbar.setCursorPosition(0)
            self.textbar.repaint()
        
                
    def change_animation_settings(self):
        self.animation_settings=QWidget();
        self.animation_settings.setWindowTitle('Change animation settings')
        animation_settings_layout=QFormLayout()
        self.animation_durationw=QLineEdit(str(ft.rifdot0(ft.r1dec(self.animation_duration))))
        self.animation_speed_minpsecw=QLineEdit(str(ft.rifdot0(ft.r1dec(self.animation_speed_minpsec))))
        self.animation_hold_lastframew=QLineEdit(str(ft.rifdot0(ft.rndec(self.animation_hold_lastframe,2))))
        animation_settings_layout.addRow(QLabel('Duration (minutes)'),self.animation_durationw)
        animation_settings_layout.addRow(QLabel('Speed (minutes/second)'),self.animation_speed_minpsecw)
        animation_settings_layout.addRow(QLabel('Hold last frame (seconds)'),self.animation_hold_lastframew)
        self.animation_durationw.editingFinished.connect(self.change_animation_duration)
        self.animation_speed_minpsecw.editingFinished.connect(self.change_animation_speed_minpsec)
        self.animation_hold_lastframew.editingFinished.connect(self.change_animation_hold_lastframe)
        self.animation_settings.setLayout(animation_settings_layout)
        self.animation_settings.resize(self.animation_settings.sizeHint())
        self.animation_settings.show()
        
    def change_animation_duration(self):
        input_duration=self.animation_durationw.text()
        number=ft.to_number(input_duration)
        if not number is None and number>0:
            self.animation_duration=int(number)
        else: self.animation_durationw.setText(str(ft.rifdot0(self.animation_duration)))
    def change_animation_speed_minpsec(self):
        input_speed=self.animation_speed_minpsecw.text()
        number=ft.to_number(input_speed)
        if not number is None and number>0:
            self.animation_speed_minpsec=number
        else: self.animation_speed_minpsecw.setText(str(ft.rifdot0(self.animation_speed_minpsec)))
    def change_animation_hold_lastframe(self):
        input_hold_lastframe=self.animation_hold_lastframew.text()
        number=ft.to_number(input_hold_lastframe)
        if not number is None and number>0:
            self.animation_hold_lastframe=number
        self.animation_hold_lastframew.setText(str(ft.rifdot0(self.animation_hold_lastframe)))
            
            
    def change_desired_timestep_minutes(self):
        input_desired_timestep_minutes=self.desired_timestep_minutesw.text()
        number=ft.to_number(input_desired_timestep_minutes)
        if input_desired_timestep_minutes.upper() == 'V':
            self.desired_timestep_minutes = 'V'
        elif not number is None and number >= 0.:
            self.desired_timestep_minutes=number
        else:
            self.desired_timestep_minutesw.setText(str(self.desired_timestep_minutes))
            
    def change_max_timestep_minutes(self):
        input_max_timestep_minutes=self.max_timestep_minutesw.text()
        if input_max_timestep_minutes == '':
            input_max_timestep_minutes = '1e10'
        number=ft.to_number(input_max_timestep_minutes)
        if not number is None and number>0:
            self.max_timestep_minutes=float(number)
        else: 
            self.max_timestep_minutesw.setText(str(ft.rifdot0(self.max_timestep_minutes))*(self.max_timestep_minutes < 1e10))        
                            
    def change_maxspeed_minpsec(self):
        input_maxspeed_minpsec=self.maxspeed_minpsecw.text()
        if input_maxspeed_minpsec == '':
            input_maxspeed_minpsec = '1e10'
        number=ft.to_number(input_maxspeed_minpsec)
        if not number is None and number>0:
            self.maxspeed_minpsec=number
        else:
            self.maxspeed_minpsecw.setText(str(ft.rifdot0(self.maxspeed_minpsec))*(self.maxspeed_minpsec < 1e10))
                    
            
    def change_use_storm_following_view(self):
        self.use_storm_following_view = not self.use_storm_following_view
        
    def change_view_nearest_radar(self):
        self.view_nearest_radar = not self.view_nearest_radar
        
    def select_radar_bands_for_view_nearest_radar(self):
        self.radar_bands_for_view_nearest_radar = QWidget()
        self.radar_bands_for_view_nearest_radar.setWindowTitle('Radar wavelength bands')
                
        self.radar_bands_view_nearest_radarw = {}
        hbox = QHBoxLayout()
        for j in ('S', 'C', 'X'):
            self.radar_bands_view_nearest_radarw[j] = QCheckBox(j)
            self.radar_bands_view_nearest_radarw[j].setTristate(False)
            self.radar_bands_view_nearest_radarw[j].setCheckState(2*(j in self.radar_bands_view_nearest_radar))
            self.radar_bands_view_nearest_radarw[j].stateChanged.connect(self.change_radar_bands_view_nearest_radar)
            hbox.addWidget(self.radar_bands_view_nearest_radarw[j])
            
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Select which radar wavelength bands to include in search for nearest radar'))
        layout.addLayout(hbox)       
        self.radar_bands_for_view_nearest_radar.setLayout(layout)
        self.radar_bands_for_view_nearest_radar.resize(self.radar_bands_for_view_nearest_radar.sizeHint())
        self.radar_bands_for_view_nearest_radar.show()
        
    def change_radar_bands_view_nearest_radar(self):
        self.radar_bands_view_nearest_radar = [i for i,j in self.radar_bands_view_nearest_radarw.items() if j.checkState() == 2]
        
        
    def select_dealiasing_settings(self):
        self.select_dealiasing_settings = QWidget()
        self.select_dealiasing_settings.setWindowTitle('Select dealiasing settings')
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Setting used for velocity dealiasing'))
        
        self.dealiasing_settingsw = {}
        group = QButtonGroup()
        hbox = QHBoxLayout()
        for j in gv.velocity_dealiasing_settings:            
            self.dealiasing_settingsw[j] = QRadioButton(j)
            self.dealiasing_settingsw[j].setChecked(j == self.dealiasing_setting)
            self.dealiasing_settingsw[j].toggled.connect(lambda state, j=j: self.change_dealiasing_setting(j))
            group.addButton(self.dealiasing_settingsw[j])
            hbox.addWidget(self.dealiasing_settingsw[j])
        layout.addLayout(hbox)
        self.dealiasing_max_nyquist_velw = QLineEdit(str(ft.rifdot0(ft.r1dec(self.dealiasing_max_nyquist_vel*self.pb.scale_factors['v']))))
        self.dealiasing_max_nyquist_velw.editingFinished.connect(self.change_dealiasing_max_nyquist_vel)
        layout.addWidget(QLabel("Select maximum Nyquist velocity for which to apply the Unet VDA model. For higher Nyquist velocities"))
        layout.addWidget(QLabel("the model will not be applied, since aliasing becomes unlikely. You can request Nyquist velocities for"))
        layout.addWidget(QLabel(f"the current radar volume by pressing ALT+P. Unit: {self.pb.productunits['v']}."))
        layout.addWidget(self.dealiasing_max_nyquist_velw)
        
        self.dualprfdealiasing_n_itw = QLineEdit(str(self.dealiasing_dualprf_n_it))
        self.dualprfdealiasing_n_itw.editingFinished.connect(self.change_dualprfdealiasing_n_it)
        layout.addWidget(QLabel("Maximum number of iterations for dual-PRF dealiasing. Dealiasing continues until either convergence or"))
        layout.addWidget(QLabel("this number of iterations is reached."))
        layout.addWidget(self.dualprfdealiasing_n_itw)
            
        self.select_dealiasing_settings.setLayout(layout)
        self.select_dealiasing_settings.resize(self.select_dealiasing_settings.sizeHint())
        self.select_dealiasing_settings.show()
            
    def change_dealiasing_setting(self, setting):
        self.dealiasing_setting = setting
        
        if self.pb.firstplot_performed:
            panels_update = [j for j in self.pb.panellist if gv.i_p[self.crd.products[j]] == 'v']
            self.pb.set_newdata(panels_update)
            
    def change_dealiasing_max_nyquist_vel(self):
        input_val = self.dealiasing_max_nyquist_velw.text()
        number = ft.to_number(input_val)
        if not number is None and number > 0:
            self.dealiasing_max_nyquist_vel = number/self.pb.scale_factors['v']
        else:
            self.dealiasing_max_nyquist_velw.setText(str(ft.rifdot0(ft.r1dec(self.dealiasing_max_nyquist_vel*self.pb.scale_factors['v']))))
                    
    def change_dualprfdealiasing_n_it(self):
        number = ft.to_number(self.dualprfdealiasing_n_itw.text())
        if not number is None and number > 0:
            self.dealiasing_dualprf_n_it = int(number)
        self.dualprfdealiasing_n_itw.setText(str(self.dealiasing_dualprf_n_it))
        if any([self.crd.products[j] in ('v','s') for j in self.pb.panellist]) and self.crd.apply_dealiasing:
            self.pb.set_newdata(self.pb.panellist)
        
          
    def change_plainproducts_parameters(self):
        product=self.crd.products[self.pb.panel]
        if product in gv.plain_products_with_parameters:
            self.plainproducts_parameters=QWidget()
            self.plainproducts_parameters.setWindowTitle('Change product parameters')
            plainproducts_parameters_layout=QFormLayout()
            
            use_list = isinstance(gv.plain_products_parameter_description[product], list)
            hbox = QHBoxLayout() if not use_list else [QHBoxLayout() for j in gv.plain_products_parameter_description[product]]
            self.parameter_valuesw = {}
            for j in range(1,5):
                if use_list:
                    self.parameter_valuesw[j] = []
                    if product in ('h', 'l'):
                        # When more products use more than one parameter or use something different than a QLineEdit, then a better way of handling
                        # this is probably needed
                        self.parameter_valuesw[j] += [QCheckBox()]
                        self.parameter_valuesw[j][0].setTristate(False)
                        self.parameter_valuesw[j][0].setCheckState(2 if self.PP_parameter_values[product][j][0] else 0)
                        self.parameter_valuesw[j][0].stateChanged.connect(lambda state, j=j: self.change_parameter_value('stateChanged', j, 0))
                        self.parameter_valuesw[j] += [QLineEdit(str(ft.rifdot0(self.PP_parameter_values[product][j][1])))]
                        self.parameter_valuesw[j][1].editingFinished.connect(lambda j=j: self.change_parameter_value('editingFinished', j, 1))
                        self.parameter_valuesw[j][1].returnPressed.connect(lambda j=j: self.change_parameter_value('returnPressed', j, 1)) 
                    for i in range(len(hbox)):
                        hbox[i].addWidget(self.parameter_valuesw[j][i])
                else:
                    self.parameter_valuesw[j] = QLineEdit(str(ft.rifdot0(self.PP_parameter_values[product][j])))
                    self.parameter_valuesw[j].editingFinished.connect(lambda j=j: self.change_parameter_value('editingFinished', j))
                    self.parameter_valuesw[j].returnPressed.connect(lambda j=j: self.change_parameter_value('returnPressed', j)) 
                    hbox.addWidget(self.parameter_valuesw[j])
                    
            if use_list:
                for i in range(len(hbox)):
                    plainproducts_parameters_layout.addRow(QLabel(''), QLabel(gv.plain_products_parameter_description[product][i]))
                    plainproducts_parameters_layout.addRow(QLabel(''), hbox[i])
            else:
                plainproducts_parameters_layout.addRow(QLabel(''), QLabel(gv.plain_products_parameter_description[product]))
                plainproducts_parameters_layout.addRow(QLabel(''), hbox)
            self.plainproducts_parameters.setLayout(plainproducts_parameters_layout)
            self.plainproducts_parameters.resize(self.plainproducts_parameters.sizeHint())
            self.plainproducts_parameters.show()
            
    def change_PP_parameters_panels(self, product):
        panels_with_plainproduct = [i for i in self.pb.panellist if self.crd.products[i] == product]
        for i in range(0, len(panels_with_plainproduct)):
            self.PP_parameters_panels[panels_with_plainproduct[i]] = min([4, i+1])
        return panels_with_plainproduct
            
    def change_parameter_value(self, action, j, i=None):
        product = self.crd.products[self.pb.panel]
        if action == 'stateChanged':
            input_state = ft.from_list_or_nolist(self.parameter_valuesw[j], i).checkState()
            ft.to_list_or_nolist(self.PP_parameter_values[product], j, input_state == 2, i)
        else:
            input_value = ft.from_list_or_nolist(self.parameter_valuesw[j], i).text()
            number = ft.to_number(input_value)
            if not number is None:
                ft.to_list_or_nolist(self.PP_parameter_values[product], j, number, i)
        
        if action == 'returnPressed':
            self.plainproducts_parameters.close()
        if action in ('stateChanged', 'returnPressed'):
            panels_with_plainproduct = self.change_PP_parameters_panels(product)
            self.pb.set_newdata(panels_with_plainproduct,plain_products_parameters_changed=True)
                
                  
    def startstop_download_oldercurrentdata(self):
        if not gv.data_sources[self.crd.selected_radar] in self.cds.source_classes:
            self.set_textbar('Downloading not implemented for radars from '+gv.data_sources[self.crd.selected_radar], 'red', 1)
            return
        
        #Setting self.dod[self.crd.selected_radar].download_times from within this function (different thread) is thread-safe, see this article:
        #http://effbot.org/pyfaq/what-kinds-of-global-value-mutation-are-thread-safe.htm
        action=self.download_startstopw.text() #action can be 'Start' or 'Stop'
        if action == 'Stop':            
            self.dod[self.crd.selected_radar].terminate()
            self.cd[self.crd.selected_radar].cd_message = None
                
            self.reset_download_widgets(self.crd.selected_radar)
        else:
            input_download_timerange=self.download_timerangew.text()
            if ft.to_number(input_download_timerange) is None or '.' in input_download_timerange or int(input_download_timerange)<0:
                #input_download_timerange must be a positive integer.
                self.set_textbar('Incorrect time range','red',1)
                return
                
            download_timerange_s=int(input_download_timerange)*60
            self.dod[self.crd.selected_radar].download_timerange_s=download_timerange_s
            self.radars_download_older_data.append(self.crd.selected_radar)
            self.pb.set_radarmarkers_data()
            self.pb.update() #For plotting the markers
            
            self.download_startstopw.setText('Stop')
            #Start the thread self.dod[self.crd.selected_radar]. If it is already running, then it is enough to update only the download timerange.
            self.dod[self.crd.selected_radar].start()
                                
    def set_download_widgets(self,timerange,startstop):
        self.download_timerangew.setText(str(timerange))
        self.download_startstopw.setText(startstop)
                
    def reset_download_widgets(self, radar):
        self.set_download_widgets('Download','Start')
        #Also reset the marker colors
        if radar in self.radars_download_older_data:
            self.radars_download_older_data.pop(self.radars_download_older_data.index(radar))
        self.pb.set_radarmarkers_data()
        self.pb.update() #For plotting the markers

    def start_automatic_download(self,radar):
        if not gv.data_sources[radar] in self.cds.source_classes:
            self.set_textbar('Downloading not implemented for radars from '+gv.data_sources[radar], 'red', 1)
            return
        
        if not radar in self.radars_automatic_download:
            self.radars_automatic_download.append(radar)
        self.pb.set_radarmarkers_data()
        self.pb.update() #For plotting the markers
        self.ad[radar].start()
        
    def stop_automatic_download(self,radar):
        self.radars_automatic_download.pop(self.radars_automatic_download.index(radar))
        self.pb.set_radarmarkers_data()
        self.pb.update()
        self.ad[radar].terminate()
        self.ad[radar].stop_timer()
        self.cd[radar].cd_message = None
        

    def cases_menu(self):
        cases_menu = QMenu(self)
        cases_menu.setStyleSheet("QMenu::item{padding-left:5px; padding-right:5px;}\
                                  QMenu::item::selected{background-color:rgb(100,255,255);}\
                                  QMenu::item:default{padding-left:5px; padding-right:5px; color:#ff00ff;}")
        position = self.casesw.geometry()
        point = position.bottomLeft()

        if len(list(self.cases_lists)) == 0:
            action = cases_menu.addAction('No case lists have been created yet')
            action.setEnabled(False)
        else:
            action = cases_menu.addAction('Settings')
            action.triggered.connect(self.show_cases_settings_window)
            
            if self.current_case_list_name is None:
                self.current_case_list_name = list(self.cases_lists)[0]
            self.current_case_list = self.cases_lists[self.current_case_list_name]
            other_case_lists = {i:j for i,j in self.cases_lists.items() if not i == self.current_case_list_name}
            
            action = cases_menu.addAction('Unselect current case')
            action.triggered.connect(self.unselect_current_case)
            
            action = cases_menu.addAction('Modify case list or click URLs')
            action.triggered.connect(self.modify_case_list)
            
            delete_menu = cases_menu.addMenu('Delete case list')
            for list_name in self.cases_lists:
                action = delete_menu.addAction(list_name)
                action.triggered.connect(lambda state, list_name=list_name: self.delete_case_list(list_name))
            
            othercases_menu = cases_menu.addMenu('Select case from other list')
            othercases_menu.setStyleSheet("QMenu::item{padding-left:5px; padding-right:15px;}\
                                               QMenu::item::selected{background-color:rgb(100,255,255);}")
            max_length = 100
            if len(other_case_lists) == 0:
                action = othercases_menu.addAction('No other lists available yet')
                action.setEnabled(False)
            else:
                for list_name in other_case_lists:
                    case_list = other_case_lists[list_name]    
                    othercase_menu = othercases_menu.addMenu(list_name+f'  ({len(case_list)})')
                    if len(case_list) == 0:
                        action = othercase_menu.addAction('This list is currently empty')
                        action.setEnabled(False)
            
                    for i, case_dict in enumerate(case_list):
                        if i % max_length == 0:
                            menu = othercase_menu.addMenu(case_dict['datetime']+' -') if len(case_list) > max_length else othercase_menu
                        action = menu.addAction(self.get_descriptor_for_case(case_dict))
                        action.triggered.connect(
                            lambda state, list_name=list_name, case_dict=case_dict: self.switch_to_case(list_name, case_dict))
       
            cases_menu.addSeparator()
            action = cases_menu.addAction(self.current_case_list_name+f' ({len(self.current_case_list)} cases)')
            action.setEnabled(False)
            
            cases = self.get_cases_as_strings()
            current_case = str(self.current_case)
            i_current_case = cases.index(current_case) if not self.current_case is None else None
            action_before = None
            for i, case_dict in enumerate(self.current_case_list):
                if i % max_length == 0:
                    if (i_current_case != None and i <= i_current_case < i+max_length) or len(self.current_case_list) <= max_length:
                        menu = cases_menu
                    else:
                        dt1 = case_dict['datetime']
                        dt2 = self.current_case_list[min(i+max_length, len(self.current_case_list))-1]['datetime']
                        menu = QMenu(dt1+' - '+dt2, cases_menu)
                        # Use insertMenu instead of addAction, in order to always put these submenus above any listed actions in the main menu
                        cases_menu.insertMenu(action_before, menu)
                action = menu.addAction(self.get_descriptor_for_case(case_dict))
                if menu == cases_menu and action_before is None:
                    action_before = action
                if cases[i] == current_case:
                    menu.setDefaultAction(action)
                action.triggered.connect(
                            lambda state, list_name=self.current_case_list_name, case_dict=case_dict: self.switch_to_case(list_name, case_dict))
                
        cases_menu.popup(self.mapToGlobal(point))
        
    def get_descriptor_for_case(self, case_dict, short_date=True):
        description = case_dict['datetime'][2 if short_date else 0:]+' '+case_dict['radar']+' '+case_dict['label']
        if len(case_dict['pos_markers_latlons']) > 1:
            description += f" ({len(case_dict['pos_markers_latlons'])})"
        return description
    
    def show_cases_settings_window(self):
        self.cases_settings=QWidget()
        self.cases_settings.setWindowTitle('Case settings')
        cases_settings_layout=QFormLayout()
        
        self.cases_use_case_zoomw = QCheckBox(tristate=False)
        self.cases_use_case_zoomw.setCheckState(2 if self.cases_use_case_zoom else 0)
        cases_settings_layout.addRow(QLabel('Use zoom-level saved with case'),self.cases_use_case_zoomw)
        self.cases_use_case_zoomw.stateChanged.connect(self.change_cases_use_case_zoom)
        
        self.cases_offset_minutesw=QLineEdit(str(int(self.cases_offset_minutes)))
        cases_settings_layout.addRow(QLabel('Time offset relative to case (minutes, - sign for before case)'),self.cases_offset_minutesw)
        self.cases_offset_minutesw.editingFinished.connect(self.change_cases_offset_minutes)
        
        self.cases_animation_windoww=QLineEdit(', '.join(list(map(str, self.cases_animation_window))))
        cases_settings_layout.addRow(QLabel('Animation window: start_offset, end_offset (like time offset)'),self.cases_animation_windoww)
        self.cases_animation_windoww.editingFinished.connect(self.change_cases_animation_window)
        
        self.cases_looping_speedw=QLineEdit(str(self.cases_looping_speed))
        cases_settings_layout.addRow(QLabel('Looping speed (cases per second)'),self.cases_looping_speedw)
        self.cases_looping_speedw.editingFinished.connect(self.change_cases_looping_speed)
        
        self.cases_loop_subsetw = QCheckBox(tristate=False)
        self.cases_loop_subsetw.setCheckState(2 if self.cases_loop_subset else 0)
        cases_settings_layout.addRow(QLabel('Show only a subset of cases when looping'),self.cases_loop_subsetw)
        self.cases_loop_subsetw.stateChanged.connect(self.change_cases_loop_subset)
        
        self.cases_loop_subset_ncasesw=QLineEdit(str(self.cases_loop_subset_ncases))
        cases_settings_layout.addRow(QLabel('Size of subset of cases to loop over'),self.cases_loop_subset_ncasesw)
        self.cases_loop_subset_ncasesw.editingFinished.connect(self.change_cases_loop_subset_ncases)
        
        self.cases_settings.setLayout(cases_settings_layout)
        self.cases_settings.resize(self.cases_settings.sizeHint())
        self.cases_settings.show()
    def change_cases_use_case_zoom(self):
        self.cases_use_case_zoom = self.cases_use_case_zoomw.checkState() == 2
    def change_cases_offset_minutes(self):
        input_offset=self.cases_offset_minutesw.text()
        number=ft.to_number(input_offset)
        if not number is None:
            self.cases_offset_minutes=int(number)
        else: self.cases_offset_minutesw.setText(str(self.cases_offset_minutes))
    def change_cases_animation_window(self):
        input_window = self.cases_animation_windoww.text()
        try:
            start_offset, end_offset = map(int, input_window.split(','))
            if start_offset < end_offset:
                self.cases_animation_window = [start_offset, end_offset]
            else: raise Exception
        except Exception:
            self.cases_animation_windoww.setText(', '.join(list(map(str, self.cases_animation_window))))
    def change_cases_looping_speed(self):
        input_speed=self.cases_looping_speedw.text()
        number=ft.to_number(input_speed)
        if not number is None:
            self.cases_looping_speed=float(number)
        else: self.cases_looping_speedw.setText(str(self.cases_looping_speed))
    def change_cases_loop_subset(self):
        self.cases_loop_subset = self.cases_loop_subsetw.checkState() == 2
        # Updating loop_start_case_index here allows for repeatedly switching between using/not using a subset, while varying the subset 
        # of cases to loop over
        self.ani.loop_start_case_index = self.get_current_case_index()
    def change_cases_loop_subset_ncases(self):
        input_ncases=self.cases_loop_subset_ncasesw.text()
        number=ft.to_number(input_ncases)
        if not number is None:
            self.cases_loop_subset_ncases=int(number)
        else: self.cases_loop_subset_ncasesw.setText(str(self.cases_loop_subset_ncases))
    
    def unselect_current_case(self):
        self.current_case = None
        self.set_textbar()
    
    def delete_case_list(self, list_name):
        msg_box = QMessageBox()
        result = msg_box.question(self, 'Case list deletion', "Are you sure that you want to delete '"+list_name+"'?")
        if result == QMessageBox.Yes:
            del self.cases_lists[list_name]
            with open(cases_lists_filename,'wb') as f:
                pickle.dump(self.cases_lists,f)
            
            if list_name == self.current_case_list_name:
                self.current_case_list_name = None
                self.current_case = None
        
    def switch_to_case(self, list_name, case_dict, time_offset='default', from_animation=False):
        # Convert to integer in the latter case, because when called from within nlr_animate.py it is given as a string
        time_offset = self.cases_offset_minutes if time_offset == 'default' else int(time_offset)
            
        if list_name != self.current_case_list_name and hasattr(self, 'modify_case_listw'):
            self.modify_case_listw.close()
            
        self.previous_case_list_name = self.current_case_list_name
        self.previous_case = self.current_case
            
        self.current_case_list_name = list_name
        # the case_list is not directly given as input, because that copies the object, and causes self.cases_lists 
        # to not get updated when self.current_case_list is updated by rearranging or deleting cases.
        self.current_case_list = self.cases_lists[list_name]
        self.current_case = case_dict
        
        self.switch_to_case_running = True
        
        date, time = case_dict['datetime'][:8], case_dict['datetime'][-4:]
        date, time = ft.next_date_and_time(date, time, time_offset)
        self.datew.setText(date); self.timew.setText(time)
        
        radar = case_dict['radar']
        if 'scannumbers_forduplicates' in case_dict and time_offset == 0: # Was not saved in the past so
            self.dsg.scannumbers_forduplicates = case_dict['scannumbers_forduplicates'].copy()
            # Also set scannumbers_forduplicates_radars[radar] in addition to scannumbers_forduplicates itself, because scannumbers_forduplicates 
            # will be reset when self.dsg.update_parameters is called
            self.dsg.scannumbers_forduplicates_radars[radar] = self.dsg.scannumbers_forduplicates.copy()
        
        if self.show_vwp and 'vvp_range_limits' in case_dict: 
            self.vvp_range_limits = case_dict['vvp_range_limits']
            self.vvp_height_limits = case_dict['vvp_height_limits']
            self.vvp_vmin_mps = case_dict['vvp_vmin_mps']
            self.vwp.display_manual_sfcobs = case_dict['display_manual_sfcobs']
            self.vwp_manual_sfcobs = case_dict['vwp_manual_sfcobs']
        
        self.update_stormmotion(case_dict['stormmotion'] if 'stormmotion' in case_dict else np.array([0, 0]))
        self.crd.change_radar(radar)
        
        self.pos_markers_latlons = case_dict['pos_markers_latlons'].copy()
        self.update_pos_marker_widgets()
        self.pb.set_sm_pos_markers()
        
        # Changing the panel views now happens in self.pb.set_newdata
        
        self.set_textbar()        
        self.switch_to_case_running = False
                
    def get_cases_as_strings(self, case_list=None):
        if not case_list:
            case_list = self.current_case_list
        return [str(j) for j in case_list]
    def get_current_case_index(self):
        cases = self.get_cases_as_strings()
        return cases.index(str(self.current_case))
    
    def get_next_case(self, direction=1):
        current_index = self.get_current_case_index()
        new_index = np.mod(current_index+direction, len(self.current_case_list))
        if 'cases' in self.ani.continue_type and self.cases_loop_subset:
            case2 = self.ani.loop_start_case_index
            case1 = max(0, case2+1-self.cases_loop_subset_ncases)
            if new_index < case1 or new_index > case2:
                new_index = case1 if direction == 1 else case2
        return self.current_case_list[new_index]
        
    def move_to_next_case(self, call_ID=None, direction=1, time_offset='default', from_animation=False):
        if not self.current_case_list is None and not self.current_case is None:
            self.move_to_next_case_running = True
            next_case_dict = self.get_next_case(direction)
            self.switch_to_case(self.current_case_list_name, next_case_dict, time_offset, from_animation)
            self.move_to_next_case_running = False
        
        if not call_ID is None:
            self.move_to_next_case_call_ID=call_ID
            
    def back_to_previous_case(self):
        if self.previous_case:
            self.switch_to_case(self.previous_case_list_name, self.previous_case)

    def hover(self, url):
        if url:
            QToolTip.showText(QCursor.pos(), url)
        else:
            QToolTip.hideText()
            
    def get_case_text(self, case_dict):
        descr = self.get_descriptor_for_case(case_dict, short_date=False)
        url_full = ""; url_short = ""
        for url in case_dict['urls']:
            url_first_part = url.replace('https://','').replace('http://','').replace('www.','')[:15]
            url_full += " <A href='"+url+"'>"+url_first_part+"</a>"
            url_short += " "+url_first_part
        return descr, url_full, url_short
        
    def modify_case_list(self):
        self.modify_case_listw=QWidget()
        self.modify_case_listw.setWindowTitle('Modify case list')
        layout=QVBoxLayout()
        layout.addWidget(QLabel('Here you can rearrange or delete cases, copy or move cases to another list, or change case labels or add or click URLs.'))
        layout.addWidget(QLabel('Rearranging can be done either by sorting them by datetime, or by dragging and dropping them manually in the list'))
        layout.addWidget(QLabel('(after enabling rearrange mode). You can select multiple cases at once by using Ctrl/Shift.'))

        self.list_cases = QListWidget()
        self.list_cases.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_cases.setDragDropMode(self.list_cases.InternalMove)
        self.list_cases.setDragEnabled(False)
        self.list_cases.setContentsMargins(0, 0, 0, 0)
        self.list_cases.model().rowsMoved.connect(self.adjust_to_case_swap)
        self.list_cases_labels = {}
        i_current_case = None; current_case = str(self.current_case)
        for i in range(len(self.current_case_list)):
            case = self.current_case_list[i]
            if str(case) == current_case:
                i_current_case = i
            case_item = ListWidgetItem()
            self.list_cases.addItem(case_item)
            descr, url_full, url_short = self.get_case_text(case)
            key = descr+']---['+url_full+']---['+url_short
            self.list_cases_labels[key] = QLabel(descr+url_full)
            self.list_cases_labels[key].setOpenExternalLinks(True)
            # Line below is deactivated, since accessing links by keyboard apparently doesn't go together with QListWidget's item selection feature
            # self.list_cases_labels[key].setTextInteractionFlags(Qt.LinksAccessibleByMouse | Qt.LinksAccessibleByKeyboard)
            self.list_cases_labels[key].linkHovered.connect(self.hover)
            self.list_cases.setItemWidget(case_item, self.list_cases_labels[key])
        if not self.current_case is None:
            self.list_cases.setCurrentRow(i_current_case)
        layout.addWidget(self.list_cases)
        
        hbox = QHBoxLayout()
        button_sort = QPushButton('Sort by datetime', autoDefault=True); hbox.addWidget(button_sort)
        self.button_rearrange = QPushButton('Rearrange mode on', autoDefault=True); hbox.addWidget(self.button_rearrange)
        button_delete = QPushButton('Delete from list', autoDefault=True); hbox.addWidget(button_delete)
        self.button_copy = QPushButton('Copy to other list', autoDefault=True); hbox.addWidget(self.button_copy)
        self.button_move = QPushButton('Move to other list', autoDefault=True); hbox.addWidget(self.button_move)
        button_edit = QPushButton('Edit label/URLs', autoDefault=True); hbox.addWidget(button_edit)
        button_switch = QPushButton('Show case', autoDefault=True); hbox.addWidget(button_switch)

        button_sort.clicked.connect(self.sort_cases)
        self.button_rearrange.clicked.connect(self.enable_or_disable_rearrange_mode)
        button_delete.clicked.connect(self.delete_case_from_list)
        self.button_copy.clicked.connect(lambda: self.open_other_case_list_menu('copy'))
        self.button_move.clicked.connect(lambda: self.open_other_case_list_menu('move'))
        button_edit.clicked.connect(self.edit_case_label)
        button_switch.clicked.connect(self.switch_to_selected_case)
        layout.addLayout(hbox)
        
        self.modify_case_listw.setLayout(layout)
        self.modify_case_listw.resize(self.modify_case_listw.sizeHint())
        self.modify_case_listw.show()

    def sort_cases(self):
        keys = [''.join(self.get_case_text(case)) for case in self.current_case_list] 
        sort_indices = np.argsort(keys)
        self.list_cases.setSortingEnabled(True)
        self.list_cases.sortItems()
        save_list = self.current_case_list.copy()
        for i in range(len(keys)):
            self.current_case_list[i] = save_list[sort_indices[i]]
        with open(cases_lists_filename,'wb') as f:
            pickle.dump(self.cases_lists,f)
            
    def adjust_to_case_swap(self, parent, start, end, destination, row):
        save = self.current_case_list[start]
        if row < start:
            self.current_case_list[row+1:start+1] = self.current_case_list[row:start]
            self.current_case_list[row] = save
        else:
            self.current_case_list[start:row-1] = self.current_case_list[start+1:row]
            self.current_case_list[row-1] = save
        with open(cases_lists_filename,'wb') as f:
            pickle.dump(self.cases_lists,f)
        
    def enable_or_disable_rearrange_mode(self):
        self.list_cases.setDragEnabled(not self.list_cases.dragEnabled())
        for key in self.list_cases_labels:
            descr, url_full, url_short = key.split(']---[')
            if self.list_cases.dragEnabled():
                self.list_cases_labels[key].setText(descr+url_short)
                self.button_rearrange.setText('URL mode on')
            else:
                self.list_cases_labels[key].setText(descr+url_full)
                self.button_rearrange.setText('Rearrange mode on')
        
    def delete_case_from_list(self):
        selected_items = self.list_cases.selectedItems()
        if len(selected_items) == 0:
            return

        for case_item in selected_items:
            index = self.list_cases.row(case_item)
            self.list_cases.takeItem(index)
            self.current_case_list.pop(index)
            cases = self.get_cases_as_strings()
            if not str(self.current_case) in cases:
                if len(self.current_case_list) > 0:
                    self.current_case = self.current_case_list[0]
                else:
                    self.current_case = None
            
        with open(cases_lists_filename,'wb') as f:
            pickle.dump(self.cases_lists,f)

    def open_other_case_list_menu(self, case_action='copy'): # or 'move'
        menu = QMenu(self)
        position = self.button_copy.geometry()
        point = position.bottomLeft()
        
        other_case_lists = {i:j for i,j in self.cases_lists.items() if not i == self.current_case_list_name}
        for list_name in other_case_lists:
            action = menu.addAction(list_name+f'  ({len(self.cases_lists[list_name])})')
            function = self.copy_to_other_list if case_action == 'copy' else self.move_to_other_list
            action.triggered.connect(lambda state, list_name=list_name: function(list_name))
        menu.addSeparator()
        action = menu.addAction('Create a new list')
        action.triggered.connect(lambda: self.define_name_new_list(case_action+'_to_other_list'))
                
        menu.popup(self.modify_case_listw.mapToGlobal(point))
    def copy_to_other_list(self, list_name):
        selected_items = self.list_cases.selectedItems()
        for item in selected_items:        
            index = self.list_cases.row(item)
            self.add_new_case_to_list(self.current_case_list[index], list_name)
        with open(cases_lists_filename,'wb') as f:
            pickle.dump(self.cases_lists,f)
    def move_to_other_list(self, list_name):
        self.copy_to_other_list(list_name)
        self.delete_case_from_list()
            
    def edit_case_label(self):
        selected_items = self.list_cases.selectedItems()
        if len(selected_items) == 0:
            return
        index = self.list_cases.row(selected_items[0])
        self.set_label_for_case(self.current_case_list_name, self.current_case_list[index])
        
    def switch_to_selected_case(self):
        selected_items = self.list_cases.selectedItems()
        if len(selected_items) == 0:
            return
        index = self.list_cases.row(selected_items[0])
        self.switch_to_case(self.current_case_list_name, self.current_case_list[index])
        
        
    def show_radar_menu(self, radars, pos):
        menu = QMenu(self)
        for radar in radars:
            action = menu.addAction(radar)
            action.triggered.connect(lambda state, radar=radar: self.crd.change_selected_radar(radar))
        menu.popup(self.mapToGlobal(pos))
        
    def showrightclickMenu(self, pos):
        self.rightmouseclick_Qpos = pos
                
        """The order in which events are handled differs among different PC's (different for Linux and Windows at least). For Windows, showing the right-click menu 
        occurs after the mouse_release event in nlr_plotting has been handled, whereas for Linux it occurs immediately after the mouse_press event has been handled.
        Because this latter order causes problems, the menu is not immediately drawn in this case, but the command to draw it is given in the function mouse_info
        from nlr_plotting, and occurs after the mouse_release event has been handled.
        """
        if ft.point_inside_rectangle(self.pb.last_mousepress_pos,self.pb.wpos['main'])[0] and not self.pb.mouse_moved_after_press:
            if self.pb.mouse_hold_right: self.need_rightclickmenu=True; return #The function gets called from self.pb.on_mouse_release
            else: self.need_rightclickmenu=False
            
            if self.pb.radar_mouse_selected:
                # Check whether multiple radars are located very close to the selected one. In that case it is hard or impossible to select
                # each radar by mouse, so in this case the right-click menu lists all these close-proximity radars instead of the actions below.
                selected_radar_xy = self.pb.radarcoords_xy[gv.radars_all.index(self.pb.radar_mouse_selected)]
                distances_to_radars = np.linalg.norm(self.pb.radarcoords_xy-selected_radar_xy, axis=1)
                max_distance = self.pb.get_max_dist_mouse_to_marker(f=0.5)
                radars = [j for i,j in enumerate(gv.radars_all) if distances_to_radars[i] < max_distance]
                if len(radars) > 1:
                    self.show_radar_menu(radars, pos)
                    return
            
            menu=QMenu(self)
                            
            selected_radar = self.pb.radar_mouse_selected if self.pb.radar_mouse_selected else self.crd.selected_radar
            if self.ad[selected_radar].isRunning():
                action = menu.addAction('Stop automatic download for '+selected_radar)
                action.triggered.connect(lambda: self.stop_automatic_download(selected_radar))
            else:
                action = menu.addAction('Start automatic download for '+selected_radar)
                action.triggered.connect(lambda: self.start_automatic_download(selected_radar))
            
            menu.addSeparator()     
            
            action = menu.addAction('Set position marker: Mouse position')
            action.triggered.connect(lambda: self.set_pos_markers_properties('Mouse'))
                                                                                        
            action = menu.addAction('Set position marker: Coordinate input')
            action.triggered.connect(self.set_marker_coordinates)
            
            marker_selected = not self.pb.marker_mouse_selected_index is None
            if len(self.pos_markers_latlons) or len(self.pos_markers_latlons_save) == 0:
                action = menu.addAction('Remove this marker' if marker_selected else 'Remove all markers')
                action.triggered.connect(lambda: self.remove_marker('pos' if marker_selected else 'all'))
                if len(self.pos_markers_latlons) == 0:
                    action.setEnabled(False)
            else:
                action = menu.addAction('Restore position markers')
                action.triggered.connect(self.restore_pos_markers)
            
            menu.addSeparator()
            
            action = menu.addAction('Set SM marker')
            if not self.pb.firstplot_performed:
                action.setEnabled(False)
            action.triggered.connect(lambda: self.set_sm_marker_properties())
            
            action = menu.addAction('Remove SM marker')
            if not self.sm_marker_present: 
                action.setEnabled(False)
            action.triggered.connect(lambda: self.remove_marker('sm'))
            
            action = menu.addAction('Calculate SM vector from marker')
            if not self.sm_marker_present or\
            self.pb.data_attr['scantime'].get(self.pb.panel, self.sm_marker_scantime) == self.sm_marker_scantime:
                action.setEnabled(False)
            action.triggered.connect(self.set_stormmotion_from_marker)
            
            action = menu.addAction('Set SM vector manually')
            action.triggered.connect(self.set_stormmotion_manually)
            
            if self.stormmotion[1] != 0. or self.stormmotion_save['sm'][1] == 0.:
                action = menu.addAction('Reset SM vector')
                action.triggered.connect(lambda: self.change_stormmotion(np.array([0, 0])))
                if self.stormmotion[1] == 0.:
                    action.setEnabled(False)
            else:
                action = menu.addAction('Restore SM vector')
                action.triggered.connect(lambda: self.change_stormmotion(self.stormmotion_save, False))
            
            menu.addSeparator()
            
            action = menu.addAction('Adjust start azimuth of scans (certain radars)')
            if not self.crd.radar in gv.radars_with_adjustable_startazimuth:
                action.setEnabled(False)
            action.triggered.connect(self.set_data_selected_startazimuth)
            
            menu.addSeparator()

            if not self.current_case_list_name is None:
                action = menu.addAction(f'Add case to current list ({self.current_case_list_name})')
                action.triggered.connect(lambda state, list_name=self.current_case_list_name: self.set_label_for_case(list_name))
                if not self.pb.firstplot_performed: 
                    action.setEnabled(False)
                submenu = menu.addMenu('Add case to other list')
            else:
                submenu = menu.addMenu('Add case to list')            
            if not self.pb.firstplot_performed: 
                submenu.setEnabled(False)
                
            for list_name in self.cases_lists:
                if not list_name == self.current_case_list_name:
                    action = submenu.addAction(list_name+f'  ({len(self.cases_lists[list_name])})')
                    action.triggered.connect(lambda state, list_name=list_name: self.set_label_for_case(list_name))                    
            submenu.addSeparator()
            action = submenu.addAction('Create new list')
            action.triggered.connect(lambda: self.define_name_new_list('set_label_for_case'))
            
            action = menu.addAction('Update current case')
            action.triggered.connect(lambda: self.add_case_to_list(self.current_case_list_name, self.current_case, False))
            if not self.current_case_shown():
                action.setEnabled(False)
                
            action = menu.addAction('Edit current case label')
            action.triggered.connect(lambda: self.set_label_for_case(self.current_case_list_name, self.current_case))
            if not self.current_case_shown():
                action.setEnabled(False)
                
            if self.current_case_shown() and 'extra_datetimes' in self.current_case and self.crd.date+self.crd.time in self.current_case['extra_datetimes']:
                action = menu.addAction('Edit label for this extra time')
            else:
                action = menu.addAction('Add extra time to case for animating')
            action.triggered.connect(self.set_label_for_extra_datetime)
            if not self.current_case_shown():
                action.setEnabled(False)
                
            if self.current_case_shown() and 'extra_datetimes' in self.current_case and self.crd.date+self.crd.time in self.current_case['extra_datetimes']:
                action = menu.addAction('Remove this extra time from case')
                action.triggered.connect(lambda: self.add_or_remove_extra_datetimes_case('remove single'))
            else:
                action = menu.addAction('Remove extra times from case')
                action.triggered.connect(lambda: self.add_or_remove_extra_datetimes_case('remove all'))
            if not self.current_case_shown():
                action.setEnabled(False)
            
            menu.popup(self.mapToGlobal(pos))
        elif self.show_vwp and ft.point_inside_rectangle(self.pb.last_mousepress_pos,self.pb.wpos['vwp'])[0] and not self.pb.mouse_moved_after_press:
            self.gui_vwp.showrightclickMenu(pos)

    def current_case_shown(self, mode='loose'):
        if self.current_case is None:
            return False
        
        datetime = self.crd.date+self.crd.time
        case_dts = [self.current_case['datetime']]+list(self.current_case.get('extra_datetimes', {}))
        max_datetimediff_s = max([7200, 60*np.abs(self.cases_animation_window).max()])
        if mode == 'strict':
            return datetime == self.current_case['datetime']
        elif mode == 'loose':
            return any(abs(ft.datetimediff_s(datetime, j)) <= max_datetimediff_s for j in case_dts)
                    
            
    def set_marker_coordinates(self):
        self.marker_coordinates_widget = QWidget()
        self.marker_coordinates_widget.setWindowTitle('Set marker coordinates')
        layout = QVBoxLayout()
        formlayout = QFormLayout()
        
        self.pos_markers_latlonsw = {}
        n = len(self.pos_markers_latlons)
        for j in range(max([10, n+1])):
            self.pos_markers_latlonsw[j] = QLineEdit()
            if j < n:
                self.pos_markers_latlonsw[j].setText(', '.join(format(i, '.6f') for i in self.pos_markers_latlons[j]))
            formlayout.addRow(QLabel(f'{j+1}'), self.pos_markers_latlonsw[j])
            self.pos_markers_latlonsw[j].editingFinished.connect(lambda j=j: self.set_pos_markers_properties('Coordinates', j))
        
        layout.addWidget(QLabel('Input can be either in decimal or DMS format, and a - sign or the suffix N/S/E/W'))
        layout.addWidget(QLabel('is allowed. Supported formats are at least those of Google Maps, ESWD, Wikipedia,'))
        layout.addWidget(QLabel('SPC storm reports and Tornado Archive.'))
        layout.addWidget(QLabel("Input should be specified in the format 'latitude, longitude'."))
        layout.addLayout(formlayout)
        self.marker_coordinates_widget.setLayout(layout)
        self.marker_coordinates_widget.resize(self.marker_coordinates_widget.sizeHint())
        self.marker_coordinates_widget.show()
                
    def set_pos_markers_properties(self, input_type, index=None):
        n = len(self.pos_markers_positions)
        list_index = min([index, n]) if not index is None else n
        
        if input_type == 'Coordinates':
            input_marker_latlon = self.pos_markers_latlonsw[index].text()
            try:
                lat, lon = ft.determine_latlon_from_inputstring(input_marker_latlon)
                x, y = ft.aeqd(gv.radarcoords[self.crd.radar], np.array([lat, lon]))
            except Exception:
                if list_index < n and input_marker_latlon == '':
                    self.remove_marker('pos', list_index)
                return
        elif input_type == 'Mouse':
            x, y = self.pb.screencoord_to_xy(self.pb.last_mousepress_pos)
            lat, lon = ft.aeqd(gv.radarcoords[self.crd.radar], np.array([x, y]), inverse=True)
        
        if n <= list_index:
            self.pos_markers_positions.append([])
            self.pos_markers_latlons.append([])
        self.pos_markers_positions[list_index] = np.array([x, y])
        self.pos_markers_latlons[list_index] = np.array([lat, lon])
        self.update_pos_marker_widgets()
            
        self.pb.set_sm_pos_markers()
        self.pb.update()
        
    def update_pos_marker_widgets(self):
        try:
            for j in range(len(self.pos_markers_latlonsw)):
                if j < len(self.pos_markers_latlons):
                    self.pos_markers_latlonsw[j].setText(', '.join(format(i, '.6f') for i in self.pos_markers_latlons[j]))
                else:
                    self.pos_markers_latlonsw[j].clear()
        except Exception:
            # Should only occur when the widgets don't exist
            pass
            
    def set_sm_marker_properties(self):
        self.sm_marker_position = self.pb.screencoord_to_xy(self.pb.last_mousepress_pos)
        self.sm_marker_latlon = ft.aeqd(gv.radarcoords[self.crd.radar],self.sm_marker_position,inverse=True)
        self.sm_marker_scantime = self.pb.data_attr['scantime'][self.pb.panel]
        self.sm_marker_scandatetime = self.pb.data_attr['scandatetime'][self.pb.panel]
        
        self.sm_marker_present=True
        self.pb.set_sm_pos_markers()
        self.pb.update()
        
    def remove_marker(self, mode, index=None):
        self.pos_markers_latlons_save = self.pos_markers_latlons.copy()
        
        if mode in ('sm', 'all'):
            self.sm_marker_present = False    
        if mode == 'pos':
            if index is None:
                index = self.pb.marker_mouse_selected_index
            del self.pos_markers_positions[index]
            del self.pos_markers_latlons[index]
        elif mode == 'all':
            self.pos_markers_positions, self.pos_markers_latlons = [], []        
            
        self.update_pos_marker_widgets()
        self.pb.set_sm_pos_markers()
        self.pb.update()
        
    def restore_pos_markers(self):
        self.pos_markers_latlons = self.pos_markers_latlons_save.copy()
        self.update_pos_marker_widgets()
        self.pb.set_sm_pos_markers()


    def set_stormmotion_from_marker(self):
        mouse_position=self.pb.screencoord_to_xy(self.pb.last_mousepress_pos)
        timediff_seconds=ft.datetimediff_s(self.sm_marker_scandatetime, self.pb.data_attr['scandatetime'][self.pb.panel])
        position_diff=mouse_position-self.sm_marker_position
        sm_direction=ft.calculate_azimuth(position_diff)
        sm_speed_mps=np.linalg.norm(position_diff)*1000./timediff_seconds
        if timediff_seconds<0:
            sm_direction=np.mod(sm_direction+180,360); sm_speed_mps*=-1
        self.change_stormmotion(np.array([sm_direction, sm_speed_mps]), convert_units=False)
                
    def set_stormmotion_manually(self):
        self.set_stormmotion=QWidget()
        self.set_stormmotion.setWindowTitle('Storm motion')
        layout=QFormLayout()
        self.sm_direction=QLineEdit(format(self.stormmotion[0], '.1f')); self.sm_speed=QLineEdit(format(self.stormmotion[1]*self.pb.scale_factors['s'], '.1f'))

        layout.addRow(QLabel('A storm speed of 0 kts removes the storm motion from the title,'))
        layout.addRow(QLabel('except when displaying the storm-relative velocity.'))
        layout.addRow(QLabel('From (degrees)'),self.sm_direction)
        layout.addRow(QLabel('Speed ('+self.pb.productunits['s']+')'),self.sm_speed)

        self.sm_direction.editingFinished.connect(self.change_stormmotion); self.sm_speed.editingFinished.connect(self.change_stormmotion)  
        
        self.set_stormmotion.setLayout(layout)
        self.set_stormmotion.resize(self.set_stormmotion.sizeHint())
        self.set_stormmotion.show()
        
    def update_stormmotion_change_radar(self):
        """The storm motion vector will be slightly different for an AEQD projection centered on a different radar.
        The method used here is slightly less accurate than reprojecting the full position difference vector,
        and calculating the new SM from this. But it works well enough, especially together with some measures in
        self.ani.update_datetimes_and_perform_firstplot that ensure that each animation iteration starts at the original
        panel view.
        """
        sm = self.stormmotion_save['sm']*np.array([np.pi/180, 1])
        # Always use the radar for which the storm motion was set (i.e. self.stormmotion_save['radar']) as reference.
        # Not doing that leads to slightly different SM when reprojecting back and forth between radars.
        old_radar, new_radar = self.stormmotion_save['radar'], self.crd.radar
        pos1, pos2 = np.zeros(2), np.array([np.sin(sm[0]), np.cos(sm[0])])
        pos1_latlon, pos2_latlon = ft.aeqd(gv.radarcoords[old_radar], np.array([pos1, pos2]), inverse=True)
        pos1, pos2 = ft.aeqd(gv.radarcoords[new_radar], np.array([pos1_latlon, pos2_latlon]))
        position_diff = pos1-pos2
        sm_direction=ft.calculate_azimuth(position_diff)
        sm_speed_mps=np.linalg.norm(position_diff)*sm[1]
        self.stormmotion = np.array([sm_direction, sm_speed_mps])
        
    def update_stormmotion(self, stormmotion):
        # Updates only storm motion, without performing additional (e.g. plotting) actions
        # stormmotion can be either a 2-element array, or a dictionary of the form {'sm':array, 'radar':radar}
        if type(stormmotion) == dict:
            self.stormmotion_save = stormmotion
            self.update_stormmotion_change_radar()
        else:
            self.stormmotion = stormmotion
            if stormmotion[1] != 0.:
                self.stormmotion_save = {'sm':stormmotion, 'radar':self.crd.radar}
        
    def change_stormmotion(self, stormmotion=None, convert_units=True):
        if type(stormmotion) == dict:
            self.update_stormmotion(stormmotion)
        else: 
            if stormmotion is None:
                input_sm_direction, input_sm_speed = self.sm_direction.text(), self.sm_speed.text()
            else:
                # stormmotion should be a 2-element array
                input_sm_direction, input_sm_speed = stormmotion.astype(str)
            
            number1, number2 = ft.to_number(input_sm_direction), ft.to_number(input_sm_speed)
            if not number1 is None and not number2 is None:
                sm_speed_mps = number2
                if convert_units:
                    sm_speed_mps /= self.pb.scale_factors['s']
                self.stormmotion = np.array([number1 % 360, sm_speed_mps])
                if self.stormmotion[1] == 0.: 
                    self.stormmotion[0] = 0
                else:
                    # Save also the current radar with the storm motion, since the function update_stormmotion_change_radar
                    # requires information about the radar for which the saved storm motion is valid.
                    self.stormmotion_save = {'sm':self.stormmotion, 'radar':self.crd.radar}
            else:
                self.sm_direction.setText(format(self.stormmotion[0], '.1f'))
                self.sm_speed.setText(format(self.stormmotion[1]*self.pb.scale_factors['s'], '.1f'))

        if self.pb.firstplot_performed:
            redraw_panellist = [j for j in self.pb.panellist if self.crd.products[j]=='s' or self.crd.products[j] in gv.plain_products_correct_for_SM]
            self.pb.set_newdata(redraw_panellist)
            if self.show_vwp:
                self.vwp.set_newdata()
                self.pb.set_draw_action('plotting_vwp')
                self.pb.update()
            
            
    def define_name_new_list(self, dest_function = 'set_label_for_case'):
        self.define_new_name=QWidget()
        self.define_new_name.setWindowTitle('Create new list')
        layout=QFormLayout()
        self.list_new_name=QLineEdit()

        layout.addRow(QLabel('Name of new list'),self.list_new_name)

        self.list_new_name.returnPressed.connect(lambda: self.create_new_list(dest_function))
        
        self.define_new_name.setLayout(layout)
        self.define_new_name.resize(self.define_new_name.sizeHint())
        self.define_new_name.show()
                                                                       
    def create_new_list(self, dest_function):
        list_name = self.list_new_name.text()
        if not list_name in self.cases_lists:
            self.set_textbar('')
            self.cases_lists[list_name] = []
            with open(cases_lists_filename,'wb') as f:
                pickle.dump(self.cases_lists,f)
            self.define_new_name.close()
            if dest_function == 'set_label_for_case':
                self.set_label_for_case(list_name)
            else:
                function = self.copy_to_other_list if dest_function == 'copy' else self.move_to_other_list
                function(list_name)
        else:
            self.set_textbar('This name already exists. Either choose a different name, or first delete the existing list.','red',2)
    
    def set_label_for_case(self, list_name, case_dict = None):
        # Set case_dict if you want to update an existing case
        self.set_case_label=QWidget()
        self.set_case_label.setWindowTitle('Case info')
        
        layout=QVBoxLayout()
        form_layout=QFormLayout()
        self.case_label=QLineEdit()
        if not case_dict is None:
            self.case_label.setText(case_dict['label'])
        form_layout.addRow(QLabel('Case label (optional, can be left empty)'),self.case_label)
        self.case_URLs = {}
        for i in range(5):
            self.case_URLs[i] = QLineEdit()
            if not case_dict is None and len(case_dict['urls']) > i:
                self.case_URLs[i].setText(case_dict['urls'][i])
            form_layout.addRow(QLabel(f'URL {i+1} (optional, can be left empty)'),self.case_URLs[i])
        self.case_button = QPushButton('Add case' if case_dict is None else 'Update label', autoDefault=True)
        self.case_button.clicked.connect(lambda: self.add_case_to_list(list_name, case_dict))
        layout.addLayout(form_layout)    
        layout.addWidget(self.case_button)
            
        self.set_case_label.setLayout(layout)
        self.set_case_label.resize(self.set_case_label.sizeHint())
        self.set_case_label.show()
            
    def add_new_case_to_list(self, case_dict, list_name=None):
        case_list = self.current_case_list if list_name is None else self.cases_lists[list_name]
            
        keys = [''.join(self.get_case_text(case)) for case in case_list]
        cases_sorted_before = (keys == np.sort(keys)).all()
        if cases_sorted_before:
            case_list.insert(bisect.bisect(keys, ''.join(self.get_case_text(case_dict))), case_dict)
        else:
            case_list += [case_dict]
                    
    def add_case_to_list(self, list_name, case_dict = None, update_label = True):
        # Set case_dict if you want to update an existing case
        # Set update_label to False if you want to keep using the current label for the case, and only want to update the case specs themselves
        if update_label:
            label = self.case_label.text()
            urls = []
            for i in self.case_URLs:
                if not self.case_URLs[i].text() == '':
                    urls.append(self.case_URLs[i].text())
        else:
            label, urls = case_dict['label'], case_dict['urls']
        
        if case_dict is None or not update_label:
            info = {'datetime': self.crd.date+self.crd.time,
                    'scannumbers_forduplicates': self.dsg.scannumbers_forduplicates,
                    'radar': self.crd.radar,
                    'translate': self.pb.panels_sttransforms[0].translate,
                    'scale': self.pb.panels_sttransforms[0].scale,
                    'panel_center': self.pb.panel_centers[0],
                    'pos_markers_latlons': self.pos_markers_latlons.copy()
                    }
            if not self.pb.data_empty[0] and not self.pb.data_isold[0]:
                info['scantime'] = self.pb.data_attr['scantime'][0]
                info['scandatetime'] = self.pb.data_attr['scandatetime'][0]
            if self.stormmotion[1] != 0.:
                info['stormmotion'] = self.stormmotion_save
            if self.show_vwp:
                info['vvp_range_limits'] = self.vvp_range_limits.copy()
                info['vvp_height_limits'] = self.vvp_height_limits.copy()
                info['vvp_vmin_mps'] = self.vvp_vmin_mps
                info['display_manual_sfcobs'] = self.vwp.display_manual_sfcobs
                info['vwp_manual_sfcobs'] = self.vwp_manual_sfcobs
            if case_dict and 'extra_datetimes' in case_dict:                
                info['extra_datetimes'] = case_dict['extra_datetimes']
        else: 
            info = case_dict.copy()
        info['label'] = label
        info['urls'] = urls
        
        self.current_case_list_name = list_name
        self.current_case_list = self.cases_lists[list_name]
        self.current_case = info
        self.set_textbar()
        
        cases = self.get_cases_as_strings()
        if not str(info) in cases or not case_dict is None:
            if not case_dict is None:
                index = cases.index(str(case_dict))
                self.current_case_list[index] = info
                
                if hasattr(self, 'modify_case_listw'):
                    descr, url_full, url_short = self.get_case_text(case_dict)
                    old_key = descr+']---['+url_full+']---['+url_short
                    descr, url_full, url_short = self.get_case_text(info)
                    new_key = descr+']---['+url_full+']---['+url_short
                    self.list_cases_labels[new_key] = self.list_cases_labels[old_key]
                    if not new_key == old_key:
                        del self.list_cases_labels[old_key]
                    
                    if self.list_cases.dragEnabled():
                        self.list_cases_labels[new_key].setText(descr+url_short)
                    else:
                        self.list_cases_labels[new_key].setText(descr+url_full)
                if hasattr(self, 'set_case_label'):
                    # This is always done when this widget is open, even when updating only the case and not the label. Reason is that a change
                    # in case_dict means that line self.case_button.clicked.connect(lambda: self.add_case_to_list(list_name, case_dict)) in
                    # self.set_label_for_case now refers to an old version of case_dict, which doesn't exist anymore. And this would lead to errors
                    # in this function. 
                    self.set_case_label.close()
            else:
                self.add_new_case_to_list(info, list_name)
            
            with open(cases_lists_filename,'wb') as f:
                pickle.dump(self.cases_lists,f)
        else:
            self.set_textbar('The exact same case already exists', 'red', 1)
            
    def set_label_for_extra_datetime(self):
        # Set case_dict if you want to update an existing case
        self.set_extra_datetime_label=QWidget()
        self.set_extra_datetime_label.setWindowTitle('Optional label')
        
        layout=QVBoxLayout()
        self.extra_datetime_label=QLineEdit()
        if 'extra_datetimes' in self.current_case and self.crd.date+self.crd.time in self.current_case['extra_datetimes']:
            self.extra_datetime_label.setText(self.current_case['extra_datetimes'][self.crd.date+self.crd.time]['label'])
        self.extra_datetime_label.returnPressed.connect(lambda: self.add_or_remove_extra_datetimes_case('add'))
        layout.addWidget(QLabel('Label for extra time (optional, can be left empty). Press Enter to confirm.'))
        layout.addWidget(self.extra_datetime_label)
            
        self.set_extra_datetime_label.setLayout(layout)
        self.set_extra_datetime_label.resize(self.set_extra_datetime_label.sizeHint())
        self.set_extra_datetime_label.show()
            
    def add_or_remove_extra_datetimes_case(self, mode):
        if mode == 'add':
            if not 'extra_datetimes' in self.current_case:
                self.current_case['extra_datetimes'] = {}
            label = self.extra_datetime_label.text()
            self.set_extra_datetime_label.close()
            info = {'label': label}
            if not self.pb.data_empty[0] and not self.pb.data_isold[0]:
                info['scantime'] = self.pb.data_attr['scantime'][0]
            self.current_case['extra_datetimes'][self.crd.date+self.crd.time] = info
        elif mode == 'remove single':
            del self.current_case['extra_datetimes'][self.crd.date+self.crd.time]
        elif mode == 'remove all':
            self.current_case['extra_datetimes'] = {}
            
        if len(self.current_case['extra_datetimes']) == 0:
            del self.current_case['extra_datetimes']
        
        with open(cases_lists_filename,'wb') as f:
            pickle.dump(self.cases_lists,f)
            
        self.set_textbar()

    def set_data_selected_startazimuth(self):
        self.set_startazimuth=QWidget()
        self.set_startazimuth.setWindowTitle('Start azimuth of scans')
        layout=QFormLayout()
        self.data_selected_startazimuthw=QLineEdit(format(self.data_selected_startazimuth, '.0f'))

        layout.addRow(QLabel('Change the start azimuth of the scans. It might be desired to change this'))
        layout.addRow(QLabel('azimuth since there is a discontinuity in the data where the scan starts,'))
        layout.addRow(QLabel('which is undesired when interesting echoes are located there.'))
        layout.addRow(QLabel('Start azimuth (degrees)'),self.data_selected_startazimuthw)

        self.data_selected_startazimuthw.editingFinished.connect(self.change_data_selected_startazimuth)
        
        self.set_startazimuth.setLayout(layout)
        self.set_startazimuth.resize(self.set_startazimuth.sizeHint())
        self.set_startazimuth.show()
        
    def change_data_selected_startazimuth(self):
        number = ft.to_number(self.data_selected_startazimuthw.text())
        if not number is None:
            self.data_selected_startazimuth = number
        self.data_selected_startazimuthw.setText(format(self.data_selected_startazimuth, '.0f'))
        self.pb.set_newdata(self.pb.panellist)
                        

    def change_plot_mode(self, new_mode):
        self.crd.plot_mode = new_mode
        
    def import_choices(self):
        try:
            try:
                with open(opa(os.path.join(gv.programdir+'/Generated_files/','saved_choices.pkl')),'rb') as f:
                    choices = pickle.load(f)
            except Exception: 
                with open(opa(os.path.join(gv.programdir+'/Input_files/','saved_choices_default.pkl')),'rb') as f:
                    choices = pickle.load(f)
        except Exception:
            choices = {}
        return choices
        
    def save_choice(self,choice_ID):
        if self.pb.firstplot_performed:
            choices = self.import_choices()
            dist_to_radar = {j:np.linalg.norm(self.pb.corners[j].mean(axis=0)) for j in self.pb.panellist}
            selected_heights = self.dsg.get_panel_center_heights(dist_to_radar, self.dsg.selected_scanangles, 
                                                                 self.dsg.get_scanangles_allproducts(self.dsg.scanangles_all_m), self.dsg.scanpair_present)
            selected_heights = {i:j for i,j in selected_heights.items() if self.crd.products[i] in gv.products_with_tilts}
            selected_scanangles = {i:j for i,j in self.dsg.selected_scanangles.items() if self.crd.products[i] in gv.products_with_tilts}
            choices[choice_ID] = {'panels':self.pb.panels, 'products':self.crd.products, 'selected_scanangles':selected_scanangles, 
                                  'selected_heights':selected_heights, 'range_nyquistvelocity_scanpairs_indices':self.dsg.range_nyquistvelocity_scanpairs_indices}
            with open(gv.programdir+'/Generated_files/saved_choices.pkl', 'wb') as f:
                pickle.dump(choices, f, protocol=2)

    def set_choice(self,choice_ID): 
        choices = self.import_choices()
        if not choice_ID in choices: 
            return
        
        choice = choices[choice_ID]
        # For backward compatibility:
        for attr in ('selected_scanangles', 'selected_heights'):
            choice[attr] = {i:j for i,j in choice[attr].items() if not j is None}
        
        panels_new = choice['panels']
        new_panellist = [self.pb.plotnumber_to_panelnumber[panels_new][j] for j in range(panels_new)]
        for j in new_panellist:
            self.crd.products[j] = choice['products'][j]
            if self.crd.scan_selection_mode in ('scan', 'scanangle') and j in choice['selected_scanangles']:
                self.dsg.selected_scanangles[j] = choice['selected_scanangles'][j]
            if j in choice['range_nyquistvelocity_scanpairs_indices']:
                #Using self.dsg.range_nyquistvelocity_scanpairs_indices makes it possible to determine which scan is desired when there is 
                #a scan pair in which one has a large range but low Nyquist velocity, and the other a smaller range but larger Nyquist velocity.
                self.dsg.range_nyquistvelocity_scanpairs_indices[j] = choice['range_nyquistvelocity_scanpairs_indices'][j]
        if self.crd.scan_selection_mode == 'height':
            self.dsg.manually_set_panel_center_heights(choice['selected_heights'])
        
        for j in new_panellist:
            if self.crd.products[j] in gv.plain_products_with_parameters:
                self.change_PP_parameters_panels(self.crd.products[j])
        
        self.setting_saved_choice=True #Is used in the function self.pb.change_panels to let the program know that it is not desired to call
        #the functions self.crd.row_mode or self.crd.column_mode, as this could change the products and scans.
        #Is also used in the function self.dsg.check_need_scans_change, to let the program know that it is desired to choose a scan
        #with an appropriate Nyquist velocity, irregardless of what the current scan is.        
        
        if panels_new != self.pb.panels:
            if not self.pb.firstplot_performed:
                self.crd.process_datetimeinput()
            self.pb.panellist = new_panellist
            self.pb.change_panels(panels_new)
        else:
            self.crd.process_datetimeinput()
        self.setting_saved_choice=False

    def view_choices(self):
        choices = self.import_choices()
        
        choices_text = {'angle':{}, 'height':{}}
        for i in range(1, 13):
            choices_text['angle'][i] = choices_text['height'][i] = ''
            if i in choices:
                panels = choices[i]['panels']
                panellist = [self.pb.plotnumber_to_panelnumber[panels][j] for j in range(panels)]
                for k in choices_text:
                    for j in panellist:
                        product, scanangle = choices[i]['products'][j], choices[i]['selected_scanangles'].get(j, None)
                        height = choices[i].get('selected_heights', {}).get(j, scanangle)
                        specs = product
                        if not product in gv.plain_products:
                            specs += ft.format_nums(scanangle) if k == 'angle' else ft.format_nums(height)
                        choices_text[k][i] += ' '*bool(j and specs) + specs
                    choices_text[k][i] = self.compress_choice_text_and_split_rows(choices_text[k][i])

        self.choices_widget = QWidget()
        self.choices_widget.setWindowTitle('Saved panel configurations')  

        layout = QGridLayout()
        layout.addWidget(QLabel('Choice'), 0, 0)
        layout.addWidget(QLabel('Elevation angle'), 0, 1)
        layout.addWidget(QLabel('Height'), 0, 4)
        self.choice_edits = {'angle':{}, 'height':{}}
        for i in choices:
            for j in self.choice_edits:
                self.choice_edits[j][i] = {}
                for k in (0, 1):
                    self.choice_edits[j][i][k] = QLineEdit(choices_text[j][i][k])
                    self.choice_edits[j][i][k].editingFinished.connect(lambda j=j, i=i: self.process_choice_edit(j, i))
            layout.addWidget(QLabel(str(i)), i, 0)
            layout.addWidget(self.choice_edits['angle'][i][0], i, 1)
            layout.addWidget(self.choice_edits['angle'][i][1], i, 2)
            layout.addWidget(QLabel(' '), i, 3)
            layout.addWidget(self.choice_edits['height'][i][0], i, 4)
            layout.addWidget(self.choice_edits['height'][i][1], i, 5)
        
        self.choices_widget.setLayout(layout)
        self.choices_widget.resize(self.choices_widget.sizeHint())
        self.choices_widget.show()
        
    def compress_choice_text_and_split_rows(self, text):
        def get_product(item):
            return ''.join([k for k in item if k.isalpha()])
        def get_value(item):
            return ''.join([k for k in item if k.isdigit() or k == '.'])
        def get_product_upper(product):
            return 'u'*(len(product) > 1)+product[-1].upper()
        def is_product_upper(product):
            return product and product == get_product_upper(product)
        items = text.split(' ')
        panels = len(items)
        cmpr_text = {0:'', 1:''}
        for i,j in enumerate(items):
            row = self.pb.rows_panels[panels][i]
            ncols = self.pb.cols_panels[panels][panels-1]+1
            product, value = get_product(j), get_value(j)
            value_before = items[i-1][1:]
            if i and value and product in items[i-1]:
                specs = value
            elif i and value and value == value_before:
                specs = product
            else:
                specs = product+value
            value_row0 = get_value(items[i % ncols])
            if row and specs != product and value and value == value_row0:
                last_item_row = cmpr_text[row].split(' ')[-1]
                if last_item_row and get_product(last_item_row) == get_product_upper(product):
                    n_present = cmpr_text[row][-1].isdigit()
                    n = int(cmpr_text[row][-1])+1 if n_present else 2
                    cmpr_text[row] = cmpr_text[row][:(-1 if n_present else None)]+str(n)
                    specs = ''
                else:
                    specs = get_product_upper(product)
            cmpr_text[row] += ' '*bool(cmpr_text[row] and specs) + specs
        all_upper = all(is_product_upper(get_product(j)) for j in cmpr_text[1].split(' ')) if cmpr_text[1] else False
        if all_upper:
            cmpr_text[1] = cmpr_text[1][:-1]
        return cmpr_text
    def combine_rows_and_decompress_choice_text(self, cmpr_text):
        items = {}
        for i in (0, 1):
            items[i] = cmpr_text[i].split(' ')
            if items[i][0]:
                for j,k in enumerate(items[i]):
                    if k[0].isalpha():
                        product = k[0]
                    product = product.lower() if k[0] != product else product
                    k_notproduct = k.replace(product, '')
                    if k_notproduct.replace('.', '').isdigit():
                        value = k_notproduct
                    elif product.isupper() or product in gv.plain_products:
                        value = ''
                    items[i][j] = product+value
        text = ' '.join(items[0])
        if items[1][0]:
            for i,j in enumerate(items[1]):
                if j[0].isupper():
                    product = j[0].lower()
                    n = int(j[1:]) if j[1:] else 1+len(items[0])-len(items[1])
                    specs = ' '.join([product+k[1:] for k in items[0][i:i+n]])
                else:
                    specs = j
                text += ' '*bool(text and specs) + specs
        return text
        
    def process_choice_edit(self, j, i): #j is value type ('angle'/'height'), i is choice
        choice_text = self.choice_edits[j][i][0].text().strip(), self.choice_edits[j][i][1].text().strip()
        choice_text = self.combine_rows_and_decompress_choice_text(choice_text)
        print(choice_text)
        items = choice_text.split(' ')
        panels = len(items)
        error = True
        if panels in self.pb.rows_panels:
            panellist = [self.pb.plotnumber_to_panelnumber[panels][j] for j in range(panels)]
            products, values = {}, {}
            for idx,k in enumerate(items):
                p = panellist[idx]
                if k[0] in gv.products_all and (k in gv.plain_products or k[1:].replace('.', '').isdigit()):
                    products[p] = k[0]
                    if k[1:]:
                        values[p] = float(k[1:])
            error = len(products) < panels
        for k in (0, 1):
            self.choice_edits[j][i][k].setStyleSheet('QLineEdit {color:'+('red' if error else 'black')+'}')
        if error:
            return
        
        choices = self.import_choices()
        key = 'selected_scanangles' if j == 'angle' else 'selected_heights'
        choices[i].update({'panels':panels, 'products':products, key:values})
        with open(opa(os.path.join(gv.programdir+'/Generated_files','saved_choices.pkl')),'wb') as f:
            pickle.dump(choices,f,protocol=2)
            

        
    def show_archiveddays(self):
        dates=self.dsg.get_dates_with_archived_data(self.crd.selected_radar,self.crd.selected_dataset)
        
        input_date=self.datew.text()
        if ft.correct_datetimeinput(input_date,'0000') and not input_date=='c':
            radars=self.dsg.get_radars_with_archived_data_for_date(input_date)
        else:
            radars=[]
        
        text='Radars with archived data for '+(input_date[1:] if 'c' in input_date and input_date!='c' else input_date)+': \n'
        for j in range(0,len(radars)):
            text+=radars[j]
            if j!=len(radars)-1:
                text+=', '
        if len(radars)==0:
            text+='-'
        text+='\n\n'
        
        text+='Dates with archived data for '+self.crd.selected_radar
        if self.crd.selected_radar in gv.radars_with_datasets:
            text+=' '+self.crd.selected_dataset+': \n'
        else: text+=': \n'
        for j in range(0,len(dates)):
            text+=str(dates[j])
            if j!=len(dates)-1:
                text+=', '
        if len(dates)==0:
            text+='- \n'

        msgbox_archiveddata=QMessageBox()
        msgbox_archiveddata.about(self, 'Dates with archived data', text)
                
    def show_scans_properties(self):
        if self.pb.firstplot_performed:
            properties_text='Scan angle, slant range, radial resolution, Nyquist velocity, low and high dual PRF Nyquist velocity. \n' +\
            'The low and high Nyquist velocity determine the correction terms that are applied to aliased velocities during dual-PRF dealiasing. '+\
            'If these are equal, then the possible correction terms are the same for both even and odd radials.\n'+\
            '--------------------------------------------------------------------------------------------------------------------\n'
            # The dashed line above is added to force QMessageBox to have a certain width, since without this it can be undesirably narrow. 
            for j in self.dsg.scanangles_all['z']:
                properties_text += 'Scan '+str(j)+': '+"%.2f" % self.dsg.scanangles_all_m['z'][j]+" \xb0, "+"%.1f" % self.dsg.radial_range_all['z'][j]+' km, '+"%.4f" % self.dsg.radial_res_all['z'][j]+' km, '
                if not self.dsg.nyquist_velocities_all_mps[j] is None:
                    properties_text+="%.1f" % (self.dsg.nyquist_velocities_all_mps[j]*self.pb.scale_factors['v'])+' '+self.pb.productunits['v'] +\
                    ', '+(("%.1f" % (self.dsg.low_nyquist_velocities_all_mps[j]*self.pb.scale_factors['v'])+' '+self.pb.productunits['v'] +\
                    ', '+"%.1f" % (self.dsg.high_nyquist_velocities_all_mps[j]*self.pb.scale_factors['v'])+' '+self.pb.productunits['v']) \
                    if not self.dsg.low_nyquist_velocities_all_mps[j] is None else '--, --')
                else: 
                    #If it was impossible to obtain the Nyquist velocities, then they are set to '--'
                    properties_text+='--, --, --'
                    
                if j!=list(self.dsg.scanangles_all['z'])[-1]:
                    properties_text+='\n'
                    
            msgbox_scanproperties=QMessageBox()
            msgbox_scanproperties.about(self, 'Scan properties', properties_text)

        
    def show_fullscreen(self):
        self.changing_fullscreen=True #Is set to False in the function self.pb.on_resize.
        if not self.fullscreen:
            self.showFullScreen(); self.fullscreen=True
        else:
            self.showMaximized(); self.fullscreen=False
        
    def change_continue_savefig(self, animation=False):
        if (self.creating_animation and not animation) or (self.continue_savefig and not self.creating_animation and animation):
            # When creating an animated gif this function should not respond to a "regular" continue savefig event, and vice versa.
            return
        
        self.continue_savefig=not self.continue_savefig
        if animation:
            self.creating_animation = not self.creating_animation
            if self.creating_animation:
                qm = QMessageBox
                choice = qm.No
                if hasattr(self, 'ani_widget') and self.ani_widget.isVisible():
                    choice = qm.question(self, 'NLradar', 'Do you want to retain any existing animation frames?', qm.Yes | qm.No)
                if choice == qm.No:
                    self.starting_animation = True
                    if os.path.exists(gv.animation_frames_directory):
                        shutil.rmtree(gv.animation_frames_directory)
                os.makedirs(gv.animation_frames_directory, exist_ok=True)
            else:
                self.save_animation()
        
        self.set_textbar() #Change the color into blue when self.continue_savefig=True, else the previous color
        if self.continue_savefig and not animation:
            save_continue_type=self.ani.continue_type; save_direction=self.ani.direction
            self.ani.continue_type='None' #Stop an animation or continuation to the left or right, because this would continue when the 
            #widget for selecting a directory is opened, which is not desired.
            self.savefig()
            if not save_continue_type is None: #Restart the animation or continuation to the left/right.
                self.ani.change_continue_type(save_continue_type,save_direction)
                
                
    def save_animation(self):
        self.ani_widget = QWidget()
        self.ani_widget.setWindowTitle('Save animation')
        
        self.ani_directoryw = QPushButton(os.path.dirname(self.animation_filename))
        self.ani_filenamew = QLineEdit(os.path.basename(self.animation_filename))
        self.ani_delay_framew = QRadioButton('frame')
        self.ani_delay_minutew = QRadioButton('minute')
        self.ani_delay_framew.setChecked(True) if self.ani_delay_ref == 'frame' else self.ani_delay_minutew.setChecked(True)
        group = QButtonGroup(); group.addButton(self.ani_delay_framew); group.addButton(self.ani_delay_minutew)
        hbox1 = QHBoxLayout(); hbox1.addWidget(self.ani_delay_framew); hbox1.addWidget(self.ani_delay_minutew)
        self.ani_delayw = QLineEdit(str(self.ani_delay))
        self.ani_delay_endw = QLineEdit(str(self.ani_delay_end))
        self.ani_create_gifw = QPushButton('GIF')
        self.ani_sort_filesw = QCheckBox()
        self.ani_sort_filesw.setTristate(False)
        self.ani_sort_filesw.setCheckState(2 if self.ani_sort_files else 0)
        self.ani_group_datasetsw = QCheckBox()
        self.ani_group_datasetsw.setTristate(False)
        self.ani_group_datasetsw.setCheckState(2 if self.ani_group_datasets else 0)
        self.ani_qualityw = QLineEdit(str(self.ani_quality))
        self.ani_create_mp4w = QPushButton('MP4')
        hbox2 = QHBoxLayout(); hbox2.addWidget(self.ani_create_gifw); hbox2.addWidget(self.ani_create_mp4w)
        
        self.ani_directoryw.clicked.connect(self.select_ani_directory)
        self.ani_delay_minutew.released.connect(lambda: self.check_delay_ref_sort_files_consistency('delay_ref'))
        self.ani_sort_filesw.stateChanged.connect(lambda: self.check_delay_ref_sort_files_consistency('sort_files'))
        self.ani_create_gifw.clicked.connect(lambda: self.create_ani('gif'))
        self.ani_create_mp4w.clicked.connect(lambda: self.create_ani('mp4'))
        
        layout = QFormLayout()
        layout.addRow(QLabel('Directory'), self.ani_directoryw)
        layout.addRow(QLabel('Filename (no extension needed)'), self.ani_filenamew)
        layout.addRow(QLabel('Set delay per'), hbox1)
        layout.addRow(QLabel('Delay between frames or minutes (cs)'), self.ani_delayw)
        layout.addRow(QLabel('Delay for final frame (cs)'), self.ani_delay_endw)
        layout.addRow(QLabel('All frames in the directory '+gv.animation_frames_directory))
        layout.addRow(QLabel('will be added to the animation. If you want to exclude some frames,'))
        layout.addRow(QLabel('you can delete them from this directory. Below you can also choose'))
        layout.addRow(QLabel('to sort frames chronologically, and to group frames by dataset'))
        layout.addRow(QLabel('(i.e. by radar/dataset/subdataset). Frames for different cases'))
        layout.addRow(QLabel('are always grouped by case.'))
        layout.addRow(QLabel('When setting delay per minute, frames will always be sorted chronologically.'))
        layout.addRow(QLabel('Sort frames chronologically'), self.ani_sort_filesw)
        layout.addRow(QLabel('Group frames by dataset'), self.ani_group_datasetsw)
        layout.addRow(QLabel('MP4 video quality (0-10)'), self.ani_qualityw)
        layout.addRow(QLabel('Create animation'), hbox2)
        layout.addRow(QLabel('As long as you keep this window open, you can recreate the animation'))
        layout.addRow(QLabel('with different settings, or press CTRL+SHIFT+S again and start adding'))
        layout.addRow(QLabel('more frames to the animation.'))
         
        self.ani_widget.setLayout(layout)
        self.ani_widget.resize(self.ani_widget.sizeHint())
        self.ani_widget.show()
        
    def select_ani_directory(self):
        text = 'Select a directory'
        selected_dir = str(QFileDialog.getExistingDirectory(None,text,os.path.dirname(self.animation_filename)))
        if selected_dir:
            self.animation_filename = opa(selected_dir+'/'+os.path.basename(self.animation_filename))
            self.ani_directoryw.setText(selected_dir)
            
    def check_delay_ref_sort_files_consistency(self, change):
        if change == 'delay_ref':
            self.ani_sort_filesw.setCheckState(2)
        elif change == 'sort_files' and not self.ani_sort_filesw.isChecked():
            self.ani_delay_framew.setChecked(True)
        
    def set_ani_parameters(self):
        filename = self.ani_filenamew.text()
        if any(filename.endswith(j) for j in ('.gif', '.mp4')):
            filename = filename[:-4]
        self.animation_filename = opa(os.path.dirname(self.animation_filename)+'/'+filename)
        
        self.ani_delay_ref = 'frame' if self.ani_delay_framew.isChecked() else 'minute'
        
        number = ft.to_number(self.ani_delayw.text())
        if not number is None:
            self.ani_delay = ft.rifdot0(float(number))
        self.ani_delayw.setText(str(self.ani_delay))
        
        number = ft.to_number(self.ani_delay_endw.text())
        if not number is None:
            self.ani_delay_end = int(number)
        self.ani_delay_endw.setText(str(self.ani_delay_end))
        
        self.ani_sort_files = self.ani_sort_filesw.checkState() == 2
        self.ani_group_datasets = self.ani_group_datasetsw.checkState() == 2
        
        number = ft.to_number(self.ani_qualityw.text())
        if not number is None:
            self.ani_quality = ft.rifdot0(float(number))
        self.ani_qualityw.setText(str(self.ani_quality))
        
    def create_ani(self, ext='gif'):
        self.set_ani_parameters()
        
        try:
            frames_all = os.listdir(gv.animation_frames_directory)
            frame_numbers = [int(j[5:-4]) for j in frames_all]
            sort_output = sorted(zip(frame_numbers, frames_all))
            frames_all = [gv.animation_frames_directory+'/'+j[1] for j in sort_output]
            frame_numbers = np.array([j[0] for j in sort_output])
            
            frames_all, frames_datetimes, frames_datasets = np.array(frames_all), np.array(self.ani_frames_datetimes), np.array(self.ani_frames_datasets)
            if not len(frames_all) == len(frames_datetimes):
                # Some frames might have been removed manually from the directory, in which case the corresponding datasets
                # and datetimes should also be removed
                frames_datetimes = frames_datetimes[frame_numbers-1]
                frames_datasets = frames_datasets[frame_numbers-1]
            
            datasets, indices = np.unique(frames_datasets, return_index=True)
            first_digits = [''.join([c for i,c in enumerate(j) if j[:i+1].isdigit()]) for j in datasets]
            viewing_cases = all(len(j) > 0 for j in first_digits) and len(np.unique(first_digits)) > 1
            if not self.ani_group_datasets and not viewing_cases:
                # Set all datasets equal, in order to prevent grouping different datasets separately.
                # Different cases are always grouped separately.
                frames_datasets = np.full(len(frames_datasets), '')
                datasets, indices = [''], [0]
            if self.ani_group_datasets and not viewing_cases:
                # Retain the original order for the unique datasets unless the datasets contain case indices
                datasets = frames_datasets[np.sort(indices)]
            
            frames, deltas = [], []
            for dataset in datasets:
                select = frames_datasets == dataset
                if self.ani_sort_files:
                    datetimes = frames_datetimes[select]
                    datetimes = sorted([list(dt_i)+[i] for i,dt_i in enumerate(datetimes)])
                    indices = [j[-1] for j in datetimes]
                    new_frames = list(frames_all[select][indices])
            
                    timediffs = []
                    for i,dt_i in enumerate(datetimes[:-1]):
                        n = min(len(dt_i), len(datetimes[i+1])) - 1 # -1, since the last column of datetimes contains an index, see above
                        for j in range(n):
                            timediff = ft.datetimediff_s(dt_i[j], datetimes[i+1][j])
                            if timediff > 0:
                                break
                        timediffs.append(timediff)
                    
                    i_retain = range(len(timediffs))
                    if self.ani_delay_ref == 'minute':
                        mean_timediff = np.mean(timediffs)
                        i_retain = [i for i,dt in enumerate(timediffs) if dt > mean_timediff/4]
                    frames += [new_frames[0]]+[new_frames[i+1] for i in i_retain]
                    deltas += [j/60*self.ani_delay for j in (timediffs[i] for i in i_retain)]+['ani_delay_end']
                else:
                    frames += list(frames_all[select])
                    deltas += [self.ani_delay]*(np.count_nonzero(select)-1)+['ani_delay_end']
            
            deltas = [(self.ani_delay if self.ani_delay_ref == 'frame' else j) if j != 'ani_delay_end' else self.__dict__[j] for j in deltas]
        
            animation_filename = self.animation_filename+f'.{ext}'
            if ext == 'gif':
                filenames_string = ' '.join(frames)+f' -o "{animation_filename}"'
                subprocess.run(f'gifsicle/gifsicle -d{self.ani_delay:.0f} --loop {filenames_string}'+' --optimize'*(self.ani_delay_ref == 'frame'))
                
                if self.ani_delay_ref == 'minute':
                    string = ' '.join([f'-d{j:.0f} "#{i}"' for i,j in enumerate(deltas)])
                    subprocess.run(f'gifsicle/gifsicle -b "{animation_filename}" {string} --optimize')
            elif ext == 'mp4':
                container = av.open(animation_filename, mode='w')
                shape = Image.open(frames[0]).size
                crf = 1+5*(10-self.ani_quality)
                stream = container.add_stream('libx264', width=shape[0], height=shape[1], pix_fmt='yuv420p', options={"crf":str(crf)})
                
                # Use a time resolution of ms
                csum = 10*np.cumsum(np.concatenate(([0], deltas)))
                stream.codec_context.time_base = Fraction(1, 1000)
                # ffmpeg time is "complicated". read more at https://github.com/PyAV-Org/PyAV/blob/main/docs/api/time.rst
                
                for i,frame in enumerate(frames+[frames[-1]]):
                    img = Image.open(frame)
                    frame = av.VideoFrame.from_image(img)
                    frame.pts = csum[i]
                    for packet in stream.encode(frame):
                        container.mux(packet)
                
                for packet in stream.encode(): # Flush stream
                    container.mux(packet)
                container.close()
            
            self.set_textbar('Animation created', 'green', 1)
        except Exception as e:
            self.set_textbar(str(e), 'red', 1)
    
    def change_savefig_include_menubar(self, state):
        self.savefig_include_menubar = state == 2
        
    def savefig(self, select_filename=True):
        #select_filename should be False when self.continue_savefig=True, after initially the directory and filename format have been chosen.
        
        img_arr1 = gloo.read_pixels(alpha=False)
        width, height = self.plotwidget.width(), self.plotwidget.height()
        if self.savefig_include_menubar:
            x, y = self.plotwidget.pos().x(), self.plotwidget.pos().y()
            full_width, full_height = self.width(), self.height()
            
            img = self.grab(QRect(QPoint(0, 0), QSize(full_width, y)))
            im = img.toImage()
            channels_count = 4
            s = im.bits().asstring(img.width() * img.height() * channels_count)
            img_arr2 = np.frombuffer(s, dtype=np.uint8).reshape((img.height(), img.width(), channels_count)).copy() # copy because otherwise read-only
            img_arr2[:,:,:3] = img_arr2[:,:,:3][:,:,::-1]
            
            # Make image width and height integer multiples of 2, since H264 video format (MP4) requires that
            im_width, im_height = int(np.ceil(full_width/2)*2), int(np.ceil(full_height/2)*2)
            img_arr = np.full((im_height, im_width, 3), img_arr2[0,0,0], dtype='uint8')
            img_arr[y:y+height,x:x+width] = img_arr1
            img_arr[:y,:] = img_arr2[:,:,:3]
        else:
            # Make image width and height integer multiples of 2, since H264 video format (MP4) requires that
            im_width, im_height = int(np.ceil(width/2)*2), int(np.ceil(height/2)*2)
            img_arr = np.full((im_height, im_width, 3), img_arr1[0,0,0], dtype='uint8')
            img_arr[:height, :width] = img_arr1
            
        im = Image.fromarray(img_arr)
        
        if self.creating_animation:
            radar_dataset = self.crd.radar+('_'+self.crd.dataset if self.crd.radar in gv.radars_with_datasets else '')
            s = self.radardata_product_versions[radar_dataset]
            radar_dataset_subdataset = radar_dataset+(' '+s if s else '')
            case_id = f'{self.get_current_case_index():04d}_' if self.current_case_shown() else ''
            dataspecs = case_id+''.join([self.dsg.get_dataspecs_string_panel(j) for j in self.pb.panellist])
            datetimes = [self.pb.data_attr['scandatetime'][j] for j in self.pb.panellist]
            if self.starting_animation:
                self.ani_frame_number = 1
                self.ani_frames_specs = [dataspecs]
                self.ani_frames_datetimes = [datetimes]
                self.ani_frames_datasets = [case_id+radar_dataset_subdataset]
                self.starting_animation = False
            else:
                if dataspecs in self.ani_frames_specs:
                    return
                else:
                    self.ani_frame_number += 1
                    self.ani_frames_specs += [dataspecs]
                    self.ani_frames_datetimes += [datetimes]
                    self.ani_frames_datasets += [case_id+radar_dataset_subdataset]
                
            filepath = opa(gv.animation_frames_directory+f'/frame{self.ani_frame_number}.gif')
        else:
            if select_filename:
                try:
                    self.get_user_selected_filepath()
                except Exception as e:
                    print(e, 'get_user_filepath_savefig')
                    return
            
            savefig_filename = self.expand_filename()
                    
            filepath = opa(self.savefig_dirname+'/'+savefig_filename)
        
        try:
            save_format=filepath[-3:]
            if save_format=='jpg':
                im=im.convert('RGB')
                im.save(filepath,quality=95,subsampling=0)
            elif save_format=='png':
                im=im.convert('RGB') #If this is not done, then the parts of the image that are transparent get a wrong color. 
                im.save(filepath,optimize=True,dpi=(72,72))
            elif save_format=='gif': 
                #I should make saving as GIF using dithering optional, by adding the possibility to set dithering on/off in the settings.
                #Converting to RGB before applying dithering is necessary, because otherwise it doesn't work.
                im=im.convert('RGB').convert('P', palette=Image.WEB, dither=Image.FLOYDSTEINBERG,colors=255)
                im.save(filepath)
        except Exception as e: print(e,'savefig'); pass
    
    def get_user_selected_filepath(self):
        text='Select a folder and filename (with extension, otherwise jpg) or file extension (without dot).'
        selected_dir_plus_filename_or_extension=str(QFileDialog.getSaveFileName(None,text,self.savefig_filename,options=QFileDialog.DontUseNativeDialog)[0])

        if selected_dir_plus_filename_or_extension:
            self.savefig_dirname=os.path.dirname(selected_dir_plus_filename_or_extension)
            basename=os.path.basename(selected_dir_plus_filename_or_extension)
            self.file_extension=basename[-3:]
            if not self.file_extension in ('gif','png','jpg'):
                self.file_extension='jpg'
                
            if basename in ('gif','png','jpg'):
                self.use_own_filename=False
            else:
                self.file_name=basename.replace('#', '*')
                if '.' in self.file_name:
                    self.file_name=self.file_name[:self.file_name.find('.')]
                self.use_own_filename=True
                
                self.file_number = None
                if '*' in self.file_name:
                    files_pattern = sorted([os.path.basename(j).split('.')[0] for j in glob.glob(self.savefig_dirname+'/'+self.file_name+'.'+self.file_extension)])
                    files_pattern = [j for j in files_pattern if j[len(self.file_name)-1:].isdigit()]
                    self.file_number = int(files_pattern[-1][len(self.file_name)-1:])+1 if len(files_pattern) > 0 else 1
                    self.file_name=self.file_name.replace('*', '')
                elif self.continue_savefig:
                    self.file_number = 1
                
            self.savefig_filename=selected_dir_plus_filename_or_extension
        else:
            self.continue_savefig=False
            raise Exception
                    
    def expand_filename(self):
        if self.use_own_filename:
            savefig_filename=self.file_name
            if not self.file_number is None:
                savefig_filename+=str(self.file_number)+'.'+self.file_extension
                self.file_number+=1
            else: 
                savefig_filename+='.'+self.file_extension
        else:
            def product_and_scan_string(panel,product,scan):
                string='u'*self.crd.using_unfilteredproduct[panel]
                string+=product+('('+self.crd.polarization[panel]+')')*self.crd.using_verticalpolarization[panel]
                if not product in gv.plain_products: 
                    string+=str(scan)
                return string
            
            duplicate_number = self.check_duplicates()

            savefig_filename = gv.radars_nospecialchar_names[self.crd.radar].replace(' ','')+f'_{self.crd.dataset}'*(self.crd.radar in gv.radars_with_datasets)+'_'
            savefig_filename += self.crd.date[2:]+self.crd.time+'_'
            products_scans = ' '.join([product_and_scan_string(j,self.crd.products[j],self.crd.scans[j]) for j in self.pb.panellist])
            savefig_filename += ''.join(list(self.compress_choice_text_and_split_rows(products_scans).values())).replace(' ', '')
            savefig_filename += '_vwp'*self.show_vwp+f'_{duplicate_number}'*(duplicate_number > 0)+'.'+self.file_extension
        return savefig_filename
    
    def check_duplicates(self):
        duplicate_number = 0
        for j in self.pb.panellist:
            key = self.crd.products[j] if self.crd.products[j] in gv.plain_products else self.crd.scans[j]
            if len(self.dsg.scannumbers_all['z'][key]) > 1:
                duplicate_number = max([duplicate_number, self.dsg.scannumbers_forduplicates[key]+1])
        return duplicate_number
    
    
    def change_show_vwp(self):
        self.show_vwp = not self.show_vwp
        self.pb.on_resize()
        if self.show_vwp:
            self.vwp.set_newdata()
            self.pb.set_draw_action('plotting_vwp')
        self.pb.update()
        
    
    
    def extra(self):
        self.extra=QTabWidget()
        self.extra.setWindowTitle('NLradar extra')
        self.extrafiles=QWidget()
        self.extra.addTab(self.extrafiles,'Files')
        self.extra_tabfiles()
        self.extra.resize(self.extra.sizeHint())
        self.extra.show()        
        
    def extra_tabfiles(self):
        layout=QFormLayout()
        self.movefilesw=QPushButton('Change directory structure, by moving files from one directory structure to another one', autoDefault=True)
        self.remove_volumeattributesw=QPushButton('Remove saved volume attributes for particular radars and datasets', autoDefault=True)
        self.movefilesw.clicked.connect(self.movefiles_select)
        self.remove_volumeattributesw.clicked.connect(self.remove_volume_attributes_select)
        layout.addWidget(self.movefilesw)
        layout.addWidget(self.remove_volumeattributesw)
        self.extrafiles.setLayout(layout)
        
        
    def remove_volume_attributes_select(self):
        self.removeattributes=QWidget()
        
        hbox_datetimes=QHBoxLayout()
        self.removeattributes_startdatew=QLineEdit(self.crd.selected_date)
        self.removeattributes_startdatew.setToolTip('Start date (YYYYMMDD)')
        self.removeattributes_starttimew=QLineEdit('0000')
        self.removeattributes_starttimew.setToolTip('Start time (HHMM)')
        self.removeattributes_enddatew=QLineEdit(self.crd.selected_date)
        self.removeattributes_enddatew.setToolTip('End date (YYYYMMDD)')
        self.removeattributes_endtimew=QLineEdit('2359')
        self.removeattributes_endtimew.setToolTip('End time (HHMM)')
        
        hbox_datetimes.addWidget(self.removeattributes_startdatew); hbox_datetimes.addWidget(self.removeattributes_starttimew)
        hbox_datetimes.addWidget(self.removeattributes_enddatew); hbox_datetimes.addWidget(self.removeattributes_endtimew)
                
        vbox_sources=QVBoxLayout()
        vbox_sources.addWidget(QLabel('Data sources from which to include radars. No selection implies selecting current radar.'))
        self.removeattributes_sourcesw = {}
        hbox = QHBoxLayout()
        n = 5
        for i,j in enumerate(gv.data_sources_all):
            self.removeattributes_sourcesw[j] = QCheckBox(j)
            self.removeattributes_sourcesw[j].setTristate(False)
            hbox.addWidget(self.removeattributes_sourcesw[j])
            if (i+1) % n == 0:
                vbox_sources.addLayout(hbox)
                hbox = QHBoxLayout()
        if len(gv.data_sources_all) % n != 0:
            vbox_sources.addLayout(hbox)
                     
        layout=QVBoxLayout()
        layout.addWidget(QLabel('Remove volume attributes for a particular radar and dataset for the selected date and time range. These volume attributes include e.g. the scanangles for all'))
        layout.addWidget(QLabel('scans in the volume, and multiple other attributes. They are saved to a file after determining them once, to increase the speed of reading data.'))
        layout.addWidget(QLabel('It could however occur that the wrong attributes are saved for particular dates and times, e.g. because data from different datasets is mixed up.'))
        layout.addWidget(QLabel('If this occurs, then you can here remove the wrong attributes, without having to remove the complete file (which is NLradar/Generated_files/attribute_IDs.pkl).'))
        layout.addLayout(hbox_datetimes)
        
        layout.addLayout(vbox_sources) 
        
        self.removeattributes_removew=QPushButton('Remove', autoDefault=True)
        self.removeattributes_infow=QLineEdit(); self.removeattributes_infow.setReadOnly(True)
        layout.addWidget(self.removeattributes_removew)
        layout.addWidget(self.removeattributes_infow)
        
        self.removeattributes_startdatew.editingFinished.connect(self.update_removeattributes_enddate)
        self.removeattributes_removew.clicked.connect(self.remove_attributes)
        
        self.removeattributes.setLayout(layout)
        self.removeattributes.resize(self.removeattributes.sizeHint())
        self.removeattributes.show()
                        
    def update_removeattributes_enddate(self):
        #Doing this speeds up the selection part when moving files for only one day.
        self.removeattributes_enddatew.setText(self.removeattributes_startdatew.text())
        
    def remove_attributes(self):
        self.update_removeattributes_infow('','black')
        
        startdate=self.removeattributes_startdatew.text(); starttime=self.removeattributes_starttimew.text()
        enddate=self.removeattributes_enddatew.text(); endtime=self.removeattributes_endtimew.text()
        if not ft.correct_datetimeinput(startdate,starttime) or not ft.correct_datetimeinput(enddate,endtime): 
            self.update_removeattributes_infow('Incorrect dates and/or times','red')
            return
        startdatetime=startdate+starttime; enddatetime=enddate+endtime
       
        selected_sources = [i for i,j in self.removeattributes_sourcesw.items() if j.checkState() == 2]
        if selected_sources:
            selected_radars = [j for j in gv.radars_all if gv.data_sources[j] in selected_sources]
        else:
            selected_radars = [self.crd.selected_radar]

        for i in selected_radars:
            for j in ('Z', 'V'):
                key = i+(' '+j)*(i in gv.radars_with_datasets)
                for sub in self.dsg.attributes_IDs[key].copy():
                    for date in self.dsg.attributes_IDs[key][sub].copy():
                        if int(startdate) <= int(date) <= int(enddate):
                            for time in self.dsg.attributes_IDs[key][sub][date].copy():
                                if int(startdatetime) <= int(date+time) <= int(enddatetime):
                                    del self.dsg.attributes_IDs[key][sub][date][time]
                        if len(self.dsg.attributes_IDs[key][sub][date]) == 0:
                            del self.dsg.attributes_IDs[key][sub][date]
        
        #Also update the file that contains attributes_IDs
        with open(self.dsg.attributes_IDs_filename,'wb') as f:
            pickle.dump(self.dsg.attributes_IDs,f)
        
        self.time_last_removal_volumeattributes = pytime.time()
        self.update_removeattributes_infow('Done','green')
                                
    def update_removeattributes_infow(self,text,color):
        self.removeattributes_infow.setStyleSheet('QLineEdit {color:'+color+'}')
        self.removeattributes_infow.setText(text); self.removeattributes_infow.repaint()
                    
       
    def movefiles_select(self):
        self.movefiles=QWidget()
               
        dirs_layout=QFormLayout()                                     
        self.movefiles_oldstructurew=QLineEdit(self.movefiles_parameters['oldstructure'])
        self.movefiles_oldstructurew.setToolTip('Specify the old directory structure for the data that you want to move')
        self.movefiles_newstructurew=QLineEdit(self.movefiles_parameters['newstructure'])
        self.movefiles_newstructurew.setToolTip('Specify the new directory structure for the data that you want to move')
        dirs_layout.addRow(QLabel('From'),self.movefiles_oldstructurew)
        dirs_layout.addRow(QLabel('To'),self.movefiles_newstructurew)   
                
        hbox_datetimes=QHBoxLayout()
        self.movefiles_startdatew=QLineEdit(self.movefiles_parameters['startdate'])
        self.movefiles_startdatew.setToolTip('Start date (YYYYMMDD)/Emtpy for all files')
        self.movefiles_starttimew=QLineEdit(self.movefiles_parameters['starttime'])
        self.movefiles_starttimew.setToolTip('Start time (HHMM)/Emtpy for all files')
        self.movefiles_enddatew=QLineEdit(self.movefiles_parameters['enddate'])
        self.movefiles_enddatew.setToolTip('End date (YYYYMMDD)/Emtpy for all files')
        self.movefiles_endtimew=QLineEdit(self.movefiles_parameters['endtime'])
        self.movefiles_endtimew.setToolTip('End time (HHMM)/Emtpy for all files')
        
        hbox_datetimes.addWidget(self.movefiles_startdatew); hbox_datetimes.addWidget(self.movefiles_starttimew)
        hbox_datetimes.addWidget(self.movefiles_enddatew); hbox_datetimes.addWidget(self.movefiles_endtimew)
                        
        vbox_sources=QVBoxLayout()
        vbox_sources.addWidget(QLabel('Data sources from which to include radars. No selection implies selecting current radar.'))
        self.movefiles_sourcesw = {}
        hbox = QHBoxLayout()
        n = 5
        for i,j in enumerate(gv.data_sources_all):
            self.movefiles_sourcesw[j] = QCheckBox(j)
            self.movefiles_sourcesw[j].setTristate(False)
            hbox.addWidget(self.movefiles_sourcesw[j])
            if (i+1) % n == 0:
                vbox_sources.addLayout(hbox)
                hbox = QHBoxLayout()
        if len(gv.data_sources_all) % n != 0:
            vbox_sources.addLayout(hbox)
                    
        self.movefiles_removefilesinpreviousdirsw=QCheckBox('Remove files from Current folder'); self.movefiles_removefilesinpreviousdirsw.setTristate(False)
        self.movefiles_movew=QPushButton('Move files', autoDefault=True)
        self.movefiles_stopw=QPushButton('Stop', autoDefault=True)
        self.movefiles_undow=QPushButton('Undo moving files', autoDefault=True); self.movefiles_undow.setEnabled(False)
        self.movefiles_infow=QLineEdit(); self.movefiles_infow.setReadOnly(True)
                
        layout=QVBoxLayout()
        layout.addWidget(QLabel('Change the directory structure for a series of files. First select the old (current) base directory and directory structure, and then select the new ones.'))
        layout.addWidget(QLabel('Then select the date and time range for which you want to move data (leave empty if you want to move all files), and the radars for which you want to do this.'))
        layout.addLayout(dirs_layout)
        layout.addLayout(hbox_datetimes)
        
        layout.addLayout(vbox_sources)
                
        layout.addWidget(self.movefiles_removefilesinpreviousdirsw)
        layout.addWidget(self.movefiles_movew)
        layout.addWidget(self.movefiles_stopw)
        layout.addWidget(self.movefiles_undow)
        layout.addWidget(self.movefiles_infow)
        self.movefiles.setLayout(layout)     
        
        self.movefiles_startdatew.editingFinished.connect(self.update_movefiles_enddate)
        for j in (self.movefiles_startdatew,self.movefiles_starttimew,self.movefiles_enddatew,self.movefiles_endtimew,self.movefiles_oldstructurew,self.movefiles_newstructurew):
            j.editingFinished.connect(self.change_movefiles_parameters)
        self.movefiles_movew.clicked.connect(self.start_movefiles)
        self.movefiles_stopw.clicked.connect(self.stop_movefiles)
        self.movefiles_undow.clicked.connect(self.undo_movefiles)
        
        #Reset self.move_filenames and self.move_datetimes, which is required to let undoing the movement of files work correctly.
        self.move_filenames={}; self.move_datetimes={}
        
        self.movefiles.resize(self.movefiles.sizeHint())
        self.movefiles.show()
        
    def change_movefiles_parameters(self):
        for j in self.movefiles_parameters:
            exec('self.movefiles_parameters[j]=self.movefiles_'+j+'w.text()')        
        
    def update_movefiles_enddate(self):
        #Doing this speeds up the selection part when moving files for only one day.
        self.movefiles_enddatew.setText(self.movefiles_startdatew.text())
    """
    Before starting to undo the movement of files, it is required to first press the stop button, that stops the previous action (movement
    of files).
    """
    def start_movefiles(self):
        self.movefiles_movew.setEnabled(False); self.movefiles_undow.setEnabled(False)
        self.movefiles_stopw.setEnabled(True)
        self.move_files()
    def stop_movefiles(self,set_stop_movingfiles=True):
        self.movefiles_movew.setEnabled(True); self.movefiles_undow.setEnabled(True)
        self.movefiles_stopw.setEnabled(False)
        self.stop_movingfiles=True
    def undo_movefiles(self):
        self.movefiles_movew.setEnabled(False); self.movefiles_undow.setEnabled(False)
        self.movefiles_stopw.setEnabled(True)
        self.move_files(undo=True)
    def update_movefiles_infow(self,text,color):
        self.movefiles_infow.setStyleSheet('QLineEdit {color:'+color+'}')
        self.movefiles_infow.setText(text); self.movefiles_infow.repaint()
        if color=='red': #Enable/disable clicking at particular widgets
            self.stop_movefiles()
    def move_files(self,undo=False):
        self.update_movefiles_infow('','black')
        self.stop_movingfiles=False
                
        old_dir=self.movefiles_oldstructurew.text()
        old_dir=old_dir.replace('\\','/') #Replace backslashes by forward slashes, as this the rest of the code is built for dealing with
        #forward slashes.

        if not bg.check_correctness_dir_string(old_dir):
            self.update_movefiles_infow('Old directory structure is incorrect','red')     
        new_dir=self.movefiles_newstructurew.text()
        new_dir=new_dir.replace('\\','/') #Replace backslashes by forward slashes, as this the rest of the code is built for dealing with
        #forward slashes.
        if not bg.check_correctness_dir_string(new_dir):
            self.update_movefiles_infow('New directory structure is incorrect','red')     
        
        startdate=self.movefiles_startdatew.text(); starttime=self.movefiles_starttimew.text()
        enddate=self.movefiles_enddatew.text(); endtime=self.movefiles_endtimew.text()
        if not all([startdate=='',starttime=='',enddate=='',endtime=='']) and (
        not ft.correct_datetimeinput(startdate,starttime) or not ft.correct_datetimeinput(enddate,endtime)): 
            self.update_movefiles_infow('Incorrect dates and/or times','red')
            return
        startdatetime=startdate+starttime; enddatetime=enddate+endtime
        if startdatetime=='':
            #In this case all dates and times are included.
            startdatetime=None; enddatetime=None
        
        selected_sources = [i for i,j in self.movefiles_sourcesw.items() if j.checkState() == 2]
        if selected_sources:
            selected_radars = [j for j in gv.radars_all if gv.data_sources[j] in selected_sources]
        else:
            selected_radars = [self.crd.selected_radar]
              
        if not undo:
            removefiles_initialfolder=True if self.movefiles_removefilesinpreviousdirsw.checkState()==2 else False
        else:
            #Always remove files from the Archived folder when undoing the movement of files.
            removefiles_initialfolder=True
            
        old_directories=[]; new_directories=[]
        for i in selected_radars:
            if self.stop_movingfiles: return
                        
            if not undo:
                self.completely_selected_directories,self.move_filenames[i],self.move_datetimes[i]=self.dsg.get_filenames_and_datetimes_in_datetime_range(i,dir_string=old_dir,startdatetime=startdatetime,enddatetime=enddatetime,return_completely_selected_directories=True)
            else:
                if not i in self.move_filenames: 
                    #In this case there are no files to set back.
                    continue  
              
            old_directories, new_directories=self._move_files(self.completely_selected_directories,self.move_filenames[i],self.move_datetimes[i],old_directories,new_directories,removefiles_initialfolder,undo,i,old_dir=old_dir,new_dir=new_dir)
            if self.stop_movingfiles: return

        if self.movefiles_infow.text()=='':
            self.update_movefiles_infow('No files found','red')
             
        #Remove empty directories
        directories=np.unique(old_directories if not undo else new_directories)
        for j in directories:
            if os.path.exists(j) and bg.check_dir_empty(j):
                os.removedirs(j)
        self.movefiles_movew.setEnabled(True); self.movefiles_undow.setEnabled(True)
        self.movefiles_stopw.setEnabled(False)
        
    def _move_files(self,completely_selected_directories,filenames,datetimes,old_directories,new_directories,removefiles_initialfolder,undo,radar,dataset=None,old_dir=None,new_dir=None):
        #old_dir and new_dir are only used when move_type=='movefiles'.
        moved_directories=[]
        for k in range(0,len(filenames)):
            if self.stop_movingfiles: 
                return None,None
            
            date=str(datetimes[k])[:8]; time=str(datetimes[k])[-4:]
            
            old_directory=self.dsg.get_directory(date,time,radar,dir_string=old_dir)
            new_directory=self.dsg.get_directory(date,time,radar,dir_string=new_dir)

            if not old_directory in old_directories:
                old_directories.append(old_directory)
            if not new_directory in new_directories:
                new_directories.append(new_directory)
            
            if removefiles_initialfolder and old_directory in completely_selected_directories and (
            os.path.dirname(old_directory)==os.path.dirname(new_directory)):
                """In this case the complete directory is moved/renamed, because all files in the directory need to be replaced.
                This is much faster than renaming files individually.
                It is only done when only the name of the last subfolder is changed, because otherwise there is not much of an improvement
                in speed compared to moving files individually.
                """
                if old_directory in moved_directories:
                    continue
                else:
                    moving_text=radar+(' '+dataset if not dataset is None else '')+', '+date+time
                    moving_text+=', '+(old_directory+'->'+new_directory if not undo else new_directory+'->'+old_directory)
                    try:
                        if not undo: os.rename(old_directory,new_directory)
                        else: os.rename(new_directory,old_directory)
                        moved_directories.append(old_directory)
                        self.update_movefiles_infow(moving_text,'green')
                    except Exception as e:
                        self.update_movefiles_infow(str(e),'red')
            else:
                if undo and not os.path.exists(old_directory):
                    os.makedirs(old_directory)
                if not undo and not os.path.exists(new_directory):
                    os.makedirs(new_directory)

                if not undo:
                    old_path=opa(os.path.join(old_directory,filenames[k]))
                    new_path=opa(os.path.join(new_directory,filenames[k]))
                else:
                    #Reverse directories when undoing.
                    old_path=opa(os.path.join(new_directory,filenames[k]))
                    new_path=opa(os.path.join(old_directory,filenames[k]))
                                              
                moving_text=radar+(' '+dataset if not dataset is None else '')+', '+date+time
                moving_text+=', '+(old_directory+'->'+new_directory if not undo else new_directory+'->'+old_directory)
                try:
                    if removefiles_initialfolder:
                        if not os.path.exists(new_path):
                            shutil.move(old_path,new_path)
                        elif os.path.exists(old_path):
                            os.remove(old_path)
                    elif not os.path.exists(new_path):
                        shutil.copyfile(old_path,new_path)
                    self.update_movefiles_infow(moving_text,'green')
                except Exception as e:
                    if os.path.exists(new_path):
                        #In this case the error occurred while attempting to delete the old path, and this is not regarded as a failed movement.
                        self.update_movefiles_infow(moving_text,'green')
                    else:
                        self.update_movefiles_infow(str(e),'red')
                    
            QApplication.processEvents()
        return old_directories, new_directories


            
    def settings(self):
        self.settings=QTabWidget()
        self.settings.setWindowTitle('NLradar settings')
        self.settingsmain=QWidget(); self.settingsmap=QWidget(); self.settingsdownload=QWidget(); self.settingsdatastorage=QWidget(); self.settingscolortables=QWidget(); self.settingsalgorithms = QWidget(); self.settingsmiscellaneous=QWidget()
        self.settings.addTab(self.settingsmain,'Main')
        self.settings.addTab(self.settingsmap,'Map')
        self.settings.addTab(self.settingsdownload,'Download')
        self.settings.addTab(self.settingsdatastorage,'Data storage')
        self.settings.addTab(self.settingscolortables,'Color tables')
        self.settings.addTab(self.settingsalgorithms,'Algorithms')
        self.settings.addTab(self.settingsmiscellaneous,'Miscellaneous')
        self.settings_tabmain(); self.settings_tabmap(); self.settings_tabdownload(); self.settings_tabdatastorage(); self.settings_tabcolortables(); self.settings_tabalgorithms(); self.settings_tabmiscellaneous()
        self.settings.resize(self.settings.sizeHint())
        self.settings.show()
        
    def settings_tabmain(self):
        layout=QFormLayout()
        
        self.dimensions_mainw={}; self.fontsizes_mainw={}
        hboxes={'dimensions':{},'fontsizes':{}}
        for j in self.dimensions_main:
            self.dimensions_mainw[j]=QLineEdit(format(self.pb.scale_physicalsize(self.dimensions_main[j]), '.2f'))
            self.dimensions_mainw[j].editingFinished.connect(lambda j=j: self.change_wsizes(j))
            hboxes['dimensions'][j]=QHBoxLayout()
            hboxes['dimensions'][j].addWidget(self.dimensions_mainw[j]); hboxes['dimensions'][j].addStretch(30)       
        for j in self.fontsizes_main:
            self.fontsizes_mainw[j]=QLineEdit(format(self.pb.scale_pointsize(self.fontsizes_main[j]), '.1f'))
            self.fontsizes_mainw[j].editingFinished.connect(lambda j=j: self.change_fontsize(j))
            hboxes['fontsizes'][j]=QHBoxLayout()
            hboxes['fontsizes'][j].addWidget(self.fontsizes_mainw[j]); hboxes['fontsizes'][j].addStretch(30)       
            
        b_size=5
        hbox_bgcolor=QHBoxLayout(); hbox_panelbdscolor=QHBoxLayout()
        hboxes_colors=[hbox_bgcolor,hbox_panelbdscolor]
        for hbox in hboxes_colors:
            hbox.addStretch(0)      
            
        self.bgcolorw=QLineEdit()
        self.bgcolor_select=QPushButton('Select', autoDefault=True)
        hbox_bgcolor.addWidget(self.bgcolorw,b_size+5); hbox_bgcolor.addWidget(self.bgcolor_select,b_size+4)
        self.bgcolorw.setText(str(int(self.bgcolor[0]))+','+str(int(self.bgcolor[1]))+','+str(int(self.bgcolor[2])))
        
        self.panelbdscolorw=QLineEdit()
        self.panelbdscolor_select=QPushButton('Select', autoDefault=True)
        hbox_panelbdscolor.addWidget(self.panelbdscolorw,b_size+5); hbox_panelbdscolor.addWidget(self.panelbdscolor_select,b_size+4)
        self.panelbdscolorw.setText(str(int(self.panelbdscolor[0]))+','+str(int(self.panelbdscolor[1]))+','+str(int(self.panelbdscolor[2])))
        
        for hbox in hboxes_colors:
            hbox.addStretch(40)
        
        widgets=[[QLabel('Width color bar areas (cm)'),hboxes['dimensions']['width']],
                      [QLabel('Height title areas (cm)'),hboxes['dimensions']['height']],
                      [QLabel('Titles font size'),hboxes['fontsizes']['titles']],
                      [QLabel('Color bar labels font size'),hboxes['fontsizes']['cbars_labels']],
                      [QLabel('Color bar ticks font size'),hboxes['fontsizes']['cbars_ticks']],
                      [QLabel(''),QLabel('')],
                      [QLabel('Background color (RGB)'),hbox_bgcolor],
                      [QLabel('Color of panel borders (RGB)'),hbox_panelbdscolor]]
        for j in range(0,len(widgets)):
            layout.addRow(widgets[j][0],widgets[j][1])
            
        self.bgcolorw.editingFinished.connect(lambda: self.change_bgcolor('QLineEdit'))
        self.bgcolor_select.clicked.connect(lambda: self.change_bgcolor('QPushButton'))
        self.panelbdscolorw.editingFinished.connect(lambda: self.change_panelbdscolor('QLineEdit'))
        self.panelbdscolor_select.clicked.connect(lambda: self.change_panelbdscolor('QPushButton'))
                        
        self.settingsmain.setLayout(layout)
        
    def change_wsizes(self,dimension):
        input_size=self.dimensions_mainw[dimension].text()
        number=ft.to_number(input_size)
        if not number is None:
            self.dimensions_main[dimension] = number/self.pb.scale_physicalsize(1)
            self.pb.wdims[0 if dimension=='width' else 1] = self.dimensions_main[dimension]
            self.pb.on_resize()
            self.pb.update()
            
    def change_fontsize(self,visual):
        input_size=self.fontsizes_mainw[visual].text()
        number=ft.to_number(input_size)
        if not number is None:
            self.fontsizes_main[visual]=number/self.pb.scale_pointsize(1)
            self.pb.visuals[visual].font_size=number
            if visual in ('cbars_ticks', 'cbars_labels'): #cbars should be updated since the position of some of the cbar ticks and labels
                #is dependent on the font size.
                self.pb.set_cbars(resize=True) #set resize=True as otherwise no update takes place
            elif visual == 'titles': #Also here the position is font size dependent
                self.pb.set_titles()
            self.pb.update()
            
    def change_color(self,color_qlineedit,color_object,source,alpha=False):
        if source=='QLineEdit':
            inputcolor=color_qlineedit.text()
        else:
            colorwidget=QColorDialog()
            inputcolor=colorwidget.getColor(initial=QColor(*color_object[:4 if alpha else 3].astype(int)),
                                            options=QColorDialog.ShowAlphaChannel if alpha else QColorDialog.ColorDialogOptions())
        if (source=='QLineEdit' and ft.rgb(inputcolor,alpha=alpha)!=False) or (source=='QPushButton' and inputcolor.isValid()):
            if source=='QLineEdit':
                color_object=np.array(list(ft.rgb(inputcolor,alpha=alpha)))
            else:
                color_object=np.array(list(inputcolor.getRgb()))
                if not alpha: color_object=color_object[:3]
        return color_object                      
    
    def change_bgcolor(self,source):
        self.bgcolor = self.change_color(self.bgcolorw,self.bgcolor,source)
        self.bgcolorw.setText(str(int(self.bgcolor[0]))+','+str(int(self.bgcolor[1]))+','+str(int(self.bgcolor[2])))
        self.pb.visuals['background'].color=self.bgcolor/255.
        self.pb.update()    
        
    def change_panelbdscolor(self,source):
        self.panelbdscolor = self.change_color(self.panelbdscolorw,self.panelbdscolor,source)
        self.panelbdscolorw.setText(str(int(self.panelbdscolor[0]))+','+str(int(self.panelbdscolor[1]))+','+str(int(self.panelbdscolor[2])))
        self.pb.visuals['panel_borders'].set_data(color=self.panelbdscolor/255.)
        self.pb.update()        

    def settings_tabmap(self):
        map_layout=QFormLayout()
        hbox_bgmapcolor=QHBoxLayout(); hbox_mapvisibility=QHBoxLayout(); hbox_mapcolorfilter=QHBoxLayout()
        hbox_maptiles_update_time = QHBoxLayout(); hbox_boxtitles=QHBoxLayout()
        b_size=5
        hboxes_background=[hbox_bgmapcolor,hbox_mapvisibility,hbox_mapcolorfilter,hbox_maptiles_update_time]
        for hbox in hboxes_background:
            hbox.addStretch(0)      
        
        self.bgmapcolorw=QLineEdit()
        self.bgmapcolor_select=QPushButton('Select', autoDefault=True)
        hbox_bgmapcolor.addWidget(self.bgmapcolorw,b_size+5); hbox_bgmapcolor.addWidget(self.bgmapcolor_select,b_size+4)
        self.bgmapcolorw.setText(str(int(self.bgmapcolor[0]))+','+str(int(self.bgmapcolor[1]))+','+str(int(self.bgmapcolor[2])))

        self.mapvis_false=QRadioButton('False'); self.mapvis_true=QRadioButton('True')
        self.mapvis_group=QButtonGroup(); self.mapvis_group.addButton(self.mapvis_false); self.mapvis_group.addButton(self.mapvis_true)
        hbox_mapvisibility.addWidget(self.mapvis_false,b_size); hbox_mapvisibility.addWidget(self.mapvis_true,b_size)
        self.mapvis_true.setChecked(True) if self.mapvisibility else self.mapvis_false.setChecked(True)
        
        self.mapcolorfilterw=QLineEdit(ft.list_to_string(self.mapcolorfilter))
        hbox_mapcolorfilter.addWidget(self.mapcolorfilterw)
        
        self.maptiles_update_timew = QLineEdit(str(self.maptiles_update_time))
        self.maptiles_update_timew.setToolTip('Map tiles are updated when the last occurrence of panning/zooming was this number of seconds ago.')
        hbox_maptiles_update_time.addWidget(self.maptiles_update_timew)
        
        for hbox in hboxes_background:
            hbox.addStretch(40)
        hbox_mapcolorfilter.addStretch(10)
        
        hbox_radars=QHBoxLayout()
        self.radars_selectcolorsw=QPushButton('Select size and colors radar markers', autoDefault=True)
        hbox_radars.addWidget(self.radars_selectcolorsw,b_size+5); hbox_radars.addStretch(50)
        
        hbox_boxtitles.addStretch(1); hbox_boxtitles.addWidget(QLabel('       Visibility'),b_size+3)
        hbox_boxtitles.addWidget(QLabel('                Colors (RGBA)'),b_size+3)
        hbox_boxtitles.addStretch(40)
        
        hboxes_lines={}; self.lines_widgets={}; self.lines_groups={}
        for j in self.lines_names:
            hboxes_lines[j]=QHBoxLayout()
            self.lines_widgets[j]=[QRadioButton('False'),QRadioButton('True'),QLineEdit(),QPushButton('Select', autoDefault=True)]
            self.lines_groups[j]=QButtonGroup()
            for i in range(0,2):
                self.lines_groups[j].addButton(self.lines_widgets[j][i])
                hboxes_lines[j].addWidget(self.lines_widgets[j][i])
                self.lines_widgets[j][1].setChecked(True) if j in self.lines_show else self.lines_widgets[j][0].setChecked(True)
            if j in self.lines_names:
                self.lines_widgets[j][2].setText(ft.list_to_string(self.lines_colors[j].astype(int)))

            hboxes_lines[j].addStretch(1)
            hboxes_lines[j].addWidget(self.lines_widgets[j][2],b_size+8)
            hboxes_lines[j].addWidget(self.lines_widgets[j][3],b_size+4)
            hboxes_lines[j].addStretch(30)
            
        self.lines_widthw=QLineEdit(str(self.lines_width))
        hbox_linewidth=QHBoxLayout(); hbox_linewidth.addWidget(self.lines_widthw); hbox_linewidth.addStretch(30)
        
        self.lines_antialiasw=QCheckBox(); self.lines_antialiasw.setTristate(False)
        self.lines_antialiasw.setCheckState(2 if self.lines_antialias else 0)
        hbox_antialias=QHBoxLayout(); hbox_antialias.addWidget(self.lines_antialiasw); hbox_antialias.addStretch(30)
            
        hbox_heightrings_derivedproducts=QHBoxLayout()
        self.show_heightrings_derivedproductsw={}
        for j in gv.plain_products:
            self.show_heightrings_derivedproductsw[j]=QCheckBox(j.upper())
            self.show_heightrings_derivedproductsw[j].setCheckState(2 if self.show_heightrings_derivedproducts[j] else 0)
            hbox_heightrings_derivedproducts.addWidget(self.show_heightrings_derivedproductsw[j])
        hbox_heightrings_derivedproducts.addStretch(30)
            
        self.showgridheightrings_panzoomw=QCheckBox(); self.showgridheightrings_panzoomw.setTristate(False)
        self.showgridheightrings_panzoomw.setCheckState(2 if self.showgridheightrings_panzoom else 0)
        self.showgridheightrings_panzoom_timew=QLineEdit(); self.showgridheightrings_panzoom_timew.setText(str(ft.rifdot0(self.showgridheightrings_panzoom_time)))
        
        self.gridheightrings_fontcolorw={}; self.gridheightrings_fontcolor_select={}
        for j in ('bottom','top'):
            self.gridheightrings_fontcolorw[j]=QLineEdit()
            self.gridheightrings_fontcolorw[j].setText(ft.list_to_string(self.gridheightrings_fontcolor[j].astype(int)))
            self.gridheightrings_fontcolor_select[j]=QPushButton('Select', autoDefault=True)
        self.gridheightrings_fontsizew=QLineEdit(format(self.pb.scale_pointsize(self.gridheightrings_fontsize), '.1f'))
        self.grid_showtextw=QCheckBox(); self.grid_showtextw.setTristate(False)
        self.grid_showtextw.setCheckState(2 if self.grid_showtext else 0)
        
        hbox_fontcolor=QHBoxLayout(); hbox_fontsize=QHBoxLayout(); hbox_showtextgrid=QHBoxLayout(); 
        
        hbox_showgridheightrings_panzoom=QHBoxLayout()
        hbox_showgridheightrings_panzoom_time=QHBoxLayout()
        hboxes=[hbox_fontcolor,hbox_fontsize,hbox_showtextgrid,hbox_showgridheightrings_panzoom,hbox_showgridheightrings_panzoom_time]
        for hbox in hboxes:
            hbox.addStretch(0)      
        hbox_fontcolor.addWidget(self.gridheightrings_fontcolorw['bottom']); hbox_fontcolor.addWidget(self.gridheightrings_fontcolor_select['bottom'])
        hbox_fontcolor.addWidget(self.gridheightrings_fontcolorw['top']); hbox_fontcolor.addWidget(self.gridheightrings_fontcolor_select['top'])
        hbox_fontsize.addWidget(self.gridheightrings_fontsizew)
        hbox_showtextgrid.addWidget(self.grid_showtextw)
        hbox_showgridheightrings_panzoom.addWidget(self.showgridheightrings_panzoomw)
        hbox_showgridheightrings_panzoom_time.addWidget(self.showgridheightrings_panzoom_timew)
        for hbox in hboxes:
            hbox.addStretch(10) if hbox==hbox_fontcolor else hbox.addStretch(30)
        
        map_widgets=[[QLabel('<b>Map</b>'),QLabel('')],
                     [QLabel('Background color (RGB)'),hbox_bgmapcolor],
                     [QLabel('Visibility'),hbox_mapvisibility],
                     [QLabel('Color filter (RGBA)'),hbox_mapcolorfilter],  
                     [QLabel('Map tiles update time (s)'), hbox_maptiles_update_time],
                     [QLabel('Radars'),hbox_radars],
                     [QLabel(''),QLabel('')],
                     [QLabel('<b>Lines</b>'),hbox_boxtitles],
                     [QLabel('Countries'),hboxes_lines['countries']],
                     [QLabel('Provinces'),hboxes_lines['provinces']],
                     [QLabel('Rivers'),hboxes_lines['rivers']],
                     [QLabel('Grid'),hboxes_lines['grid']],
                     [QLabel('Height rings'),hboxes_lines['heightrings']],
                     [QLabel('Line width'),hbox_linewidth],
                     [QLabel('Apply antialiasing to lines'),hbox_antialias],
                     [QLabel('Show height rings derived products'),hbox_heightrings_derivedproducts],
                     [QLabel('Show grid/height rings pan/zoom'),hbox_showgridheightrings_panzoom],
                     [QLabel('If not, update them after x seconds'),hbox_showgridheightrings_panzoom_time],
                     [QLabel('<b>Text grid and height rings</b>'),QLabel('')],
                     [QLabel('Font color bottom, top (RGB)'),hbox_fontcolor],
                     [QLabel('Font size'),hbox_fontsize],
                     [QLabel('Show grid coordinates'),hbox_showtextgrid]]
        for j in range(0,len(map_widgets)):
            map_layout.addRow(map_widgets[j][0],map_widgets[j][1])
        self.settingsmap.setLayout(map_layout)
        
        
        self.bgmapcolorw.editingFinished.connect(lambda: self.change_bgmapcolor('QLineEdit'))
        self.bgmapcolor_select.clicked.connect(lambda: self.change_bgmapcolor('QPushButton'))
        self.mapvis_false.toggled.connect(lambda: self.pb.change_mapvisibility(False))
        self.mapvis_true.toggled.connect(lambda: self.pb.change_mapvisibility(True))
        self.mapcolorfilterw.editingFinished.connect(self.pb.change_mapcolorfilter)
        self.maptiles_update_timew.editingFinished.connect(self.change_maptiles_update_time)
        self.radars_selectcolorsw.clicked.connect(self.select_properties_radar_markers)
        for line in self.lines_names:
            self.lines_widgets[line][0].toggled.connect(lambda state, line=line: self.change_line_state(line,False))
            self.lines_widgets[line][1].toggled.connect(lambda state, line=line: self.change_line_state(line,True))
            #No state for QLineEdit
            self.lines_widgets[line][2].editingFinished.connect(lambda line=line: self.change_line_state(line,'QLineEdit'))    
            self.lines_widgets[line][3].clicked.connect(lambda state, line=line: self.change_line_state(line,'QPushButton'))
        self.lines_widthw.editingFinished.connect(self.change_lines_width)
        self.lines_antialiasw.stateChanged.connect(self.change_lines_antialias)
        self.showgridheightrings_panzoomw.stateChanged.connect(self.change_showgridheightrings_panzoom)
        self.showgridheightrings_panzoom_timew.editingFinished.connect(self.change_showgridheightrings_panzoom_time)
        for j in ('bottom','top'):
            self.gridheightrings_fontcolorw[j].editingFinished.connect(lambda j=j: self.change_gridheightrings_fontcolor(j,'QLineEdit'))
            self.gridheightrings_fontcolor_select[j].clicked.connect(lambda state, j=j: self.change_gridheightrings_fontcolor(j,'QPushButton'))
        self.gridheightrings_fontsizew.editingFinished.connect(self.change_gridheightrings_fontsize)
        self.grid_showtextw.stateChanged.connect(self.change_grid_showtext)
        for j in gv.plain_products:
            self.show_heightrings_derivedproductsw[j].stateChanged.connect(lambda state, j=j: self.change_show_heightrings_derivedproducts(j))
        
        
    def select_properties_radar_markers(self):
        self.radar_markers_properties_widget=QWidget()
        self.radar_markers_properties_widget.setWindowTitle('Select colors and size radar markers')
        layout=QFormLayout()
        
        b_size=5
        
        self.radar_markersizew=QLineEdit(str(ft.round_float(self.pb.scale_pixelsize(self.radar_markersize))))
        layout.addRow(QLabel('Size'),self.radar_markersizew)
        types=('Default','Selected','Automatic download','Automatic download + selected')
        hboxes_types={}; self.radar_colorsw={}; self.radar_colors_selectw={}
        for j in types:
            hboxes_types[j]=QHBoxLayout()
            self.radar_colorsw[j]=QLineEdit()
            self.radar_colors_selectw[j]=QPushButton('Select', autoDefault=True)
            hboxes_types[j].addWidget(self.radar_colorsw[j],b_size+5); hboxes_types[j].addWidget(self.radar_colors_selectw[j],b_size+4)
            self.radar_colorsw[j].setText(ft.list_to_string(self.radar_colors[j]))
            layout.addRow(QLabel(j),hboxes_types[j])
            
        self.radar_markersizew.editingFinished.connect(self.change_radar_markersize)
        for j in types:
            self.radar_colorsw[j].editingFinished.connect(lambda j=j: self.change_radar_colors(j,'QLineEdit'))
            self.radar_colors_selectw[j].clicked.connect(lambda state,j=j: self.change_radar_colors(j,'QPushButton'))
            
        self.radar_markers_properties_widget.setLayout(layout)
        self.radar_markers_properties_widget.resize(self.radar_markers_properties_widget.sizeHint())
        self.radar_markers_properties_widget.show()
        
    def change_bgmapcolor(self,source):
        self.bgmapcolor = self.change_color(self.bgmapcolorw,self.bgmapcolor,source)
        self.bgmapcolorw.setText(str(int(self.bgmapcolor[0]))+','+str(int(self.bgmapcolor[1]))+','+str(int(self.bgmapcolor[2])))
        self.pb.visuals['background_map'].color=self.bgmapcolor/255.
        self.pb.update()
        
    def change_maptiles_update_time(self):
        inputtime=self.maptiles_update_timew.text()
        number=ft.to_number(inputtime)
        if not number is None and number>0: 
            self.maptiles_update_time = number
        
    def change_radar_markersize(self):
        inputsize=self.radar_markersizew.text()
        number=ft.to_number(inputsize)
        if not number is None and number>0: 
            self.radar_markersize=number/self.pb.scale_pixelsize(1)
        else: self.radar_markersizew.setText(str(ft.round_float(self.pb.scale_pixelsize(self.radar_markersize))))
        self.pb.set_radarmarkers_data()
        self.pb.update()
        
    def change_radar_colors(self,radartype,source):
        self.radar_colors[radartype]=self.change_color(self.radar_colorsw[radartype],self.radar_colors[radartype],source)
        self.radar_colorsw[radartype].setText(ft.list_to_string(self.radar_colors[radartype]))
        self.pb.set_radarmarkers_data()
        self.pb.update()
        
    def change_line_state(self,linetype,source):
        #Using isChecked() is required, because a signal is emitted when a radiobutton gets toggled, as well as when it gets untoggled
        if source==False and self.lines_widgets[linetype][0].isChecked(): 
            self.lines_show.remove(linetype)
            if linetype in self.ghtext_show: self.ghtext_show.remove(linetype)
        elif source==True and self.lines_widgets[linetype][1].isChecked(): 
            self.lines_show.append(linetype)
            if linetype in self.ghtext_names and not (linetype=='grid' and not self.grid_showtext): self.ghtext_show.append(linetype)            
        elif source in ('QLineEdit','QPushButton'):
            self.lines_colors[linetype]=self.change_color(self.lines_widgets[linetype][2],self.lines_colors[linetype],source,alpha=True)
            if linetype in self.lines_names:
                self.lines_widgets[linetype][2].setText(ft.list_to_string(self.lines_colors[linetype].astype(int)))
                self.pb.update_combined_lineproperties(range(10),changing_colors=True)
        
        if self.pb.firstplot_performed and linetype=='grid' and source==True: self.pb.set_grid()
        if self.pb.firstplot_performed and linetype=='heightrings' and source==True: self.pb.set_heightrings()
        if linetype in ('grid', 'heightrings'):
            self.pb.set_ghlineproperties(self.pb.panellist)
        else:
            self.pb.set_maplineproperties(self.pb.panellist)
        if linetype in self.ghtext_names:
            self.pb.set_ghtextproperties(self.pb.panellist)
        self.pb.update()
        
    def change_lines_width(self):
        input_value=self.lines_widthw.text()
        number=ft.to_number(input_value)
        if not number is None and number>0:
            self.lines_width=number
            self.pb.set_maplineproperties(self.pb.panellist)
            self.pb.update()
        self.lines_widthw.setText(str(self.lines_width))
        
    def change_lines_antialias(self):
        self.lines_antialias = True if self.lines_antialiasw.checkState()==2 else False
        #Drawing only needed for first panel, as the line visuals in the other panels are views of the first
        self.pb.visuals['map_lines'][0].antialias=self.lines_antialias
        self.pb.visuals['gh_lines'][0].antialias=self.lines_antialias
        self.pb.update()
            
    def change_show_heightrings_derivedproducts(self,product):
        self.show_heightrings_derivedproducts[product]=True if self.show_heightrings_derivedproductsw[product].checkState()==2 else False
        panellist_product=[j for j in self.pb.panellist if self.crd.products[j]==product]
        if len(panellist_product)>0:
            self.pb.set_heightrings(panellist_product)
            self.pb.set_ghlineproperties(panellist_product)
            self.pb.set_ghtextproperties(panellist_product)
            self.pb.update()
            
    def change_showgridheightrings_panzoom(self):
        self.showgridheightrings_panzoom=True if self.showgridheightrings_panzoomw.checkState()==2 else False
    def change_showgridheightrings_panzoom_time(self):
        input_showgridheightrings_panzoom_time=self.showgridheightrings_panzoom_timew.text()
        number=ft.to_number(input_showgridheightrings_panzoom_time)
        if not number is None and number>0.:
            self.showgridheightrings_panzoom_time=number
        else: self.showgridheightrings_panzoom_timew.setText(str(ft.rifdot0(self.showgridheightrings_panzoom_time)))
            
    def change_gridheightrings_fontcolor(self,bottom_top,source):
        self.gridheightrings_fontcolor[bottom_top]=self.change_color(self.gridheightrings_fontcolorw[bottom_top],self.gridheightrings_fontcolor[bottom_top],source)
        self.gridheightrings_fontcolorw[bottom_top].setText(ft.list_to_string(self.gridheightrings_fontcolor[bottom_top].astype(int)))
        if bottom_top == 'bottom': 
            self.pb.text_hor_bottom_colorfilter.filter = np.append(self.gridheightrings_fontcolor['bottom']/255., 1.)
        else:
            self.pb.text_hor_top_colorfilter.filter = np.append(self.gridheightrings_fontcolor['top']/255., 1.)
        self.pb.update()
    def change_gridheightrings_fontsize(self):
        inputsize=self.gridheightrings_fontsizew.text()
        number=ft.to_number(inputsize)
        if not number is None and number>0: 
            self.gridheightrings_fontsize=number/self.pb.scale_pointsize(1)
            for j in range(10):
                # Since 'text_hor2' and 'text_vert2' are visualviews, they don't need to have their font_size set too.
                self.pb.visuals['text_hor1'][j].font_size = number
                if j in self.pb.visuals['text_vert1']:
                    self.pb.visuals['text_vert1'][j].font_size = number
        else: self.gridheightrings_fontsizew.setText(str(ft.round_float(self.pb.scale_pointsize(self.gridheightrings_fontsize))))
        if 'grid' in self.lines_show:
            self.pb.set_grid() #Update the grid since the positions are dependent on the font size
            self.pb.set_ghtextproperties(self.pb.panellist)
            self.pb.update()
    def change_grid_showtext(self):
        self.grid_showtext=True if self.grid_showtextw.checkState()==2 else False
        if 'grid' in self.lines_show:
            self.ghtext_show.append('grid') if self.grid_showtext else self.ghtext_show.remove('grid') 
            self.pb.set_ghtextproperties(self.pb.panellist)
            self.pb.update()

        
        
    def settings_tabdownload(self):
        layout = QVBoxLayout()
        self.networktimeoutw=QLineEdit(); self.networktimeoutw.setText(str(ft.rifdot0(self.networktimeout)))
        self.minimum_downloadspeedw=QLineEdit(); self.minimum_downloadspeedw.setText(str(ft.rifdot0(self.minimum_downloadspeed)))
        self.networktimeoutw.editingFinished.connect(self.change_networktimeout)
        self.minimum_downloadspeedw.editingFinished.connect(self.change_minimum_downloadspeed)
        hbox = {}
        for widget in ('networktimeoutw', 'minimum_downloadspeedw'):
            hbox[widget] = QHBoxLayout()
            hbox[widget].addStretch(1); hbox[widget].addWidget(eval('self.'+widget)); hbox[widget].addStretch(50)
        formlayout=QFormLayout()
        formlayout.addRow(QLabel('Download timeout limit (seconds)'),hbox['networktimeoutw'])
        formlayout.addRow(QLabel('Minimum download speed (MB/minute)'),hbox['minimum_downloadspeedw'])
        layout.addLayout(formlayout)
            
        layout.addWidget(QLabel(''))
        self.api_keysw = copy.deepcopy(self.api_keys) # Is only done to get the same keys, values will be updated below
        labels = {'KNMI': "Set the API keys for the <b>KNMI</b> Data Platform. You can request these <A href='https://developer.dataplatform.knmi.nl/apis/'>here</a>.",
                  'DMI': "Set the API key for the <b>DMI</b> open data service. Follow the <A href='https://opendatadocs.dmi.govcloud.dk/en/Authentication'>following</a> guide to obtain a key for the radar data service.",
                  'Météo-France': "Set the API key for de <b>Météo-France</b> open data service. Subscribe to the radar data API <A href='https://portail-api.meteofrance.fr/web/en/api/DonneesPubliquesRadar'>here</a>, then click on 'configure the API' and generate a token."}
        labels_keys = {'KNMI': {'opendata': 'Open Data', 'sfcobs': 'Current 10 Minute Data KNMI Stations'},
                       'DMI': {'radardata': 'Radar Data'},
                       'Météo-France': {'radardata': 'Radar Data'}}
        for datasource in self.api_keys:
            label = QLabel(labels[datasource])
            label.setOpenExternalLinks(True)
            layout.addWidget(label)
            formlayout=QFormLayout()
            for key in self.api_keys[datasource]:
                self.api_keysw[datasource][key] = QLineEdit(self.api_keys[datasource][key])
                self.api_keysw[datasource][key].editingFinished.connect(lambda datasource=datasource, key=key: self.change_api_keys(datasource, key))
                hbox = QHBoxLayout()
                hbox.addWidget(self.api_keysw[datasource][key], 5); hbox.addStretch(2)
                formlayout.addRow(QLabel('API key '+labels_keys[datasource][key]),hbox)
            layout.addLayout(formlayout)
            
        layout.addStretch(50)
        self.settingsdownload.setLayout(layout)
        
                
    def change_networktimeout(self):
        input_networktimeout=self.networktimeoutw.text()
        number=ft.to_number(input_networktimeout)
        if not number is None: 
            self.networktimeout=number if number>=0. else 0.
            self.networktimeoutw.setText(str(ft.rifdot0(self.networktimeout)))
        else: self.networktimeoutw.setText(str(ft.rifdot0(self.networktimeout)))
    def change_minimum_downloadspeed(self):
        input_minimum_downloadspeedw=self.minimum_downloadspeedw.text()
        number=ft.to_number(input_minimum_downloadspeedw)
        if not number is None: self.minimum_downloadspeed=number
        else: self.minimum_downloadspeedw.setText(str(ft.rifdot0(self.minimum_downloadspeed)))
    def change_api_keys(self, datasource, key):
        self.api_keys[datasource][key] = self.api_keysw[datasource][key].text().strip()
        

    def settings_tabdatastorage(self):
        datastorage_layout=QVBoxLayout()
        
        datastorage_text=["The program reads files from your disk, and here you can specify the directories in which your data is located. You can use both '/' and '\\' as the path separator.",
                          "If the directory names contain dates and times, then you can use variables like ${date} to specify this.",
                          "The following variables are allowed: ${date}, ${date+}, ${timeX}, ${timeX+}, ${datetimeX} and ${datetimeX+}. Here, dates and times have the formats YYYYMMDD and HHMM (and datetime the format",
                          "YYYYMMDDHHMM). Further, X should be a number, and represents the number of minutes to which a time will be floored to get the time in the name of the directory. When e.g. X=5 and",
                          "time=1442, then ${timeX} becomes 1440. When X=60, then ${timeX}=1400 etc. A condition for X is that dividing 1440 by X must give an integer.",
                          "Further, a variable with a plus indicates that the next date/time/datetime is used, where e.g.${timeX+} becomes ${timeX}+X minutes etc. Variables with a plus are not allowed to appear before the",
                          "corresponding variable without the plus.",
                          "Finally, time variables are not allowed to appear before the first date variable. Date and time variables are also not allowed to appear in more than 2 substrings of the directory, where a substring is",
                          "a part separated by '/' operators. And when a datetime variable is included, they may appear in only one substring.",
                          "You can also include ${radar} and ${radarID} variables. ${radar} represents the radar name without spaces, and ${radarID} is the identifier that is used in filenames to indicate which radar is used.",
                          "These identifiers can be found for the radars by mouse hovering over the labels in the left column of the window for selecting directories.",
                          "",
                          "More than one directory structure can be specified per radar and dataset, and different structure should be separated by a ';'. If the first part of the directory structures is repeated in all structures, then",
                          "you can put it in front, and separate it by a ';;' from the rest. This part is then put in front of all directory structures that are specified next.",
                          "",
                          "If the specified directory structures do not satisfy the requirements, then this is indicated by a red color.",
                          "You can see examples of the resulting complete directory path for the current date and time input by mouse hovering over the directory structures, if the input structure is correct.",
                          "You should preferentially not put files for multiple radars in the same directories. There should preferentially also be no empty directories."]
        for j in range(0,len(datastorage_text)):
            datastorage_layout.addWidget(QLabel(datastorage_text[j],font=self.help_font))  
        
        datastorage_widgets_layout=QFormLayout()
        
        self.dirselectw=QPushButton('Set', autoDefault=True)
        datastorage_widgets_layout.addRow(QLabel('Directory structures'),self.dirselectw)
        self.dirselectw.clicked.connect(self.selectdirs)

        self.derivedproducts_dirselectw=QPushButton(self.derivedproducts_dir, autoDefault=True)
        self.derivedproducts_defaultdirw=QPushButton('Default', autoDefault=True)
        hbox_derivedproducts=QHBoxLayout(); 
        hbox_derivedproducts.addWidget(self.derivedproducts_dirselectw,6); hbox_derivedproducts.addWidget(self.derivedproducts_defaultdirw,1)
        datastorage_widgets_layout.addRow(QLabel('Derived products'),hbox_derivedproducts)
        self.derivedproducts_dirselectw.clicked.connect(self.selectdir_derivedproducts)
        self.derivedproducts_defaultdirw.clicked.connect(self.usedefaultdir_derivedproducts)
        
        datastorage_layout.addLayout(datastorage_widgets_layout)
        self.settingsdatastorage.setLayout(datastorage_layout)
                    
    def selectdir_derivedproducts(self):
        derivedproducts_dir_input=str(QFileDialog.getExistingDirectory(None, 'Select the folder:',self.derivedproducts_dir))
        if derivedproducts_dir_input!='':
            self.derivedproducts_dir=derivedproducts_dir_input    
            self.derivedproducts_dirselectw.setText(self.derivedproducts_dir)
    def usedefaultdir_derivedproducts(self):
        self.derivedproducats_dir=gv.derivedproducts_dir_Default
        self.derivedproducts_dirselectw.setText(self.derivedproducts_dir)
        
        
    def selectdirs(self):
        self.dirselect=QTabWidget()
        self.dirselect.setWindowTitle('Select directory structures')
        self.dirselecttabs={}
        self.radardirs_widgets, self.default_dirs_widgets = {}, {}
        self.additionaldirs_rds_widgets, self.additionaldirs_widgets = {}, {}
        for j in gv.data_sources_all:
            self.dirselecttabs[j]=QWidget()
            self.dirselect.addTab(self.dirselecttabs[j],j)
            self.dirs_dstabs(j)
        self.dirselect.resize(self.dirselect.sizeHint())
        self.dirselect.show()
        
    def generate_dir_example_date_and_time(self):
        input_date=self.datew.text(); input_time=self.timew.text()
        if ft.correct_datetimeinput(input_date,input_time) and not input_time=='c':
            example_date=input_date
            example_time=input_time
        else:
            example_date=''.join(ft.get_ymdhm(pytime.time())[:3]); example_time=''.join(ft.get_ymdhm(pytime.time())[-2:])
        return example_date, example_time
        
    def get_radars_datasets_source(self, datasource):
        radars = gv.radars[datasource]
        rds = [] # All available radar_datasets for datasource
        for j in radars:
            rds += [f'{j}_Z', f'{j}_V'] if j in gv.radars_with_datasets else [j]
        return radars, rds

    def dirs_dstabs(self,datasource):
        layout=QVBoxLayout()
        grid_layout=QGridLayout()

        radars, rds = self.get_radars_datasets_source(datasource)
        datasets = np.unique(np.concatenate([['Z','V'] if j in gv.radars_with_datasets else [''] for j in radars]))
        
        layout.addWidget(QLabel('General directory structure used for all radars'+', per dataset'*(len(datasets) > 1)))

        radar = self.crd.selected_radar if self.crd.selected_radar in radars else radars[0]
        for i,dataset in enumerate(datasets):
            key = datasource+f'_{dataset}'*len(dataset)
            dir_string = self.radarsources_dirs[key]
            self.radardirs_widgets[key] = QLineEdit(dir_string)
            
            tool_tip = self.get_example_dir(dir_string, radars)
            self.radardirs_widgets[key].setToolTip(tool_tip)
            
            self.default_dirs_widgets[key] = QPushButton('Default', autoDefault=True)
            
            label = QLabel('All radars '+dataset)
            label.setToolTip(gv.rplaces_to_ridentifiers[radar])
            grid_layout.addWidget(label,i,0,1,1) #row, column, rowspan, colspan
            grid_layout.addWidget(self.radardirs_widgets[key],i,1,1,10)
            grid_layout.addWidget(self.default_dirs_widgets[key],i,11,1,1)
            self.default_dirs_widgets[key].clicked.connect(lambda state, key=key: self.use_default_dir(key))
            self.radardirs_widgets[key].editingFinished.connect(lambda source=datasource, key=key: self.change_radardata_dir(source, key))
        layout.addLayout(grid_layout)
        
        layout.addWidget(QLabel('Optional additional directory structures for individual radars'))
            
        additional = [[i,j] for i,j in self.radardirs_additional.items() if self.dsg.split_radar_dataset(i)[0] in radars]
        
        grid_layout = QGridLayout()
        n = min(len(rds), 10)
        for i in range(n):
            key = datasource+f'_{i}'
            self.additionaldirs_rds_widgets[key] = QComboBox()
            self.additionaldirs_rds_widgets[key].addItems(['']+rds)
            self.additionaldirs_widgets[key] = QLineEdit()
            if i < len(additional):
                rd, additional_dir_string = additional[i]
                rd_radar = self.dsg.split_radar_dataset(rd)[0]
                
                self.additionaldirs_rds_widgets[key].setCurrentText(rd)
                self.additionaldirs_rds_widgets[key].setToolTip(gv.rplaces_to_ridentifiers[rd_radar])
                
                self.additionaldirs_widgets[key].setText(additional_dir_string)
                tool_tip = self.get_example_dir(self.radardata_dirs[rd], radars)
                self.additionaldirs_widgets[key].setToolTip(tool_tip)
                
            grid_layout.addWidget(self.additionaldirs_rds_widgets[key],i,0,1,1) #row, column, rowspan, colspan
            grid_layout.addWidget(self.additionaldirs_widgets[key],i,1,1,10)
            grid_layout.addWidget(QLabel(),i,11,1,1)
            
            self.additionaldirs_rds_widgets[key].currentTextChanged.connect(lambda text, source=datasource: self.change_individual_dirstrings(source))
            self.additionaldirs_widgets[key].editingFinished.connect(lambda source=datasource: self.change_individual_dirstrings(source))
        layout.addLayout(grid_layout)
        layout.addStretch(13-n-len(datasets))
              
        self.dirselecttabs[datasource].setLayout(layout)
        
    def get_example_dir(self, dir_string, radars):
        radar = self.crd.selected_radar if self.crd.selected_radar in radars else radars[0]
        example_date, example_time=self.generate_dir_example_date_and_time()
             
        example_dir_list = bg.dirstring_to_dirlist(dir_string)
        return ', '.join([bg.convert_dir_string_to_real_dir(d, radar, example_date, example_time)
                              for j,d in enumerate(example_dir_list)])

    def change_radardata_dir(self, source, key): # key has format source_dataset
        # Replace backslashes by forward slashes, as the rest of the code is built for dealing with forward slashes.
        input_str=self.radardirs_widgets[key].text().replace('\\','/')
        input_dir_strings=bg.dirstring_to_dirlist(input_str)
            
        input_correct = all(bg.check_correctness_dir_string(j) for j in input_dir_strings)
        if input_correct:
            self.radardirs_widgets[key].setStyleSheet('QLineEdit {color:black}')
            self.radarsources_dirs[key] = input_str
            
            radars, rds = self.get_radars_datasets_source(source)
            self.update_radardirs_source(radars, rds)
            
            tool_tip = self.get_example_dir(input_str, radars)
            self.radardirs_widgets[key].setToolTip(tool_tip)
        else:
            self.radardirs_widgets[key].setStyleSheet('QLineEdit {color:red}')
            self.radardirs_widgets[key].setToolTip('')
            
    def update_radardirs_source(self, radars, rds):
        for rd in rds:
            radar, dataset = self.dsg.split_radar_dataset(rd)
            source = gv.data_sources[radar]
            key = source+f'_{dataset}'*len(dataset)
            self.radardata_dirs[rd] = self.radarsources_dirs[key]
            if rd in self.radardirs_additional:
                general_dirstring = self.radardata_dirs[rd].strip()
                self.radardata_dirs[rd] += '; '*(len(general_dirstring) and general_dirstring[-1] != ';')+self.radardirs_additional[rd]
            
            dir_strings = bg.dirstring_to_dirlist(self.radardata_dirs[rd])
            if self.radardata_dirs_indices[rd] >= len(dir_strings):
                self.radardata_dirs_indices[rd] = len(dir_strings)-1
                
    def use_default_dir(self, key):
        self.radarsources_dirs[key] = gv.radarsources_dirs_Default[key]
        self.radardirs_widgets[key].setText(self.radarsources_dirs[key])
        
    def change_individual_dirstrings(self, source):
        radars, rds = self.get_radars_datasets_source(source)        
        for rd in rds:
            if rd in self.radardirs_additional:
                del self.radardirs_additional[rd]

        for key in self.additionaldirs_rds_widgets:
            rd = self.additionaldirs_rds_widgets[key].currentText()
            rd_radar = self.dsg.split_radar_dataset(rd)[0]
            
            input_str = self.additionaldirs_widgets[key].text().replace('\\','/')
            input_dir_strings = bg.dirstring_to_dirlist(input_str)
                
            input_correct = all(bg.check_correctness_dir_string(j) for j in input_dir_strings)
            if rd and input_correct:
                self.additionaldirs_widgets[key].setStyleSheet('QLineEdit {color:black}')
                if input_str:
                    self.radardirs_additional[rd] = input_str
                
                self.update_radardirs_source(radars, rds)
                
                tool_tip = self.get_example_dir(self.radardata_dirs[rd], radars)
                self.additionaldirs_widgets[key].setToolTip(tool_tip)
                self.additionaldirs_rds_widgets[key].setToolTip(gv.rplaces_to_ridentifiers[rd_radar])
            else:
                self.additionaldirs_widgets[key].setStyleSheet('QLineEdit {color:'+('red' if not input_correct else 'black')+'}')
                self.additionaldirs_widgets[key].setToolTip('')
                self.additionaldirs_rds_widgets[key].setToolTip('')


    def settings_tabcolortables(self):
        colortables_layout=QVBoxLayout()
        colortables_form=QFormLayout()
        self.colortablesw={j:QPushButton(os.path.basename(self.colortables_dirs_filenames[j]), autoDefault=True) for j in gv.products_all}
        
        self.cmaps_minvaluesw={}; self.cmaps_maxvaluesw={}
        for j in gv.products_all:
            self.cmaps_minvaluesw[j]=QLineEdit(); self.cmaps_maxvaluesw[j]=QLineEdit()
            if not self.cmaps_minvalues[j]=='':
                #The scaling is necessary for velocities, to convert values in self.cmaps_minvalues (which have units of m/s) to the chosen unit.
                self.cmaps_minvaluesw[j].setText(str(ft.rifdot0(ft.r1dec(self.cmaps_minvalues[j]*self.pb.scale_factors[j]))))
            if not self.cmaps_maxvalues[j]=='':
                self.cmaps_maxvaluesw[j].setText(str(ft.rifdot0(ft.r1dec(self.cmaps_maxvalues[j]*self.pb.scale_factors[j]))))
            
        for j in gv.products_all:
            self.cmaps_minvaluesw[j].setToolTip('Minimum product value to display in the color map. Leave empty for use of color table minimum.')
            self.cmaps_maxvaluesw[j].setToolTip('Maximum product value to display in the color map. Leave empty for use of color table maximum.')
            hbox=QHBoxLayout()
            hbox.addWidget(self.colortablesw[j],20); hbox.addWidget(self.cmaps_minvaluesw[j],2); hbox.addWidget(self.cmaps_maxvaluesw[j],2)
            colortables_form.addRow(QLabel(gv.productnames_cmapstab[j]),hbox)
            self.colortablesw[j].clicked.connect(lambda state, j=j: self.change_colortables(j))
            self.cmaps_minvaluesw[j].editingFinished.connect(lambda j=j: self.change_cmaps_minvalues(j))
            self.cmaps_maxvaluesw[j].editingFinished.connect(lambda j=j: self.change_cmaps_maxvalues(j))
        
        self.set_default_colortables=QPushButton('Default', autoDefault=True)
        self.set_NWS_colortables=QPushButton('NWS', autoDefault=True)
        colortables_type_hbox=QHBoxLayout()
        colortables_type_hbox.addWidget(self.set_default_colortables); colortables_type_hbox.addWidget(self.set_NWS_colortables)
        colortables_layout.addLayout(colortables_form); colortables_layout.addLayout(colortables_type_hbox)
        colortables_layout.addStretch(100)
        self.settingscolortables.setLayout(colortables_layout)
        
        self.set_default_colortables.clicked.connect(lambda: self.change_colortables('Default'))
        self.set_NWS_colortables.clicked.connect(lambda: self.change_colortables('NWS'))
                
    def change_colortables(self,product):
        if product in ('Default','NWS'):
            new_colortables_dirs_filenames=gv.colortables_dirs_filenames_Default if product=='Default' else gv.colortables_dirs_filenames_NWS
            for j in new_colortables_dirs_filenames.keys():
                self.colortables_dirs_filenames[j]=new_colortables_dirs_filenames[j]
                self.colortablesw[j].setText(os.path.basename(self.colortables_dirs_filenames[j]))
        else:    
            colortables_dirs_filenames_input=str(QFileDialog.getOpenFileName(None, 'Select the color table:',os.path.dirname(self.colortables_dirs_filenames[product]),filter='*.csv')[0])
            if colortables_dirs_filenames_input!='': 
                self.colortables_dirs_filenames[product]=colortables_dirs_filenames_input
                self.colortablesw[product].setText(os.path.basename(self.colortables_dirs_filenames[product]))
        
        if self.pb.firstplot_performed:
            self.pb.set_newdata(self.pb.panellist)
        else:
            self.pb.set_cbars()
            self.pb.update()
                
    def change_cmaps_minvalues(self,product):
        input_text=self.cmaps_minvaluesw[product].text()
        input_value=ft.to_number(input_text)
        if input_text=='' or not input_value is None:
            self.cmaps_minvalues[product]='' if input_text=='' else input_value/self.pb.scale_factors[product]
            if self.pb.firstplot_performed:
                self.pb.set_newdata([j for j in self.pb.panellist if self.crd.products[j]==product])
        else:
            self.cmaps_minvaluesw[product].setText(str(self.cmaps_minvalues[product]))
                    
    def change_cmaps_maxvalues(self,product):
        input_text=self.cmaps_maxvaluesw[product].text()
        input_value=ft.to_number(input_text)
        if input_text=='' or not input_value is None:
            self.cmaps_maxvalues[product]='' if input_text=='' else input_value/self.pb.scale_factors[product]
            if self.pb.firstplot_performed:
                self.pb.set_newdata([j for j in self.pb.panellist if self.crd.products[j]==product])
        else:
            self.cmaps_maxvaluesw[product].setText(str(self.cmaps_maxvalues[product]))
        
        
    def settings_tabalgorithms(self):
        layout = QFormLayout()
        
        self.cartesian_product_resw= QLineEdit(str(self.cartesian_product_res))
        hbox= QHBoxLayout(); hbox.addWidget(self.cartesian_product_resw); hbox.addStretch(30)
        layout.addRow(QLabel('Product resolution of Cartesian derived products (km)'), hbox)
        self.cartesian_product_maxrangew= QLineEdit(str(self.cartesian_product_maxrange))
        hbox= QHBoxLayout(); hbox.addWidget(self.cartesian_product_maxrangew); hbox.addStretch(30)
        layout.addRow(QLabel('Maximum range of Cartesian derived products (km)'), hbox)
        
        self.settingsalgorithms.setLayout(layout)
        self.cartesian_product_resw.editingFinished.connect(lambda: self.change_cartesian_product_attr('res'))
        self.cartesian_product_maxrangew.editingFinished.connect(lambda: self.change_cartesian_product_attr('maxrange'))
                    
    def change_cartesian_product_attr(self, attr):
        number = ft.to_number(getattr(self, f'cartesian_product_{attr}w').text())
        if not number is None and number > 0:
            setattr(self, f'cartesian_product_{attr}', float(number))
        getattr(self, f'cartesian_product_{attr}w').setText(str(getattr(self, f'cartesian_product_{attr}')))
        if any([self.crd.products[j] in gv.plain_products_correct_for_SM for j in self.pb.panellist]) and self.stormmotion[1] != 0.:
            self.pb.set_newdata(self.pb.panellist)


    def settings_tabmiscellaneous(self):
        miscellaneous_layout=QFormLayout()
        hbox_max_radardata_in_memory_GBs=QHBoxLayout(); hbox_sleeptime_after_plotting=QHBoxLayout(); hbox_use_scissor=QHBoxLayout()
        hboxes=[hbox_max_radardata_in_memory_GBs,hbox_sleeptime_after_plotting,hbox_use_scissor]
        for hbox in hboxes:
            hbox.addStretch(1)
        
        self.max_radardata_in_memory_GBsw=QLineEdit(); self.max_radardata_in_memory_GBsw.setText(str(ft.rifdot0(self.max_radardata_in_memory_GBs)))
        hbox_max_radardata_in_memory_GBs.addWidget(self.max_radardata_in_memory_GBsw)
        self.sleeptime_after_plottingw=QLineEdit(); self.sleeptime_after_plottingw.setText(str(ft.rifdot0(self.sleeptime_after_plotting)))
        hbox_sleeptime_after_plotting.addWidget(self.sleeptime_after_plottingw)
        self.use_scissorw=QCheckBox(); self.use_scissorw.setTristate(False)
        self.use_scissorw.setCheckState(2 if self.use_scissor else 0)
        hbox_use_scissor.addWidget(self.use_scissorw)
        
        for hbox in hboxes:
            hbox.addStretch(50)

        miscellaneous_widgets=[[QLabel('Maximum amount of radar data kept in memory (GB)'),hbox_max_radardata_in_memory_GBs],
                               [QLabel('Sleep time after plotting'),hbox_sleeptime_after_plotting],
                               [QLabel('Enable partial drawing of screen'),hbox_use_scissor]]
        for j in range(0,len(miscellaneous_widgets)):
            miscellaneous_layout.addRow(miscellaneous_widgets[j][0],miscellaneous_widgets[j][1])

        self.settingsmiscellaneous.setLayout(miscellaneous_layout)
        self.max_radardata_in_memory_GBsw.editingFinished.connect(self.change_max_radardata_in_memory_GBs)
        self.sleeptime_after_plottingw.editingFinished.connect(self.change_sleeptime_after_plotting)
        self.use_scissorw.stateChanged.connect(self.change_use_scissor)
        
    def change_max_radardata_in_memory_GBs(self):
        input_max_radardata_in_memory_GBs=self.max_radardata_in_memory_GBsw.text()
        number=ft.to_number(input_max_radardata_in_memory_GBs)
        if not number is None and number>=0:
            self.max_radardata_in_memory_GBs=float(number)
        else: self.max_radardata_in_memory_GBsw.setText(str(ft.rifdot0(self.max_radardata_in_memory_GBs)))
    def change_sleeptime_after_plotting(self):
        input_sleeptime_after_plotting=self.sleeptime_after_plottingw.text()
        number=ft.to_number(input_sleeptime_after_plotting)
        if not number is None and number>=0:
            self.sleeptime_after_plotting=number
        else: self.sleeptime_after_plottingw.setText(str(ft.rifdot0(self.sleeptime_after_plotting)))
    def change_use_scissor(self):
        self.use_scissor=True if self.use_scissorw.checkState()==2 else False
                
                
    def helpwidget(self):
        self.help=QTabWidget()
        self.help.setWindowTitle('NLradar help')
        self.helpgeneral=QWidget(); self.helpkeyboard=QWidget(); self.helpradardata=QWidget()
        self.helpproducts=QWidget(); self.helpcolortables=QWidget(); self.helpsettings=QWidget(); self.helpextra=QWidget()
        self.help.addTab(self.helpgeneral,'General')
        self.help.addTab(self.helpkeyboard,'Keyboard')
        self.help.addTab(self.helpradardata,'Radar data')
        self.help.addTab(self.helpproducts,'Products')
        self.help.addTab(self.helpcolortables,'Color tables')
        self.help.addTab(self.helpsettings,'Settings')
        self.help.addTab(self.helpextra,'Extra remarks')
                
        self.help_tabgeneral(); self.help_tabkeyboard(); self.help_tabradardata(); self.help_tabproducts(); self.help_tabcolortables(); self.help_tabsettings(); self.help_tabextra()
        self.help.resize(self.help.sizeHint())
        self.help.show()
        
    def help_tabgeneral(self):
        general_layout=QVBoxLayout()
        general_text=["It is recommended to read through this help before starting to use the program regularly.",
                      '',
                      "Current data can be obtained automatically by right-clicking at a radar, and selecting 'start automatic download for ...'. It can be stopped in a similar way.",
                      "Older data for the current day (and for the selected radar) can be obtained by using the download widgets in the menu. Both processes can run for multiple radars simultaneously.",
                      "IMPORTANT: For downloading KNMI data you will have to set API keys at <b>Settings/Download</b>.",
                      "For more information about obtaining current and storing archived data, see the tab <b>Radar data</b>.",
                      "",
                      "Switching to a different radar takes place by clicking on a radar marker. If you hold the CTRL key when clicking, the new radar is only selected, without plotting.",
                      "Most of the navigation takes place by means of keyboard shortcuts. It is therefore important to read the help section <b>Keyboard</b>.",
                      '',
                      'Zooming can not only be done with the scroll wheel of the mouse, but also by holding the right mouse button and moving the mouse. This way of zooming is more suitable when you need to make',
                      'small steps.',
                      '',
                      'The menu that pops up when right-clicking at the map gives you the option to place a marker at the mouse position. This can be done to mark a location, but the marker can also be used to determine a storm motion vector',
                      '(which is used for the storm-relative velocity). For the latter option place the marker at the starting point of the storm, and then go backward/forward in time (possibly changing other parameters',
                      'like the scan or radar), and then right-click again at the current position of the storm. You can then choose to calculate the storm motion vector. The storm motion vector can also be set manually via',
                      'the same right-click menu, or can be reset.',
                      'This menu is also used to provide radar-specific options. Currently there is one such option, which is the option to change the start azimuth of the scans, currently only available for Cabauw.',
                      'It also provides the option to add the current radar view to a list of cases, which can be assessed by clicking at the Cases widget in the menu bar.',
                      '',
                      "Color tables can be changed, as described at <b>Color tables</b>.",
                      '',
                      "Incomplete files cannot be opened by the program, and if you try so, an error is raised in the error log (the attached black window). In the case of unfinished downloads they are usually automatically",
                      "removed, but if this does not occur then you can manually remove them from the folder in which they are located. This folder is a subfolder with the name 'Download' of your base directory (selected at",
                      "<b>Settings/Radar data</b>). Unfinished downloads are automatically removed when closing the program.",
                      '',
                      "The color of a radar marker indicates whether the radar is selected, whether automatic download takes place and whether download of older data occurs. These colors can be selected at ",
                      "<b>Settings/Map</b>. A darker color indicates that download of older data occurs.",
                      "Multiple other properties of the map can be changed too in the <b>Settings</b>.",
                      '',
                      "Radar data gets stored in memory after it has been read from your hard disk, and the maximum amount of memory that the program can use for storing data can be selected at",
                      "<b>Settings/Miscellaneous</b> (default is 2 GB). Reading data from memory is clearly faster than reading them from the hard disk.",
                      "The program also saves a number of 'volume attributes' to a file after determining them once, to increase the speed. These volume attributes include e.g. the scanangles for all scans in the volume, and",
                      "they are used internally by the program. It could occur that wrong attributes get saved, resulting in incorrect plots. If this is the case, then you can remove these attributes under the menu item <b>Extra</b>.",
                      '',
                      "Basemap source: Bing."]
        for j in range(0,len(general_text)):
            qlabel=QLabel(general_text[j],font=self.help_font); qlabel.setOpenExternalLinks(True)
            general_layout.addWidget(qlabel)
        general_layout.addStretch(1)
        self.helpgeneral.setLayout(general_layout)
               
    def help_tabkeyboard(self):
        global plottimes_max
        keyboard_layout=QFormLayout()
        keyboard_text=[['ENTER','Plot for current input'],
                  ['SPACE',"Start/stop animation, or stop continuing backward/forward in time. Animation ends at input date and time. If both 'c', then end time gets updated when new data available."],
                  ['ALT+C/ALT+SHIFT+C',"Set the date and time equal to 'c', with/without plotting the data."],
                  ['ALT+LEFT/RIGHT','Go continuously backward/forward in time.'],
                  ['LEFT/RIGHT, SHIFT+LEFT/RIGHT','Go to previous/next radar volume, go one hour backward/forward in time.'],
                  ['(SHIFT+)BACKSPACE','Go back to the previous combination of radar and dataset (and also date and time).'],
                  ['CTRL+SPACE, (CTRL+)SHIFT+SPACE','Loop through case list, animate case by looping through the animation window specified in Cases/Settings (while looping through case list too)'],
                  ['CTRL+(ENTER/LEFT/RIGHT, BACKSPACE)','Switch to current/previous/next case in currently selected case list, switch back to previously shown case (can be from other list)'],
                  ['(SHIFT+)HOME, CTRL+HOME','Reset view to radar-centered and (not) reset zoom, or reset zoom without resetting view'],
                  ['F','Enable/disable storm-following view, which moves view with currently set storm motion'],
                  ['ALT+1,2,3,4,6,8,0','Show 1,2,3,4,6,8 or 10 panels.'],
                  ['CTRL+1-10','Switch to the radar that is nth nearest to the panel centers.'],
                  ['N/ALT+N','Automatically select nearest radar after a change of view/Select which radar wavelength bands to include for automatic selection'],
                  ['SHIFT+D','Switch between the Z and V datasets, if available for the radar.'],
                  ['CTRL+D','Switch between directory structures if you specified multiple at <b>Settings/Data storage</b>.*'],
                  ['CTRL+P','Switch between different versions of products when available.*'],
                  ['SHIFT+S/E','Influences which scan is chosen when switching to new radar. Keep using same scan/choose scan with nearest elevation angle.'],
                  ['SHIFT+H','Similar to above, but choose scan with nearest height at panel center. This mode is however also applied when moving view with storm.*'],
                  ['SHIFT+N/A/R/C','Go to normal/all/row/column mode*'],
                  ['Z/R/V/S/W/D/P/K/C/X/E/A/M/L/Q/T/Y','Change the product of the selected panel, see the <b>Products</b> tab for more information.'],
                  ['SHIFT+Q','Change a parameter for some derived products (E/A/M), but only if that product is shown in the selected panel.*'],
                  ['SHIFT+U','View the unfiltered/filtered variant of the product. The unfiltered variant might not be available.'],
                  ['SHIFT+P','Change the polarization of the product. Only for dual-polarization radars.'],
                  ['SHIFT+V/ALT+V','Apply dealiasing to the velocity field/Select which dealiasing procedures are applied'],
                  ['SHIFT+I','Apply bilinear interpolation, only available for particular products'],
                  ['SHIFT+Z','Hide/show radar data.'],
                  ['1-9, 0 and SHIFT+1-5','Change scan for the selected panel to 1-15.*'],
                  ['DOWN/UP, SHIFT+DOWN/UP','Go one scan down/up for all panels, go one scan down/up for the selected panel.'],
                  ['SHIFT+F1-F12','Save current panel configuration as panel choice 1-12.'],
                  ['F1-F12','Show panel choice 1-12.'],
                  ['ALT+F1','Show all saved panel configurations, with possibility to edit them.'],
                  ['ALT+A','Show radars for which archived data is available for the selected date, and dates for which archived data is available for the selected radar.'],
                  ['ALT+P','Show the scan angle, range, radial resolution and Nyquist velocity for all scans in the current volume.'],
                  ['SHIFT+F','View at full screen.'],
                  ['CTRL+(S/ALT+S/SHIFT+S)','Save figure/continuously save figures (until pressing CTRL+ALT+S again)/add images to animation, which can be saved after pressing CTRL+SHIFT+S again.'],
                  ['*',"See the tab <b>Extra remarks</b> for more info"]]
        for j in range(0,len(keyboard_text)):
            keyboard_layout.addRow(QLabel(keyboard_text[j][0],font=self.help_font),QLabel(keyboard_text[j][1],font=self.help_font))  
        self.helpkeyboard.setLayout(keyboard_layout)
        
    def help_tabradardata(self):
        radardata_layout=QVBoxLayout()
        radardata_text=['<b>Current data</b>:',
                  "Most users will need to download the radar data, which can be done with the methods described at <b>General</b>. The data is stored in the directories that are selected at <b>Settings/Data storage</b>.",
                  "",
                  "<b>Archived data</b>:",
                  "The following description assumes that you need to download the archived data, and do not have it already at your hard disk. If the latter is the case, then it's enough to specify the location of the data",
                  "at <b>Settings/Data storage</b>. Also mentioned here are the file formats that are supported.",
                  'KNMI:',
                  "Data must be manually downloaded from the KNMI website, which can be done via these links; <A href='https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=radar_tar_volume_debilt&x-dataset-version=1.0'>De Bilt</a>, <A href='https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=radar_tar_volume_denhelder&x-dataset-version=1.0'>Den Helder</a>, <A href='https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=radar_tar_vol_full_herwijnen&x-dataset-version=1.0'>Herwijnen</a>.",
                  "The downloaded data must subsequently be put in the right directory, that has the structure given at <b>Settings/Data storage</b>. As an example:",
                  "If your directory structure is NLradar/Radar_data/KNMI/RAD${radarID}_OPER_O___TARVOL__L2__${date}T000000_${date+}T000000_0001, then you put the",
                  "downloaded .tar files in the directory NLradar/Radar_data/KNMI, and subsequently unpack the .tar file in the folder that is shown when right-clicking",
                  "at the file (this folder has the format RAD${radarID}_OPER_O___TARVOL__L2__${date}T000000_${date+}T000000_0001).",
                  "When done correctly you should then be able to plot the data. The files need to be of HDF5 format, and must have a name like RAD_NL60_VOL_NA_200806220010.h5.",
                  "KMI:",
                  "There is no open source for KMI data yet, but if you do have data, then it can be read if it has one of the formats described here.",
                  "The program can read both the hdf and binary .vol files that the KMI provides, at least if they have names like: 2003061015000300z.vol/2012060714002400Z.vol/2014081000000400dBZ.vol (same for",
                  "both datasets), 2014081000000400dBZ.vol.h5 (same for both datasets), 20170615100000.rad.bewid.pvol.dbzh.scanv.hdf/20170615100000.rad.bewid.pvol.dbzh.scanz.hdf (different for both datasets).",
                  "skeyes:",
                  "There is also no open source for skeyes data yet, but the program can read the data for Zaventem if it is of hdf/hdf5 format, at least if the files have names like",
                  "20170511134500.rad.bezav.pvol.dbzh.scan_abc.hdf (in fact delivered by the KMI), and EBBR140808152208.RAW0555.h5. hdf and hdf5 format files should not be put in the same folders,",
                  "because they need to be treated in different ways!",
                  "DWD:",
                  "There is also no open source for archived DWD data yet, but if you do have data, then it can be read if it is of BUFR format, with filenames such as",
                  "sweep_vol_v_0-20190809150057_10605--buf.",
                  "TU Delft:",
                  "Data can be manually downloaded from the <A href='https://opendap.tudelft.nl/thredds/catalog/IDRA/catalog.html'>TU Datacenter</a>. You need to pick the 'near_range', 'standard_range' or 'far_range' NetCDF files,",
                  "and download them into the right directory, with structure given at <b>Settings/Data storage</b>. The program assumes that at most one such NetCDF file is available per date."]
        
        for j in range(0,len(radardata_text)):
            qlabel=QLabel(radardata_text[j],font=self.help_font); qlabel.setOpenExternalLinks(True)
            radardata_layout.addWidget(qlabel)
        radardata_layout.addStretch(1)
        self.helpradardata.setLayout(radardata_layout)
        
    def help_tabproducts(self):
        products_layout=QFormLayout()
        products_text=[['Z/V/S/W/D/C/X/P/K','Reflectivity, radial velocity, storm-relative radial velocity, spectrum width, differential reflectivity, correlation coefficient, linear depolarisation ratio,',
                        "differential phase and specific differential phase."],
                       ['Q/T/Y', 'Signal quality index, clutter correction, clutter phase alignment. These products are currently only available for the new KNMI radars (Herwijnen and Den Helder). '],
                       ['R',"Rain intensity, calculated using the <A href='http://glossary.ametsoc.org/wiki/Marshall-palmer_relation'>Marshall-Palmer relation</a>."],
                       ['E','Echo tops, where the height of the highest reflectivity bin that satisfies Z>=dBZ_threshold (press SHIFT+Q) is shown.',
                        'Heights are calculated assuming the 4/3 Earth radius model (Doviak and Zrnic, 1993).'],
                       ['A','Pseudo-CAPPI (Constant Altitude Plan Position Indicator) at the height given by PCAPPI height (SHIFT+Q). The PCAPPI reflectivities',
                        'are calculated by linear interpolation between the (logarithmic) reflectivities at the scan below and above the PCAPPI height. If the PCAPPI height lies below',
                        'the lowest scan or above the highest scan, then the reflectivity for the lowest scan resp. highest scan is shown.'],
                       ['M',"Maximum reflectivity for the complete volume scan, where only reflectivity bins at a height above the Zmax minimum height (SHIFT+Q) are used",
                       'for determining the maximum reflectivity. '],
                       ['L',"Vertically integrated liquid, calculated by vertical integration of <A href='http://glossary.ametsoc.org/wiki/Vertically_integrated_liquid'>this</a> relation. The reflectivity below the lowest scan",
                        'is assumed to be constant, which might not be a good assumption. One can choose (SHIFT+Q) whether to cap reflectivity at 56 dBZ in the integration, which is commonly done to reduce',
                        'contribution from hail.']]
        for j in range(0,len(products_text)):
            qlabel2=QLabel(products_text[j][1],font=self.help_font); qlabel2.setOpenExternalLinks(True)
            products_layout.addRow(QLabel(products_text[j][0],font=self.help_font),qlabel2)
            if len(products_text[j])>2:
                for i in range(2,len(products_text[j])):
                    qlabel2=QLabel(products_text[j][i],font=self.help_font); qlabel2.setOpenExternalLinks(True)
                    products_layout.addRow(QLabel('',font=self.help_font),qlabel2)
        self.helpproducts.setLayout(products_layout)        
        
    def help_tabcolortables(self):
        colortables_layout=QVBoxLayout()
        colortables_text=["The color tables can be changed manually by selecting another file under <b>Settings/Color tables</b>, or by changing the currently selected file. The files can be found in the folder",
                      "<i>NLradar/Input_files</i>.",
                      "It is recommended to edit/create them in Notepad or a comparable application, because other applications might add undesired characters. Be sure to save them as CSV, and preferentially",
                      "under a name different from the defaultLast 5 color table's name.",
                      "The colors must be given as RGB combinations with values ranging from 0 to 255. You can view the colors corresponding to RGB combinations at <A href='https://www.colorschemer.com/rgb-color-codes'>Colorschemer</a>.",
                      'When you want to have discontinuous steps in them, you must specify 2 RGB combinations per product value. This is for example the case in the default color table for the reflectivity.',
                      "By adding the line 'Step: x' in the file you can choose to use a fixed step between subsequent ticks.",
                      "If you want to exclude some product values or values arising from the choice of a fixed step size for a tick label, then you need to add a line 'Exclude for ticks: x1,x2...' (with xi product values) to the",
                      "file.",
                      "When you want to include ticks at a position different from those regarded before, then you need to add a line 'Include for ticks: x1,x2...'.",
                      'In the case of the velocity and spectrum width you can choose between the units m/s, kts, mph and km/h. The default units are kts.',
                      "Finally, you can choose to shorten the first and/or last color segment of the colorbar. This could be useful when the colorbar should support quite extreme values,",
                      "but you don't want to have for example half of the colorbar showing this extreme range. To prevent this you can add a line below or above the first or last color definition.",
                      "That line should contain a single value, and the length of the corresponding color segment will then be based on this value instead of on the actual value (which will still",
                      "be used for the ticks). This option is used in the default color table for VIL."]
        for j in range(0,len(colortables_text)):
            qlabel=QLabel(colortables_text[j],font=self.help_font); qlabel.setOpenExternalLinks(True)
            colortables_layout.addWidget(qlabel)
        colortables_layout.addStretch(1)
        self.helpcolortables.setLayout(colortables_layout)

    def help_tabsettings(self):
        settings_layout=QFormLayout()
        settings_text=[['Map:',''],
                       ['Map tiles update time', 'Map tiles are updated when the last occurrence of panning/zooming was this number of seconds ago. You might need to increase this time in case',
                        'of a slow PC, for fast panning/zooming.'],
                       ['Apply antialiasing to lines','Antialiasing causes lines to look smoother, but it comes at the cost of increased GPU usage, and therefore could decrease the speed.'],
                       ['Height rings',"In the case of products with different scans, the numbers near this rings give the approximate height (above radar antenna level) at which the center of the radar beam is scanning",
                       "(calculated assuming the 4/3 earth radius model). This is also the case for two 'plain products' (products without different scans), which are the PCAPPI and the rain intensity.",
                       "In the case of other plain products, the rings are located at distances where particular scans are at there maximum range. The number on the inner side of the ring",
                       "now gives the height of the beam center for the scan that reaches its maximum range, and the number on the outer side of the ring gives the height of the beam center",
                       "for the highest scan that scans at a greater range."],               
                       ['Show grid/height rings pan/zoom','If false, then the grid and height rings are not visible during panning/zooming, which increases the plotting speed.'],
                       ['If not, update them after x seconds','Time after which they will again be visible, measured from the last pan/zoom action.'],
                       ['Download:',''],
                       ['Network timeout','The program stops trying to reach the KNMI server after this time. It could be necessary to increase this value when you have a slow internet connection.'],
                       ['Minimum download speed','The program stops a download when the speed is lower than this minimum speed. It could be necessary to decrease this value when you have a slow internet',
                        'connection.'],
                       ['Color tables:','A black color table is shown when the selected color table has an incorrect format.',
                        'By choosing a minimum and/or maximum product value to display, you can change the range of product values that is shown without the need to update the color table.',
                        "By clicking the buttons 'Default' and 'NWS' you can choose to use a set of default color tables (my own choices) or a set of tables used (at least for most",
                        'products) in some NWS offices.'],
                       ['Miscellaneous:','- The sleep time after plotting determines the amount of time during which no new plot command can be given after a particular plot is finished. Setting this to zero',
                        "can lead to plots not appearing immediately after they have been created, but only after the last plot in a plotting series has been created. This does not occur at all",
                        "PC's however, and you should experiment a little to find a value that works at your PC. It should be the smallest time that does not lead to the problem mentioned.",
                        'Furthermore, after the calculation of a derived product this sleep time is automatically increased with a factor of 2.'],
                       ['','- Partly drawing of the screen should be enabled if it does not lead to problems, because it increases the plotting speed. With some drivers it could lead to',
                        'the problem of a flickering background (white and black) however, and in this case it should be disabled.']]
        for j in range(0,len(settings_text)):
            settings_layout.addRow(QLabel(settings_text[j][0],font=self.help_font),QLabel(settings_text[j][1],font=self.help_font))  
            if len(settings_text[j])>2:
                for i in range(2,len(settings_text[j])):
                    settings_layout.addRow(QLabel('',font=self.help_font),QLabel(settings_text[j][i],font=self.help_font))
        self.helpsettings.setLayout(settings_layout)
                                
    def help_tabextra(self):
        extra_layout=QFormLayout()
        extra_text=[['CTRL+D','A switch of directory structure will only take place when there is data available for the currently selected date.',
                     'Regarding product versions: They are available for DWD hdf5 (reflectivity+velocity) and NWS NEXRAD L2 (reflectivity) files.'],
                    ['SHIFT+H','The program determines the beam elevation for the current scan at the center of the panel. After changing from radar, or from time',
                     'when moving view with storm, it finds the scan for which the new beam elevation at the center of the panel is closest to the value it was before.'],
                    ['SHIFT+N/A/R/C','- Normal mode; any change of product, scan, polarization, filtering or dealiasing only affects the currently selected panel.',
                     '- All-panel mode; when choosing another scan, the scan for each panel is increased by the same number as the scan for the selected panel.',
                     '- Row mode; when changing a product, the panels in the same row share it too. When changing a scan, the panel in the same column shares it too.',
                     '- Column mode; when changing a product, the panel in the same column shares it too. When changing a scan, the panels in the same row share it too.',
                     '- When the number of panels is equal to 2 or 3, then column mode is used when row mode is selected, because it is likely not desired to have all panels showing the same product,',
                     'as would be the case when using row mode.',
                     '- When using row mode when changing the number of panels, then all panels in the same row get assigned the product in the leftmost panel, and all panels in the same column',
                     'get assigned the scan of the upper panel in the column. When using column mode the same happens, except that the roles of products and scans are reversed.',
                     '- When changing the polarization, filtering or dealiasing settings, then these modes have the same effect as described above for products and scans, except that using the',
                     'all-panel mode implies that the polarization, filtering or dealiasing setting for all panels is set equal to that for the selected panel.'],
                    ['1-9, 0 and SHIFT+1-5','In the case of the new radars of KNMI, the last scan is a vertical scan, which is plotted horizontal however. This means that the distance to the radar is in fact the height',
                     'above radar level.'],
                    ['SHIFT+Q','Derived products are saved to a file after their calculation. Per product for which a setting can be changed, data will be written to the file for up to 4 different values of',
                     'the setting. If you use a fifth one, then one of the previous 4 will be removed, which is the one that has been displayed the least amount of times.'],
                    ['SHIFT+V','It is here attempted to remove the typical dual PRF aliasing errors. Most C-band radars scan with two PRFs, which usually differ for even and odd radials,',
                     'in order to extend the maximum velocity that can be detected. This comes however at the cost of introducing these aliasing errors, where the measured velocity deviates by',
                     'an integer multiple of a certain velocity from the true velocity. This certain velocity is (usually) twice the low/high Nyquist velocity for that particular radial, and',
                     'this low/high Nyquist velocity can be requested by pressing ALT+P. Even and odd radials are alternately scanned with the low/high Nyquist velocity, but whether even or odd',
                     'radials are scanned with the low Nyquist velocity differs per radar (or even per scan). If you want to know which Nyquist velocity is used for a particular radial, then you should',
                     'look at the correction terms that have been applied by the dealiasing algorithm.',
                     'In the case of some radars, the radar switches from PRF halfway a radial. If this is the case, then the possible correction terms are the same for all radials, and usually based on',
                     'the low Nyquist velocity. If this is the case, then the low and high Nyquist velocities that are shown by pressing ALT+P are equal.',
                     'To increase the complexity further: The possible correction terms can for some radars take on all integer multiples of once the low Nyquist velocity, instead of twice that velocity.',
                     'This increases the number of possible correction terms substantially, and makes it more difficult to correctly dealias the velocities. The result is then often an oversmoothed velocity field.'],
                    ['CTRL+S/CTRL+ALT+S',"You can choose to use a fixed name format, which has the form radar_dataset_datetime_productsandscans. You can also choose for using your own filename, to which increasing numbers",
                     "will be appended when you pressed CTRL+ALT+S. If you want to continue from numbering in already existing files, then you can append a '#' to the filename (before a possible extension).",
                     "This also works when pressing CTRL+S.",
                     "If you want to use the fixed name format, then you should only give the file extension as input (gif, png or jpg, without dot). If you want to use your own file format, then you should",
                     "give the filename as input, including the file extension (with dot). If you don't include the file extension, then the image is saved as jpg."]]
        
        for j in range(0,len(extra_text)):
            qlabel2=QLabel(extra_text[j][1],font=self.help_font); qlabel2.setOpenExternalLinks(True)
            extra_layout.addRow(QLabel(extra_text[j][0],font=self.help_font),qlabel2)
            if len(extra_text[j])>2:
                for i in range(2,len(extra_text[j])):
                    qlabel2=QLabel(extra_text[j][i],font=self.help_font); qlabel2.setOpenExternalLinks(True)
                    extra_layout.addRow(QLabel('',font=self.help_font),qlabel2)
        self.helpextra.setLayout(extra_layout)
                                
                                

    def closeEvent(self, event=None):
        QCoreApplication.instance().quit()
        
        self.derivedproducts_filename_version = self.dp.filename_version
        
        settings={}
        for j in range(0,len(variables_names_raw)): 
            settings[variables_names_raw[j]]=eval(variables_names_withclassreference[j])                
        with open(settings_filename,'wb') as f:
            pickle.dump(settings,f)
            
            
        #Dump info over the volume attributes into pickle files, which will be loaded during the next startup of the program in nlr_datasourcegeneral.py
        with open(self.dsg.attributes_descriptions_filename,'wb') as f:
            pickle.dump(self.dsg.attributes_descriptions,f)
        with open(self.dsg.attributes_IDs_filename,'wb') as f:
            pickle.dump(self.dsg.attributes_IDs,f)
        with open(self.dsg.attributes_variable_filename,'wb') as f:
            pickle.dump(self.dsg.attributes_variable,f)
             




def main():
    app = QApplication(sys.argv)
    gui = GUI()
    
    print(app.exec_())

    # sys.exit(app.exec_())
    # gui.closeEvent()
    print('finish')
        
if __name__ == '__main__':
    main()