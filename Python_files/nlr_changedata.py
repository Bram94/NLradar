# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

from nlr_datasourcegeneral import DataSource_General
from nlr_animate import Animate
import nlr_functions as ft
import nlr_background as bg
import nlr_globalvars as gv

from PyQt5.QtCore import QObject, pyqtSignal, QTimer    

import time as pytime
import numpy as np
import os
import copy
import traceback


"""
The first plotting function that is called must always be process_datetimeinput, because some initialization takes place there.

When the date changes, a list with all datetimes for which data for that date, and the previous and next date is available is stored in 
self.filedatetimes. When a particular date and time is selected, the program searches for the file for which the datetime is closed to that particular
date and time. When going x steps backward/forward, the program chooses a new datetime that lies x*volume_timestep_radars[radar] minutes away from the
previous datetime, where volume_timestep_radars[radar] is the typical timestep between subsequent volumes for that particular radar. The program then
determines the file for which the datetime is closest to this new datetime. To assure that the performed step is large enough, the program chooses
a datetime that lies one further backward/forward when the time step resulting from choosing the closest datetime is too small (except when not possible).
In the case of current data, almost everything is handled in CurrentData, although that class makes use of some functions here.

self.radar is always used when plotting new data.
When selecting a new radar, this radar gets stored in self.selected_radar. When data is present for the selected date and time for this new radar,
then the plot will be updated, and self.radar becomes equal to self.selected_radar. The old plot remains visible if this is not the case and
self.radar does not change, although the new radar remains stored in self.selected_radar.
When subsequently a different date and time is chosen for which data is present for the new radar, then the program switches to this new radar. When
however selecting another product or time by using the keyboard, the program falls back to the old radar, which then gets also stored in
self.selected_radar.
The same principle holds for self.selected_dataset, self.selected_date and self.selected_time. 

For the dataset there is however also self.save_selected_dataset, which is only used in this script. It is used because it is possible that a
particular dataset is available for one combination of radar and date, while it isn't available for the same radar with another date. 
If self.selected_dataset is not available while the other dataset is, then self.selected_dataset is set equal to the other dataset. 
self.save_selected_dataset saves the 'true' selected dataset however, and self.selected_dataset will be set equal to self.save_selected_dataset when
calling self.process_datetimeinput (which occurs when switching from radar or date (in the text bar)). 
"""

class Change_RadarData(QObject):
    """A signal must be used for calling set_datetimewidgets from outside the main thread, because otherwise the program crashes.
    """
    signal_set_datetimewidgets = pyqtSignal(object,object)
    def __init__(self, gui_class, radar, scan_selection_mode, date, time, products, dataset, productunfiltered, polarization, apply_dealiasing, scans, plot_mode, parent = None): 
        super(Change_RadarData, self).__init__(parent) 
        self.signal_set_datetimewidgets.connect(self.set_datetimewidgets)
        
        self.gui=gui_class
        self.dod=None #Gets defined in nlr.py, and becomes self.dod
        self.pb=self.gui.pb
        self.dsg=DataSource_General(gui_class=self.gui, crd_class=self)
        self.ani=Animate(crd_class=self)
        
        self.directory=None
        self.selected_radar=radar; self.selected_date=date; self.selected_time=time
        self.radar=radar; self.date=date; self.time=time
        if date == 'c':
            # self.date and self.time should not be 'c'
            datetime = ft.get_datetimes_from_absolutetimes(pytime.time())
            self.date, self.time = datetime[:8], datetime[-4:]
        self.products=products.copy(); self.scans = scans.copy()
        
        self.plot_mode=plot_mode
        self.scan_selection_mode=scan_selection_mode
        self.dataset=dataset; self.selected_dataset=dataset; self.save_selected_dataset=self.selected_dataset
        #Is currently only used for KMI data, where the volumes are divided in 2 datasets; one with a large range and small Nyquist
        #velocity, and one with a smaller range but large Nyquist velocity. 'Z' refers to the former dataset, 'V' to the latter.
        self.productunfiltered=productunfiltered      
        self.polarization=polarization
        self.apply_dealiasing = apply_dealiasing
        
        self.using_unfilteredproduct={j:False for j in range(10)}
        self.using_verticalpolarization={j:False for j in range(10)}
        
        """These 'before' variables get updated in the function set_newdata in nlr_plottingbasic.py
        """
        self.current_variables={'radar':self.radar,'dataset':self.dataset,'radardir_index':None,'product_version':None,'date':self.date,'time':self.time,'datetime':self.date+self.time,'scannumbers_forduplicates':{}}
        self.before_variables=copy.deepcopy(self.current_variables)
        self.rd_before_variables=copy.deepcopy(self.current_variables) #Is only updated when the radar, radardir_index or dataset changes.
        
        self.filedatetimes=[]
        # self.filedatetimes_errors (updated in self.pb.set_newdata) will contain per directory a dictionary with datetimes with errors 
        # as keys and corresponding total file sizes as values.
        self.filedatetimes_errors = {}
        self.volume_timestep_m = gv.volume_timestep_radars[self.radar]

        self.download_selection='Download'
        self.performing_download_current=False
        self.start_time=0
        self.end_time=0
        #A different end time is used for change_dataset etc., to ensure that a change in dataset etc. is always executed when it has not been done
        #a short while ago.
        self.process_keyboardinput_timebetweenfunctioncalls=0.  
        self.previous_timebetweenfunctioncalls=0.
        self.process_keyboardinput_finished_before=True
        
        self.changing_subdataset=False #Is used in nlr_plottingbasic.py to determine whether self.before_variables['radardir_index']
        #or self.before_variables['product_version'] should be updated.
        self.going_back_to_previous_plot=False
        #self.going_back_to_previous_plot is used in self.dsg.get_scans_information, in order to check whether it is 
        #desired to update self.dsg.scannumbers_forduplicates.
        self.change_radar_running = False
        self.process_datetimeinput_running=False
        self.process_keyboardinput_running=False
        self.requesting_latest_data = False
        self.lrstep_beingperformed=None #Is used within nlr_importdata.py, to determine the values for self.dsg.scannumbers_forduplicates
        
        self.change_radar_call_ID=0
        self.process_datetimeinput_call_ID=0
        self.process_keyboardinput_call_ID=None # See nlr_animate.py for reason for setting this to None
        self.update_current_call_ID=0
        
        self.selected_scanangles={} #selected_scanangles is used when the radar volume structure changes during the course of a day, as can be the case
        #for the radar in Zaventem. When selecting a particular scanangle, that is not available with the next volume structure, then it is chosen 
        #automatically again when the structure changes back to something that contains the selected scanangle.
        #Should not be updated when going to the left/right.
        #Gets initialized in the function self.dsg.get_scans_information, because its initialization requires information that is not present before calling that
        #function for the first time.
        
        self.time_last_change_scan_selection_mode=0 #is used in nlr_datasourcegeneral.py
        self.time_last_leftright=0
        self.time_last_productchange=0
        self.time_last_individualscanchange=0 #When changing the scan through a number key, or by pressing SHIFT+UP/DOWN.
        self.time_last_downup=0
        self.time_last_up_first_scan_scanpair=0
        
        self.timer_process_keyboardinput=QTimer()
        self.timer_process_keyboardinput.setSingleShot(True)
        self.timer_process_keyboardinput.timeout.connect(self.process_keyboardinput)
        
        

    def change_scan_selection_mode(self, new_mode):
        self.scan_selection_mode = new_mode
        self.time_last_change_scan_selection_mode=pytime.time()

    def switch_to_nearby_radar(self, n, check_date_availability=True, delta_time=0, source=None):
        # t = pytime.time()
        # from cProfile import Profile
        # profiler = Profile()
        # profiler.enable() 

        # switch to the nth nearest radar
        date, time = self.gui.datew.text().replace(' ',''), self.gui.timew.text().replace(' ','')
        radar_keys = [i for i,j in gv.radar_bands.items() if j in self.gui.radar_bands_view_nearest_radar]
        if (check_date_availability and (not ft.correct_datetimeinput(date,time) or date=='c' or time=='c')) or not radar_keys:
            return False

        panel_center_xy = self.pb.screencoord_to_xy(self.pb.panel_centers[0])
        if delta_time:
            panel_center_xy += self.pb.translation_dist_km(delta_time)
        
        panel_center_coords = ft.aeqd(gv.radarcoords[self.radar], panel_center_xy, inverse = True)
        distances = {i:ft.calculate_great_circle_distance_from_latlon(panel_center_coords, gv.radarcoords[i]) for i in radar_keys}
        if self.lrstep_beingperformed and self.radar in radar_keys:
            for i in radar_keys:
                # Without this measure it sometimes happens that the program flips back and forth between 2 radars that are nearly as close.
                # This is not desired, therefore in case of very small distance differences the current radar will remain selected. This 
                # is done by slightly increasing the distance of the slightly closer radar.
                if i != self.radar and 0 <= distances[self.radar]-distances[i] < 1:
                    distances[i] += 1                    
        i_sorted_distances = np.argsort(list(distances.values()))
        
        if check_date_availability:
            i = j = 0
            selected_radar = self.radar
            while j < n and i < len(i_sorted_distances):
                desired_radar = radar_keys[i_sorted_distances[i]]
                directory = self.dsg.get_directory(date,time,desired_radar,self.selected_dataset)
                if os.path.exists(directory):
                    dir_string = self.dsg.get_dir_string(desired_radar,self.selected_dataset)
                    datetimes = self.dsg.get_datetimes_directory(desired_radar, directory)
                    
                    if '${date' in dir_string:
                        dir_dtbounds = bg.get_datetime_bounds_dir_string(dir_string, date, time)
                        
                        threshold = 60*30
                        if self.lrstep_beingperformed:
                            direction = np.sign(self.lrstep_beingperformed)
                            ref_datetime = dir_dtbounds[0 if direction == -1 else 1]
                            include_next_dir = direction*ft.datetimediff_s(date+time, ref_datetime) < threshold
                        else:
                            diffs = np.abs([ft.datetimediff_s(date+time, k) for k in dir_dtbounds])
                            include_next_dir = diffs.min() < threshold
                            direction = -1 if diffs[0] < diffs[1] else 1
                            
                        if include_next_dir:
                            next_dir = bg.get_next_possible_dir_for_dir_string(dir_string, desired_radar, date, time, direction)
                            if os.path.exists(next_dir):
                                datetimes = np.append(datetimes, self.dsg.get_datetimes_directory(desired_radar, next_dir))

                    if not self.lrstep_beingperformed:
                        date_present = any(abs(ft.datetimediff_s(date+time, k)) < 60*20 for k in datetimes)
                    else:
                        vt = self.determine_volume_timestep_m(datetimes, desired_radar)                                
                        date_present = any(0 < direction*ft.datetimediff_s(date+time, k) < 60*max(2*vt, 15) for k in datetimes)\
                                       if desired_radar == self.radar else\
                                       (sum(0 < direction*ft.datetimediff_s(date+time, k) < 60*30 for k in datetimes) > 1)
                    if date_present:
                        selected_radar = desired_radar
                        j += 1
                i += 1
        else:
            selected_radar = radar_keys[i_sorted_distances[n-1]]
                                
        radar_changed = self.radar != selected_radar
        # If called from self.process_datetimeinput, then continuation will be handled there
        if radar_changed and not source == self.process_datetimeinput:
            if self.lrstep_beingperformed:
                # Set date+time back by 1 volume timestep. This is done to prevent that switching radar comes with an unnecessarily large
                # timestep (which could happen when the closest datetime for the new radar is ahead of the former radar's datetime).
                # Steps in self.pb.set_newdata ensure that the program doesn't actually go backward in time.
                date, time = ft.next_date_and_time(date, time, -vt*np.sign(self.lrstep_beingperformed))
                self.set_datetimewidgets(date, time)
            self.change_radar(selected_radar)
            
            if not check_date_availability:
                # Restore original date and time in widgets, to enable for example easy downloading of older data
                self.signal_set_datetimewidgets.emit(date, time)
        else:
            self.selected_radar = selected_radar
            
        # profiler.disable()
        # import pstats
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(15)  
        # print(pytime.time()-t, 't_swith_nearest_radar')
        return radar_changed
        
    def change_radar(self, new_radar, call_ID=None, set_data=True, limit_update_download=True):
        self.selected_radar=new_radar
        self.change_radar_running = True
        
        if set_data:
            self.update_download_widgets(limit_update_download)
            
        if set_data and self.check_viewing_most_recent_data():
            self.plot_mostrecent_data()
        else:
            self.process_datetimeinput(set_data=set_data)
            
        self.end_time=pytime.time()
        self.change_radar_running = False
        if not call_ID is None:
            self.change_radar_call_ID=call_ID
            
    def check_viewing_most_recent_data(self):
        timediff = pytime.time() - ft.get_absolutetimes_from_datetimes(self.date+self.time)
        if self.pb.firstplot_performed and timediff < 900:
            # First entry is newest datetime, second is second-newest
            # The second-newest datetime is also used, since it can happen that for the latest datetime only
            # a very incomplete volume is available.
            newest_datetimes = self.dsg.get_newest_datetimes_currentdata(self.radar,self.selected_dataset)
            current_datetime = self.selected_date+self.selected_time
            duplicate = self.dsg.scannumbers_forduplicates[self.scans[0]]
            max_duplicate = len(self.dsg.scannumbers_all['z'][self.scans[0]])-1
            # Also consider which duplicates are currently shown. For the newest datetime this should be either the
            # last or second-to-last duplicate (the last duplicate scan might not be fully available yet), while for 
            # the second-newest datetime the last duplicate should be shown.
            if current_datetime == newest_datetimes[0] and duplicate in (max_duplicate, max_duplicate-1) or\
            current_datetime == newest_datetimes[1] and duplicate == max_duplicate:
                return True
        return False
            
    def update_download_widgets(self, limit_update_download=True):
        # When switching to another radar, the current item of self.downloadw must be set to the value corresponding to the new radar.
        # The exception is when limit_update_download=True, in which case the current download timerange is retained when this timerange
        # is 0 for the selected radar
        if limit_update_download and self.dod[self.radar].download_timerange_s > 0 and self.dod[self.selected_radar].download_timerange_s == 0:
            self.gui.set_download_widgets(self.gui.download_timerangew.text(), 'Start') 
        elif self.dod[self.selected_radar].download_timerange_s == 0:
            self.gui.reset_download_widgets(self.selected_radar)
        else: 
            self.gui.set_download_widgets(self.dod[self.selected_radar].download_timerange_s//60, 'Stop') 
            
    def change_selected_radar(self, new_radar):
        # Changes just the selected radar without plotting
        self.selected_radar = new_radar
        self.update_download_widgets()
        self.pb.set_radarmarkers_data()
        self.pb.update()


    def change_dataset(self):
        self.selected_dataset='V' if self.selected_dataset=='Z' else 'Z'
        self.save_selected_dataset=self.selected_dataset
        #self.process_datetimeinput is called instead of self.pb.set_newdata, because the list with available datetimes needs to be updated, because
        #the datasets don't always have the same number of files
        self.process_datetimeinput()
        self.end_time=pytime.time()
                
    def change_dir_index(self):
        """Update the index of the current directory string in self.gui.radardata_dirs[radar_dataset].
        This means that we switch to the next directory string for which data is available for self.selected_date. 
        If there is no other directory string for which data is available for self.selected_date, then the index is not changed.
        """
        radar_dataset, dir_string_list = self.dsg.get_variables(self.selected_radar, self.selected_dataset)[:2]
        n_dirs=len(dir_string_list)
                                            
        dir_index = self.gui.radardata_dirs_indices[radar_dataset]
        new_dir_index=int((dir_index+1) % n_dirs)
        while new_dir_index != dir_index:
            try:
                next_dir_string=dir_string_list[new_dir_index]
                nearest_dir=bg.get_nearest_directory(next_dir_string,self.selected_radar,self.selected_date,self.selected_time,self.dsg.get_filenames_directory,self.dsg.get_datetimes_from_files)
                date_nearest_dir=bg.get_date_and_time_from_dir(nearest_dir,next_dir_string,radar = self.selected_radar)[0]
                if date_nearest_dir is None: #happens when the date variable is not in dir_string
                    datetimes = self.dsg.get_datetimes_directory(self.selected_radar,nearest_dir)
                    if any([k.startswith(self.selected_date) for k in datetimes]):
                        date_nearest_dir = self.selected_date
                if date_nearest_dir!=self.selected_date:
                    new_dir_index=int((new_dir_index+1) % n_dirs)
                else: 
                    break
            except Exception:
                # This at least happens when a certain subdataset is not available for any date for the selected radar
                new_dir_index = dir_index
                
        if new_dir_index != dir_index:
            self.gui.radardata_dirs_indices[radar_dataset] = new_dir_index
                
            self.changing_subdataset=True
            self.process_datetimeinput()
            self.changing_subdataset=False
            
    def change_product_version(self):
        """Update the product version in self.gui.radardata_product_versions[radar_dataset].
        This means that we switch to the next product version if there is more than one available.
        """
        radar_dataset = self.dsg.get_radar_dataset(self.selected_radar, self.selected_dataset)
        
        pv_index = 0; new_pv_index = 0
        if not self.dsg.product_versions_directory is None:
            n = len(self.dsg.product_versions_directory)
            if self.gui.radardata_product_versions[radar_dataset] in self.dsg.product_versions_directory:
                pv_index = np.where(self.dsg.product_versions_directory == self.gui.radardata_product_versions[radar_dataset])[0][0]
                new_pv_index = (pv_index + 1) % n
            else:
                pv_index = 0
                new_pv_index = n-1
                
        if new_pv_index != pv_index:
            new_pv = self.dsg.product_versions_directory[new_pv_index]
            self.gui.radardata_product_versions[radar_dataset] = new_pv
            if new_pv in self.gui.selected_product_versions_ordered:
                self.gui.selected_product_versions_ordered.remove(new_pv)
            self.gui.selected_product_versions_ordered.append(new_pv)
            
            self.changing_subdataset=True
            self.process_datetimeinput()
            self.changing_subdataset=False
            
        
    def change_productunfiltered(self):
        if pytime.time()-self.end_time<self.gui.sleeptime_after_plotting: return
        self.productunfiltered[self.pb.panel] = not self.productunfiltered[self.pb.panel] #True implies filtered product, False unfiltered product
        if self.plot_mode=='Row':
            self.productunfiltered, panellist_change=self.change_variable_in_row(self.productunfiltered)
        elif self.plot_mode=='Column': 
            self.productunfiltered, panellist_change=self.change_variable_in_column(self.productunfiltered)
        elif self.plot_mode=='All':
            for j in self.pb.panellist:
                self.productunfiltered[j]=self.productunfiltered[self.pb.panel]
            panellist_change=self.pb.panellist
        else:
            panellist_change=[self.pb.panel]
            
        self.pb.set_newdata(panellist_change)
        self.end_time=pytime.time()

    def change_polarization(self):
        if pytime.time()-self.end_time<self.gui.sleeptime_after_plotting: return
        self.polarization[self.pb.panel]='V' if self.polarization[self.pb.panel]=='H' else 'H'
        if self.plot_mode=='Row':
            self.polarization, panellist_change=self.change_variable_in_row(self.polarization)
        elif self.plot_mode=='Column': 
            self.polarization, panellist_change=self.change_variable_in_column(self.polarization)
        elif self.plot_mode=='All':
            for j in self.pb.panellist:
                self.polarization[j]=self.polarization[self.pb.panel]
            panellist_change=self.pb.panellist
        else:
            panellist_change=[self.pb.panel]
    
        self.pb.set_newdata(panellist_change)
        self.end_time=pytime.time()
        
    def change_apply_dealiasing(self):
        if pytime.time()-self.end_time<self.gui.sleeptime_after_plotting: return
        self.apply_dealiasing[self.pb.panel] = not self.apply_dealiasing[self.pb.panel]
        if self.plot_mode=='Row':
            self.apply_dealiasing, panellist_change=self.change_variable_in_row(self.apply_dealiasing)
        elif self.plot_mode=='Column': 
            self.apply_dealiasing, panellist_change=self.change_variable_in_column(self.apply_dealiasing)
        elif self.plot_mode=='All':
            for j in self.pb.panellist:
                self.apply_dealiasing[j]=self.apply_dealiasing[self.pb.panel]
            panellist_change=self.pb.panellist
        else:
            panellist_change=[self.pb.panel]
        
        self.pb.set_newdata(panellist_change)
        self.end_time=pytime.time()

            
    def set_datetimewidgets(self,date,time):
        self.gui.datew.setText(date); self.gui.datew.repaint()
        self.gui.timew.setText(time); self.gui.timew.repaint()
        
    def plot_mostrecent_data(self,plot_data=True):
        self.signal_set_datetimewidgets.emit('c','c')
        if self.pb.firstplot_performed:
            for j in self.dsg.scannumbers_all['z']:
                try: #Exception occurs when self.dsg.scannumbers_all['z'] is not complete
                    self.dsg.scannumbers_forduplicates[j]=len(self.dsg.scannumbers_all['z'][j])-1
                except Exception: pass
        if plot_data:
            self.requesting_latest_data = True
            self.process_datetimeinput()
            self.requesting_latest_data = False
        
    def back_to_previous_plot(self,change_datetime=False):
        if not self.pb.firstplot_performed or (
        self.ani.continue_type == 'None' and pytime.time()-self.end_time<self.gui.sleeptime_after_plotting): return
        # The latter only when self.ani.continue_type == 'None', to prevent that this function does not work at least half of the time
        # during an animation or a continuation to the left or right.
        
        if change_datetime:
            self.going_back_to_previous_plot=True
            #Is only set to True when also changing the datetime, because when only going back to the previous radar, radardir_index and dataset, then it is
            #possible that the values for self.rd_before_variables['scannumbers_forduplicates'] are not anymore valid for the new combination of radar, dataset 
            #and datetime, such that there might be the need to update self.dsg.scannumbers_forduplicates. This is done in the function self.dsg.update_parameters.
            
            #Go back to the previous date and time
            if len(self.before_variables['scannumbers_forduplicates'])>0: 
                self.dsg.scannumbers_forduplicates=self.before_variables['scannumbers_forduplicates'].copy()
            self.signal_set_datetimewidgets.emit(self.before_variables['date'], self.before_variables['time'])
        else:            
            self.dsg.scannumbers_forduplicates=self.rd_before_variables['scannumbers_forduplicates'].copy()
        
        
        back_variables=self.before_variables if change_datetime else self.rd_before_variables
        
        #If self.radar and/or self.dataset are not equal to self.selected_radar and self.selected_dataset, then they are restored to
        #self.radar and self.dataset.
        if self.radar!=self.selected_radar:
            self.selected_radar=self.radar
            self.pb.set_radarmarkers_data()
            self.pb.update()
        elif self.dataset!=self.selected_dataset:
            self.selected_dataset=self.dataset
        elif self.radar!=back_variables['radar']: 
            self.change_radar(back_variables['radar'])
        elif self.dataset!=back_variables['dataset']: self.change_dataset()
        else: 
            #This does not need to be done when changing the radar or dataset, because it cannot occur that the radar/dataset and the 
            #directory string index get changed during the same function call.
            radar_dataset=self.dsg.get_variables(self.radar,self.dataset)[0]
            if not back_variables['radardir_index'] is None and self.gui.radardata_dirs_indices[radar_dataset]!=back_variables['radardir_index']:
                self.changing_subdataset=True
                self.gui.radardata_dirs_indices[radar_dataset]=back_variables['radardir_index']
            elif not back_variables['product_version'] is None and self.gui.radardata_product_versions[radar_dataset]!=back_variables['product_version']:
                self.changing_subdataset=True
                self.gui.radardata_product_versions[radar_dataset]=back_variables['product_version']
        
            self.process_datetimeinput(change_datetime=change_datetime) #Setting change_datetime=True when self.dsg.scannumbers_forduplicates has changed is important,
            #because otherwise the function self.pbtthis.set_newdata does not know that it should call self.pb.update_before_variables.
            self.changing_subdataset=False
        self.end_time=pytime.time()
        
        self.going_back_to_previous_plot=False




    def get_filedatetimes(self,dataset,date=None,time=None):
        """This function gets only called to update self.filedatetimes for self.selected_radar, so it does not require the radar as
        input.
        """
        files_datetimes=np.array([],dtype='int64')
        try:
            if not (date is None or time is None):
                self.directory = self.dsg.get_nearest_directory(self.selected_radar,dataset,date,time)
                
            files_datetimes=self.dsg.get_files(self.selected_radar,self.directory,return_datetimes=True)
            if self.directory in self.filedatetimes_errors:
                # Exclude datetimes in self.filedatetimes_errors[self.directory], unless the total file size has changed
                # for this datetime.
                datetimes_remove = [dt for dt,size in self.filedatetimes_errors[self.directory].items() if 
                                    dt in files_datetimes and size == self.dsg.get_total_volume_files_size(dt)]
                files_datetimes = [j for j in files_datetimes if not j in datetimes_remove]
            
            """Ensure that a switch of product version persists when switching from radar/dataset. After a call of the function
            self.dsg.get_files we know the number of product version that is available within the directory (stored in
            self.dsg.product_versions_directory). We can thus check whether an already selected product version within the
            variable self.gui.selected_product_versions_ordered is contained within self.dsg.product_versions_directory.
            If that is the case, we select the latest product version from self.gui.selected_product_versions_ordered
            for which this is the case."""
            if not self.dsg.product_versions_directory is None:
                for j in self.gui.selected_product_versions_ordered[::-1]:
                    if j in self.dsg.product_versions_directory:
                        radar_dataset = self.dsg.get_radar_dataset(self.selected_radar, dataset)
                        self.gui.radardata_product_versions[radar_dataset] = j
                        break

        except Exception as e:
            print(e, 'get_filedatetimes')
        files_available=True if len(files_datetimes)>0 else False
        return files_available, files_datetimes
            
    def determine_list_filedatetimes(self,date=None,time=None,source=None):     
        """This function gets only called to update self.filedatetimes for self.selected_radar, so it does not require the radar as
        input.
        If date=None and time=None, then self.directory is used.
        """
        files_available, files_datetimes=self.get_filedatetimes(self.selected_dataset,date,time)
        if not files_available and self.selected_radar in gv.radars_with_datasets:
            #Try the other dataset, and if data is available for that dataset, then change self.selected_dataset. The true selected dataset remains
            #stored in self.save_selected_dataset.
            new_dataset='V' if self.selected_dataset=='Z' else 'Z'
            files_available, files_datetimes=self.get_filedatetimes(new_dataset,date,time)
            if files_available:
                self.selected_dataset=new_dataset 

        if not files_available: 
            self.filedatetimes=[np.array([]),np.array([])]
        else:
            files_available=True if len(files_datetimes)>0 else False
            if files_available: #It is possible that removing datetimes of files that caused errors leads to an empty datetimes list.
                files_absolutetimes=np.array(ft.get_absolutetimes_from_datetimes(files_datetimes))
                self.filedatetimes=[files_datetimes,files_absolutetimes]
            #Else self.filedatetimes remains unchanged
            self.volume_timestep_m = self.determine_volume_timestep_m()
            
        if not files_available:
            if self.directory is None or not os.path.exists(self.directory):
                text='No directory found with selected directory format. Downloading data might still work, due to automatic creation of directory.'
            else:
                text='Directory is empty: '+self.directory
            self.gui.set_textbar(text,'red',minimum_display_time=2)
            
            if not source == self.reset_dirinfo:
                # Prevent recursive errors when also the restored directory is empty
                self.reset_dirinfo()
            
        return files_available
    
    def determine_volume_timestep_m(self, datetimes=None, radar=None):
        abstimes = self.filedatetimes[1] if datetimes is None else ft.get_absolutetimes_from_datetimes(datetimes)
        radar = self.radar if radar is None else radar
        return np.median(np.diff(abstimes))/60 if len(abstimes) > 1 else gv.volume_timestep_radars[radar]
            
    def get_closestdatetime(self,date,time,lr_step=0):
        #self.filedatetimes shouldn't be empty. When it is, then an exception is raised.
        if len(self.filedatetimes[0])==0:
            raise Exception
        
        if time!='c':
            timediffs=self.filedatetimes[1]-ft.get_absolutetimes_from_datetimes(date+time)
            index=np.argmin(np.abs(timediffs))
            mintimediff = timediffs[index]
            if lr_step == 0 and self.ani.starting_iteration_action:
                # When determining start and end datetime at the start of an animation iteration, it is not allowed that the end datetime is (much)
                # later than the inputted value. Similarly, it is not allowed that the start datetime is (much) earlier than the inputted value. 
                if self.ani.starting_iteration_action == 'start' and mintimediff < -1800:
                    if index < len(timediffs)-1:
                        index += 1
                    else: 
                        lr_step = 1
                elif self.ani.starting_iteration_action == 'end' and mintimediff > 1800:
                    if index > 0:
                        index -= 1
                    else:
                        lr_step = -1
            
            original_directory = self.directory
            #If this is the case, then the first or last file in self.filedatetimes is reached.
            if (lr_step<0 and index==0 and mintimediff>0) or (lr_step>0 and index==len(timediffs)-1 and mintimediff<0):
                if self.ani.starting_iteration_action:
                    current_date, current_time = date, time
                    desired_newdate = desired_newtime = None
                else:
                    current_date, current_time = self.date, self.time
                    desired_newdate, desired_newtime = date, time
                    if abs(ft.datetimediff_s(self.date+self.time, date+time)) <= 300:
                        # Setting to None prevents that first self.dsg.get_nearest_directory is called before determining the next directory
                        desired_newdate = desired_newtime = None
                print(self.filedatetimes[0][0], self.filedatetimes[0][-1], lr_step)
                self.directory=self.dsg.get_next_directory(self.selected_radar,self.selected_dataset,current_date,current_time,int(np.sign(lr_step)),desired_newdate,desired_newtime)
                print('get_next_dir', current_date, current_time, desired_newdate, desired_newtime, self.directory)
                self.determine_list_filedatetimes()

                #If not files_available, then the 'old' list with datetimes is still used.
                timediffs=np.abs(self.filedatetimes[1]-ft.get_absolutetimes_from_datetimes(date+time))
                #Determine the index of the datetime that is closest to date and time.
                index=np.argmin(timediffs)      
                
            # Check whether the timestep is at least nonzero. If not, then if possible go one volume backward/forward.
            datetime, current_datetime = int(self.filedatetimes[0][index]), int(self.date+self.time)
            if (lr_step > 0 and datetime <= current_datetime and index+1 < len(self.filedatetimes[0])) or\
               (lr_step < 0 and datetime >= current_datetime and index > 0):
                index += np.sign(lr_step)
            # Check whether the timestep made is smaller than the maximum allowed timestep. If not, then go 1 step backward/forward in time 
            # and reset directory if needed
            datetime = self.filedatetimes[0][index]
            if lr_step != 0 and abs(ft.datetimediff_s(self.date+self.time,datetime)/60) > self.gui.max_timestep_minutes:
                if original_directory != self.directory:
                    self.reset_dirinfo()
                    index = -1 if lr_step > 0 else 0
                else:
                    index += -1 if lr_step > 0 else 1

            closest_datetime=self.filedatetimes[0][index]
        else:
            closest_datetime=self.filedatetimes[0][-1]
            
        closest_date, closest_time = closest_datetime[:8], closest_datetime[-4:]
        return closest_date, closest_time
    
    def reset_dirinfo(self):
        #Reset the working directory, list of file datetimes etc.
        self.directory = self.dsg.get_directory(self.date,self.time,self.radar,self.dataset)
        self.determine_list_filedatetimes(source=self.reset_dirinfo)
        
    
    def process_datetimeinput(self,call_ID=None,set_data=True,change_datetime=None):
        # change_datetime=True should be given as input when self.dsg.scannumbers_forduplicates has changed in the function self.back_to_previous_plot, 
        # because there is otherwise no easy way to check for such a change in scannumbers_forduplicates.
        """If set_data=False, then only dates and times are updated, and volume attributes are retrieved and returned (and automatically restored after
        retrieving them, in nlr_datasourcegeneral.py). Except for dates and times, no data gets updated in this case.
        """
        self.process_datetimeinput_running=True #Must be set to False before every return
        if self.selected_radar in gv.radars_with_datasets:
            self.selected_dataset=self.save_selected_dataset
                    
        self.start_time=pytime.time()
        date=str(self.gui.datew.text()).replace(' ','')
        time=str(self.gui.timew.text()).replace(' ','')
        #Remove spaces, because a date or time with an extra space should not be regarded as incorrect input.
        if not ft.correct_datetimeinput(date,time): 
            self.process_datetimeinput_call_ID=call_ID
            self.process_datetimeinput_running=False
            self.gui.datew.setText(self.date); self.gui.timew.setText(self.time)
            return
        
        if time != 'c' and self.gui.use_storm_following_view and self.gui.view_nearest_radar and not self.change_radar_running:
            delta_time = ft.datetimediff_s(self.date+self.time, date+time)
            print('switch to nearby')
            self.switch_to_nearby_radar(1, delta_time=delta_time, source=self.process_datetimeinput)

        files_available=self.determine_list_filedatetimes(date,time)
        #When time=='c', the last file at the disk is always used.
                
        if files_available:
            self.selected_date,self.selected_time=self.get_closestdatetime(date,time)
            self.signal_set_datetimewidgets.emit(self.selected_date,self.selected_time) 
                     
            if change_datetime is None:
                change_datetime=True if (self.selected_date!=self.date or self.selected_time!=self.time or not self.pb.firstplot_performed) else False 
            self.requesting_latest_data = time == 'c'
            returns = self.pb.set_newdata(self.pb.panellist,change_datetime,source_function=self.process_datetimeinput,set_data=set_data,
                                          apply_storm_centering=True)
            self.requesting_latest_data = False
            if not set_data:
                retrieved_attrs = returns
            
            self.end_time=pytime.time()
                
        #When showing an animation (except for during the start), then self.selected_radar and self.selected_dataset are reset when there is no
        #data available for the current radar and dataset. This is done because this makes it possible to continue the animation for the previous 
        #radar and dataset, which are self.radar and self.dataset.
        #Only when set_data=True, because this should only be done at the end of the evaluation of the function self.ani.update_datetimes_and_perform_firstplot.
        if not files_available:
            self.reset_dirinfo()
            
            if set_data and self.ani.continue_type == 'ani':
                if self.ani.starting_animation:
                    self.ani.continue_type='None'
                else:
                    if self.selected_radar!=self.radar: 
                        self.selected_radar=self.radar; self.pb.set_radarmarkers_data()
                    self.selected_dataset=self.dataset
                    self.ani.update_animation=True
            
        self.process_datetimeinput_call_ID=call_ID
        self.process_datetimeinput_running=False
        if files_available and not set_data:
            return retrieved_attrs
        
        
    def reset_datetime(self, date, time):
        """This function should be called after calling self.process_datetimeinput with set_data=False, restore the date and time
        for which data is shown.
        It is not needed when self.process_datetimeinput or self.pb.set_newdata is called afterwards with set_data = True. 
        """
        self.selected_date = date; self.selected_time = time
        self.signal_set_datetimewidgets.emit(self.selected_date,self.selected_time)
        self.process_datetimeinput(set_data = False)
        
            
    def perform_leftrightstep(self,lr_step):
        """Check for the presence of scans that are performed more than once during a complete volume scan. If they are not present, or if the
        first (step backward)/last (step forward) of those scans is currently visible, then the time is changed.
        
        Further, the desired timestep, self.desired_timestep_minutes(), must be taken into account. When scans with duplicates are currently
        visible, this is handled different compared to the case in which they are not. In the case of visible scans with duplicates, it is assumed that
        those scans are distributed proportionally through the volume, with a time between them given by
        volume_timestep_radars[self.radar]/visible_scans_max_occurrences.
        """                                                            
        use_same_volumetime = False
        # vt is (typical) time between volumes for self.radar
        vt, input_dt = self.volume_timestep_m, self.desired_timestep_minutes()
        consider_duplicates = input_dt == 0. or np.mod(input_dt, vt) > 0.
        # dt >= 1 is to ensure that self.filedatetimes_list gets updated in the function
        # self.get_closestdatetime when another date is reached. If not included, then the condition with mintimediff would
        # not be satisfied.
        dt = max(1, input_dt)
        
        if abs(lr_step) == 1 and consider_duplicates and len(self.visible_productscans_with_duplicates):
            scans = self.visible_productscans_with_duplicates
            scannumbers = [self.dsg.scannumbers_all['z'][j] for j in scans]            
            scan_ndup_max = scans[np.argmax(len(j) for j in scannumbers)]
            ndup_max = len(self.dsg.scannumbers_all['z'][scan_ndup_max])
            idup_max = self.dsg.scannumbers_forduplicates[scan_ndup_max]
            delta_i = max(1, round(input_dt/vt*ndup_max))
            idup_max += lr_step*delta_i
            ratio = idup_max/ndup_max
            
            for scan in self.productscans_with_duplicates:
                ndup = len(self.dsg.scannumbers_all['z'][scan])
                self.dsg.scannumbers_forduplicates[scan] = np.mod(int(np.floor(ratio*ndup)), ndup)
            
            use_same_volumetime = 0 <= ratio < 1
            if dt < vt:
                timestep_m = lr_step*dt
            else:                
                volume_shift = abs(ratio//1)
                timestep_m = lr_step*volume_shift*vt
            estimated_timestep = lr_step*delta_i/ndup_max*vt
        else:
            if abs(lr_step) == 1:
                timestep_m = lr_step*dt
            elif abs(lr_step) == 12: #Use a time step of 60 minutes.
                timestep_m = np.sign(lr_step)*60
            estimated_timestep = timestep_m
            
        if not use_same_volumetime:
            trial_date, trial_time=ft.next_date_and_time(self.date,self.time,timestep_m)
            self.selected_date, self.selected_time = self.get_closestdatetime(trial_date,trial_time,lr_step=lr_step)

            if self.selected_date==self.date and self.selected_time==self.time:
                #No change of time in this case, and the scannumbers_forduplicates must be restored to 
                #their previous values. self.current_variables['scannumbers_forduplicates'] is used instead of 
                #self.before_variables['scannumbers_forduplicates'], because the latter contains not values for the most recent plot, 
                #but for the one before.
                self.dsg.scannumbers_forduplicates=self.current_variables['scannumbers_forduplicates'].copy()
                use_same_volumetime=None
                
            self.signal_set_datetimewidgets.emit(self.selected_date,self.selected_time)
        return use_same_volumetime, estimated_timestep
    
    def desired_timestep_minutes(self):
        return self.gui.desired_timestep_minutes if not self.gui.desired_timestep_minutes == 'V' else self.volume_timestep_m
           
    def process_keyboardinput(self,leftright_step=0,downup_step=0,new_scan=0,new_product='0',call_ID=None,from_timer=True): 
        # from cProfile import Profile
        # profiler = Profile()
        # profiler.enable() 
        if not call_ID is None:
            self.process_keyboardinput_call_ID=call_ID
        self.process_keyboardinput_running=True
        """Important: self.process_keyboardinput_running must be set to False before every return in this function.
        """

        if not from_timer and leftright_step!=0. and self.timer_process_keyboardinput.isActive(): 
            #In this case it is not desired to evaluate this function.
            #The last condition is only for leftright_step!=0., to enable the program to process other events when self.ani.continue_type!='None'.
            #These are events like a change in scan or change in product.
            self.process_keyboardinput_running=False
            return 
        elif not from_timer:
            self.process_keyboardinput_arguments=[leftright_step,downup_step,new_scan,new_product,call_ID]
        elif from_timer:
            leftright_step,downup_step,new_scan,new_product,call_ID=self.process_keyboardinput_arguments
            
        
        #A try-except clausule is used to handle exceptions that occur when self.dsg.scannumbers_all is incomplete
        #(e.g. because there is no data for a particular radar)
        if True:
            if not self.pb.firstplot_performed and (leftright_step!=0 or downup_step!=0 or new_scan!=0): 
                #This is done because in these cases this function needs information about the scan attributes, that is not available before
                #the first plot is performed.
                self.process_keyboardinput_running=False
                return
            
            if not self.ani.continue_type == 'None' and not call_ID is None and call_ID>1 and self.process_keyboardinput_finished_before:
                #self.process_keyboardinput_timebetweenfunctioncalls should be approximately equal to the time it takes to finish all commands
                #that are executed (by OpenGL etc.) after finishing this function call.
                #It is only updated when continuing to the left or when showing an animation, because in other cases this method does not work.
                #This is because it is possible that a new call of this function does not follow immediately after finishing the previous one,
                #such that evaluation the following command would cause self.process_keyboardinput_timebetweenfunctioncalls to become (way) too
                #large.
                if call_ID>2:
                    #Use the minimum of the current and previous value of pytime.time()-self.end_time, because it is observed that it sometimes
                    #takes much longer (say 0.05 s) to go from emitting a function call in nlr_animate.py to processing it, which should not
                    #increase self.process_keyboardinput_timebetweenfunctioncalls by this same amount.
                    self.process_keyboardinput_timebetweenfunctioncalls=np.min([self.previous_timebetweenfunctioncalls,pytime.time()-self.end_time])
                else:
                    self.process_keyboardinput_timebetweenfunctioncalls=pytime.time()-self.end_time
                self.previous_timebetweenfunctioncalls=pytime.time()-self.end_time
            else:
                self.process_keyboardinput_timebetweenfunctioncalls=self.gui.sleeptime_after_plotting
                
    
            self.scans_currentlyvisible_notplain=[self.scans[j] for j in self.pb.panellist if self.products[j] not in gv.plain_products]
            self.products_currentlyvisible=[self.products[j] for j in self.pb.panellist]
            
            if leftright_step != 0:
                plain_products = [self.products[j] for j in self.pb.panellist if self.products[j] in gv.plain_products]
                notplain_scans = [self.scans[j] for j in self.pb.panellist if not self.products[j] in gv.plain_products]
                if any(i not in self.dsg.scannumbers_all['z'] for i in plain_products+notplain_scans):
                    print(plain_products, notplain_scans, self.scans, self.dsg.scannumbers_all, self.dsg.scanangles_all)
                visible_scans_max_occurrences = max([len(self.dsg.scannumbers_all['z'][i]) for i in plain_products+notplain_scans])
                
                vt = self.volume_timestep_m
                min_dt = vt/visible_scans_max_occurrences
                if visible_scans_max_occurrences == 0:
                    #This can at least occur when viewing Cabauw data, in which case no data is present for a given time,
                    #while still no error is raised in the data obtaining procedure. It is desired to not raise that error,
                    #since otherwise the old volume attributes will be used, which might contain different numbers of duplicates
                    #when going back in time vs when going forward in time, leading to different values for timestep_m and
                    #thereby different speeds with which the program loops through the data.
                    timestep_m = vt
                elif abs(leftright_step) == 1.:
                    timestep_m = max(min_dt, self.desired_timestep_minutes())
                else: 
                    #When pressing SHIFT+LEFT/RIGHT
                    timestep_m = vt
        
                maxspeed_minpsec = self.gui.animation_speed_minpsec if 'ani' in self.ani.continue_type else self.gui.maxspeed_minpsec
                time_to_wait = max([timestep_m/maxspeed_minpsec-(pytime.time()-self.start_time),
                                    self.process_keyboardinput_timebetweenfunctioncalls-(pytime.time()-self.end_time)])
            else:
                time_to_wait=self.gui.sleeptime_after_plotting-(pytime.time()-self.end_time)
                if downup_step!=0.:
                    #Prevent going up and down from occurring annoyingly fast
                    time_to_wait=max(time_to_wait, 0.06-(pytime.time()-self.start_time))
                    
            if self.gui.continue_savefig:
                # Always sleep some time, since otherwise it might happen that some plots are skipped, which leads to missing frames in an animation
                time_to_wait = max([time_to_wait, 0.02])
    
            if not from_timer and time_to_wait>0.:
                self.process_keyboardinput_running=False; self.process_keyboardinput_finished_before=False
                
                if self.timer_process_keyboardinput.isActive():
                    self.timer_process_keyboardinput.stop()
                    
                self.timer_process_keyboardinput.start(int(time_to_wait*1000.)) #Unit is ms
                return
            
            
            self.start_time=pytime.time()
            self.lrstep_beingperformed = leftright_step if leftright_step else None
    
            if self.selected_radar!=self.radar:
                self.selected_radar=self.radar; self.pb.set_radarmarkers_data()
            self.selected_dataset=self.dataset; self.selected_date=self.date; self.selected_time=self.time
    
            if not self.pb.firstplot_performed: 
                files_available = self.determine_list_filedatetimes(self.date,self.time)
                if files_available:
                    try:
                        #Catch exceptions that could arise when the list self.filedatetimes is empty.
                        self.selected_date,self.selected_time=self.get_closestdatetime(self.date,self.time,lr_step=leftright_step)
                        self.date=self.selected_date; self.time=self.selected_time
                    except Exception:
                        self.process_keyboardinput_running=False; self.process_keyboardinput_finished_before=False
                        return
                else:
                    self.process_keyboardinput_running=False; self.process_keyboardinput_finished_before=False
                    return
                    
                
            if leftright_step!=0:
                #Contains duplicate scans and plain products in plain_products_affect_by_double_volume in the case of a double volume.
                self.productscans_with_duplicates=[i for i,j in self.dsg.scannumbers_all['z'].items() if len(j)>1]
                self.visible_productscans_with_duplicates=[j for j in self.scans_currentlyvisible_notplain if j in self.productscans_with_duplicates]
                self.visible_productscans_with_duplicates+=[j for j in gv.plain_products_affected_by_double_volume if j in self.products_currentlyvisible and j in self.productscans_with_duplicates]
    
                radar_changed = False
                try:
                    use_same_volumetime, estimated_timestep = self.perform_leftrightstep(leftright_step)
                    if self.gui.use_storm_following_view and self.gui.view_nearest_radar and not self.change_radar_running:
                        delta_time = estimated_timestep*60
                        radar_changed = self.switch_to_nearby_radar(1, delta_time=delta_time, source=self.process_keyboardinput)
                        use_same_volumetime = False if radar_changed else use_same_volumetime
                except Exception as e:
                    print(e, 'perform_leftrightstep')
                    return
                
                # print('print3', use_same_volumetime, radar_changed)
                if use_same_volumetime is None or radar_changed: 
                    # In first case the date and time haven't changed, so there is no need to continue. In second case the change of
                    # radar is processed in self.change_radar.
                    self.process_keyboardinput_running = self.process_keyboardinput_finished_before = False
                    self.lrstep_beingperformed = None
                    if use_same_volumetime is None and self.ani.continue_type=='leftright':
                        self.ani.continue_type='None'
                    if radar_changed:
                        self.end_time=pytime.time()
                    return
                         
            if new_scan!=0:
                if self.plot_mode=='All':
                    downup_step=new_scan-self.scans[self.pb.panel]; new_scan=0
                else:
                    self.scans[self.pb.panel]=new_scan
                              
            if abs(downup_step)==1.1 and self.products[self.pb.panel] not in gv.plain_products: 
                self.scans[self.pb.panel] = min(max(1, self.scans[self.pb.panel]+int(downup_step)), len(self.dsg.scanangles_all['z']))
            elif downup_step!=0 and self.scans_currentlyvisible_notplain:
                step=downup_step
                if min(self.scans_currentlyvisible_notplain)<=-downup_step and downup_step<0: 
                    step=1-min(self.scans_currentlyvisible_notplain)
                elif max(self.scans_currentlyvisible_notplain)+downup_step>len(self.dsg.scanangles_all['z']) and downup_step>0: 
                    step=len(self.dsg.scanangles_all['z'])-max(self.scans_currentlyvisible_notplain)
                                
                """It is possible that there is a scan pair of which one scan has a large range and low Nyquist velocity, and one has a smaller range
                and higher Nyquist velocity. In this case it can be desired to show both scans of this scan pair, with e.g. scan 1 for the 
                reflectivity (largest range) and scan 2 for the velocity (largest Nyquist velocity). 
                In order to prevent that going up/down in this case leads to showing reflectivity and velocity for cleary different
                scanangles, I choose to only update the panel with the first of those 2 scans in this case, and scans in other panels remain
                the same. The exception is when for 1 product both of those 2 scans are shown, because doing the above thing in this case
                would lead to 2 panels showing the same product for the same scan.
                When starting with both scans in the scan pair, and going up and down without changing a scan/product in 
                another way in the mean time, then both scans in the scan pair will again be shown when going down and reaching
                the 'bottom'.
                
                The method below does not only include cases in which both scans of the scan pair are shown, but also cases in which only the first
                scan of the scan pair is shown. In this case also only the panel(s) that display this first scan will be updated. These cases are
                included because otherwise the difference in scanangle between this panel and other panels that do not show this scan can increase
                substantially. This is because the scanangle in the panel that shows the first scan of the scan pair can increase by at most 0.1 
                degree, while this increase can be much larger for panels that show other scans.
                """
                scanpair_present=self.dsg.check_presence_large_range_large_nyquistvelocity_scanpair()
                
                if scanpair_present:
                    products_panels_in_scanpair=[self.products[j] for j in self.pb.panellist if self.scans[j] in (1,2) and not self.products[j] in gv.plain_products]
                    panels_with_first_scan_of_scanpair=[j for j in self.pb.panellist if self.scans[j]==1]
                else:
                    products_panels_in_scanpair=[]; panels_with_first_scan_of_scanpair=[]
                
                #The condition for the length is necessary, because the part with the all function will always evaluate to True when working
                #with an empty list.
                if step>0 and len(panels_with_first_scan_of_scanpair)>0 and (
                len(np.unique(products_panels_in_scanpair))==len(products_panels_in_scanpair)):
                    self.time_last_up_first_scan_scanpair=pytime.time()
                    
                    self.savescans_withduplicatescanangles=self.scans.copy()
                    for j in self.pb.panellist:
                        if j in panels_with_first_scan_of_scanpair:
                            self.scans[j]=2
                else:
                    self.scans_currentlyvisible_notplain=[self.scans[j] for j in self.pb.panellist if self.products[j] not in gv.plain_products]
                    if step<0 and scanpair_present and any([j==2 for j in self.scans_currentlyvisible_notplain]) and not any([j==1 for j in self.scans_currentlyvisible_notplain]):
                        #If step<0 and a scan pair is present and some scans are equal to 2 while there are no scans yet that are 1, then 
                        #update only the scans that are equal to 2.
                        for j in self.pb.panellist:
                            if self.scans[j]==2: self.scans[j]=1
                    else:
                        #Use the 'normal' method for going up/down.
                        for j in self.pb.panellist: #This is purposely also done for plain products.
                            if step<0:
                                self.scans[j]=np.max([self.scans[j]+step,1])
                            else:
                                self.scans[j]=np.min([self.scans[j]+step,len(self.dsg.scanangles_all['z'])])  
                    self.scans_currentlyvisible_notplain=[self.scans[j] for j in self.pb.panellist if self.products[j] not in gv.plain_products]
                    
                    if step<0 and scanpair_present and any([j==1 for j in self.scans_currentlyvisible_notplain]) and (
                    self.time_last_up_first_scan_scanpair>np.max([self.time_last_productchange,self.time_last_individualscanchange])):
                        #When going down and reaching the 'bottom', then use the set of scans that where also used before initially going 
                        #up, at least when the above conditions are satisfied.
                        self.scans=self.savescans_withduplicatescanangles.copy()
                        
                
            if new_product!='0': 
                self.products[self.pb.panel]=new_product
            
            panellist_change=[]
            if new_product!='0' or new_scan!=0:
                if self.plot_mode=='Row':
                    panellist_change=self.row_mode(variables_change='products' if new_product!='0' else 'scans')
                elif self.plot_mode=='Column':
                    panellist_change=self.column_mode(variables_change='products' if new_product!='0' else 'scans')
                else: panellist_change=[self.pb.panel]
                
                if self.products[self.pb.panel] in gv.plain_products_with_parameters or self.pb.products_before[self.pb.panel] in gv.plain_products_with_parameters:
                    for j in (self.products[self.pb.panel], self.pb.products_before[self.pb.panel]):
                        #Handle the cases in which only one of self.products[self.pb.panel] and self.pb.products_before[self.pb.panel]
                        #is a plain product, and also the case in which both are a plain product.
                        if j in gv.plain_products_with_parameters:
                            panellist_change += self.gui.change_PP_parameters_panels(j)
                    panellist_change = list(np.unique(panellist_change)) 
            elif abs(downup_step)>0. and abs(downup_step)!=1.1: 
                panellist_change=[j for j in self.pb.panellist if self.products[j] not in gv.plain_products]
            elif abs(downup_step)==1.1 and self.products[self.pb.panel] not in gv.plain_products:
                panellist_change=[self.pb.panel]
            elif leftright_step!=0 and use_same_volumetime and not self.change_radar_running: # radar could change when combining nearest
                # radar selection with storm-following view. In that case update all panels.
                d, d_before = self.dsg.scannumbers_forduplicates, self.current_variables['scannumbers_forduplicates']
                for j in self.pb.panellist:
                    try:
                        if (self.products[j] in gv.products_with_tilts and d[self.scans[j]] != d_before[self.scans[j]]) or (
                        self.products[j] in gv.plain_products_affected_by_double_volume and d[self.products[j]] != d_before[self.products[j]]):
                            panellist_change.append(j)
                    except Exception as e:
                        print(e, d, d_before, [self.scans[j] for j in self.pb.panellist])
                        1/0
            elif leftright_step!=0:
                panellist_change=[j for j in self.pb.panellist]
            
            if new_scan!=0 or downup_step!=0: 
                self.update_selected_scanangles()
            
            if panellist_change:
                change_datetime = leftright_step != 0
                self.pb.set_newdata(panellist_change,change_datetime=change_datetime,source_function=self.process_keyboardinput,
                                    apply_storm_centering=True)
                

            if new_scan!=0 or abs(downup_step)==1.1: self.time_last_individualscanchange=pytime.time()
            elif downup_step!=0: self.time_last_downup=pytime.time()
            elif leftright_step!=0: self.time_last_leftright=pytime.time()
            elif new_product!='0': self.time_last_productchange=pytime.time()
            
            self.end_time=pytime.time()
            self.lrstep_beingperformed=None
            self.process_keyboardinput_running=False
            self.process_keyboardinput_finished_before=True
            # profiler.disable()
            # import pstats
            # stats = pstats.Stats(profiler).sort_stats('cumtime')
            # stats.print_stats(1)  
        else:
            self.process_keyboardinput_running=False; self.process_keyboardinput_finished_before=False
            print(e,'process_keyboardinput')
        
        
    def update_selected_scanangles(self, update_allpanels=False):
        panellist = range(self.pb.max_panels) if update_allpanels else self.pb.panellist
        for j in panellist:
            #'z' is always available as key for the volume attributes
            product = gv.i_p[self.products[j]] if self.dsg.scanangles_all_m[gv.i_p[self.products[j]]] else 'z'
            max_scanangle = max(self.dsg.scanangles_all_m[product].values())
            self.selected_scanangles[j] = self.dsg.scanangles_all_m[product].get(self.scans[j], max_scanangle)
        
    def change_variable_in_row(self,variable,panel='selected_panel'):
        if panel=='selected_panel':
            panel=self.pb.panel #Else the reference panel (whose scan or product is used for other panels) should be given as input

        if self.pb.panels<4: row_panels=(0,1,2,5)
        elif panel<5: row_panels=(0,1,2,3,4)
        else: row_panels=(5,6,7,8,9)
        panellist_change=[panel]

        scanpair_present=False if not self.pb.firstplot_performed else self.dsg.check_presence_large_range_large_nyquistvelocity_scanpair()
        for j in row_panels:
            if not j==panel and j in self.pb.panellist:  
                    
                if variable==self.scans and self.pb.changing_panels:
                    """When changing panels it is not desired to change scans when a scan pair is present and both the self.scans[j] and
                    self.scans[panel] are in the scan pair.
                    """
                    if not (scanpair_present and self.scans[j] in (1,2) and self.scans[panel] in (1,2)):     
                        variable[j]=variable[panel]; panellist_change.append(j)
                else:
                    variable[j]=variable[panel]; panellist_change.append(j)
                
        return variable, panellist_change
    
    def change_variable_in_column(self,variable,panel='selected_panel'):
        if panel=='selected_panel':
            panel=self.pb.panel #Else the reference panel (whose scan or product is used for other panels) should be given as input
            
        panellist_change=[panel]
        
        scanpair_present=False if not self.pb.firstplot_performed else self.dsg.check_presence_large_range_large_nyquistvelocity_scanpair()
        if self.pb.panels>3:
            j=np.mod(panel+5,10) #Index of the other panel in the same column
            
            if variable==self.scans and self.pb.changing_panels:
                """When changing panels it is not desired to change scans when a scan pair is present and both the self.scans[j] and
                self.scans[panel] are in the scan pair.
                """
                if not (scanpair_present and self.scans[j] in (1,2) and self.scans[panel] in (1,2)):     
                    variable[j]=variable[panel]; panellist_change.append(j)
            else:
                variable[j]=variable[panel]; panellist_change.append(j)
            
        return variable, panellist_change
        
    def row_mode(self,panel='selected_panel',variables_change='both'): #variables_change can be one of ('products','scans','both')
        #When the number of panels is equal to 3, then self.column_mode is called, because it does not seem to be desired to have all panels showing
        #the same product, as would be the case when calling this function. This implies that for self.panels=3, it does not matter whether the row
        #or column mode is used.
        if self.pb.panels in (2,3): return self.column_mode(panel,variables_change)
        
        if variables_change in ('products','both'):
            self.products, panellist_change1=self.change_variable_in_row(self.products,panel)
        else: panellist_change1=[]
        if variables_change in ('scans','both'):
            self.scans, panellist_change2=self.change_variable_in_column(self.scans,panel)
        else: panellist_change2=[]
        panellist_change=list(np.unique(panellist_change1+panellist_change2))
            
        return panellist_change
                
    def column_mode(self,panel='selected_panel',variables_change='both'): #variables_change can be one of ('products','scans','both')
        if variables_change in ('scans','both'):
            self.scans, panellist_change1=self.change_variable_in_row(self.scans,panel)
        else: panellist_change1=[]
        if variables_change in ('products','both'):
            self.products, panellist_change2=self.change_variable_in_column(self.products,panel)
        else: panellist_change2=[]
        panellist_change=list(np.unique(panellist_change1+panellist_change2))
        
        return panellist_change

            
    def update_current(self, radar, call_ID=None, ani_iteration_end=None, date='c', time='c'):
        """Switches to the most recent datetime when all scans available.
        Is called either during automatic downloading (with call_ID = None) or when updating the end datetime of an animation.
        A date and time should be given when requesting the second-newest datetime (as is done below when the desired scans are not yet 
        available for the newest datetime).
        """
        print('update current')
        if not radar == self.selected_radar:
            #It could be that the radar changes in between emitting the signal in nlr_currentdata.py and actually calling this function.
            self.update_current_call_ID=call_ID
            return
        save_date = self.date; save_time = self.time
        
        self.pb.set_radarmarkers_data() #To assure that the correct radarmarker is colored red
        
        if gv.fileperscan_autodownload[self.dsg.data_source(radar)] and not self.pb.firstplot_performed:
            # When scans are delivered in multiple files, it is well possible that the desired scans are not yet available for the newest
            # datetime. So in that case choose the 2nd-newest datetime, because when no 1st plot has been performed it is not possible to 
            # check whether these desired scans are already available.
            _, second_newest_datetime = self.dsg.get_newest_datetimes_currentdata(self.selected_radar,self.selected_dataset)
            if not second_newest_datetime:
                return
            date, time = second_newest_datetime[:8], second_newest_datetime[-4:]
            
        self.signal_set_datetimewidgets.emit(date, time)
        
        if not self.pb.firstplot_performed:
            self.process_datetimeinput(set_data = True)
            self.update_current_scannumbers_all = self.dsg.scannumbers_all
        else:
            scanangles_all_m_before = copy.deepcopy(self.dsg.scanangles_all_m)
            retrieved_attrs = self.process_datetimeinput(set_data=False) #Determine volume attributes
            # self.update_current_scannumbers_all is used in self.ani.update_datetimes_and_perform_firstplot to update 
            # self.ani.animation_end_duplicates. It cannot simply be set equal to self.dsg.scannumbers_forduplicates, since this variable
            # doesn't get updated with set_data=False.
            self.update_current_scannumbers_all = retrieved_attrs.get('scannumbers_all', {}) if retrieved_attrs else {}
            
            availibility_scans_panels = {j:False for j in self.pb.panellist}
            for j in self.pb.panellist:
                if not retrieved_attrs:
                    # Happens when obtaining volume attributes raised an error, in which case retrieved_attrs is an empty dict
                    continue
                """Check whether data is available for the new time for all scans that are currently shown. If not, then
                the canvas is not updated.
                Checking whether data is available for the scans is not simply done by checking whether self.scans[j] is
                present in the new dictionary retrieved_attrs['scanangles_all_m'], because when the number of scans differs between the old
                and new volume, then self.scans[j] might be present, but could point to other data.
                """
                i_p = gv.i_p[self.products[j]]
                if self.products[j] in gv.plain_products and len(retrieved_attrs['scanangles_all_m'][i_p]) >= len(scanangles_all_m_before[i_p]):
                    availibility_scans_panels[j] = True
                
                if self.products[j] not in gv.plain_products:
                    # It can happen that no scan at all is available for product i_p in scanangles_all_m_before,
                    # in which case self.scans[j] is not within scanangles_all_m_before[i_p]. In that case any
                    # download of new data should lead to plotting, so set availibility_scans_panels[j] = True                    
                    if not self.scans[j] in scanangles_all_m_before[i_p]:
                        availibility_scans_panels[j] = True
                    elif any(abs(scanangles_all_m_before[i_p][self.scans[j]]-i) < 0.2 for i in retrieved_attrs['scanangles_all_m'][i_p].values()):
                        availibility_scans_panels[j] = True
                        
            if not all(availibility_scans_panels.values()):
                if date == 'c' and time == 'c':
                    _, second_newest_datetime = self.dsg.get_newest_datetimes_currentdata(self.selected_radar,self.selected_dataset)
                    if not second_newest_datetime is None:
                        date, time = second_newest_datetime[:8], second_newest_datetime[-4:]
                        if not (date == save_date and time == save_time):
                            print('use second newest time')
                            return self.update_current(radar, call_ID, ani_iteration_end, date, time)
                if call_ID is None:
                    return self.reset_datetime(save_date, save_time)
            elif call_ID is None or ani_iteration_end:
                # Doing this also at the end of an iteration ensures that when a new scan comes available it is already included in the same iteration
                self.requesting_latest_data = True
                self.pb.set_newdata(self.pb.panellist, change_datetime = save_date+save_time!=self.selected_date+self.selected_time,
                                    apply_storm_centering=True)
                self.requesting_latest_data = False
    
        self.update_current_call_ID=call_ID