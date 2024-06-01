# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import nlr_globalvars as gv
import nlr_functions as ft

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication   

import time as pytime
import numpy as np



class Animate(QThread):
    signal_process_keyboardinput=pyqtSignal(int,int,int,str,int,bool)
    signal_process_datetimeinput=pyqtSignal(int,bool) #int for call ID, bool for set_data
    signal_plot_current=pyqtSignal(str,int,bool) #str for radar, int for call ID, bool for ani_start
    signal_change_radar = pyqtSignal(str,int,bool)
    signal_move_to_next_case = pyqtSignal(int,int,str,bool)
    signal_set_panels_sttransforms_manually = pyqtSignal(np.ndarray, np.ndarray, bool, bool)
    def __init__(self, crd_class, parent=None):
        super(Animate, self).__init__(parent) 
        self.crd=crd_class
        self.dsg=self.crd.dsg
        self.gui=self.crd.gui
        self.pb=self.gui.pb
        
        self.signal_process_keyboardinput.connect(self.crd.process_keyboardinput)
        self.signal_process_datetimeinput.connect(self.crd.process_datetimeinput)
        self.signal_plot_current.connect(self.crd.plot_current)
        self.signal_change_radar.connect(self.crd.change_radar)
        self.signal_move_to_next_case.connect(self.gui.move_to_next_case)
        self.signal_set_panels_sttransforms_manually.connect(self.pb.set_panels_sttransforms_manually)
        
        self.continue_type='None' # Make it a string, to enable string iteration for checking continue_type
        self.direction=0
        self.last_ani_time=0
        self.process_datetimeinput_call_ID=0
        self.process_keyboardinput_call_ID=0
        self.plot_current_call_ID=0
        self.change_radar_call_ID=0
        self.loop_start_case_index=None
        self.move_to_next_case_call_ID=0
        #self.process_keyboardinput_call_ID must always be set to 1 when starting a sequence of calls of the function
        #self.crd.process_keyboardinput, because the condition that this call ID should be greater than 1 is used to determine whether it
        #is possible to determine the time that it takes to finish the commands that are given after finishing the call of the function
        #self.crd.process_keyboardinput.
        
        self.starting_animation=False
        self.update_animation=False #Is set to True when the animation parameters like start datetime and end datetime must be updated
        self.sleep_time=0.005
        
        """self.continue_type can be set to 'None' manually by pressing SPACE, or is set to 'None' in nlr_changedata.py, where this is done when the date
        and time have not been changed during a left/right step.
        """
        self.continue_functions={'leftright':self.continue_leftright, 'cases':self.continue_leftright,
                                 'ani':self.animate, 'ani_case':self.animate, 'ani_cases':self.animate}
                
        
                
    def change_continue_type(self,continue_type,direction): #direction=-1 for left,+1 for right
        if continue_type=='ani': 
            if not (self.continue_type == 'None' and pytime.time()-0.2 < self.last_ani_time):
                self.continue_type = 'ani' if self.continue_type == 'None' else 'None'
                self.last_ani_time = pytime.time()
            else:
                # Prevent that an animation starts when SPACE has been pressed more than once for stopping an animation or continuation to
                # the left/right.
                return
        else: 
            self.continue_type=continue_type
        self.direction=direction
        self.start()
            
    def run(self):
        if not self.continue_type == 'None':
            self.continue_functions[self.continue_type]()
            # self.continue_type has changed at this point, so run again for the new continue_type
            self.run()
        
    def continue_leftright(self):
        self.process_keyboardinput_call_ID=0
        
        while self.continue_type in ('leftright', 'cases'):
            if self.continue_type == 'leftright':
                #Do not emit a new signal before the previous one is processed.
                if not self.crd.process_keyboardinput_running and pytime.time()-self.crd.end_time>0.005 and self.process_keyboardinput_call_ID in (0,self.crd.process_keyboardinput_call_ID):
                    self.process_keyboardinput_call_ID+=1
                    self.signal_process_keyboardinput.emit(self.direction,0,0,'0',self.process_keyboardinput_call_ID,False)
            else:
                if not self.gui.move_to_next_case_running and pytime.time()-self.crd.end_time>1./self.gui.cases_looping_speed and self.move_to_next_case_call_ID in (0,self.gui.move_to_next_case_call_ID):
                    self.move_to_next_case_call_ID+=1
                    self.signal_move_to_next_case.emit(self.move_to_next_case_call_ID, 1, 'default', True)
            pytime.sleep(self.sleep_time)
                    
            
    def update_datetimes_and_perform_firstplot(self, start=False):
        use_current_case = start or self.update_animation or self.continue_type == 'ani_case'
        if start or 'ani_case' in self.continue_type:
            self.starting_animation=True
            
            if self.continue_type == 'ani':
                #Remove spaces, as the presence of a space should not lead to regarding the input as incorrect.
                show_current_animation = self.crd.check_viewing_most_recent_data()
                input_enddate = 'c' if show_current_animation else self.gui.datew.text().replace(' ','')
                input_endtime = 'c' if show_current_animation else self.gui.timew.text().replace(' ','')
            else:
                if start or getattr(self, 'case_list_name', None) != self.gui.current_case_list_name:
                    # loop_start_case_index is used for looping over a subset of cases. Besides here, it also is updated in self.gui.change_cases_loop_subset
                    self.loop_start_case_index = self.gui.get_case_index()

                self.case_list_name = self.gui.current_case_list_name
                self.case_dict = self.gui.current_case if use_current_case else self.gui.get_next_case()
                case_datetime = self.case_dict['datetime']
                
                extra_time_offsets = np.array([0, 0])
                if 'extra_datetimes' in self.case_dict:
                    datetimes = [int(j) for j in list(self.case_dict['extra_datetimes'])+[case_datetime]]
                    min_datetime, max_datetime = str(min(datetimes)), str(max(datetimes))
                    extra_time_offsets[0] = int(ft.datetimediff_s(case_datetime, min_datetime)/60)
                    extra_time_offsets[1] = int(ft.datetimediff_s(case_datetime, max_datetime)/60)
                
                self.animation_window = self.gui.cases_animation_window
                end_datetime = ft.next_datetime(case_datetime, self.animation_window[1]+extra_time_offsets[1])
                input_enddate, input_endtime = end_datetime[:8], end_datetime[-4:]
                               
            if ft.correct_datetimeinput(input_enddate,input_endtime):
                self.animation_enddate=input_enddate; self.animation_endtime=input_endtime
                self.animation_end_duplicates = self.dsg.scannumbers_forduplicates.copy()
            else: 
                self.continue_type='None'; self.starting_animation=False; self.update_animation = False
                return
                                
        
        self.radar, self.dataset = self.crd.radar, self.crd.dataset
        self.duration = self.gui.animation_duration if self.continue_type == 'ani' else\
                        np.diff(self.animation_window)[0]+np.diff(extra_time_offsets)[0]
            
        save_datetime = self.crd.selected_date+self.crd.selected_time
        self.crd.signal_set_datetimewidgets.emit(self.animation_enddate, self.animation_endtime)
        
        if self.continue_type == 'ani':
            if self.animation_enddate == 'c':
                self.plot_current_call_ID += 1
                ani_iteration_end = start
                self.signal_plot_current.emit(self.crd.selected_radar, self.plot_current_call_ID, ani_iteration_end)
                while self.crd.plot_current_call_ID != self.plot_current_call_ID:
                    pytime.sleep(0.01)
                # Update self.animation_end_duplicates in this case
                self.animation_end_duplicates = {i:len(j)-1 for i,j in self.crd.plot_current_scannumbers_all.get('z', {}).items()}
            else:
                self.process_datetimeinput_call_ID += 1
                #set_data=False, because it is not desired that data is plotted for the end date and time at this point.
                self.signal_process_datetimeinput.emit(self.process_datetimeinput_call_ID, False)
                while self.crd.process_datetimeinput_call_ID != self.process_datetimeinput_call_ID:
                    pytime.sleep(0.01)
        else:
            self.move_to_next_case_call_ID += 1
            direction = 0 if use_current_case else 1
            time_offset = str(ft.datetimediff_m(case_datetime, self.animation_enddate+self.animation_endtime))
            self.signal_move_to_next_case.emit(self.move_to_next_case_call_ID, direction, time_offset, False)
            while self.gui.move_to_next_case_call_ID != self.move_to_next_case_call_ID:
                pytime.sleep(0.01)
                     
        end_datetime = self.crd.selected_date+self.crd.selected_time
        ref_datetime = end_datetime if self.animation_enddate == 'c' else self.animation_enddate+self.animation_endtime
        start_datetime = ft.next_datetime(ref_datetime, -self.duration)
        # print(start_datetime, end_datetime, 'test')
        self.startdatetime, self.enddatetime = int(start_datetime), int(end_datetime)
            
        self.datetime = int(save_datetime) if (start or self.update_animation) and self.continue_type in ('ani', 'ani_case') else self.startdatetime
        if not self.startdatetime < self.datetime < self.enddatetime:
            self.datetime = self.startdatetime
            
        if self.continue_type == 'ani':
            self.crd.signal_set_datetimewidgets.emit(str(self.datetime)[:8], str(self.datetime)[-4:])
            self.process_datetimeinput_call_ID += 1
            #Now set_data=True
            self.signal_process_datetimeinput.emit(self.process_datetimeinput_call_ID, True)
            while self.crd.process_datetimeinput_call_ID != self.process_datetimeinput_call_ID:
                pytime.sleep(0.01)
        else:
            self.move_to_next_case_call_ID += 1
            direction = 0 # When needed the next case has already been selected above
            time_offset = str(ft.datetimediff_m(case_datetime, str(self.datetime)))
            self.signal_move_to_next_case.emit(self.move_to_next_case_call_ID, direction, time_offset, True)
            while self.gui.move_to_next_case_call_ID != self.move_to_next_case_call_ID:
                pytime.sleep(0.01)
                
        if self.datetime == self.startdatetime:
            # Update to the actual start datetime
            self.datetime = self.startdatetime = int(self.crd.date+self.crd.time)
        # print(self.startdatetime, self.enddatetime, self.datetime, self.animation_end_duplicates)
        
        if start and self.startdatetime == self.enddatetime and self.continue_type in ('ani', 'ani_case'):
            # Not when self.continue_type == 'ani_cases', since then there might be other cases with data available
            self.continue_type='None'
            
        self.update_animation = False
        self.starting_animation = False
            
        
    def check_need_update_animation(self):
        if 'ani_case' in self.continue_type:
            self.update_animation = self.case_list_name != self.gui.current_case_list_name or str(self.case_dict) != str(self.gui.current_case) or\
                                    self.animation_window != self.gui.cases_animation_window
            if self.update_animation:
                print(str(self.case_dict) != str(self.gui.current_case), self.update_animation)
        else:
            self.update_animation = (self.radar != self.crd.radar and not (self.gui.use_storm_following_view and self.gui.view_nearest_radar)) or\
                                     self.dataset != self.crd.dataset or self.duration != self.gui.animation_duration
                                                 
    def animate(self):
        try:
            self.update_animation=False
            self.calling_update_datetime_and_perform_firstplot=True
            self.update_datetimes_and_perform_firstplot(start=True)
            pytime.sleep(self.sleep_time)
            self.process_keyboardinput_call_ID=0
            self.previous_request = {'datetime':None, 'scannumbers_forduplicates':None, 'radar':None}
            
            while 'ani' in self.continue_type:
                if not self.update_animation and self.datetime==self.enddatetime:
                    try: 
                        #This is set in a try-except clausule, because it is possible that an error is generated here when self.dsg.scannumbers_all gets
                        #modified in the main thread. If this occurs, then this threads pauses a while, and continues with the next iteration of the while loop.
                        plain_products = [self.crd.products[j] for j in self.pb.panellist if self.crd.products[j] in gv.plain_products]
                        notplain_scans = [self.crd.scans[j] for j in self.pb.panellist if not self.crd.products[j] in gv.plain_products]
                        visible_scans_max_occurrences = max([len(self.dsg.scannumbers_all['z'][i]) for i in plain_products+notplain_scans])
                        vt = self.crd.volume_timestep_m #(Typical) time between volumes for radar
                        min_timestep_condition = 0. <= self.crd.desired_timestep_minutes() <= vt/visible_scans_max_occurrences
                        duplicates_condition = min_timestep_condition and any([self.dsg.scannumbers_forduplicates.get(i, 0) <
                                                                               self.animation_end_duplicates.get(i, 0) for i in plain_products+notplain_scans])
                    except Exception:
                        pytime.sleep(0.002); continue #Continue with next iteration of while loop
                #duplicates_condition ensures that when some scan is present more than once in the volume, all of those scans are plotted for the last 
                #volume of the animation.
                #self.update_animation is set to True when the enddate and time of the animation have been changed, and then 
                #self.update_datetimes_and_perform_firstplot must be called.
                
                self.check_need_update_animation()

                # Allow self.datetime to be a bit before self.startdatetime. This is done since when combining viewing nearest radar with storm-following
                # view, it can happen that a switch from radar leads to a slightly earlier datetime.
                timerange_check = ft.datetimediff_m(str(self.startdatetime), str(self.datetime)) > -10 and self.datetime < self.enddatetime
                repeat = self.datetime == self.previous_request['datetime'] and\
                         self.dsg.scannumbers_forduplicates == self.previous_request['scannumbers_forduplicates'] and\
                         self.crd.radar == self.previous_request['radar']
                if repeat:
                    print('repeating frames, restarting animation', self.datetime, self.dsg.scannumbers_forduplicates, 
                          self.crd.scans, self.crd.radar)
                if not self.update_animation and not repeat and (timerange_check or (self.datetime == self.enddatetime and duplicates_condition)):
                    running_condition=not self.crd.process_keyboardinput_running and not self.crd.process_datetimeinput_running and pytime.time()-self.crd.end_time>0.005
                    if running_condition and self.process_keyboardinput_call_ID in (0,self.crd.process_keyboardinput_call_ID):
                        self.previous_request = {'datetime':self.datetime, 'scannumbers_forduplicates':self.dsg.scannumbers_forduplicates.copy(), 'radar':self.crd.radar}
                        self.process_keyboardinput_call_ID+=1
                        # print(pytime.time(), 'emit')
                        self.signal_process_keyboardinput.emit(1,0,0,'0',self.process_keyboardinput_call_ID,False)
                    
                        # Make sure that the iteration is finished before updating self.datetime
                        while self.process_keyboardinput_call_ID != self.crd.process_keyboardinput_call_ID or\
                        self.crd.process_keyboardinput_running or self.crd.timer_process_keyboardinput.isActive():
                            pytime.sleep(self.sleep_time)
                        self.datetime=int(self.crd.date+self.crd.time)
                else:
                    self.process_keyboardinput_call_ID=0 #It is important to set it to 0, because this indicates that a new sequence of
                    #calls of self.crd.process_keyboardinput starts, and this information is used when determining
                    #self.crd.process_keyboardinput_timebetweenfunctioncalls.
                    # Also set self.crd.process_keyboardinput_call_ID to None, because otherwise with 2-frame animations you get that
                    # self.crd.process_keyboardinput_call_ID is equal to 1 at the end of the animation, which is also the value it has
                    # after executing the first iteration. And this causes the program to execute one more iteration than desired.
                    self.crd.process_keyboardinput_call_ID=None
                    
                    #Sleep some time, to hold the last frame of the animation. When the requested sleeptime is set smaller than the time that
                    #other frames are displayed, then the latter time is taken.
                    if not self.update_animation:
                        requested_sleep_time = self.gui.animation_hold_lastframe if self.continue_type != 'ani_cases' else 1/self.gui.cases_looping_speed
                        pytime.sleep(np.max([5/self.gui.maxspeed_minpsec, requested_sleep_time]))
                    #This prevents the program from always going back to the starting time when stopping an animation at the last frame.
                    if 'ani' in self.continue_type:
                        self.previous_request = {'datetime':None, 'scannumbers_forduplicates':None, 'radar':None}
                        self.update_datetimes_and_perform_firstplot()
                        
                pytime.sleep(self.sleep_time)
        except Exception as e: 
            self.continue_type='None'; print(e,'animate')
        #Always end with self.update_animation=False, as the function process_keyboardinput in nlr_changedata.py requires that info.
        self.update_animation=False