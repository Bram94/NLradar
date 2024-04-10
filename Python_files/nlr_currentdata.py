# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import nlr_globalvars as gv
import nlr_functions as ft

from PyQt5.QtCore import QThread,QObject,pyqtSignal    

import numpy as np
import os
opa=os.path.abspath
import time as pytime
import datetime as dt
import copy
import re
from urllib.request import urlopen
import requests
import warnings
import threading
import nexradaws





class AutomaticDownload(QThread): 
    """
    Automatically downloads data for a particular radar in a thread separate from the main thread. Different instances of this class should be used
    for different radars.
    """
    textbar_signal=pyqtSignal()

    def __init__(self, gui_class, radar, parent=None):
        QThread.__init__(self, parent)
        self.gui=gui_class
        self.crd=self.gui.crd
        self.dsg=self.crd.dsg
        self.dp=self.gui.dsg.dp
        self.cd=self.gui.cd
        
        self.radar=radar
        self.index=1 #1 is the index used in the class CurrentData to distinguish processes related to automatic download from those related to download 
        #of old data.
        
        """Within a download interval (typically the duration of a radar volume) multiple attempts to download new files will occur. The times at which 
        these occur are determined by a combination of self.download_interval, self.time_offsets and self.random_offset_range. Every self.download_interval
        seconds new attempts will occur at time_offset + random_offset_range seconds relative to the start of the download interval.
        """
        
        self.download_interval=gv.intervals_autodownload[gv.data_sources[self.radar]]
        self.time_offsets=gv.timeoffsets_autodownload[gv.data_sources[self.radar]]
        self.random_offset_range=10 #A random number in the range (-self.random_offset_range,self.random_offset_change) is added to 
        #self.starttime, to avoid that programs of different users start downloading all at the same time
        self.multifilevolume = gv.multifilevolume_autodownload[gv.data_sources[self.radar]]
        self.running=False
        self.timer_info=None
        
        self.textbar_signal.connect(self.gui.set_textbar)
        
        
                
    def run(self):
        self.running=True
        # Start at -2, in order to execute at least 2 download attempts before downloading becomes timer-based.
        # The reason to execute 2 instead of 1 is that the 1st attempt might take quite long, in which case new
        # files might already come available in the meantime for radars with multiple files per volume.
        self.time_offset_index=-2
        self.set_timer()
        
        self.exec_()
                         
    def set_timer(self):
        #An index of 0 indicates that a time offset given by self.time_offsets[0] is used, etc.
        if self.running:
            currenttime_s=pytime.time()
            if self.time_offset_index<0: #time_offset_index<0 at the start of AutomaticDownload
                self.starttime=currenttime_s
                # Set both times equal, since it is desired that the next iteration uses the floored reference time.
                self.floor_time = self.ceil_time = np.floor(currenttime_s/self.download_interval)*self.download_interval
            else:
                # print(self.floor_time, self.ceil_time, pytime.time(), self.time_offset_index)
                self.base_time=self.ceil_time if self.time_offset_index==0 else self.floor_time
                self.starttime=self.base_time+self.time_offsets[self.time_offset_index]
                self.starttime+=(np.random.random(1)[0]-0.5)*2*self.random_offset_range
                if self.time_offset_index == 0:
                    self.floor_time=np.floor(currenttime_s/self.download_interval)*self.download_interval
                    self.ceil_time=np.ceil(currenttime_s/self.download_interval)*self.download_interval
            
            if self.starttime-pytime.time()>0 or self.time_offset_index<0:    
                self.check_time_and_emit_timer_info(); return
            elif pytime.time() < self.ceil_time+self.time_offsets[-1]:
                # When the automatic downloading process is somehow interrupted (e.g. because the PC is put to sleep),
                # self.ceil_time could be quite outdated. In that case start again with a negative self.time_offset_index.
                if self.time_offset_index<len(self.time_offsets)-1:
                    self.time_offset_index+=1
                else:
                    self.time_offset_index=0
            else:
                self.time_offset_index = -1
            self.set_timer()
                
    def check_time_and_emit_timer_info(self):        
        time_seconds=round(self.starttime-pytime.time())
        minutes=int(np.floor(time_seconds/60)); seconds=int(np.round(np.mod(time_seconds,60)))
        self.timer_info=('0' if minutes<10 else '')+str(minutes)+':'+('0' if seconds<10 else '')+str(seconds)
        #Emit info about the time until the next attempt to find a new file.
        while time_seconds>0 and self.running:
            if self.radar == self.crd.selected_radar:
                self.textbar_signal.emit()
            pytime.sleep(1)
            
            time_seconds=round(self.starttime-pytime.time())
            minutes=int(np.floor(time_seconds/60)); seconds=int(np.round(np.mod(time_seconds,60)))
            self.timer_info=('0' if minutes<10 else '')+str(minutes)+':'+('0' if seconds<10 else '')+str(seconds)
        else:
            self.timer_info=None
            if self.running:
                #A thread is used, because otherwise the maximum recursion depth could get exceeded when repeatingly calling functions without a
                #pause inbetween.
                #threading.Timer is used here instead of QTimer, because use of QTimer causes the program to crash when calling urlopen in 
                #the function update_downloadlist of the datasource specific classes below.
                self.try_download_starter=threading.Timer(0,self.try_download)
                self.try_download_starter.start()
        
    def stop_timer(self):
        #Stop the timer that could be running
        self.timer_info=None
        self.running=False
            
    def try_download(self):  
        currenttime_s=pytime.time()
        
        file_at_disk = self.cd[self.radar].run(date='c',time='c',index=self.index)
        # print(file_at_disk, 'at this')
        if file_at_disk: #Means that the file for the most recent time for which data is available is now located at the disk.
            absolute_filetime=self.cd[self.radar].datetimes_downloadlist[self.index][1][-1]
            # print(absolute_filetime-(currenttime_s-600), 'difference')
            if absolute_filetime<=currenttime_s-600 or self.multifilevolume or True:
                #If the first is condition is met, then the most recent possible file is not present yet, so it is tried again after some time.
                #If the latter condition is met, then it is possible that not all files for the volume could be downloaded at once, so in this
                #case always the maximum number of download attempts is executed. 
                if self.time_offset_index<len(self.time_offsets)-1:
                    self.time_offset_index+=1
                else:
                    self.time_offset_index=0
            else: #New file is present at disk
                self.time_offset_index=0
        else: #Unable to obtain the most recent file, due to an error or because there are no files.
            if self.time_offset_index<len(self.time_offsets)-1:
                self.time_offset_index+=1
            else:
                self.time_offset_index=0
        self.set_timer()
                      
            
            
class DownloadOlderData(QThread):
    reset_download_widgets_signal=pyqtSignal(str)
    """Downloads data for a range of times for a particular radar, at least if the files are not already at the disk. It starts with the current date
    and time input, and then continues backward in time until a time range given by self.download_timerange_s has been spanned. 
    Different instances of this class should be used for different radars.
    """
    
    def __init__(self,gui_class,radar,parent=None):
        super(DownloadOlderData, self).__init__(parent) 
        self.gui=gui_class
        self.crd=self.gui.crd
        self.cd=self.gui.cd
        
        self.radar=radar
        self.index=2 #2 is the index used in the class CurrentData to distinguish processes related to download of old data from those related to
        #automatic download.
        
        self.download_timerange_s=0 #gets updated in the function self.gui.startstop_download_oldercurrentdata
        self.reset_download_widgets_signal.connect(self.gui.reset_download_widgets)
            
        
        
    def run(self):
        #Use the current date and time input to start with.
        timeinput=str(self.gui.timew.text())
        dateinput=self.crd.selected_date if (str(self.gui.datew.text())=='c' and not timeinput=='c') else str(self.gui.datew.text())
        if ft.correct_datetimeinput(dateinput,timeinput):
            if dateinput[0]=='c' and len(dateinput)>1: #This is the case when the dateinput has the format cYYYYMMDD.
                dateinput=dateinput[1:]
            
            if not (dateinput=='c' and timeinput=='c'):
                enddatetime=dateinput+timeinput
                startdatetime=ft.get_datetimes_from_absolutetimes(ft.get_absolutetimes_from_datetimes(enddatetime)-self.download_timerange_s)
                allowed_datetimerange=[int(startdatetime),int(enddatetime)]
            else:
                allowed_datetimerange=None
            file_at_disk=self.cd[self.radar].run(dateinput,timeinput,self.index,allowed_datetimerange)
            if not file_at_disk: 
                self.reset_download_widgets_signal.emit(self.radar); return #An index of 0 indicates that no older data gets downloaded.
        else:
            self.reset_download_widgets_signal.emit(self.radar); return
               
        date=self.cd[self.radar].date[self.index]; time=self.cd[self.radar].time[self.index]
        self.datetime_index=np.where(self.cd[self.radar].datetimes_downloadlist[self.index][0]==int(date+time))[0][0]
        self.absolutetime=ft.get_absolutetimes_from_datetimes(date+time)
        enddatetime=date+time if allowed_datetimerange is None else enddatetime
        self.end_absolutetime=ft.get_absolutetimes_from_datetimes(enddatetime)
            
        self.continue_downloading()
        
        
    def continue_downloading(self):
        while True:
            self.datetime_index-=1
            if self.datetime_index<0: break
            
            self.absolutetime=self.cd[self.radar].datetimes_downloadlist[self.index][1][self.datetime_index]
            if self.end_absolutetime-self.absolutetime>self.download_timerange_s:
                break
            else:
                datetime=str(self.cd[self.radar].datetimes_downloadlist[self.index][0][self.datetime_index])
                date=datetime[:8]; time=datetime[-4:]
                
            self.cd[self.radar].run(date,time,self.index)
        
        self.download_timerange_s=0 #Reset this value, because this value is used in the function change_radar in nlr_changedata.py
        self.reset_download_widgets_signal.emit(self.radar)
            

        

class CurrentData(QObject):
    plot_signal=pyqtSignal(str)
    textbar_signal=pyqtSignal()
    determine_list_filedatetimes_signal = pyqtSignal()
    
    """For each radar a different instance of this class is used. Two processes can run in such an instance, where the first is an automatic
    download started from the AutomaticDownload class, and the second is the download of older data started from the DownloadOlderData class. These
    processes are separated from each other by means of an index, where 1 is used for automatic downloads, and 2 for download of older data. 
    All variables that can be different for these 2 processes are put in dictionaries, with these indices as keys. 
    """

    def __init__(self,gui_class,radar,parent=None):
        super(CurrentData, self).__init__(parent)         
        self.gui=gui_class
        self.crd=self.gui.crd
        self.dsg=self.crd.dsg
        self.cds=CurrentData_DatasourceSpecific(gui_class)
        self.pb = self.gui.pb #Gets assigned in nlr.py
        
        self.radar=radar
        
        self.date={}; self.time={}
        self.file_at_disk={}
        self.download_succeeded={}
        self.urls={}; self.url_kwargs={}; self.datetimes={}; self.savenames={}; self.download_savenames={}
        self.datetimes_downloadlist={}
        
        self.cd_message=None; self.cd_message_type=None
        #self.cd_message_type is one of 'Error_info', 'Progress_info', 'Download_info'. The latter is only used for information about download progress
        self.cd_message_updating_downloadlist='Determining file to download..'
        self.cd_message_tooslow='Download too slow. Maybe you should change parameters at Settings/Download'
        self.cd_message_incorrect_apikey='Incorrect API key. Set the correct key at Settings/Download'
        self.cd_message_run_nofilespresent='No files available to download'
        
        self.isrunning = {1:False, 2:False}
        
        self.plot_signal.connect(self.crd.plot_current)
        self.textbar_signal.connect(self.gui.set_textbar)
        self.determine_list_filedatetimes_signal.connect(self.crd.determine_list_filedatetimes)



    def run(self,date,time,index,allowed_datetimerange=None):        
        """
        date can be either 'c' or a string with the format YYYYMMDD.
        time can be either 'c' or a string with the format HHMM.
        index=1 for automatic download, 2 for download of archived data.
        allowed_datetimerange should be None or a list with a startdatetime and an enddatetime in integer format.
        
        When date=='c', then it is assumed that the data that must be obtained is from the last 24 hours. If if has the format YYYMMDD, then this
        does not have to be the case, although downloading data that is more than 24 hours old is not yet implemented.
        
        This function first checks whether it is necessary to update the downloadlist, which is a list with all datetimes for which data can be 
        downloaded (dependent on what is available at the source for the particular radar). If so, it updates this list.
        If date=='c' and/or time=='c', it also determines the date and/or time for which data must be downloaded.
        If the downloadlist is empty, then it returns file_at_disk=False. If it is not empty, then it is checked for which file in the downloadlist 
        the date and time are closest to the requested date and time. This is the file that is chosen, and it is then checked whether this file is 
        already at the disk or not. If not, then its download is started.
        """
        other_index = 2 if index == 1 else 1
        
        self.date[index]=date; self.time[index]=time
        self.isrunning[index] = True
        
        self.set_time_attributes()
        if self.time[index]=='c':
            files_available=self.cds.run(self,'update_downloadlist',index)

            if files_available:
                self.date[index]=str(self.datetimes_downloadlist[index][0][-1])[:-4]
                self.time[index]=str(self.datetimes_downloadlist[index][0][-1])[-4:]
        else:
            if self.date[index]=='c':
                if int(self.time[index])>=int(self.currenttime):
                    self.date[index]=self.previousdate
                else: self.date[index]=self.currentdate
            trial_datetime=int(self.date[index]+self.time[index])

            if not index in self.datetimes_downloadlist or len(self.datetimes_downloadlist[index][0])==0:
                files_available=self.cds.run(self,'update_downloadlist',index)
            else:
                # if self.cds.source_with_partial_last_file(self.radar) or trial_datetime>self.datetimes_downloadlist[index][0][-1] or\
                if trial_datetime>self.datetimes_downloadlist[index][0][-1] or\
                (trial_datetime < self.datetimes_downloadlist[index][0][0] and self.dsg.data_source(self.radar) != 'DWD'):
                    # For DWD always the full download file list will be determined, so there's no need to update when going beyond 
                    # the first datetime in the list
                    files_available=self.cds.run(self,'update_downloadlist',index)
                else: files_available=True
        
        file_at_disk=False
        if files_available: 
            """Find the date and time of the file in the downloadlist whose date and time are closest to the requested date and time.
            If allowed_datetimerange is None then self.datetimes_downloadlist[index] is used completely, and if not then only datetimes in the range
            of allowed_datetimerange are included. If the list is 
            """
            if allowed_datetimerange is None:
                datetimes_list=self.datetimes_downloadlist[index]
            else:
                dtdl=self.datetimes_downloadlist[index]
                in_datetimerange=(dtdl[0]>=allowed_datetimerange[0]) & (dtdl[0]<=allowed_datetimerange[1])
                datetimes_list=[j[in_datetimerange] for j in dtdl]
                
            if len(datetimes_list[0])==0:
                if allowed_datetimerange is None:
                    self.show_error_info('No downloadable files found')
                else:
                    self.show_error_info('No downloadable files found between '+str(allowed_datetimerange[0])+' and '+str(allowed_datetimerange[1]))
                self.isrunning[index] = False
                return False
            
            self.date[index], self.time[index] = self.get_closest_datetime(datetimes_list, self.date[index], self.time[index])
            
            #Obtain lists with URL's and names under which files should be saved. Lists are used, because it is possible that more than
            #one file is available per volume.
            self.savenames[index], self.datetimes[index], self.urls[index], self.url_kwargs[index], self.download_savenames[index] = [], [], [], [], []
            # Prevent that both automatic download and download of older data run the same datetime
            if not (self.isrunning[other_index] and self.date[index]+self.time[index] == self.date[other_index]+self.time[other_index]):
                self.cds.run(self,'get_urls_and_savenames_downloadfile', index)
            else:
                print('dont run 1!!!!!!!!!!!!!', self.date[index]+self.time[index])
            if index == 1:
                # Include also previous datetime for automatic downloading, since radar volume for last datetime might not be complete yet
                self.date[index], self.time[index] = self.get_previous_datetime(datetimes_list, self.date[index], self.time[index])
                if not (self.isrunning[other_index] and self.date[index]+self.time[index] == self.date[other_index]+self.time[other_index]):
                    self.cds.run(self,'get_urls_and_savenames_downloadfile', index)
                else:
                    print('dont run 2!!!!!!!!!!!!!', self.date[index]+self.time[index])
            
            self.file_at_disk[index]=False; self.download_succeeded[index]=False
            save_directory_before=None
            n_downloads = 0
            # Newest datetime first, second-newest second
            newest_datetimes = list(self.dsg.get_newest_datetimes_currentdata(self.radar,self.crd.selected_dataset))
            mod_base = min([10, (len(self.savenames[index])+1)//2])
            for j in range(len(self.savenames[index])):
                save_directory = os.path.dirname(self.savenames[index][j])
                download_directory = os.path.dirname(self.download_savenames[index][j])
                if not save_directory == save_directory_before:
                    current_files_disk = os.listdir(save_directory) if os.path.exists(save_directory) else []
                    current_files_disk_download = os.listdir(download_directory) if os.path.exists(download_directory) else []
                save_directory_before = save_directory
                filename = os.path.basename(self.savenames[index][j])
                self.file_at_disk[index] = filename in current_files_disk or filename in current_files_disk_download
                download_file = not self.file_at_disk[index] or self.cds.source_with_partial_last_file(self.radar)
                if download_file:
                    self.start_download(index,j)  
                    n_downloads += 1
                    if not self.datetimes[index][j] in newest_datetimes:
                        newest_datetimes = [self.datetimes[index][j], newest_datetimes[0]]
                        if not newest_datetimes[1] is None:
                            newest_datetimes = sorted(newest_datetimes, reverse=True)
                    
                if index == 1 and ((download_file and n_downloads % mod_base == 0) or j == len(self.savenames[index])-1):
                    # When currently showing the most recent scans or when not having plot any data yet,
                    # a new file is automatically plotted after it is downloaded.
                    if not self.pb.firstplot_performed:
                        self.plot_signal.emit(self.radar)
                    elif self.crd.selected_radar == self.radar and\
                    any([j in (None, self.crd.selected_date+self.crd.selected_time) for j in newest_datetimes]):
                        """Always emit a plot_signal when these conditions are satisfied, but additional constraints for plotting are present in
                        the function self.crd.plot_current. These contraints require that all scans for which data is currently shown, are present in the new volume.
                        """
                        self.plot_signal.emit(self.radar)
                        
            file_at_disk = self.file_at_disk[index] or self.download_succeeded[index] or (
                           self.cds.source_with_partial_last_file(self.radar) and len(self.savenames[index]) == 0)
        self.isrunning[index] = False
        return file_at_disk
    
    def set_time_attributes(self):
        currenttime_s=pytime.time()
        self.currenttime=''.join(ft.get_ymdhm(currenttime_s)[3:5])
        self.currentdate=''.join(ft.get_ymdhm(currenttime_s)[:3])
        self.previousdate=''.join(ft.get_ymdhm(currenttime_s-24*3600)[:3])
        
    def get_closest_datetime(self, datetimes_list, date, time):
        if time!='c':
            timediffs=datetimes_list[1]-ft.get_absolutetimes_from_datetimes(date+time)
            index=np.argmin(np.abs(timediffs))
            closest_datetime=datetimes_list[0][index]
        else:
            closest_datetime=datetimes_list[0][-1]
        return str(closest_datetime)[:8], str(closest_datetime)[-4:]
    
    def get_previous_datetime(self, datetimes_list, date, time):
        index = np.where(datetimes_list[0] == int(date+time))[0][0]
        previous_datetime = datetimes_list[0][index-1]
        return str(previous_datetime)[:8], str(previous_datetime)[-4:]

    def display_dlProgress(self, current_size, total_size, download_datetime):
        MBs_alreadydownloaded = current_size/1048576
        timePassed = pytime.time()-self.download_start_time             
        transferRate = MBs_alreadydownloaded/timePassed*60 if timePassed>0 else 1000 # mbytes per minute
        
        time = download_datetime[-4:]
        if total_size:
            percent = int(current_size*100/total_size)
            message = '%3s%%' % percent+' ('+time+'Z, '+self.radar+')'
        else:
            message = '%.1f MB' % MBs_alreadydownloaded+' ('+time+'Z, '+self.radar+')'
        if self.cd_message_type!='Error_info' and pytime.time()-self.time_last_download_message > 1:
            # Emitting this info too often interferes with other activity in the GUI thread
            self.time_last_download_message = pytime.time()
            self.emit_info(message, 'Download_info')
                                                
        if transferRate < self.gui.minimum_downloadspeed and timePassed > 5: # download will be slow at the beginning, hence wait 5 seconds
            self.errors_tooslow+=1
            pytime.sleep(1) # let's not hammer the server
            print('tooslow')
            if self.errors_tooslow<=gv.max_download_errors_tooslow: 
                print('too slow')
                raise TooSlowException
            else: 
                self.errors_nottooslow = gv.max_download_errors_nottooslow #This is done to immediately stop downloading attempts
                print('not too slow')
                raise CustomException(self.cd_message_tooslow)

    def download_file(self,index,file_index): 
        try:            
            download_directory=os.path.dirname(self.download_savenames[index][file_index])
            if not os.path.exists(download_directory):
                os.makedirs(download_directory)
            save_directory=os.path.dirname(self.savenames[index][file_index])
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)            

            if self.url_kwargs[index] and self.url_kwargs[index][file_index]:
                response = requests.get(self.urls[index][file_index], stream=True, **self.url_kwargs[index][file_index])
            else:
                response = requests.get(self.urls[index][file_index], stream=True)
            
            if response.status_code == 416: 
                # Requested range not satisfiable, implies that file doesn't extend into requested range. Happens e.g. with
                # downloading latest NEXRAD L2 file, in which case one doesn't know beforehand whether the remote file size
                # has increased or not.
                self.emit_info(None, None)
                self.download_succeeded[index] = True # Not really succeeded, but kind of since there's nothing to download
                return
            elif not response.status_code in (200, 206):
                if len(self.url_kwargs[index]) > file_index:
                    print(self.url_kwargs[index][file_index])
                print(response.headers.get('content-length', 0), response.status_code)
                raise Exception('request error ', response.status_code, response.reason)
                
            total_bytes = int(response.headers.get('content-length', 0))
            current_bytes = 0
            
            partial_HTTP_request = len(self.url_kwargs[index]) and self.url_kwargs[index][file_index].get('headers', {}).get('Range', False)
            if partial_HTTP_request:
                with open(self.savenames[index][file_index], 'ab') as f:
                    for data in response.iter_content(10*1024):
                        current_bytes += len(data)
                        self.display_dlProgress(current_bytes, total_bytes, self.datetimes[index][file_index])
                        f.write(data)
            else:
                with open(self.download_savenames[index][file_index], 'wb') as f:
                    for data in response.iter_content(10*1024):
                        current_bytes += len(data)
                        self.display_dlProgress(current_bytes, total_bytes, self.datetimes[index][file_index])
                        f.write(data)
                os.replace(self.download_savenames[index][file_index], self.savenames[index][file_index])
            
            #Update list with datetimes of available files that is used in nlr_changedata.py
            date, time = self.datetimes[index][file_index][:8], self.datetimes[index][file_index][-4:]
            directory=self.dsg.get_directory(date, time,self.radar,self.crd.selected_dataset,dir_index = 0)
            if directory==self.crd.directory:
                #Only if the directory for the radar, date and time for which data has been downloaded, is the same as the current working directory.
                self.determine_list_filedatetimes_signal.emit()
        except PermissionError:
            print('permissiontry:', partial_HTTP_request)
            self.download_savenames[index][file_index] += '2'
            return self.download_file(index, file_index)
        except TooSlowException:
            self.show_error_info('Download too slow, repeat...')
            self.download_file(index,file_index); return
        except Exception as e:
            self.errors_nottooslow += 1
            print(e, f'download_file for {self.radar}', pytime.time())
            if self.errors_nottooslow <= gv.max_download_errors_nottooslow:
                self.download_file(index, file_index); return
            else:
                if str(e)!='None': self.show_error_info(str(e)+',download_file')
                else: self.emit_info(None,None)
                
                if os.path.exists(self.download_savenames[index][file_index]):
                    try:
                        os.remove(self.download_savenames[index][file_index])
                    except Exception: pass
                self.download_succeeded[index]=False
        else:
            self.emit_info(None,None)
            self.download_succeeded[index]=True
            
    def start_download(self,index,file_index):
        time = self.datetimes[index][file_index][-4:]
        self.emit_info(f'   % ({time}Z, {self.radar})', 'Progress_info')
        self.errors_tooslow = 0
        self.errors_nottooslow = 0
        self.download_start_time = pytime.time()
        self.time_last_download_message = pytime.time()
        self.download_file(index,file_index)
        
        
        
    def emit_info(self,info,info_type):
        self.cd_message_type=info_type; self.cd_message=info
        if self.radar == self.crd.selected_radar:
            self.textbar_signal.emit()
        
    def show_error_info(self,message):
        if message!=None:
            self.emit_info(message,'Error_info')
        #The error message is shown for a time of 2 seconds
        self.cd_message_timer=threading.Timer(2,self.set_cd_message_to_None)
        self.cd_message_timer.start()
        
    def set_cd_message_to_None(self):
        self.cd_message=None; self.cd_message_type=None
        self.cd_message_timer.cancel()
        
        
class TooSlowException(Exception):
    pass 
class CustomException(Exception):
    pass










class CurrentData_DatasourceSpecific():
    def __init__(self,gui_class,parent=None):
        self.gui=gui_class
        self.cd=None
        
        self.source_KNMI=Source_KNMI(gui_class=self.gui, cds_class=self)
        self.source_DWD=Source_DWD(gui_class=self.gui, cds_class=self)
        self.source_IMGW=Source_IMGW(gui_class=self.gui, cds_class=self)
        self.source_DMI=Source_DMI(gui_class=self.gui, cds_class=self)
        self.source_MeteoFrance=Source_MeteoFrance(gui_class=self.gui, cds_class=self)
        self.source_NWS=Source_NWS(gui_class=self.gui, cds_class=self)
        self.source_classes={'KNMI':self.source_KNMI,'DWD':self.source_DWD,'IMGW':self.source_IMGW,'DMI':self.source_DMI,
                             'Météo-France':self.source_MeteoFrance,'NWS':self.source_NWS}
        self.sources_with_partial_last_file = ['NWS']

    def source_with_partial_last_file(self, radar):
        data_source = gv.data_sources[radar]
        return data_source in self.sources_with_partial_last_file
        
    def run(self,currentdata_sourceclass,function,index):
        self.cd=currentdata_sourceclass
        data_source = gv.data_sources[self.cd.radar]
        if data_source in self.source_classes:
            self.source_classes[data_source].cd = self.cd
            return getattr(self.source_classes[data_source], function)(index)
        else:
            self.cd.show_error_info('Downloading not implemented for this radar')
            

            

    
class Source_KNMI():
    def __init__(self,gui_class,cds_class,parent=None):
        self.gui=gui_class
        self.dsg=self.gui.dsg
        self.cds=cds_class
            
    
            
    def update_downloadlist(self,index):
        self.cd.emit_info(self.cd.cd_message_updating_downloadlist,'Progress_info')
        
        url = gv.download_sourceURLs_KNMI[self.cd.radar]
        if not (self.cd.date[index] == 'c' or self.cd.time[index] == 'c'):
            startdatetime = ft.next_datetime(self.cd.date[index]+self.cd.time[index], -720)
        else:
            startdatetime = ft.next_datetime(self.cd.currentdate+self.cd.currenttime, -1440)
        startAfterFilename = 'RAD_NL'+gv.rplaces_to_ridentifiers[self.cd.radar]+'_VOL_NA_'+startdatetime+'.h5'
        
        output = None
        try:
            output = requests.get(url, headers={'Authorization': self.gui.api_keys['KNMI']['opendata']}, params = {'maxKeys': 300, 'startAfterFilename': startAfterFilename}, timeout = self.gui.networktimeout)
            files = output.json().get('files')
            filenames = [file['filename'] for file in files]
            
            if len(filenames) > 0:
                datetimes = self.dsg.get_datetimes_from_files(self.cd.radar, filenames)
                absolutetimes = ft.get_absolutetimes_from_datetimes(datetimes)
                self.cd.datetimes_downloadlist[index]=[datetimes.astype('uint64'), absolutetimes]
                self.cd.emit_info(None,None) #Remove the message given self.cd.cd_message_updating_downloadlist
                return True #files_availabe=True
            else:
                self.cd.show_error_info(self.cd.cd_message_run_nofilespresent)
                return False #files_available=False
        except Exception as e: 
            if not output is None and output.reason == 'Unauthorized':
                self.cd.show_error_info(self.cd.cd_message_incorrect_apikey)
            else:
                self.cd.show_error_info(str(e)+', update_downloadlist')
            filenames = []
            return False #files_available=False 

    def get_urls_and_savenames_downloadfile(self,index):
        """Obtain the urls and names under which the files will be saved.
        The file will first be saved under the name given in self.cd.download_savenames[index], and after the download is finished this
        will be renamed to the name given in self.cd.savenames[index].
        """
        datetime = self.cd.date[index]+self.cd.time[index]
        url = gv.download_sourceURLs_KNMI[self.cd.radar]+'/RAD_NL'+gv.rplaces_to_ridentifiers[self.cd.radar]+'_VOL_NA_'+datetime+'.h5/url'
        try:
            get_file_response = requests.get(url, headers={"Authorization": self.gui.api_keys['KNMI']['opendata']})
            self.cd.urls[index] += [get_file_response.json().get("temporaryDownloadUrl")]
        except Exception as e:
            print(e)
            self.cd.show_error_info('Could not obtain download URL')
        self.cd.datetimes[index] += [datetime]
    
        date, time = datetime[:8], datetime[-4:]
        directory=self.dsg.get_directory(date, time,self.cd.radar,None,dir_index = 0)
        filename='RAD_NL'+gv.rplaces_to_ridentifiers[self.cd.radar]+'_VOL_NA_'+datetime+'.h5'
        self.cd.savenames[index] += [opa(os.path.join(directory,filename))]
        download_directory=self.dsg.get_download_directory(self.cd.radar,None)
        self.cd.download_savenames[index] += [opa(os.path.join(download_directory,filename))]
        




class Source_DWD():
    def __init__(self,gui_class,cds_class,parent=None):
        self.gui=gui_class
        self.dsg=self.gui.dsg
        self.cds=cds_class
        
        self.files={j:{} for j in gv.radars['DWD']}
        self.files_datetimes={j:{} for j in gv.radars['DWD']}

                       
            
    def update_downloadlist(self,index):
        self.cd.emit_info(self.cd.cd_message_updating_downloadlist,'Progress_info')
                
        dtc=np.array([],dtype='int64')
        urls=self.get_urls_downloadlist()
        self.files[self.cd.radar][index]=[]

        error_received = False
        for j in urls:
            try: 
                contents = requests.get(j).content.decode('UTF-8')
            except Exception as error: 
                self.cd.show_error_info(str(error)+',update_downloadlist')
                error_received = True
                continue
        
            start_indices=[s.start() for s in re.finditer('="ras', contents)]
            end_indices=[s.end() for s in re.finditer('hd5"', contents)]
            self.files[self.cd.radar][index]+=[j+'/'+contents[start_indices[i]+2:end_indices[i]-1] for i in range(0,len(start_indices))]
                        
        self.files[self.cd.radar][index]=np.array(self.files[self.cd.radar][index])
            
        #Contains the datetimes for all files
        self.files_datetimes[self.cd.radar][index]=self.dsg.get_datetimes_from_files(self.cd.radar,self.files[self.cd.radar][index],dtype='int64',return_unique_datetimes=False)
        dtc=np.unique(self.files_datetimes[self.cd.radar][index]) #Contains the unique datetimes of the files
        
        if len(dtc)>0:
            absolutetimes=ft.get_absolutetimes_from_datetimes(dtc.astype('str'))
            self.cd.datetimes_downloadlist[index]=[dtc,absolutetimes]
            
            self.cd.emit_info(None,None) #Remove the message given self.cd.cd_message_updating_downloadlist
            files_available=True
        else:
            if not error_received:
                self.cd.show_error_info(self.cd.cd_message_run_nofilespresent)
            files_available=False
            
        return files_available
    
    def get_urls_and_savenames_downloadfile(self, index):
        """Obtain the urls and names under which the files will be saved.
        The file will first be saved under the name given in self.cd.download_savenames[index], and after the download is finished this
        will be renamed to the name given in self.cd.savenames[index].
        """
        date, time = self.cd.date[index], self.cd.time[index]
        datetimes = self.files_datetimes[self.cd.radar][index]
        select = datetimes == int(date+time)
        self.cd.urls[index] = np.append(self.cd.urls[index], self.files[self.cd.radar][index][select])
        self.cd.datetimes[index] = np.append(self.cd.datetimes[index], datetimes[select].astype('str'))
        
        file_ids, self.cd.savenames[index], self.cd.download_savenames[index] = [], [], []
        for datetime, url in zip(self.cd.datetimes[index], self.cd.urls[index]):
            date, time = datetime[:8], datetime[-4:]
            directory=self.dsg.get_directory(date, time,self.cd.radar,'Z' if 'pcp' in url else 'V',dir_index = 0)
            filename=os.path.basename(url)
            self.cd.savenames[index] += [opa(os.path.join(directory,filename))]
            download_directory=self.dsg.get_download_directory(self.cd.radar,'Z' if 'pcp' in url else 'V')
            self.cd.download_savenames[index] += [opa(os.path.join(download_directory,filename))]
            i = filename.index(date)
            file_ids += [int(filename[i-3:i-1])]
            
        elevations_map = {0:5, 1:4, 2:3, 3:2, 4:1, 5:0, 6:6, 7:7, 8:8, 9:9}
        elevations = [elevations_map[j] for j in file_ids]
        # Sort files based on elevation, in order to download the lowest scans first
        indices = np.argsort(elevations)
        self.cd.urls[index], self.cd.datetimes[index] = np.array(self.cd.urls[index])[indices], np.array(self.cd.datetimes[index])[indices]
        self.cd.savenames[index], self.cd.download_savenames[index] = np.array(self.cd.savenames[index])[indices], np.array(self.cd.download_savenames[index])[indices]
                        
    def get_urls_downloadlist(self):
        urls=[]
        basename='https://opendata.dwd.de/weather/radar/sites/'
        for i in ('sweep_pcp_z','sweep_pcp_v','sweep_vol_z','sweep_vol_v'):
            for j in ('filter_polarimetric', 'filter_simple'):
                urls.append(basename+i+'/'+gv.rplaces_to_ridentifiers[self.cd.radar]+'/hdf5/'+j)
        return urls 
    
    
    
    
    
class Source_IMGW():
    def __init__(self,gui_class,cds_class,parent=None):
        self.gui=gui_class
        self.dsg=self.gui.dsg
        self.cds=cds_class

        self.files={j:{} for j in gv.radars['IMGW']}; self.files_datetimes={j:{} for j in gv.radars['IMGW']}
        
        self.http_dirs = []
        
        
        
    def update_downloadlist(self,index):
        self.cd.emit_info(self.cd.cd_message_updating_downloadlist,'Progress_info')
                
        dtc=np.array([],dtype='int64')
        urls=self.get_urls_downloadlist()
        self.files[self.cd.radar][index]=[]

        error = None
        for j in urls:
            try: 
                #Suppress ResourceWarnings, to prevent that they are raised when the socket is not closed.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    text=urlopen(j,timeout=self.gui.networktimeout)
                    contents=str(text.read())
            except Exception as error: 
                self.cd.show_error_info(str(error)+',update_downloadlist')
                continue
        
            end_indices=[s.end()-1 for s in re.finditer('.vol<', contents)]
            start_indices=[]
            for i in range(len(end_indices)):
                part1 = contents[(0 if i==0 else end_indices[i-1]):end_indices[i]]
                start_indices += [(0 if i==0 else end_indices[i-1])+part1.rfind('>')+1]
            #The list at the DWD open data site includes files that contain the latest data, but these should not be included in the list with files,
            #as they do not contain a datetime. Further, this last entry in the list with files at the DWD open data site is in fact a map of the
            #form Oct 29 11:29 sweep_vol_v_9-latest_10410--buf.bz2 -> sweep_vol_v_9-20171029112904_10410--buf.bz2, and none of these filenames
            #should be included. For the second name this is achieved with the second condition.
            self.files[self.cd.radar][index]+=[j+'/'+contents[start_indices[i]:end_indices[i]] for i in range(0,len(start_indices))]
                        
        self.files[self.cd.radar][index]=np.array(self.files[self.cd.radar][index])
            
        #Contains the datetimes for all files
        self.files_datetimes[self.cd.radar][index]=self.dsg.get_datetimes_from_files(self.cd.radar,self.files[self.cd.radar][index],dtype='int64',return_unique_datetimes=False)
        dtc=np.unique(self.files_datetimes[self.cd.radar][index]) #Contains the unique datetimes of the files
        if len(dtc)>0:
            absolutetimes=ft.get_absolutetimes_from_datetimes(dtc.astype(str))
            self.cd.datetimes_downloadlist[index]=[dtc,absolutetimes]
            
            self.cd.emit_info(None,None) #Remove the message given self.cd.cd_message_updating_downloadlist
            files_available=True
        else:
            if error is None:
                self.cd.show_error_info(self.cd.cd_message_run_nofilespresent)
            files_available=False
            
        return files_available
    
    def get_urls_and_savenames_downloadfile(self,index):
        """Obtain the urls and names under which the files will be saved.
        The file will first be saved under the name given in self.cd.download_savenames[index], and after the download is finished this
        will be renamed to the name given in self.cd.savenames[index].
        """
        datetime = self.cd.date[index]+self.cd.time[index]
        datetimes = self.files_datetimes[self.cd.radar][index]
        select = datetimes == int(datetime)
        urls = list(self.files[self.cd.radar][index][select])

        date, time = datetime[:8], datetime[-4:]
        for url in urls:
            self.cd.urls[index] += [url]
            self.cd.datetimes[index] += [datetime]
            directory=self.dsg.get_directory(date, time,self.cd.radar,'V' if 'ZVW' in url else 'Z',dir_index = 0)
            filename=os.path.basename(url)
            self.cd.savenames[index] += [opa(os.path.join(directory,filename))]
            download_directory=self.dsg.get_download_directory(self.cd.radar,'V' if 'ZVW' in url else 'Z')
            self.cd.download_savenames[index] += [opa(os.path.join(download_directory,filename))]
                        
    def get_urls_downloadlist(self):
        if len(self.http_dirs) == 0:
            self.get_http_dirs()
        
        urls=[]
        basename='https://datavis.daneradarowe.pl/volumes/'
        for j in [i for i in self.http_dirs if i.startswith(gv.rplaces_to_ridentifiers[self.cd.radar])]:
            urls.append(basename+j)
        return urls
    
    def get_http_dirs(self):
        url = 'https://datavis.daneradarowe.pl/volumes'
        try: 
            with urlopen(url,timeout=self.gui.networktimeout) as text:
                contents=str(text.read())
        except Exception as e:
            self.cd.show_error_info(str(e)+',get_http_dirs')
            return
        
        self.http_dirs = []
        end_indices=[s.end()-2 for s in re.finditer('.vol/<', contents)]
        for i_end in end_indices:
            part1 = contents[:i_end]
            i_start = part1.rfind('>')+1
            self.http_dirs += [contents[i_start:i_end]]
    
    
    
    
    
class Source_DMI():
    def __init__(self,gui_class,cds_class,parent=None):
        self.gui=gui_class
        self.dsg=self.gui.dsg
        self.cds=cds_class
            
        self.radar_ids = {'Juvre':'60960','Sindal':'06036','Bornholm':'06194','Stevns':'06177','Virring Skanderborg':'06103'}
    
    
    
    def update_downloadlist(self,index):
        self.cd.emit_info(self.cd.cd_message_updating_downloadlist,'Progress_info')
        
        url = 'https://dmigw.govcloud.dk/v1/radardata/collections/volume/items'
        if not (self.cd.date[index] == 'c' or self.cd.time[index] == 'c'):
            startdatetime = ft.next_datetime(self.cd.date[index]+self.cd.time[index], -720)
            enddatetime = self.cd.date[index]+self.cd.time[index]
        else:
            startdatetime = ft.next_datetime(self.cd.currentdate+self.cd.currenttime, -1440)
            enddatetime = self.cd.currentdate+self.cd.currenttime
        datetime = ft.format_datetime(startdatetime,'YYYYMMDDHHMM->YYYY-MM-DDTHH:MM:SSZ')+'/'+\
                   ft.format_datetime(enddatetime,'YYYYMMDDHHMM->YYYY-MM-DDTHH:MM:SSZ')
        
        output = None
        try:
            output = requests.get(url, params = {'stationId':self.radar_ids[self.cd.radar],'datetime':datetime,'api-key':self.gui.api_keys['DMI']['radardata']}, timeout = self.gui.networktimeout)
            files = output.json()['features']
            if len(files) > 0:
                datetimes = np.array([ft.format_datetime(j['properties']['datetime'],'YYYY-MM-DDTHH:MM:SSZ->YYYYMMDDHHMM') for j in files])[::-1]
                absolutetimes = ft.get_absolutetimes_from_datetimes(datetimes)
                self.cd.datetimes_downloadlist[index]=[datetimes.astype('uint64'), absolutetimes]
                self.cd.emit_info(None,None) #Remove the message given self.cd.cd_message_updating_downloadlist
                return True #files_availabe=True
            else:
                self.cd.show_error_info(self.cd.cd_message_run_nofilespresent)
                return False #files_available=False
        except Exception as e:
            if not output is None and output.reason == 'Unauthorized':
                self.cd.show_error_info(self.cd.cd_message_incorrect_apikey)
            else:
                self.cd.show_error_info(str(e)+', update_downloadlist')
            return False #files_available=False 

    def get_urls_and_savenames_downloadfile(self,index):
        """Obtain the urls and names under which the files will be saved.
        The file will first be saved under the name given in self.cd.download_savenames[index], and after the download is finished this
        will be renamed to the name given in self.cd.savenames[index].
        """
        datetime = self.cd.date[index]+self.cd.time[index]
        self.cd.datetimes[index] += [datetime]
        
        base_url = 'https://dmigw.govcloud.dk/v1/radardata/download/'
        date, time = datetime[:8], datetime[-4:]
        # When the time ends with a 0 a long-range Z scan is performed, and when it ends with a 5 a short-range V scan is performed
        dataset = 'Z' if time[-1] == '0' else 'V'
        directory = self.dsg.get_directory(date, time, self.cd.radar, dataset, dir_index=0)
        filename = gv.rplaces_to_ridentifiers[self.cd.radar]+'_'+date+time+'.vol.h5'
        
        self.cd.urls[index] += [base_url+filename+'?api-key='+self.gui.api_keys['DMI']['radardata']]
        self.cd.savenames[index] += [directory+'/'+filename]
        download_directory=self.dsg.get_download_directory(self.cd.radar,'Z' if 'pcp' in filename else 'V')
        self.cd.download_savenames[index] += [download_directory+'/'+filename]
            
            
     


class Source_MeteoFrance():
    def __init__(self,gui_class,cds_class,parent=None):
        self.gui=gui_class
        self.dsg=self.gui.dsg
        self.cds=cds_class
        
        self.base_url = 'https://public-api.meteofrance.fr/public/DPRadar/v1'
            
        self.urls = {j:{} for j in gv.radars['Météo-France']}; self.urls_datetimes = copy.deepcopy(self.urls)
    
    
    
    def update_downloadlist(self, index):
        self.cd.emit_info(self.cd.cd_message_updating_downloadlist,'Progress_info')
        
        self.urls[self.cd.radar][index], self.urls_datetimes[self.cd.radar][index] = [], []
        
        station = gv.rplaces_to_ridentifiers[self.cd.radar]
        datetimes = []
        for i in ('PAM', 'PAG'):
            try:
                out = eval(requests.get(self.base_url+f'/stations/{station}/observations/{i}', 
                                        params = {'apikey':self.gui.api_keys['Météo-France']['radardata']}).content.decode('utf-8'))
                if out.get('code', None):
                    print(out)
                if out.get('code', None) in ('900900', '900901', '900902'):
                    return self.cd.show_error_info(self.cd.cd_message_incorrect_apikey)

                for j in out.get('links', []):
                    title = j['title']      
                    if not i in title:
                        continue
                    self.urls[self.cd.radar][index].append(j['href'])
                    self.urls_datetimes[self.cd.radar][index].append(ft.format_datetime(title[-20:], 'YYYY-MM-DDTHH:MM:SSZ->YYYYMMDDHHMM'))
            except Exception as e:
                self.cd.show_error_info(str(e)+', update_downloadlist')
                  
        # -5 minutes, since end instead of start datetimes are given for files
        datetimes = np.unique([ft.next_datetime(j, -5) for j in self.urls_datetimes[self.cd.radar][index]])
        absolutetimes = ft.get_absolutetimes_from_datetimes(datetimes)
        self.cd.datetimes_downloadlist[index] = [datetimes.astype('uint64'), absolutetimes]
        self.cd.emit_info(None, None) #Remove the message given self.cd.cd_message_updating_downloadlist
        return len(self.urls[self.cd.radar][index]) > 0 # files_availabe    

    def get_urls_and_savenames_downloadfile(self, index):
        download_directory = self.dsg.get_download_directory(self.cd.radar)
        station = gv.rplaces_to_ridentifiers[self.cd.radar]
        
        url_datetimes = self.urls_datetimes[self.cd.radar][index]
        for i,url_datetime in enumerate(url_datetimes):
            # -5 minutes, since end instead of start datetimes are given for files
            datetime = ft.next_datetime(url_datetime, -5)
            if datetime != self.cd.date[index]+self.cd.time[index]:
                continue
            
            url = self.urls[self.cd.radar][index][i]
            file_type = 'PAM' if 'PAM' in url else 'PAG'
            file_id = url[-1]
            unknown = 'LFPW' if file_type == 'PAM' else 'EODC'
            filename = f'T_{file_type}{file_id}{station}_C_{unknown}_{url_datetime}00.bufr.gz'
            # Use url_datetime for directory, since that leads to the most logical ordening of files into directories (in a way that is in agreement
            # with how you would unpack an archived dataset).
            directory = self.dsg.get_directory(url_datetime[:8], url_datetime[-4:], self.cd.radar)
            
            self.cd.datetimes[index].append(datetime)
            self.cd.urls[index].append(url)
            self.cd.url_kwargs[index].append({'params':{'apikey':self.gui.api_keys['Météo-France']['radardata']}})
            self.cd.savenames[index].append(directory+'/'+filename)
            self.cd.download_savenames[index].append(download_directory+'/'+filename)     
            
        file_ids = [j[-1] for j in self.cd.urls[index]]
        # Sort files in order of decreasing file id. This is done since the last file_id always contains the lowest scan. For other file_ids 
        # the map from scan to file_id is not always this simple, and radar-dependent.
        indices = np.argsort(file_ids)[::-1]
        for attr in ('urls', 'datetimes', 'savenames', 'download_savenames'):
            self.cd.__dict__[attr][index] = [self.cd.__dict__[attr][index][i] for i in indices]

        
     
            
            
class Source_NWS():
    def __init__(self,gui_class,cds_class,parent=None):
        self.gui=gui_class
        self.dsg=self.gui.dsg
        self.cds=cds_class
        
        self.files={j:{} for j in gv.radars['NWS']}
        self.file_sizes={j:{} for j in gv.radars['NWS']}

    
    
    def update_downloadlist(self,index):
        global conn
        self.cd.emit_info(self.cd.cd_message_updating_downloadlist,'Progress_info')
        
        timediff = 0 if self.cd.time[index] == 'c' else\
                   ft.datetimediff_s(self.cd.date[index]+self.cd.time[index], self.cd.currentdate+self.cd.currenttime)
        self.use_realtime_feed = timediff < 24*3600
        if self.use_realtime_feed:
            try:
                output = requests.get(f'https://mesonet-nexrad.agron.iastate.edu/level2/raw/{self.cd.radar}/dir.list').content
            except Exception as e:
                self.cd.show_error_info(str(e)+', update_downloadlist')
                return False
            # Exclude last file if it is very small (i.e. very new with little content)
            output = [j for i,j in enumerate(ft.list_data(output.decode('utf-8'), ' ')) if i+1 < len(output) or int(j[0]) > 100000] 
            filenames = np.array([j[1] for j in output])
            self.files[self.cd.radar][index] = np.array([self.cd.radar+'/'+j for j in filenames])
            self.file_sizes[self.cd.radar][index] = [int(j[0]) for j in output]
        else:
            startdatetime = ft.next_datetime(self.cd.date[index]+self.cd.time[index], -720)
            enddatetime = self.cd.date[index]+self.cd.time[index]
            dt_start = dt.datetime.strptime(startdatetime, '%Y%m%d%H%M')
            dt_end = dt.datetime.strptime(enddatetime, '%Y%m%d%H%M')
    
            if not 'conn' in globals():
                conn = nexradaws.NexradAwsInterface()
            try:
                scans = conn.get_avail_scans_in_range(dt_start, dt_end, self.cd.radar)
            except Exception as e:
                self.cd.show_error_info(str(e)+', update_downloadlist')
                return False
            # Not sure what MDM files contain, but at least not complete volumes.
            scans = [scan for scan in scans if not scan.filename.endswith('MDM')]
            
            self.files[self.cd.radar][index] = np.array([j.key for j in scans])
            self.file_sizes[self.cd.radar][index] = np.array([j.size for j in scans])
            filenames = np.array([j.filename for j in scans])
            
        if len(filenames):
            datetimes = self.dsg.get_datetimes_from_files(self.cd.radar, filenames)
            absolutetimes = ft.get_absolutetimes_from_datetimes(datetimes)
            self.cd.datetimes_downloadlist[index]=[datetimes.astype('uint64'), absolutetimes]
            self.cd.emit_info(None, None) #Remove the message given self.cd.cd_message_updating_downloadlist
            return True #files_availabe=True
        else:
            self.cd.show_error_info(self.cd.cd_message_run_nofilespresent)
            return False #files_available=False

    def get_urls_and_savenames_downloadfile(self,index):
        """Obtain the urls and names under which the files will be saved.
        The file will first be saved under the name given in self.cd.download_savenames[index], and after the download is finished this
        will be renamed to the name given in self.cd.savenames[index].
        """
        date, time = self.cd.date[index], self.cd.time[index]
        directory = self.dsg.get_directory(date, time, self.cd.radar)
        
        base_url = 'https://mesonet-nexrad.agron.iastate.edu/level2/raw/' if self.use_realtime_feed else 'https://noaa-nexrad-level2.s3.amazonaws.com/'
        datetimes = self.cd.datetimes_downloadlist[index][0]
        i = np.where(datetimes == int(date+time))[0][0]
        filekey = self.files[self.cd.radar][index][i]
        filename = os.path.basename(filekey)
        filename_save = filename[:4]+filename[5:] if self.use_realtime_feed else filename
        filepath_save = directory+'/'+filename_save
        
        if not os.path.exists(filepath_save) or os.path.getsize(filepath_save) < self.file_sizes[self.cd.radar][index][i] or\
        (self.use_realtime_feed and i == len(datetimes)-1): # Always download the last file when using real-time feed
            self.cd.savenames[index] += [filepath_save]
            self.cd.datetimes[index] += [date+time]
            self.cd.urls[index] += [base_url+filekey]
            if self.use_realtime_feed:
                self.cd.url_kwargs[index] += [{'headers':{'Range':f'bytes={os.path.getsize(self.cd.savenames[index][-1])}-'}} if 
                                              os.path.exists(self.cd.savenames[index][-1]) else {}]
            download_directory=self.dsg.get_download_directory(self.cd.radar)
            self.cd.download_savenames[index] += [download_directory+'/'+filename_save]