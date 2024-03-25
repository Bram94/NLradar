# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import requests
from urllib.request import urlretrieve, urlopen
import os
import numpy as np
import gpxpy.geo
import zipfile
import zlib
import netCDF4 as nc
import re
import time as pytime
import warnings
import sys
platform = sys.platform

import nlr_functions as ft
import nlr_globalvars as gv
    
    

def KNMI_hour_to_time(hour):
    return format(int(hour)*100,'04d') if not hour=='24' else '0000'

def get_closest_datetime(datetimes, datetime):
    ref_datetime = ''.join(ft.next_date_and_time(datetime[:8], datetime[-4:], 10))
    """The datetimes in datetimes contain the end datetimes of 10-minute intervals, implying that
    the average datetime for a datetime interval is 5 minutes earlier. Further, datetime refers to the
    start of the radar volume. For these 2 reasons, 10 minutes are added to self.date+self.time, 
    in order to find correctly the closest datetime.
    """
    return ft.get_closest_datetime(np.asarray(datetimes), ref_datetime)


station_ids = {'De Bilt':'260','Den Helder':'235','Herwijnen':'356'}
dir_sfcobs = gv.programdir+'/Generated_files/sfc_obs'
if not os.path.exists(dir_sfcobs):
    os.makedirs(dir_sfcobs)



class SfcObsKNMI():
    def __init__(self, gui_class):
        self.gui = gui_class
        
        self.dir_KNMI = dir_sfcobs+'/KNMI/'
        os.makedirs(self.dir_KNMI, exist_ok=True)
                    
        self.message_incorrect_apikey = 'No API key provided, so cannot retrieve surface observations. Set this key at Settings/Download'
        
        self.action_lasttimes = {'datetimeslist_current':0, 'download_data':0}
        self.datetimes_download_before = []
        
        self.url_recent_data = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/Actuele10mindataKNMIstations/versions/2/files"
        
        
        
    def list_recent_files(self):
        start_datetime = ft.next_datetime(self.date+self.time, -120)
        output = requests.get(self.url_recent_data, headers={"Authorization": self.gui.api_keys['KNMI']['sfcobs']}, 
                              params = {"maxKeys": 24, 'startAfterFilename': f'KMDS__OPER_P___10M_OBS_L2_{start_datetime}'})
        if not output is None and output.reason == 'Unauthorized':
            self.gui.set_textbar(self.message_incorrect_apikey,'red',1)  
        return output.json().get("files")
            
    def get_metadata_station(self):
        if not hasattr(self, 'f'):
            filepath = self.dir_KNMI+'STD___OPER_P___OBS_____L2.nc'
            if not os.path.exists(filepath):
                file_name = self.list_recent_files()[0].get('filename')
                get_file_response = requests.get(self.url_recent_data+'/'+file_name+'/url', headers={"Authorization": self.gui.api_keys['KNMI']['sfcobs']})
                if not get_file_response is None and get_file_response.reason == 'Unauthorized':
                    self.gui.set_textbar(self.message_incorrect_apikey,'red',1)            
                url = get_file_response.json().get("temporaryDownloadUrl")
                
                warnings.simplefilter('ignore', ResourceWarning)
                try:
                    urlretrieve(url, filepath)
                except Exception as e:
                    print(e,'get_metadata_station')
                    raise Exception('Unable to obtain meta data of the station')
                    
            # For some reason, generating this NC dataset can crash the program when done repeatedly. So do it only once, is also not needed more often
            self.f = nc.Dataset(filepath)
            
        ids = list(self.f['station'])
        index = ids.index('06'+station_ids[self.radar])
        self.obs['station'] = self.radar
        self.obs['station_fullname'] = self.radar
        self.obs['station_elev'] = int(round(float(self.f['height'][index])))
        self.obs['station_coords'] = [self.f['lat'][index], self.f['lon'][index]]
        self.obs['dist'] = 0
        
        with open('stations_KNMI.txt', 'w') as f:
            l = max([len(self.f['stationname'][i]) for i in range(len(self.f['stationname']))])
            for i in range(len(self.f['station'])):
                f.write(self.f['stationname'][i]+' '*(l-len(self.f['stationname'][i]))+'\t'+str(self.f['lat'][i])+', '+str(self.f['lon'][i]))
                if not i == len(self.f['station'])-1:
                    f.write('\n')

    def get_sfcobs_filename(self, date):
        return dir_sfcobs+'/KNMI/'+station_ids[self.radar]+'_'+date+'.txt'
    
    def get_datetimeslist_current(self):
        files = self.list_recent_files()
        filenames = [file['filename'] for file in files[1:]] #Exclude the first since it contains the metadata
        times = [j[-7:-3] for j in filenames]
        lastModified = [file['lastModified'] for file in files[1:]]
        lastModified_datetimes = [ft.format_datetime(j[:16], 'YYYY-MM-DDTHH:MM->YYYYMMDDHHMM') for j in lastModified]
        self.dtc = []
        for j in range(len(times)):
            lastModified_date, lastModified_time = lastModified_datetimes[j][:8], lastModified_datetimes[j][-4:]
            actual_date = lastModified_date if int(lastModified_time) > int(times[j]) else ft.next_date(lastModified_date, -1)
            self.dtc += [actual_date+times[j]]
        self.dtc = np.array(self.dtc, dtype = 'int64')
        
        self.action_lasttimes['datetimeslist_current'] = pytime.time()
        
    def insert_data_in_txtfile(self, date, data):
        """insert data for one date into the file corresponding to that date.
        data should be a dictionary that contains the keys 'station', 'lat', 'lon', 'elev', 'datetimes', 'ff', 'dd', 't', 'td'
        The keys 'datetimes', 'ff', 'dd', 't' and 'td' should map to lists or arrays
        """
        filepath = self.get_sfcobs_filename(date)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read()

            content_data_list = np.array([j for j in ft.list_data(content, separator=',') if j[0]==station_ids[self.radar]])
            content_data = {'datetimes':content_data_list[:,1],'dd':content_data_list[:,2],'ff':content_data_list[:,3],'t':content_data_list[:,4],'td':content_data_list[:,5]}
            remove_indices = [] #Remove datetimes that are already listed in the file, which occurs usually when 10-minute data has 
            #been saved in the file. In this case the 10-minute data is retained, because it has more significant digits than the hourly
            #data.
            for j in range(len(content_data['datetimes'])):
                if content_data['datetimes'][j] in data['datetimes']:
                    remove_indices.append(j)
            content_data = {j:np.delete(content_data[j],remove_indices) for j in content_data}
                    
            data_all = {j:np.append(content_data[j],data[j]) for j in data}
            
            sort_indices = np.argsort(data_all['datetimes'])
            data_all_sorted = {j:data_all[j][sort_indices] for j in data_all}
        else:
            data_all_sorted = data
        
        content = "STN,YYYYMMDDHHMM,   DD,   FF,    T,   TD\n"
        for j in range(len(data_all_sorted['datetimes'])):
            content += station_ids[self.radar]+','+data_all_sorted['datetimes'][j]+',  '+data_all_sorted['dd'][j]+',  '+data_all_sorted['ff'][j]+',  '+data_all_sorted['t'][j]+',  '+data_all_sorted['td'][j]+'\n'
        
        with open(filepath, 'w+') as f:
            f.write(content)
    
    def download_data(self, datetimes_download): 
        """datetimes_download specifies for which dates or datetime data should be downloaded. If self.obs_type=='Archived', it should specify a list with dates for which
        data must be downloaded, and if self.obs_type=='Current' it should specify a list with the datetime for which data must be downloaded.
        
        This function downloads surface observations from the KNMI. Data for the last 24 hours is stored by the KNMI in a separate dataset than older data, hence
        the different treatment of these cases. The temporal resolution is 10 minutes for data from the last 24 hours, and 1 hour for older data.
        After downloading the data, both datasets are however combined into one, with a format that resembles the format that the KNMI uses for archived data. 
        The difference is that there are no separate columns for the date and hour, but both are combined into one column that represents datetimes.
        """
        self.data = ''
        if self.obs_type=='Archived':
            #datetimes_download is in this case in fact a list of dates
            url = 'http://projects.knmi.nl/klimatologie/uurgegevens/getdata_uur.cgi'
            
            warnings.simplefilter('ignore', ResourceWarning)  
            # This method currently doesn't work, due to issues at KNMI
            # try:
            #     r = requests.post(url, data={"stns":station_ids[self.radar], "start":datetimes_download[0]+"01", "end":datetimes_download[-1]+"24", "vars":"DD:FF:T:TD"})
            # except Exception as e:
            #     print(e,'download_data','Archived')
            #     raise Exception('Unable to download surface observations')
            # content = r.text            
            # print(r, content)
            
            files_present = [file for file in os.listdir(self.dir_KNMI) if file.startswith('uurgeg_'+station_ids[self.radar])]
            file_that_contains_datetime = [file for file in files_present if int(file[-29:-17]) <= int(self.date+self.time) <= int(file[-16:-4])]          
            if len(file_that_contains_datetime) == 0:
                ceil_year = int(np.ceil(int(self.date[:4])/10)*10)
                start_year, end_year = str(ceil_year-9), str(ceil_year)
                year_string = start_year+'-'+end_year
                                        
                url = 'https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/uurgegevens/uurgeg_'+station_ids[self.radar]+'_'+year_string+'.zip'
                zip_filename = self.dir_KNMI+os.path.basename(url)
                urlretrieve(url, zip_filename)            
                zip_ref = zipfile.ZipFile(zip_filename, 'r')
                zip_ref.extractall(self.dir_KNMI)
                zip_ref.close()
                os.remove(zip_filename)
                
                orig_filename = self.dir_KNMI+'uurgeg_'+station_ids[self.radar]+'_'+year_string+'.txt'
                with open(orig_filename, 'r') as f:
                    content = f.read()
                    s = slice(-200, None)                    
                    data_list = np.array([i for i in ft.list_data(content[s], separator=',') if i[0]==station_ids[self.radar]])                    
                start_datetime = start_year+'01010000'
                end_datetime = ft.next_date(data_list[-1][1], 1)+'0000'
                txt_filename = self.dir_KNMI+'uurgeg_'+station_ids[self.radar]+'_'+start_datetime+'_'+end_datetime+'.txt'
                os.rename(orig_filename, txt_filename)
            else:
                txt_filename = self.dir_KNMI+file_that_contains_datetime[0]
                
            with open(txt_filename, 'r') as f:
                content = f.read()
                length = len(content)
                index = content.find(self.date)
                #Listing all data consumes too much self.time
                if index!=-1:
                    s = slice(max([index-1000,0]), min([index+11000,length]))
                else:
                    s = slice(-200, None)
                    #Obtain data for the last datetime

            data_list = np.array([i for i in ft.list_data(content[s], separator=',') if i[0]==station_ids[self.radar]])
            #important: For hour=='24' does KNMI_hour_to_time return '0000', which is a time of the next day. The date should therefore be updated to the next day, which
            #is done below.
            for j in datetimes_download:
                data_list_j = np.array([i for i in data_list if i[1]==j])
                if len(data_list_j)==0: continue
                dd = data_list_j[:, 3]; dd[dd == '990'] = '' #Don't plot observations when direction = 'veranderlijk'
                ff = data_list_j[:, 5]
                select = ff != ''; ff[select] = (ff[select].astype('float')/10).astype('str')
                t = data_list_j[:, 7]
                select = t != ''; t[select] = (t[select].astype('float')/10).astype('str')
                td = data_list_j[:, 9]
                select = td != ''; td[select] = (td[select].astype('float')/10).astype('str')
                datetimes = [''.join(ft.next_date_and_time(i[1], KNMI_hour_to_time(i[2]), 1440 if i[2]=='24' else 0)) for i in data_list_j]
                data = {'datetimes':datetimes, 'dd':dd,'ff':ff,'t':t,'td':td}           
                self.insert_data_in_txtfile(j, data)               
        else:                
            datetime = datetimes_download[0]
            filename = 'KMDS__OPER_P___10M_OBS_L2_'+datetime+'.nc'
            get_file_response = requests.get(self.url_recent_data+'/'+filename+'/url', headers={"Authorization": self.gui.api_keys['KNMI']['sfcobs']})
            if not get_file_response is None and get_file_response.reason == 'Unauthorized':
                self.gui.set_textbar(self.message_incorrect_apikey,'red',1)
                
            url = get_file_response.json().get("temporaryDownloadUrl")         
            filepath = self.dir_KNMI+filename
            
            warnings.simplefilter('ignore', ResourceWarning)
            try:
                urlretrieve(url, filepath)
            except Exception as e:
                print(e,'download_data', 'Current')
                raise Exception('Unable to download surface observations')
            
            f = nc.Dataset(filepath,'r+')
            
            stations = list(f['station'])
            index = stations.index('06'+station_ids[self.radar])
            
            data = {}
            abstime = f['time'][0].item()
            #The reference time used by the KNMI is apparently January 1, 1950 instead of January 1, 1970    
            data = {'datetimes':[str(int(int(ft.get_datetimes_from_absolutetimes(abstime))-2e9))],
                    'ff': [str(f['ff'][index][0])], 'dd':[str(f['dd'][index][0])], 't':[str(f['ta'][index][0])], 'td':[str(f['td'][index][0])]}
            # f['xx'][index] returns a list instead of a scalar!
            
            date = str(data['datetimes'][0][:8]); time = data['datetimes'][0][-4:]
            if time=='0000':
                #Important: If the time is '0000', it means that the data is from the previous day, from between 23:50 and 00:00. For this reason the data is put
                #in the files for the previous date, instead of the file for date.
                date = ft.next_date_and_time(date, time, -1440)[0]
            self.insert_data_in_txtfile(date, data)
            
            f.close()
            os.remove(filepath) #Remove the netcdf file with observations, as it is not used anymore
        self.action_lasttimes['download_data'] = pytime.time()
            
    def get_datetimes_to_download(self, datetimes, required_datetimes):
        if self.obs_type=='Archived':
            return np.unique([j[:8] for j in required_datetimes if not j in datetimes]) #Return dates instead of datetimes
        else:
            return required_datetimes if not required_datetimes[0] in datetimes else [] #Return the required datetime
                                        
    def import_data(self):
        if self.obs_type=='Current' and (not hasattr(self, 'dtc') or int(self.date+self.time) > self.dtc[-1]) and\
        pytime.time()-self.action_lasttimes['datetimeslist_current']>60:
            self.get_datetimeslist_current()

        previousdate = ft.next_date_and_time(self.date,'0000',-1440)[0] #Also request the previous and next date, for in case self.time is
        #at the beginning or end of a day
        nextdate = ft.next_date_and_time(self.date,'0000',1440)[0]
        dates = (previousdate, self.date, nextdate) if self.obs_type=='Archived' else (previousdate, self.date)
        
        data = ''
        for j in dates:
            try:
                with open(self.get_sfcobs_filename(j), 'r') as f:
                    data += f.read()
            except Exception:
                continue
            
        if len(data)>0:
            data_list = np.array([j for j in ft.list_data(data, separator=',') if j[0]==station_ids[self.radar]])
            datetimes = data_list[:,1]
        else:
            datetimes = []
        
        if self.obs_type=='Archived':
            required_datetimes = []
            current_date = ft.get_datetimes_from_absolutetimes(pytime.time())[:8]
            for i in dates:
                if i==current_date: continue #Archived observations are usually available up to yesterday
                for j in range(1,24):
                    required_datetimes += [i+KNMI_hour_to_time(str(j))]
        else:
            if len(self.dtc)>0:
                required_datetimes = [str(get_closest_datetime(self.dtc.astype(str), self.date+self.time)[1])]
            else: 
                #If len(self.dtc)==0, then it was impossible to obtain a list with datetimes for current data at the KNMI
                #server, but it might still be the case that some surface observations are already present at the disk.
                #For this reason no exception is raised.
                required_datetimes = []
        
        """First check whether data is already present at the disk for the required dates and times, and download data for the dates and times for which this is not 
        the case.
        """
        datetimes_download = []
        if len(required_datetimes)>0:
            datetimes_download = self.get_datetimes_to_download(datetimes, required_datetimes)
            if len(datetimes_download)>0 and (list(datetimes_download)!=self.datetimes_download_before or 
            pytime.time()-self.action_lasttimes['download_data']>60):
                self.download_data(datetimes_download)
        self.datetimes_download_before = list(datetimes_download)
                              
        self.data = ''
        for j in dates: #dates is defined above
            try:
                with open(self.get_sfcobs_filename(j), 'r') as f:
                    self.data += f.read()
            except Exception: continue
        
        if len(self.data)==0:
            raise Exception('Unable to obtain surface observations')
                    
    def get_obs_type(self):
        abstime = ft.get_absolutetimes_from_datetimes(self.date+self.time)
        current_abstime = pytime.time()
        self.obs_type = 'Current' if current_abstime - abstime-600 < 60*24*3600 else 'Archived'
            
    def get_sfcobs_at_time(self,radar, date, time):
        self.radar, self.date, self.time = radar, date, time
        self.obs = {}
        
        t = pytime.time()
        # print(pytime.time()-t, 't1')
        self.get_metadata_station()
        # print(pytime.time()-t, 't2')
        self.get_obs_type()
        # print(pytime.time()-t, 't3')
        self.import_data()
        # print(pytime.time()-t, 't4')
            
        data_list = np.array(ft.list_data(self.data, separator=','))
        data_list_data = data_list[data_list[:,0]==station_ids[self.radar]]
        datetimes = data_list_data[:,1]

        index, datetime = get_closest_datetime(datetimes, self.date+self.time)
        if abs(ft.datetimediff_s(datetime, self.date+self.time))>3600:
            raise Exception('Nearest datetime with available surface observations is too far in time from the selected datetime')
        
        data_time = data_list_data[index]
        param_to_index_map = {'FF': 3, 'DD': 2, 'T': 4, 'Td': 5}
        for param in param_to_index_map:
            index = param_to_index_map[param]
            if not data_time[index] == '':
                self.obs[param] = float(data_time[index])
            elif param == 'DD' and 'FF' in self.obs:
                if self.obs['FF'] < 1.2:
                    self.obs['DD'] = self.obs['FF'] = 0
                else:
                    raise Exception('Wind too variable')
        self.obs['datetime'] = datetime
        
        self.update_data_availability = False
        return self.obs
    
    
    
class SfcObsKMI():
    def __init__(self, gui_class):
        self.gui = gui_class
        
        
    def get_sfcobs_at_time(self, radar, date, time):    
        params = {'service':'WFS', 'version':'2.0.0', 'request':'GetFeature', 'typenames':'synop:synop_station', 'outputformat':'csv'}
        meta = requests.get('https://opendata.meteo.be/service/ows', params=params).text
        
        stns_list = ft.list_data(meta)[1:]
        stns_coords = [ft.string_to_list(j[2].replace('POINT (', '').replace(')', ''), ' ') for j in stns_list]
        
        ref_coords = gv.radarcoords[radar]
        distances = [gpxpy.geo.haversine_distance(ref_coords[0], ref_coords[1], float(i[0]), float(i[1])) for i in stns_coords]
        argmin_dist = np.argmin(distances)
        min_dist = distances[argmin_dist]
        lat, lon = stns_coords[argmin_dist]
        
        obs = {}
        obs['station'] = stns_list[argmin_dist][1]
        obs['station_fullname'] = stns_list[argmin_dist][4].title()
        obs['station_elev'] = round(int(float(stns_list[argmin_dist][3])))
        obs['station_coords'] = [float(lat), float(lon)]
        obs['dist'] = min_dist
        
        dt_start = ft.format_datetime(ft.next_datetime(date+time, -60), 'YYYYMMDDHHMM->YYYY-MM-DD HH:MM')
        dt_end = ft.format_datetime(ft.next_datetime(date+time, 60), 'YYYYMMDDHHMM->YYYY-MM-DD HH:MM')
        
        params = {'service':'WFS', 'version':'2.0.0', 'request':'GetFeature', 'typenames':'synop:synop_data', 'outputformat':'csv', 
                  'CQL_FILTER':"((BBOX(the_geom,"+lon+","+lat+","+lon+","+lat+", 'EPSG:4326')) AND (timestamp >= '"+dt_start+":00.000' AND timestamp <= '"+dt_end+":00.000'))",
                  'sortby':'timestamp'}
        data = requests.get('https://opendata.meteo.be/service/ows', params=params).text
        
        data = np.array(ft.list_data(data))[1:]
        datetimes = np.array([ft.format_datetime(j[:16]) for j in data[:, 3]])
        
        index, datetime = get_closest_datetime(datetimes, date+time)
        if abs(ft.datetimediff_s(datetime, date+time))>3600:
            raise Exception('Nearest datetime with available surface observations is too far in time from the selected datetime')
        
        data_time = data[index]
        param_to_index_map = {'FF': 10, 'DD': 12, 'T': 6}
        for param in param_to_index_map:
            index = param_to_index_map[param]
            if not data_time[index] == '':
                obs[param] = float(data_time[index])
            elif param == 'DD' and 'FF' in obs:
                if obs['FF'] < 1.2:
                    obs['DD'] = obs['FF'] = 0
                else:
                    raise Exception('Wind too variable')
        # obs['Td'] = None
        obs['datetime'] = datetime
        
        with open('stations_KMI.txt', 'w') as f:
            l = max([len(j[4]) for j in stns_list])
            for i,j in enumerate(stns_list):
                f.write(j[4]+' '*(l-len(j[4]))+'\t'+str(stns_coords[i][0])+', '+str(stns_coords[i][1]))
                if not i == len(stns_list)-1:
                    f.write('\n')
        
        return obs
    
    

class SfcObsMETAR():
    def __init__(self, gui_class):
        self.gui = gui_class
        
        self.dir_METAR = dir_sfcobs+'/METAR/'
        os.makedirs(self.dir_METAR, exist_ok=True)
        
        with open(gv.programdir+'/Input_files/metar_stations.txt', 'r', encoding='utf-8') as f:
            self.stations_list = ft.list_data(f.read(), '\t')
            
        self.time_last_request_current = {}
        
        
        
    def get_sfcobs_at_time(self, radar, date, time, skip_stations=['KECP']):
        lat, lon = gv.radarcoords[radar]
        stations_list = [i for i in self.stations_list if not i[0] in skip_stations]
        distances = [gpxpy.geo.haversine_distance(lat, lon, float(i[1]), float(i[2])) for i in stations_list]
        argmin_dist = np.argmin(distances)
        station_meta = stations_list[argmin_dist]
        station = station_meta[0]
        
        obs = {}
        obs['station'] = station
        obs['station_fullname'] = station_meta[4]
        obs['station_elev'] = int(round(float(station_meta[3])))
        obs['station_coords'] = [float(station_meta[1]), float(station_meta[2])]
        obs['dist'] = distances[argmin_dist]
        
        previous_date, next_date = ft.next_date(date, -1), ft.next_date(date, 1)        
        limit_datetime = ''.join(ft.get_ymdhm(pytime.time()-3600))
        request_datetime = ft.round_datetime(ft.next_datetime(date+time, 720), 60)
        if int(request_datetime) < int(limit_datetime):
            if 200 <= int(time) <= 2200:
                request_dates = [date]
            else:
                request_dates = [previous_date, date] if int(time) < 200 else [date, next_date]
        else:
            request_dates = ['current']
        
        data, datetimes = [], []
        for _date in request_dates:
            sfcobs_filename = f'{self.dir_METAR}{station}_{_date}.txt'
            new_request_needed = not os.path.exists(sfcobs_filename)
            if not new_request_needed:
                with open(sfcobs_filename, 'r') as f:
                    text = f.read()
                    if _date == 'current':
                        last_datetime = text[:text.index('\n')]
                        # Do a new request when requested date+time more recent than last metar datetime, with additional requirement
                        # that previous request was at least 1 minute ago.
                        if int(last_datetime) < int(date+time) and pytime.time()-self.time_last_request_current.get(station, 0) > 60:
                            new_request_needed = True
            
            if new_request_needed:
                request_date = _date if _date == 'current' else ft.next_date(_date, 1)                
                request_hour = _date if _date == 'current' else '00'
                params = {'TYPE':'metar', 'DATE':request_date, 'HOUR':request_hour, 'UNITS':'M', 'STATION':station}
                text = requests.get('http://weather.uwyo.edu/cgi-bin/wyowx.fcgi', params=params).text
                i1, i2 = text.index('<PRE>\n')+6, text.index('</PRE>')
                text = text[i1:i2]
                
            _data = ft.list_data(text, ' ')[::-1]
            # Some stations like KPAH include some weird lines that start with PAH instead of KPAH
            _data = [j for j in _data if j[0] == station]
            data += _data
            date_start, date_end = ft.next_date(previous_date, -1), ft.next_date(next_date, 1)
            date_map = {j[-2:]:j for j in ft.get_dates_in_range(date_start, date_end, 1)}
            datetimes += [date_map[j[1][:2]]+j[1][2:6] for j in _data]

            if new_request_needed:
                if _date == 'current':
                    # Add the last available datetime to the beginning of the file
                    text = datetimes[-1]+'\n'+text
                    self.time_last_request_current[station] = pytime.time()
                with open(sfcobs_filename, 'w') as f:
                    f.write(text)
        # This is done after writing the text to a file, since when metars are available for the station, but without winds, it's
        # still desired to save these metars to a file. As this prevents unnecessary follow-up requests for this station.
        indices_keep = [i for i,j in enumerate(data) if any(len(k) > 6 and k[-2:] == 'KT' for k in j)]
        data = [data[i] for i in indices_keep]
        datetimes = [datetimes[i] for i in indices_keep]
                
        index = None
        if datetimes:
            index, datetime = get_closest_datetime(datetimes, date+time)
        if index is None or abs(ft.datetimediff_s(datetime, date+time)) > 3600:
            print('exclude station', station)
            if len(skip_stations) < 2:
                return self.get_sfcobs_at_time(radar, date, time, skip_stations+[station])
            raise Exception('Nearest datetime with available surface observations is too far in time from the selected datetime')
        print('sfc station', station)
            
        data = data[index]
        obs['datetime'] = datetime
        i = [i for i,j in enumerate(data) if len(j) > 6 and j[-2:] == 'KT'][0]
        obs['FF'] = float(data[i][3:5])/gv.scale_factors_velocities['kts']
        DD = data[i][:3]
        if DD == 'VRB':
            if obs['FF'] < 1.2:
                obs['DD'] = obs['FF'] = 0
            else:
                raise Exception('Wind too variable')
        else:
            obs['DD'] = int(DD)
        try:
            i = [i for i,j in enumerate(data) if 4 < len(j) < 8 and j.find('/') in (2, 3)][0]
            i_slash = data[i].index('/')
            obs['T'] = int(data[i][:i_slash]) if not data[i][0] == 'M' else -int(data[i][1:i_slash])
            obs['Td'] = int(data[i][i_slash+1:]) if not data[i][i_slash+1] == 'M' else -int(data[i][i_slash+2:])
        except Exception as e:
            print(e, 'setting_T_Td')
            pass
        
        return obs
            
        
class SfcObsDWD():
    def __init__(self, gui_class):
        self.gui = gui_class
        
        self.dir_DWD = dir_sfcobs+'/DWD/'
        os.makedirs(self.dir_DWD, exist_ok=True)
        
        self.data = {}
        self.data_filenames = {'wind': None, 'temp': None}
        self.data_station = {'wind': None, 'temp': None}
        self.data_list = {}
        self.data_list_filenames = {'wind': None, 'temp': None}
        self.data_list_dates = {'wind': None, 'temp': None}
        
        self.action_lasttimes = {'download_data':{'wind':{},'temp':{}}}
        
        self.obs_types = ['wind','temp']
        self.exclude_station_ids = {'wind':[],'temp':[]}



    def get_stns_lists(self):
        urls = {'wind': 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/wind/recent/zehn_min_ff_Beschreibung_Stationen.txt',
                'temp': 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/recent/zehn_min_tu_Beschreibung_Stationen.txt'} 
        
        current_date = ''.join(ft.get_ymdhm(pytime.time())[:3])
        for j in self.obs_types:
            filepath = self.dir_DWD+'/stationslist_'+j+'_'+current_date+'.txt'  
            if not os.path.exists(filepath):                    
                warnings.simplefilter('ignore', ResourceWarning)
                try:
                    urlretrieve(urls[j], filepath)
                    #Remove any old file if it exists
                    try:
                        old_file = [i for i in os.listdir(self.dir_DWD) if 'stationslist_'+j in i and not current_date in i][0]
                        os.remove(self.dir_DWD+'/'+old_file)
                    except Exception:
                        pass
                except Exception as e:
                    print(e,'get_stns_lists')
                    try:
                        #Check whether an older file exists, and if so, use that. This should keep using surface observations possible
                        #when no internet connection is available.
                        old_file = [i for i in os.listdir(self.dir_DWD) if 'stationslist_'+j in i and not current_date in i][0]
                        filepath = self.dir_DWD+'/'+old_file
                    except Exception:
                        raise Exception('Unable to obtain meta data of the stations')
                
            with open(filepath,'rb') as f:
                data = f.read().decode('latin-1')
            
            self.stns_list[j] = ft.list_data(data, separator=' ')[2:]
            
            
    def include_station(self, i, j):
        return (not i[0] in self.exclude_station_ids[j] and not 
            (int(ft.next_date(i[2], 2)) < int(self.date) or int(i[1]) > int(self.date)) and not
            abs(int(i[3]) - gv.radar_elevations[self.radar]) > 500) #The elevation difference between station and radar is not allowed
            #to be greater than 500 m.
            #And ft.next_date(i[2], 2) instead of i[2], because I've observed that sometimes the end date specified in the stations list
            #is lagging a bit behind. I have observed for example that the end date in the stations list was yesterday, while in fact
            #data for today was also already available. Further, I shift by 2 days instead of one day to add some safety margin.
                        
    def get_nearest_station(self):
        """Important: There are different station lists for temperature and wind data, implying that temperature
        data is not for all wind stations available. As wind is the most important parameter, this function determines the closest
        wind station. If temperature is available for that station or for another station within 3 km, then it is included.
        If not, then no temperature data gets obtained. In that case self.station will only contain 'wind' as key     
        """
        self.radar_coords = gv.radarcoords[self.radar]
        
        for j in self.obs_types: #'wind' must be the first key, as the wind parameters are the most important ones
            #to obtain. For temperature, the station that is closest to the wind station is chosen
            stns_list = [i for i in self.stns_list[j] if self.include_station(i, j)]
            # print(j, len(stns_list), len(self.stns_list[j]), self.exclude_station_ids[j])
            
            stns_ids = [i[0] for i in stns_list]
            stns_coords = [[float(i[4]),float(i[5])] for i in stns_list]
            ref_coords = self.radar_coords if j=='wind' else self.obs['station_coords']
            distances = [gpxpy.geo.haversine_distance(ref_coords[0],ref_coords[1],i[0],i[1]) for i in stns_coords]
                    
            argmin_dist = np.argmin(distances)
            min_dist = distances[argmin_dist]
            if j=='wind' or min_dist<3000:
                self.station[j] = format(int(stns_ids[argmin_dist]), '05d')
            if j=='wind':
                self.obs['station'] = self.station[j]
                self.obs['station_fullname'] = stns_list[argmin_dist][6]
                self.obs['station_elev'] = int(round(float(stns_list[argmin_dist][3])))
                self.obs['station_coords'] = stns_coords[argmin_dist]
                self.obs['dist'] = min_dist
            
                
    def download_data(self, obs_type):
        """There are different datasets for recent and 'now' data, where the 'now' dataset contains only data for the current date. Both datasets have the same format,
        and get downloaded and combined into one.
        """
        j = obs_type
        
        timediff = pytime.time() - ft.get_absolutetimes_from_datetimes(self.date+self.time)
        
        if timediff > 365*24*3600:
            download_datasets = ['historical']
        elif timediff > 25*3600:
            download_datasets = ['recent']
        else:
            download_datasets = ['recent', 'now']
            
        
        txt_filenames = {}
        for dataset in download_datasets:
            if dataset == 'historical': 
                #Obtain the desired URLs for historical data. 
                dataset_name = 'wind' if j == 'wind' else 'air_temperature'
                _ = requests.get('https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/'+dataset_name+'/historical/')
                text = _._content.decode('utf-8')
                var_name = 'wind' if j == 'wind' else 'TU'
                s = '<a href="10minutenwerte_'+var_name+'_'+self.station[j]
                i1 = text.index(s); i2 = text.rindex(s); i3 = text[i2:].index('&gt;</a>')+i2
                t = text[i1:i3]
                l = ft.string_to_list(t, '\r\n')
                for line in l:
                    index = line.index(self.station[j]+'_')+6
                    startdate = line[index:index+8]
                    enddate = line[index+9:index+17]
                    if int(startdate) <= int(self.date) and int(self.date) <= int(enddate):
                        filename = line[9:(61 if j == 'wind' else 59)]
                        break
                url = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/'+dataset_name+'/historical/'+filename
            else:
                urls = {'recent': {
                         #Using self.station[j] is a bit of a hack, since the correct URL is only obtained for the requested j (obs_type). But that is enough, since the other URLs are not used.
                        'wind': 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/wind/recent/10minutenwerte_wind_'+self.station[j]+'_akt.zip',
                        'temp': 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/recent/10minutenwerte_TU_'+self.station[j]+'_akt.zip'
                        },
                        'now': {
                        'wind': 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/wind/now/10minutenwerte_wind_'+self.station[j]+'_now.zip',
                        'temp': 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/now/10minutenwerte_TU_'+self.station[j]+'_now.zip'
                        }}
                url = urls[dataset][j]
                

            zip_filename = self.dir_DWD+os.path.basename(url)
            
            warnings.simplefilter('ignore', ResourceWarning)
            try:
                urlretrieve(url, zip_filename)
            except Exception as e:
                print(e, 'download_data', dataset, j)
                continue
            
            zip_ref = zipfile.ZipFile(zip_filename, 'r')
            txt_filenames[dataset] = self.dir_DWD+zip_ref.namelist()[0]
            zip_ref.extractall(self.dir_DWD)
            zip_ref.close()
            os.remove(zip_filename)
            
        #Combine the 'recent' and 'now' files
        if all([dataset in txt_filenames for dataset in ('recent','now')]):
            with open(txt_filenames['recent'], 'a') as f1:
                with open(txt_filenames['now'], 'r') as f2:
                    data2 = f2.read()
                    eor_index = data2.index('eor')
                    f1.write(data2[eor_index+4:])
            os.remove(txt_filenames['now'])
            del txt_filenames['now']

            self.action_lasttimes['download_data'][j][self.station[j]] = pytime.time()
                
            
        dataset = list(txt_filenames)[0] #There can be at most one dataset left in txt_filenames at this point
        filename = os.path.basename(txt_filenames[dataset])
        if dataset == 'historical':
            start_datetime = filename[20:28]+'0000'
            end_datetime = filename[29:37]+'2350'
        elif dataset == 'recent':
            start_datetime = filename[20:28]+'0000'
            with open(txt_filenames[dataset],'r') as f:
                data = f.read()
            data_list = [i for i in ft.list_data(data[-200:], separator=';') if i[0] in self.station[j] and i[-1]=='eor']
            end_datetime = data_list[-1][1]
        new_filename = j+'_'+self.station[j]+'_'+start_datetime+'_'+end_datetime+'.txt'
        try:
            os.rename(txt_filenames[dataset], self.dir_DWD+'/'+new_filename)
        except FileExistsError:
            os.remove(txt_filenames[dataset])
             
        #Remove outdated existing files
        files_station = [i for i in self.get_sfcobs_files_station(j)]
        start_datetimes = [int(file[11:23]) for file in files_station]
        end_datetimes = [int(file[24:36]) for file in files_station]
        for i in range(len(files_station)):
            if not files_station[i] == new_filename and end_datetimes[i] <= int(end_datetime) and\
            any([start_datetimes[j] <= start_datetimes[i] <= end_datetimes[j] for j in range(len(end_datetimes)) if not end_datetimes[j] == end_datetimes[i]]):
                #Remove a file if all its observations are contained within other files
                os.remove(self.dir_DWD+'/'+files_station[i])
                    
        
    def import_data(self, obs_type):
        j = obs_type
    
        file_available, filename = self.check_presence_sfcobs_in_dir_and_if_so_return_filename(j)
        if not filename is None and filename == self.data_filenames[j]: #No update needed
            return
        
        if file_available:
            with open(filename,'r') as f:
                self.data[j] = f.read()
        else:
            self.download_data(j)
            file_available, filename = self.check_presence_sfcobs_in_dir_and_if_so_return_filename(j)
            with open(filename,'r') as f:
                self.data[j] = f.read()
        self.data_filenames[j] = filename
        self.data_station[j] = self.station[j]
                
    def get_sfcobs_files_station(self, obs_type):
        return sorted([i for i in os.listdir(self.dir_DWD) if self.station[obs_type] in i and i.startswith(obs_type)])

    def check_presence_sfcobs_in_dir_and_if_so_return_filename(self, obs_type):
        files_station = self.get_sfcobs_files_station(obs_type)
        start_datetimes = [int(file[11:23]) for file in files_station]
        end_datetimes = [int(file[24:36]) for file in files_station]
        for i in range(len(files_station)):
            start_datetime, end_datetime = start_datetimes[i], end_datetimes[i]
            if start_datetime <= int(self.date+self.time) <= end_datetime:
                return True, self.dir_DWD+'/'+files_station[i]
            
        #If the above condition is not satisfied, then check whether the requested datetime is maybe beyond the range of datetimes
        #for which surface obs are currently available. In that case the most recent file is returned.
        #The value of 25*3600 is also used in download_data
        if pytime.time() - ft.get_absolutetimes_from_datetimes(self.date+self.time) < 25*3600 and\
        pytime.time() - self.action_lasttimes['download_data'][obs_type][self.station[obs_type]] < 60:
            return True, self.dir_DWD+'/'+files_station[np.argmax(end_datetimes)]
        else:
            return False, None
    
    
    def get_sfcobs_at_time(self,radar, date, time, start=True):            
        self.radar, self.date, self.time = radar, date, time
        
        floor_time = format(int(np.floor(int(self.time)/10)*10),'04d')
        desired_datetime = ''.join(ft.next_date_and_time(self.date,floor_time,10))
        desired_abstime = ft.get_absolutetimes_from_datetimes(desired_datetime)
        
        self.stns_list = {}
        self.station = {}
        self.obs = {}
        
        
        if start:
            self.exclude_station_ids = {'wind': [], 'temp': []}
        self.get_stns_lists()
        n = 0
        while n==0 or not all([j in self.data for j in self.station]):
            n+=1
            
            self.get_nearest_station()
            for j in self.station:
                # print(n, self.station[j], j, list(self.data), list(self.station))
                if self.data_station[j] != self.station[j] and j in self.data:
                    del self.data[j]; self.data_filenames[j] = None; self.data_station[j] = None
                
                if not self.station[j] in self.action_lasttimes['download_data'][j]:
                    self.action_lasttimes['download_data'][j][self.station[j]] = 0
                
                try:
                    self.import_data(j)
                except Exception as e:
                    print(e, 'get_sfcobs_at_time', j, self.station[j])               
                    self.exclude_station_ids[j].append(self.station[j])
                        
                    if n==5:
                        raise Exception('Unable to obtain surface observations')
                        
                    break #Stop the for loop through obs_types, also when j=='wind', the first obs_type.
                    #This is necessary because the wind station must be updated because of the missing data,
                    #and an update of the wind station might also require an update of the temp station.
                
        for j in self.station:
            if not self.data_list_filenames[j] == self.data_filenames[j] or not self.data_list_dates[j] == self.date:
                length = len(self.data[j])
                index = self.data[j].find(';'+self.date)
                #Listing all data consumes too much self.time
                if index!=-1:
                    self.data_list[j] = ft.list_data(self.data[j][max([index-1000,0]):min([index+11000,length])], separator=';')
                else:
                    #Obtain data for the last datetime
                    self.data_list[j] = ft.list_data(self.data[j][-200:], separator=';')
                self.data_list[j] = [i for i in self.data_list[j] if i[0] in self.station[j]]
                if len(self.data_list[j][-1]) < len(self.data_list[j][-2]):
                    # The last row in the list might be incomplete, and in that case it is removed
                    self.data_list[j].pop(-1)
                self.data_list_filenames[j] = self.data_filenames[j]
                self.data_list_dates[j] = self.date
                
            datetimes = np.array([i[1] for i in self.data_list[j] if len(i)>1])
            abstimes = ft.get_absolutetimes_from_datetimes(datetimes)
            len_corr = len(self.data_list[j])-len(datetimes) # There can be a length difference due to the condition of len(i)>1
            data_time = self.data_list[j][len_corr+np.argmin(np.abs(abstimes-desired_abstime))]
            
            if j=='wind':
                #Update desired_datetime and desired_abstime, such that for temperature data the same datetime is chosen as for
                #wind data.
                desired_datetime = data_time[1]
                desired_abstime = ft.get_absolutetimes_from_datetimes(desired_datetime)
            if j=='wind' and (data_time[3] == '-999' or abs(ft.datetimediff_s(data_time[1], self.date+self.time))>3600):
                self.exclude_station_ids['wind'].append(self.station['wind'])
                if len(self.exclude_station_ids['wind'])<5:
                    self.get_sfcobs_at_time(radar, date, time, start=False); break
                else:
                    raise Exception('Nearest datetime with available surface observations is too far in time from the selected datetime')
            elif j=='temp' and data_time[1]!=desired_datetime:
                #Temperature data is only added when it is available for the same datetime as for the wind data.
                continue

            if j=='wind':
                self.obs['FF'] = float(data_time[3])
                self.obs['DD'] = int(data_time[4])
            elif j=='temp':
                self.obs['T'] = float(data_time[4])
                self.obs['Td'] = float(data_time[7])
                
            self.obs['datetime'] = data_time[1]

        return self.obs
    
    
    
if __name__=='__main__':
    k = SfcObsMETAR('hello')
    obs = k.get_sfcobs_at_time('Rzesz\u00F3w', '20210210', '0400')
    1/0
    
    t = pytime.time()
    k = SfcObsKNMI()
    obs = k.get_sfcobs_at_time('Herwijnen', '20170817', '0000')
    print(obs)
    print(pytime.time()-t, 't_KNMI')
    
    t = pytime.time()
    s = SfcObsDWD()
    obs = s.get_sfcobs_at_time('Offenthal', '20180404', '0830')
    print(obs)
    print(pytime.time()-t,'t_DWD')