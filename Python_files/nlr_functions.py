# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import numpy as np
import calendar
from time import gmtime
import datetime as dt
import warnings
import re
import sys
import time as pytime
import pyproj
geod = pyproj.Geod(ellps="WGS84")



def list_to_string(l, separator=','):
    s = ''
    for j in range(len(l)):
        s += str(l[j])
        if j!=len(l)-1:
            s += separator
    return s

def string_to_list(s, separator=','):
    l = []
    index = s.find(separator)
    while index!=-1:
        l.append(s[:index].strip())
        s = s[index+len(separator):]
        s = s.strip()
        index = s.find(separator)
    l.append(s)
    return l

def get_datalines(data):
    lines = []
    index = data.find('\n')
    while index!=-1:
        lines.append(data[:index])
        data = data[index:].strip()
        index = data.find('\n')
    lines.append(data)
    if lines[-1]=='': lines.pop()
    
    return lines

def list_data(data, separator=','):
    lines = get_datalines(data)
    lines_list = [string_to_list(j, separator) for j in lines]
    return lines_list

def dict_val_arr(input_dict):
    return np.array(list(input_dict.values()))

def dict_sublists_to_list(input_dict):
    output_list = []
    for sublist in input_dict.values():
        output_list += sublist
    return output_list

def create_subdicts_if_absent(input_dict, subdicts_keys):
    if not type(subdicts_keys) in (tuple, list, np.ndarray):
        subdicts_keys = [subdicts_keys]
    for key in subdicts_keys:
        if not key in input_dict:
            input_dict[key] = {}
        input_dict = input_dict[key]
        
def initialise_dict_entries_if_absent(input_dict, keys, types):
    if not type(keys) in (tuple, list, np.ndarray):
        keys = [keys]
    if not type(types) in (tuple, list, np.ndarray):
        types = [types]*len(keys)
    for i, key in enumerate(keys):
        if not key in input_dict:
            input_dict[key] = types[i]()
        
def remove_keys_from_all_subdicts(input_dict, keys):
    if not type(keys) in (tuple, list, np.ndarray):
        keys = [keys]
    for subdict in input_dict.values():
        for key in keys:
            if key in subdict:
                del subdict[key]

def remove_star(string):
    return_string=string;
    if string[-1]=='*': return_string=string[:-1];
    return return_string
    
def dontselect_if_star(string_list):
    return_list=[];
    for string in string_list:
        if string[-1]!='*':
            return_list.append(string);
    return return_list
            
def from_list_or_nolist(input_, i=0):
    if isinstance(input_, list):
        output = input_[i]
    else: output = input_
    return output

def to_list_or_nolist(input_, index, value, i=0):
    if isinstance(input_[index], list):
        input_[index][i] = value
    else: 
        input_[index] = value

def to_number(string,allow_leading_zeros=True):
    save_string=string
    match=re.search('[a-zA-Z]', string)
    letter=None if match is None else match.group(0)
    if letter=='e': #An e is allowed, if it is put in the string like 1e10, and if there are no other letters.
        e_index=save_string.find('e')
        string=save_string[:e_index]
        if e_index<len(save_string):
            string+=save_string[e_index+1:]
        match=re.search('[a-zA-Z]', string)
        letter=None if match is None else match.group(0)
    if letter is not None: number=None
    else:
        try:
            if allow_leading_zeros and len([j for j in save_string if j in map(str,range(1,10))])>0:
                while save_string[0]=='0':
                    save_string=save_string[1:]
            number=eval(save_string)
        except Exception as e:
            number=None
    return number

def correct_datetimeinput(dateinput,timeinput):
    if not (dateinput=='c' and timeinput=='c'): 
        try:
            datetime = dt.datetime.strptime(dateinput+timeinput, '%Y%m%d%H%M')
        except Exception:
            return False
    return True

def numbers_list(input_list):
    if len(input_list)>0:
        numbers=np.array([to_number(j) for j in input_list if not to_number(j) is None])
        return numbers
    else: return []
            
def rgbq(string):
    number=to_number(string)
    if not number is None:
        if 0<=number<=255: rgbq=1
        else: rgbq=0
    else: rgbq=0
    return rgbq
    
def rgb(string,alpha=False):
    maxj=3 if not alpha else 4
    rgb_values=['x' for j in range(0,maxj)]; rgbq_rgbvalues=[0 for j in range(0,maxj)]
    remainder_string=string
    for j in range(0,maxj):
        if j<maxj-1: 
            if remainder_string.find(',')!=-1 and len(remainder_string[:remainder_string.find(',')])>0:                   
                rgb_values[j]=remainder_string[:remainder_string.find(',')]
                remainder_string=remainder_string[remainder_string.find(',')+1:]
            else: rgb_values[j+1]=''
        else:
            if rgb_values[j]!='':
                rgb_values[j]=remainder_string
        rgbq_rgbvalues[j]=rgbq(rgb_values[j])
    if all(rgbq_rgbvalues)==True: 
        output=list(map(float,rgb_values))
    else: output=False
    return output
    
def rndec(number, n):
    return np.round(np.asarray(number)*10**n)/10**n
def r1dec(number):
    return rndec(number, 1)
def fndec(number, n):
    return np.floor(np.asarray(number)*10**n)/10**n
def f1dec(number):
    return fndec(number, 1)
def cndec(number, n):
    return np.ceil(np.asarray(number)*10**n)/10**n
def c1dec(number):
    return cndec(number, 1)
            
def rifdot0(number):
    if not isinstance(number,(list,np.ndarray)): 
        return_array=False
        number=[number if not isinstance(number,str) else float(number)]
    else: return_array=True
    number=np.array(number,dtype=object)
    for j in range(0,len(number)):
        if number[j]==round(number[j]): number[j]=int(number[j])
    return number if return_array else number[0]      

def round_float(number):
    number=float(number); rounded_number=number
    first_nonzero_decimal_reached = False
    decimals = format(number, '.10f').split('.')[-1]
    for j in range(len(decimals)):
        if decimals[j] != '0':
            first_nonzero_decimal_reached = True     
        if j>0 and first_nonzero_decimal_reached and rndec(number,j-1)==rndec(number,j):
            n_equal_sequence += 1
        else:
            n_equal_sequence = 0
        if n_equal_sequence == 2:
            rounded_number=rndec(number,j); break
        elif j == len(decimals)-1:
            rounded_number = round(number)
    rounded_number=rifdot0(rounded_number)
    return rounded_number

def format_nums(numbers, dec=1, remove_0dot=True, separator=None):
    numbers = rifdot0(rndec(numbers, dec))
    numbers = numbers.astype('str') if isinstance(numbers, np.ndarray) else str(numbers)
    if remove_0dot and len(numbers):
        if isinstance(numbers, np.ndarray):
            select = np.array([j[0] == '0' for j in numbers])
            numbers[select] = [j.replace('0.', '.') for j in numbers[select]]
        else:
            numbers = numbers.replace('0.', '.')
    return numbers if separator is None else separator.join(numbers)

def halftimestring(timenumber): #time as a number
    ts=str(timenumber)
    if len(ts)<2:
        tstring='0'+ts
    else:
        tstring=ts
    return tstring  

def timestring(timenumber): #time as a number, the numbers after a possible ':' are seconds
    ts=str(timenumber)
    if ts.find(':')!=-1:
        hoursminutes=ts[:ts.find(':')]
    else: hoursminutes=ts
    if len(hoursminutes)<4:
        if len(hoursminutes)<3:
            if len(hoursminutes)<2:
                tstring='000'+ts
            else:
                tstring='00'+ts
        else:
            tstring='0'+ts
    else:
        tstring=ts
    return tstring 

def time_to_minutes(time,inverse=False): # Expects time as a string in format HHMM when inverse==False, or as an integer when inverse==True
    if not inverse:
        return int(int(time[:2])*60+int(time[2:4]))
    else:
        return halftimestring(int((time-np.mod(time,60))/60))+halftimestring(int(np.mod(time,60)))
    
def floor_time(time, n_minutes):
    time_minutes=time_to_minutes(time)
    return time_to_minutes(int(np.floor(time_minutes/n_minutes)*n_minutes),inverse=True)
def ceil_time(time, n_minutes):
    time_minutes=time_to_minutes(time)
    return time_to_minutes(int(np.ceil(time_minutes/n_minutes)*n_minutes),inverse=True)
    
def get_ymdhm(time_s):
    tstruct=gmtime(time_s)
    year=str(tstruct[0]); month=halftimestring(tstruct[1]); day=halftimestring(tstruct[2])
    hour=halftimestring(tstruct[3]); minutes=halftimestring(tstruct[4])
    return year, month, day, hour, minutes
          
def next_date_and_time(date,time,timestep_m):
    year=int(date[:4]); month=int(date[4:6]); day=int(date[6:8])
    hour=int(time[:2]); minutes=int(time[2:])
    time_s=calendar.timegm((year,month,day,hour,minutes,30,0,0,0))
    nexttimestruct=gmtime(time_s+timestep_m*60)
    nextdate=str(nexttimestruct[0])+halftimestring(nexttimestruct[1])+halftimestring(nexttimestruct[2])
    nexttime=halftimestring(nexttimestruct[3])+halftimestring(nexttimestruct[4])
    return nextdate, nexttime

def next_date(date, datestep_days):
    return next_date_and_time(date, '0000', 1440*datestep_days)[0]

def get_datetimes_in_range(start_dt, end_dt, timestep_m):
    datetimes = [start_dt]
    while int(datetimes[-1]) < int(end_dt):
        datetimes += [next_datetime(datetimes[-1], timestep_m)]
    return datetimes

def get_dates_in_range(start_date, end_date, timestep_m):
    return [dt[:-4] for dt in get_datetimes_in_range(start_date+'0000', end_date+'0000', 1440)]

def next_datetime(datetime,timestep_m):
    datetime_str=datetime
    date=datetime_str[:8]; time=datetime_str[-4:]
    nextdate, nexttime=next_date_and_time(date,time,timestep_m)
    next_datetime=nextdate+nexttime
    return next_datetime

def round_datetime(datetime, n_minutes):
    time_s = get_absolutetimes_from_datetimes(datetime)
    return get_datetimes_from_absolutetimes(round(time_s/(n_minutes*60))*(n_minutes*60))
def floor_datetime(datetime, n_minutes):
    time_s = get_absolutetimes_from_datetimes(datetime)
    return get_datetimes_from_absolutetimes(np.floor(time_s/(n_minutes*60))*(n_minutes*60))
def ceil_datetime(datetime, n_minutes):
    time_s = get_absolutetimes_from_datetimes(datetime)
    return get_datetimes_from_absolutetimes(np.ceil(time_s/(n_minutes*60))*(n_minutes*60))
    
def get_absolutetimes_from_datetimes(datetimes): # The datetimes should either have format YYYYMMDDHHMM or YYYYMMDDHHMMSS
    datetimes1=[datetimes] if not isinstance(datetimes,(list,np.ndarray)) else datetimes
    times_seconds=[calendar.timegm(tuple(map(int,(dt[:4],dt[4:6],dt[6:8],dt[8:10],dt[10:12],30 if len(dt) == 12 else dt[12:14],0,0,0)))) for dt in datetimes1]
    if isinstance(datetimes,np.ndarray): times_seconds=np.array(times_seconds,dtype='int64')
    return times_seconds[0] if not isinstance(datetimes,(list,np.ndarray)) else times_seconds

def get_datetimes_from_absolutetimes(absolutetimes,include_seconds=False):
    # Returns datetimes in format YYYYMMDDHHMM if not include_seconds else YYYYMMDDHHMMSS
    absolutetimes1=[absolutetimes] if not isinstance(absolutetimes,(list,np.ndarray)) else absolutetimes
    structs=[gmtime(j) for j in absolutetimes1]
    if not include_seconds:
        datetimes=[str(j[0])+halftimestring(j[1])+halftimestring(j[2])+halftimestring(j[3])+halftimestring(j[4]) for j in structs]
    else:
        datetimes=[str(j[0])+halftimestring(j[1])+halftimestring(j[2])+halftimestring(j[3])+halftimestring(j[4])+halftimestring(j[5]) for j in structs]
    return datetimes[0] if not isinstance(absolutetimes,(list,np.ndarray)) else datetimes

def get_closest_datetime(datetimes, datetime): #Returns the index of and the datetime in datetimes that is closest to datetime
    abstimes = get_absolutetimes_from_datetimes(datetimes)
    abstime = get_absolutetimes_from_datetimes(datetime)
    index = np.argmin(np.abs(abstimes-abstime))
    return index, datetimes[index]

def datetimediff_s(datetime1,datetime2): # The datetimes should either have format YYYYMMDDHHMM or YYYYMMDDHHMMSS
    #Input as string
    dt1=datetime1; dt2=datetime2
    dt1_seconds=int(dt1[12:]) if len(dt1) == 14 else 30
    dt2_seconds=int(dt2[12:]) if len(dt2) == 14 else 30
   
    timetuple1=tuple(map(int,(dt1[:4],dt1[4:6],dt1[6:8],dt1[8:10],dt1[10:12],dt1_seconds,0,0,0)))
    timetuple2=tuple(map(int,(dt2[:4],dt2[4:6],dt2[6:8],dt2[8:10],dt2[10:12],dt2_seconds,0,0,0)))
    time1_seconds=calendar.timegm(timetuple1); time2_seconds=calendar.timegm(timetuple2)
    datetimediff_seconds=time2_seconds-time1_seconds
    return datetimediff_seconds

def datetimediff_m(datetime1,datetime2): # The datetimes should have format YYYYMMDDHHMM 
    return int(datetimediff_s(datetime1, datetime2)/60)

def timediff_s(time1,time2): # The times should either have format HHMM or HHMMSS
    # Always calculates the smallest time difference in seconds between 2 given input times. 
    # The maximum absolute value is 43320 seconds (a half day)
    diff = datetimediff_s('20000101'+time1, '20000101'+time2)
    if np.abs(diff) <= 43200:
        return diff
    else:
        return diff-86400 if diff > 0 else 86400+diff
    
def scantimediff_s(scantime1, scantime2):
    time1 = scantime_to_flattime(scantime1)
    time2 = scantime_to_flattime(scantime2)
    return timediff_s(time1, time2)

def scantimerange_s(scantime):
    time1 = format_time(scantime[:8], 'HH:MM:SS->HHMMSS')
    time2 = format_time(scantime[-8:], 'HH:MM:SS->HHMMSS')
    return timediff_s(time1, time2)

def scantimerange_formatted(scantime):
    timerange = scantimerange_s(scantime)
    sign = np.sign(timerange)
    if timerange >= 100:
        minutes = abs(timerange) // 60
        seconds = abs(timerange) % 60
        return '-'*bool(sign == -1)+f'{minutes}m{seconds}s'
    else:
        return '-'*bool(sign == -1)+f'{timerange}s'
    
def get_scandate(date, time, scantime):
    # For a given radar volume for a certain date and time, it is possible that some scans were obtained the previous or next day.
    # This function determines the actual date that corresponds to a certain scan, based on the time difference between (the volume)
    # time and scantime
    if '-' in scantime:
        scantime = get_avg_scantime([scantime])
    if time[:2] == '23' and scantime[:2] == '00':
        return next_date(date, 1)
    elif time[:2] == '00' and scantime[:2] == '23':
        return next_date(date, -1)
    else:
        return date
    
def scantime_to_flattime(scantime): # Should be given in format HH:MM:SS
    if '-' in scantime:
        #In this case scantime represents a time range, and the average of the 2 times is returned.
        time1 = format_time(scantime[:8], 'HH:MM:SS->HHMMSS')
        timediff = scantimerange_s(scantime)
        avg_absolutetime = int(round(get_absolutetimes_from_datetimes('20000101'+time1)+timediff/2))
        return get_datetimes_from_absolutetimes(avg_absolutetime, include_seconds=True)[8:]
    else:
        return format_time(scantime, 'HH:MM:SS->HHMMSS')
    
def get_start_and_end_volumetime_from_scantimes(scantimes): # expects scantimes to be a list
    flattimes = np.array([int(scantime_to_flattime(scantime)) for scantime in scantimes])
    if flattimes.max() > 233000:
        flattimes[flattimes < 10000] += 240000 # It can happen that product times are from different days
    first_scantime, last_scantime = scantimes[flattimes.argmin()], scantimes[flattimes.argmax()]
    return (first_scantime[:8], last_scantime[9:]) if '-' in first_scantime else (first_scantime, last_scantime)
        
def get_avg_scantime(scantimes): # expects scantimes to be a list
    first_scantime, last_scantime = get_start_and_end_volumetime_from_scantimes(scantimes)
    timerange = first_scantime+'-'+last_scantime
    avg_time = scantime_to_flattime(timerange)
    return format_time(avg_time)
    
def get_timerange_from_starttime_and_antspeed(startdate,starttime,antspeed, include_dates=False):
    starttime_s = get_absolutetimes_from_datetimes(startdate+format_time(starttime, 'HH:MM:SS->HHMMSS'))
    end_datetime = get_datetimes_from_absolutetimes(starttime_s+360/antspeed,include_seconds=True)
    endtime = format_time(end_datetime[8:])
    timerange = starttime+'-'+endtime
    if not include_dates:
        return timerange
    else:
        return timerange, [startdate, end_datetime[:8]]

def format_date(date,conversion='YYYY-MM-DD->DD-MMMl-YYYY'):
    month_abbreviations={1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'}
    month_abbreviations_inverse={j:i for i,j in month_abbreviations.items()}
    if conversion=='YYYY-MM-DD->DD-MMMl-YYYY':
        return date[-2:]+'-'+month_abbreviations[int(date[5:7])]+'-'+date[:4]
    elif conversion=='YYYYMMDD->DD-MMMl-YYYY':
        return date[-2:]+'-'+month_abbreviations[int(date[4:6])]+'-'+date[:4]
    elif conversion=='YYYYMMDD->YYYY-MM-DD':
        return date[:4]+'-'+date[4:6]+'-'+date[-2:]
    elif conversion=='YYYY-MM-DD->YYYYMMDD':
        return date[:4]+date[5:7]+date[-2:]
    elif conversion in ('DD-MMMl-YYYY->YYYYMMDD', 'DD MMMl YYYY->YYYYMMDD'):
        return date[-4:]+format(month_abbreviations_inverse[date[-8:-5].lower()], '02d')+format(int(date[:-9]), '02d')
    elif conversion=='DD-MM-YY->YYYYMMDD':
        fulldate='20'+date[-2:] if int(date[-2:])<80 else '19'+date[-2:]
        return fulldate+date[-5:-3]+date[:2]
    
def format_time(time,conversion='HHMMSS->HH:MM:SS'):
    if conversion=='HHMMSS->HH:MM:SS':
        return time[:2]+':'+time[2:4]+':'+time[-2:]
    elif conversion=='HHMMSS->HHMM:SS':
        return time[:4]+':'+time[-2:]
    elif conversion=='HHMM->HH:MM':
        return time[:2]+':'+time[2:4]
    elif conversion=='HHMM:SS->HH:MM:SS':
        return time[:2]+':'+time[2:]
    elif conversion in ('HH:MM->HHMM', 'HH:MM:SS->HHMMSS'):
        return time.replace(':','')

def format_datetime(datetime,conversion='YYYY-MM-DDTHH:MM->YYYYMMDDHHMM'):
    if conversion=='YYYY-MM-DDTHH:MM->YYYYMMDDHHMM':
        return format_date(datetime[:10], 'YYYY-MM-DD->YYYYMMDD')+format_time(datetime[-5:], 'HH:MM->HHMM')
    elif conversion=='YYYY-MM-DDTHH:MM:SSZ->YYYYMMDDHHMM':
        return format_date(datetime[:10], 'YYYY-MM-DD->YYYYMMDD')+format_time(datetime[-9:-4], 'HH:MM->HHMM')
    elif conversion=='YYYYMMDDHHMM->YYYY-MM-DDTHH:MM:SSZ':
        return format_date(datetime[:8], 'YYYYMMDD->YYYY-MM-DD')+'T'+format_time(datetime[-4:], 'HHMM->HH:MM')+':00Z'
    elif conversion == 'YYYYMMDDHHMM->YYYY-MM-DD HH:MM':
        return format_date(datetime[:8], 'YYYYMMDD->YYYY-MM-DD')+' '+format_time(datetime[-4:], 'HHMM->HH:MM')
    
def beamelevation(slantrange_or_elevation,scanangle,inverse=False): #slantrange in meters,scanangle in degrees
    R_earth=6371.*10**3;k=4./3.;
    if inverse==False:
        slantrange=slantrange_or_elevation;
        elevation=(np.power(np.power(slantrange*1./np.cos(scanangle*np.pi/180.),2.)+np.power(k*R_earth,2.)+2.*k*R_earth*slantrange*1./np.cos(scanangle*np.pi/180.)*np.sin(scanangle*np.pi/180.),0.5)-k*R_earth); 
        return elevation;
    else:
        elevation=slantrange_or_elevation;
        slantrange=np.power(np.power(k*R_earth*np.sin(scanangle*np.pi/180.),2.)+elevation*(elevation+2*k*R_earth),0.5)-k*R_earth*np.sin(scanangle*np.pi/180.); 
        return slantrange
        
def var1_to_var2(var1,scanangle,conversion='sr+theta->gr+h'):
    """Converts slant range, ground range, elevation angle theta, and beam elevation between each other.
    Formulas are based on the 4/3 Earth radius model, see Doviak and Zrnic (2014), eqs. 2.28.
    Distances should be given as input in kilometers, and angles in degrees.
    """
    ke, Re = 4./3., 6371.
    kR = ke*Re
    theta = np.deg2rad(scanangle)
    if conversion in ('sr+theta->gr+h','sr+theta->h','sr+theta->gr'):
        sr = var1
        h = np.sqrt(sr**2+kR**2+2.*kR*sr*np.sin(theta))-kR
        if conversion == 'sr+theta->h':
            return h
        else:
            gr = kR*np.arcsin(sr*np.cos(theta)/(kR+h))
            return gr if conversion == 'sr+theta->gr' else (gr, h)
    elif conversion in ('gr+theta->sr+h', 'gr+theta->sr', 'gr+theta->h'):
        gr = var1
        returns = []
        if 'sr' in conversion:
            returns += [kR/np.cos(gr/kR+theta)*np.sin(gr/kR)]
        if conversion[-1] == 'h':
            returns += [kR*(-1+np.cos(theta)/np.cos(gr/kR+theta))]
        return returns[0] if len(returns) == 1 else returns
    elif conversion == 'h+theta->gr':
        h = var1
        return kR*(theta+np.arccos(kR*np.cos(theta)/(h+kR)))-2.*kR*theta  
    elif conversion == 'h+theta->sr':
        h = var1
        return -kR*np.sin(theta)+np.sqrt((kR*np.sin(theta))**2+(h**2+2.*h*kR))
                                                      
def echotops_maxelevations(radial_bins,radial_res,scanangles):
    #Function expects the scan range to increase for decreasing scanangle.
    if scanangles[-1]==90.0: radial_bins=radial_bins[:-1]; radial_res=radial_res[:-1]; scanangles=scanangles[:-1];
    radial_bins_unique=[[j,radial_bins[j]] for j in range(0,len(radial_bins)-1) if radial_bins[j]!=radial_bins[j+1]]; radial_bins_unique.append([len(radial_bins)-1,radial_bins[-1]]);
    scans_ranges=[[j,radial_bins[j]*radial_res[j]] for j in range(0,len(radial_bins)-1) if radial_bins[j]*radial_res[j]-2>radial_bins[j+1]*radial_res[j+1]]; scans_ranges.append([len(radial_bins)-1,radial_bins[-1]*radial_res[-1]]);   
    scanangles_unique=np.array([scanangles[j] for j,i in scans_ranges]);
    scans_ranges=np.array([j for i,j in scans_ranges]);

    elevations_minside=beamelevation(scans_ranges[1:]*1000,np.array([scanangles_unique[j-1] for j in range(1,len(scanangles_unique))]),False)/1000.;
    elevations_plusside=beamelevation(scans_ranges*1000,scanangles_unique,False)/1000.;
    return scans_ranges,elevations_minside,elevations_plusside
        
def find_scanangle_closest_to_beamelevation(scanangles,radius,elevation): #Scanangles in degrees, radius and elevation in meters.
    scanangles_beamelevations=np.zeros(len(scanangles))
    for j in range(0,len(scanangles)):
        scanangles_beamelevations[j]=var1_to_var2(radius,scanangles[j],conversion='gr+theta->h')
    beamelevations_minus_elevation=list(map(lambda x:np.abs(x-elevation),scanangles_beamelevations))
    nearest_elevation=min(beamelevations_minus_elevation)
    nearest_elevation_listpos=[i for i,x in enumerate(beamelevations_minus_elevation) if x==nearest_elevation]
    nearest_scanangle=scanangles[nearest_elevation_listpos[0]]
    return nearest_scanangle

def mindiff_value_array(array,value):
    mindiff=np.min(np.abs(array-value))
    return mindiff

def closest_value(array,value, return_index=False):
    index=np.argmin(np.abs(array-value))
    return array[index] if not return_index else (array[index], index)

def point_inside_rectangle(point,corners):
    rectangle_center=np.sum(corners,axis=0)/4;
    rectangle_xdim=corners[:,0].max()-rectangle_center[0];
    rectangle_ydim=corners[:,1].max()-rectangle_center[1];    
    point_xdisttocenter=abs(point[0]-rectangle_center[0]);
    point_ydisttocenter=abs(point[1]-rectangle_center[1]);
    if point_xdisttocenter<=rectangle_xdim and point_ydisttocenter<=rectangle_ydim:
        inside=True; dist_to_edge=min(rectangle_xdim-point_xdisttocenter,rectangle_ydim-point_ydisttocenter);
    else:
        inside=False; dist_to_edge=-1;
    return [inside, dist_to_edge]
    
def mindist_maxdist_maxangle(corners):
    corner_dist = np.linalg.norm(corners, axis=1)
    indices = np.argsort(corner_dist)
    corner_dist_sort, corners_sort = corner_dist[indices], corners[indices]
    min_dist = 0.
    if not point_inside_rectangle([0,0], corners)[0]:
        nearest_corner = corners_sort[0]
        nearest_corner_min1 = corners_sort[1]
        nearest_corner_min2 = corners_sort[2]
        dist1 = np.dot(nearest_corner_min1-nearest_corner, -nearest_corner)/np.linalg.norm(nearest_corner_min1-nearest_corner)
        dist2 = np.dot(nearest_corner_min2-nearest_corner, -nearest_corner)/np.linalg.norm(nearest_corner_min2-nearest_corner)
        position_mindist = nearest_corner
        if dist1 < 0 and dist2 > 0:
            position_mindist += dist2*(nearest_corner_min2-nearest_corner)/np.linalg.norm(nearest_corner_min2-nearest_corner)
        elif dist1 > 0:
            position_mindist += dist1*(nearest_corner_min1-nearest_corner)/np.linalg.norm(nearest_corner_min1-nearest_corner)
        min_dist = np.linalg.norm(position_mindist)
    max_dist = corner_dist_sort[-1]
    
    # When the radar is approximately in the middle of the view, then angle_maxdist is chosen to be that of the top-right corner, since that's
    # the corner where no grid coordinates are drawn.
    farthest_corner = corners[np.where((max_dist-corner_dist)/(max_dist-min_dist) < 0.01)[0].max()]
    angle_maxdist = np.arctan2(farthest_corner[0],farthest_corner[1]) % (2*np.pi)
    angle_mindist = None
    if min_dist > 0 and min_dist == corner_dist_sort[0]:
        angle_mindist = np.arctan2(position_mindist[0],position_mindist[1]) % (2*np.pi)
    return min_dist, max_dist, angle_mindist, angle_maxdist
    
def distance_to_rectangle_point_inside(corners,point):
    distances_to_corners = np.linalg.norm(corners-point, axis=1)
    corner_mindist = corners[np.argmin(distances_to_corners)]
    distance_to_rectangle = np.abs(corner_mindist-point).min()
    return distance_to_rectangle
    
def av_angle_circle_in_rectangle(xlim,ylim,radius):
    with np.errstate(invalid='ignore'):
        x1_plus=np.sqrt(radius**2-ylim[0]**2);x2_plus=np.sqrt(radius**2-ylim[1]**2);
        y1_plus=np.sqrt(radius**2-xlim[0]**2);y2_plus=np.sqrt(radius**2-xlim[1]**2);
        x1_min=0;x2_min=0;y1_min=0;y2_min=0;
        if np.isnan(x1_plus)==False:x1_min=-x1_plus;
        if np.isnan(x2_plus)==False:x2_min=-x2_plus;
        if np.isnan(y1_plus)==False:y1_min=-y1_plus;
        if np.isnan(y2_plus)==False:y2_min=-y2_plus;
        
    if x1_plus<xlim[0] or x1_plus>xlim[1] or np.isnan(x1_plus)==True: x1_plus=0;
    if x2_plus<xlim[0] or x2_plus>xlim[1] or np.isnan(x2_plus)==True: x2_plus=0;
    if y1_plus<ylim[0] or y1_plus>ylim[1] or np.isnan(y1_plus)==True: y1_plus=0;
    if y2_plus<ylim[0] or y2_plus>ylim[1] or np.isnan(y2_plus)==True: y2_plus=0;
    if x1_min<xlim[0] or x1_min>xlim[1] or np.isnan(x1_min)==True: x1_min=0;
    if x2_min<xlim[0] or x2_min>xlim[1] or np.isnan(x2_min)==True: x2_min=0;
    if y1_min<ylim[0] or y1_min>ylim[1] or np.isnan(y1_min)==True: y1_min=0;
    if y2_min<ylim[0] or y2_min>ylim[1] or np.isnan(y2_min)==True: y2_min=0;
    x_coordinates=[xlim[0],xlim[1],xlim[0],xlim[1],x1_plus,x2_plus,x1_min,x2_min];
    y_coordinates=[y1_plus,y2_plus,y1_min,y2_min,ylim[0],ylim[1],ylim[0],ylim[1]];
    points_of_intersection=[[x_coordinates[i],j] for i,j in enumerate(y_coordinates) if j!=0 and x_coordinates[i]!=0];
    
    angles=np.array([]);
    for j in range(0,len(points_of_intersection)):
        angles=np.append(angles,np.arctan(points_of_intersection[j][0]/points_of_intersection[j][1]));
        if points_of_intersection[j][1]<0:
            angles[j]=angles[j]+np.pi;
    if abs(max(angles)-min(angles))>np.pi:
        angles[angles<0]=angles[angles<0]+2*np.pi;
    angles=np.sort(angles);
    
    if len(angles)==4:
        anglesdiff=[[angles[1]-angles[0],0,1],[angles[3]-angles[2],2,3]];
        maxanglediff=np.max([x[0] for x in anglesdiff]);
        posmaxanglediff=[x[0] for x in anglesdiff].index(maxanglediff);
        angle1=angles[anglesdiff[posmaxanglediff][1]];
        angle2=angles[anglesdiff[posmaxanglediff][2]];
    else:
        angle1=angles[0];angle2=angles[1];
    
    if np.abs(angle1-angle2)>np.pi and angle1<angle2:
        angle1=angle1+2*np.pi;
    elif np.abs(angle1-angle2)>np.pi and angle2<angle1:
        angle2=angle2+2*np.pi;
        
    av_angle=0.5*(angle1+angle2);
    return av_angle
    
def azimuthal_angle(coords, deg=False):
    x,y=coords;
    if x!=0:
        phi=-np.arctan(y/x)+0.5*np.pi;
        if x<0:
            phi=phi+np.pi;
    else:
        if y>0: phi=0;
        else: phi=np.pi;
    return phi*180/np.pi if deg else phi

def angle_diff(angle1, angle2=None, between_0_360=False):
    diff = np.diff(angle1) if angle2 is None else angle2-angle1
    angle_diff = (diff+180) % 360 - 180
    return angle_diff % 360 if between_0_360 else angle_diff

def rotation_matrix_R2(angle,inverse=False):
    if inverse==False: R=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]); return R
    else: R_inv=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]); return R_inv
                
def angle_intersection_circle_with_corner_bisector(cornerx,cornery,cornervector,radius):
    """This problem is analytically solved for cornervector=[1,1] (upper right corner). In the case of other cornervectors, 
    the vector [cornerx,cornery] is first rotated to the upper right quadrant for which the analytical solution is valid, 
    and is finally rotated back."""
    rotate_angle=azimuthal_angle(cornervector)-0.25*np.pi;
    x1,y1=rotation_matrix_R2(rotate_angle,True).dot(np.array([cornerx,cornery]))
    r1=np.sqrt(x1**2+y1**2); r=radius;
    alpha=0.5*(x1+y1-np.sqrt((x1+y1)**2-2*(r1**2-r**2)));
    x,y=rotation_matrix_R2(rotate_angle,False).dot(np.array([x1,y1]))-alpha*np.array(cornervector);
    angle=azimuthal_angle([x,y]);
    return angle
         
def calibration(PV,calibration_a,calibration_b,scale_factor): #PV=pixel value
    RV=(calibration_a*PV+calibration_b)*scale_factor;
    return RV  

def interpolate_2colors(color1,color2,position): #color1 and color2 must be arrays. Position must lie between 0 and 1.
    color=color1+position*(color2-color1)
    return color
            
def cbar_interpolate(cbarlist,res,log=False): # product values must increase from bottom to top
    productvalues=cbarlist[:,0];
    colors1=cbarlist[:,1:4];
    colors2=cbarlist[:,4:7];
    colorsint=np.zeros((0,3));
    if not log: resolution=res; #Number of interpolation points in a reflectivity interval of 1 dBZ.
    else: resolution=50*res;
    for i in range(0,len(cbarlist)-1):
        if not log:
            productvalues_diff=productvalues[i+1]-productvalues[i];
        else: 
            productvalues_diff=np.log10(productvalues[i+1]/productvalues[i]);
        maxj=int(productvalues_diff*resolution); 
        for j in range(0,maxj):
            if int(colors2[i,0])==-1:
                colorsint=np.concatenate((colorsint,[colors1[i]+(colors1[i+1]-colors1[i])*j/(resolution*productvalues_diff)]));
            else:
                colorsint=np.concatenate((colorsint,[colors1[i]+(colors2[i]-colors1[i])*j/(resolution*productvalues_diff)]));                                 
    return colorsint

def determine_latlon_from_inputstring(input_marker_latlon):
    input_marker_latlon = input_marker_latlon.strip()
    deg = u'\u00b0'
    if ',' in input_marker_latlon:
        input_marker_lat, input_marker_lon = [j.replace(deg, '') for j in input_marker_latlon.split(',')]
    elif ' ' in input_marker_latlon:
        input_marker_lat, input_marker_lon = [j.replace(deg, '') for j in input_marker_latlon.split(' ')]
    elif '\t' in input_marker_latlon:
        # Assumes SPC storm reports notation
        input_marker_lat, input_marker_lon = input_marker_latlon.split('\t')
        input_marker_lat = input_marker_lat[:-2]+'.'+input_marker_lat[-2:]
        input_marker_lon = '-'+input_marker_lon[:-2]+'.'+input_marker_lon[-2:]
    elif deg in input_marker_latlon: 
        # Assumes DMS notation
        index=input_marker_latlon.find('N')
        if index==-1: index=input_marker_latlon.find('S')
        
        #It is assumed that the N/S in the latitudinal part is followed by a space or comma, to separate it from the 
        #longitudinal part.
        lat=input_marker_latlon[:index+1]
        lon=input_marker_latlon[index+2:]  
            
        input_marker_lat=float(lat[:lat.index(deg)])+1/60*float(lat[lat.index(deg)+1:lat.index("'")])
        if '"' in lat: 
            input_marker_lat=str(input_marker_lat+1/3600*float(lat[lat.index("'")+1:lat.index('"')]))+lat[lat.index('"')+1:]
        else: input_marker_lat=str(input_marker_lat)+lat[lat.index("'")+1:]
        
        input_marker_lon=float(lon[:lon.index(deg)])+1/60*float(lon[lon.index(deg)+1:lon.index("'")])
        if '"' in lon: 
            input_marker_lon=str(input_marker_lon+1/3600*float(lon[lon.index("'")+1:lon.index('"')]))+lon[lon.index('"')+1:]
        else: input_marker_lon=str(input_marker_lon)+lon[lon.index("'")+1:]
    else: 
        raise Exception('Incorrect input')
        
    if 'N' in input_marker_lat: input_marker_lat=input_marker_lat[:input_marker_lat.index('N')]
    elif 'S' in input_marker_lat: input_marker_lat='-'+input_marker_lat[:input_marker_lat.index('S')]
    if 'E' in input_marker_lon: input_marker_lon=input_marker_lon[:input_marker_lon.index('E')]
    elif 'W' in input_marker_lon: input_marker_lon='-'+input_marker_lon[:input_marker_lon.index('W')]
    
    number1=to_number(input_marker_lat); number2=to_number(input_marker_lon)
    if number1 is None or number2 is None:
        raise Exception('Incorrect input')
    else:
        return number1, number2
    
# print(determine_latlon_from_inputstring('33.52° N, 86.93° W'))
# print(determine_latlon_from_inputstring('32.733°N 98.3358°W'))
            
def aeqd(latlon_0,latlon_or_xy_1,degrees=True,inverse=False):
    R_earth=6371
    if degrees==True: 
        latlon_0=np.array(latlon_0)*np.pi/180
    lat_0,lon_0=latlon_0[0],latlon_0[1]
    warnings.filterwarnings('ignore') #Avoid warnings for errors caused by division by zero, because they are later corrected
    if inverse==False:
        latlon_1=np.array(latlon_or_xy_1)
        if degrees==True: latlon_1=latlon_1*np.pi/180
        if not hasattr(latlon_1[0], "__len__"): lat_1,lon_1=latlon_1[0],latlon_1[1]
        else: lat_1,lon_1=latlon_1[:,0],latlon_1[:,1]
            
        rho=R_earth*np.arccos(np.sin(lat_0)*np.sin(lat_1)+np.cos(lat_0)*np.cos(lat_1)*np.cos(lon_1-lon_0))
        theta=np.arctan(np.divide(np.cos(lat_1)*np.sin(lon_1-lon_0),np.cos(lat_0)*np.sin(lat_1)-np.sin(lat_0)*np.cos(lat_1)*np.cos(lon_1-lon_0)))
        if type(theta)==np.ndarray:
            theta[((lon_1<lon_0) & (theta>0)) | ((lon_1>lon_0) & (theta<0)) | ((lon_1==lon_0) & (lat_1<lat_0))]=theta[((lon_1<lon_0) & (theta>0)) | ((lon_1>lon_0) & (theta<0)) | ((lon_1==lon_0) & (lat_1<lat_0))]+np.pi
        elif (lon_1<lon_0 and theta>0) or (lon_1>lon_0 and theta<0) or (lon_1==lon_0 and lat_1<lat_0): theta=theta+np.pi
        
        xy_1=np.transpose(rho*np.array([np.sin(theta),np.cos(theta)]))
        xy_1[np.isnan(xy_1)]=0.0 #This happens for latlons in latlon_1 that are equal to latlon_0
    else:
        xy_1=np.array(latlon_or_xy_1)
        if not hasattr(xy_1[0], "__len__"): x_1,y_1=xy_1[0],xy_1[1]
        else: x_1,y_1=xy_1[:,0],xy_1[:,1]
        c=np.linalg.norm(xy_1, axis=-1)/R_earth
        lat=np.arcsin(np.cos(c)*np.sin(lat_0)+np.divide(y_1*np.sin(c)*np.cos(lat_0),c*R_earth))
        lon=lon_0+np.arctan(np.divide(x_1*np.sin(c),c*R_earth*np.cos(lat_0)*np.cos(c)-y_1*np.sin(lat_0)*np.sin(c)))
        latlon_1=np.transpose([lat,lon])
        if len(latlon_1[np.isnan(latlon_1)])>0:
            latlon_1[np.isnan(latlon_1)]=np.array([lat_0,lon_0])
        if degrees==True: latlon_1=latlon_1*180/np.pi
    warnings.resetwarnings() #Remove the warnings filter
    return xy_1 if not inverse else latlon_1
        
def calculate_great_circle_distance_from_latlon(latlon_0, latlon_1): #latlon_0 should be a list/array with a single latitude 
    #and longitude, and latlon_1 can either also be a List/array that contains a single latitude and longitude, 
    #or it can be a list/array of lat, lon pairs.
    lat0, lon0 = latlon_0
    latlon_1 = np.array(latlon_1)
    if isinstance(latlon_1[0], np.ndarray):
        lat1, lon1 = latlon_1[:, 0], latlon_1[:, 1]
    else:
        lat1, lon1 = latlon_1
    if isinstance(latlon_1[0], np.ndarray):
        return geod.inv(np.repeat(lon0, len(lon1)), np.repeat(lat0, len(lon1)), lon1, lat1)[2] / 1e3
    else:
        return geod.inv(lon0, lat0, lon1, lat1)[2] / 1e3

previous_radar_latlon = None
proj = None
def calculate_great_circle_distance_from_xy(radar_latlon, xy_0, xy_1):
    global previous_radar, proj
    if not radar_latlon == previous_radar_latlon:
        lat, lon = radar_latlon
        proj = pyproj.Proj('+proj=aeqd +lat_0='+str(lat)+' +lon_0='+str(lon)+' +a=6378140 +b=6356750 +x_0=0 y_0=0 +units=km')
    latlon_0 = proj(xy_0[0], xy_0[1], inverse=True)[::-1]
    latlon_1 = proj(xy_1[0], xy_1[1], inverse=True)[::-1]
    return calculate_great_circle_distance_from_latlon(latlon_0, latlon_1)

def calculate_azimuth(point): #Calculates the azimuth from which the line pointing from the origin to the point originates
    if point[0]>=0:
        if point[1]>0:
            azimuth=180+np.arctan(point[0]/point[1])*180./np.pi;
        elif point[1]==0:
            azimuth=270;
        else:
            azimuth=360+np.arctan(point[0]/point[1])*180./np.pi;
    else:
        if point[1]>0:
            azimuth=180+np.arctan(point[0]/point[1])*180./np.pi;
        elif point[1]==0: 
            azimuth=90;
        else:
            azimuth=np.arctan(point[0]/point[1])*180./np.pi;
    return azimuth


def blend_rgba_colors_3D(c1,c2,t): #Blending colors using the algorithm described by Fordi at http://stackoverflow.com/questions/726549/algorithm-for-additive-color-mixing-for-rgb-values
    c=np.zeros(c1.shape) #c1 must be an 3D m*n*4 array, c2 can have the same dimensions as c1, or must be a length 4 array, with one color. t is the mixing value which must lie between 0 and 1.
    if len(c2.shape)==1:
        c[:,:,:3]=np.sqrt((1-t)*np.power(c1[:,:,:3],2)+t*np.power(c2[:3],2))
        c[:,:,3] = (1-t)*c1[:,:,3]+t*c2[3]
    else:
        c[:,:,:3]=np.sqrt((1-t)*np.power(c1[:,:,:3],2)+t*np.power(c2[:,:,:3],2))
        c[:,:,3] = (1-t)*c1[:,:,3]+t*c2[:,:,3]
    return c

def blend_rgba_colors_1D(c1,c2,t): #Blending colors using the algorithm described by Fordi at http://stackoverflow.com/questions/726549/algorithm-for-additive-color-mixing-for-rgb-values
    c=np.zeros(c1.shape) #c1 and c2 must be 1D arrays with 4 elements (1 color). t is the mixing value which must lie between 0 and 1.
    c[:3]=np.sqrt((1-t)*np.power(c1[:3],2)+t*np.power(c2[:3],2))
    c[3] = (1-t)*c1[3]+t*c2[3]
    return c
    
def convert_float_to_uint(data,n_bits,data_limits,astype_int=True):    
    dtype='uint'+str(n_bits)
    a=(2**n_bits-1)/(data_limits[1]-data_limits[0])
    b=data_limits[0]
    data=a*(data-b)+0.5 #+0.5 ensures that flooring by converting data to integer below produces correct results
    if astype_int:
        try:
            data = data.astype(dtype)
        except Exception: 
            #Occurs when data is a number
            data=int(data)
    return data

def convert_uint_to_float(data,n_bits,data_limits):
    a=(data_limits[1]-data_limits[0])/(2**n_bits-1)
    b=data_limits[0]
    try:
        data=data.astype('float32')
    except Exception:
        pass
    data=a*data+b
    return data

def bytes_to_array(data, datadepth = 8):
    if sys.byteorder != 'big':
        byteorder = '>'
    else:
        byteorder = '<'
    
    datawidth = int(datadepth / 8)

    datatype = byteorder + 'u' + str(datawidth)

    return np.ndarray(shape=(int(len(data) / datawidth),),
                  dtype=datatype, buffer=data)
    
def add_rolled_arr(arr1, arr2, axis, shift):
    """ Add a rolled (shifted, with periodic boundaries) version of arr2 to arr1, in an efficient way.
    arr2 is rolled along the specified axis by the specified shift.
    Works only for 2D arrays
    """
    if axis==0:
        arr1[:shift] += arr2[-shift:]
        arr1[shift:] += arr2[:-shift]
    else:
        arr1[:,:shift] += arr2[:,-shift:]
        arr1[:,shift:] += arr2[:,:-shift]
    
def add_shifted_arr(arr1, arr2, axis, shift): 
    """ Add a shifted version of arr2 to arr1, without use of periodic boundaries, in an efficient way.
    arr2 is shifted along the specified axis by the specified shift.
    Works only for 2D arrays
    """
    if axis==0:
        if shift<0:
            arr1[-shift:] += arr2[:shift]
        else:
            arr1[:-shift] += arr2[shift:]
    else:
        if shift<0:
            arr1[:,-shift:] += arr2[:,:shift]
        else:
            arr1[:,:-shift] += arr2[:,shift:]
            
def get_moving_avg(arr, n, axis, mask_value=None):
    if mask_value is None:
        avg = arr.copy()
        for i in range(1, n):
            add_rolled_arr(avg, arr, axis, -i)
        avg /= n
    else:
        unmasked = ~np.isnan(arr) if mask_value == np.nan else (arr != mask_value)
        n_unmasked = unmasked.astype('uint8')
        arr = arr.copy()
        arr[~unmasked] = 0
        avg = arr.copy()
        for i in range(1, n):
            add_rolled_arr(avg, arr, axis, -i)
            add_rolled_arr(n_unmasked, unmasked, axis, -i)
        avg /= n_unmasked
        avg[n_unmasked == 0] = mask_value
    return avg
        
def get_window_sum(arr, window = [2, 2, 2, 2, 2]):
    """For each radar bin this function sums all elements in the selected window. The length of the list 'window' specifies the number of radials that the window comprises,
    and the elements in 'window' specify for each radial (centered around the central radial in which the radar bin is located) the number of radial bins in the window. 
    For each radial do 2*x+1 radial bins belong to the window, where x is the element in 'window'. The elements in window thus specify the number of radial bins above and below
    the central radius of the window. 
    """
    if len(window) % 2 == 0: raise Exception("The length of 'window' should be an odd number")
    
    window = np.array(window)
    azi_sums = {1:arr.copy()}
    n_azi = int((len(window) - 1)/2)
    for i in range(1, n_azi+1):
        azi_sums[1 + 2*i] = azi_sums[1 + 2*(i-1)].copy()
        add_rolled_arr(azi_sums[1 + 2*i], arr, 0, -i)
        add_rolled_arr(azi_sums[1 + 2*i], arr, 0, i)
    
    window_sum = azi_sums[1 + 2*n_azi].copy()
    for j in range(1, max(window)+1):
        n_azi_j = np.count_nonzero(window>=j)
        add_shifted_arr(window_sum, azi_sums[n_azi_j], 1, j)
        add_shifted_arr(window_sum, azi_sums[n_azi_j], 1, -j)
    
    return window_sum

def get_window_indices(i, window, shape, periodic = 'rows'):
    # i should be of integer dtype, not unsigned integer, because of calculations performed below!
    """For each index (row, column format) in i, this function returns indices for all elements within a 2D array with shape 'shape'
    that are located within a window 'window' centered at the grid cell with the given index.
    
    This is done in a specific order, where the function determines for each grid cell c in 'window' consecutively which indices
    it needs to select. This means that when len(i)=n, that the first n elements in the returned rows_window and cols_window refer
    to the first grid cell in 'window'.
    So when the returned indices are used to update values within an array 'array', then this should be coded as
    array[(rows_window, cols_window)] = np.tile(updated_values, window_size), where window_size = sum([j*2+1 for j in window]).
    
    'periodic' specifies whether periodic boundaries should be used. It can be either None, 'rows', 'cols' or 'both'. Default is 'rows',
    as should be used for radar data provided on a polar grid. If the grid is not periodic along a certain axis, then indices extending
    beyond the axis are put equal to zero or axis_length-1.
    """
    window = np.array(window)
    
    rows, cols = i[:, 0], i[:, 1]
    rows_window = [rows]
    cols_window = [cols]
    
    n_azi = len(window)
    for i in range(1, int((n_azi - 1)/2)+1):
        rows_window += [rows-i, rows+i]
        cols_window += [cols]*2
        
    for j in range(1, max(window)+1):
        n_azi_j = np.count_nonzero(window>=j)

        cols_window += [cols-j]*n_azi_j + [cols+j]*n_azi_j
        for n in range(2):
            rows_window += [rows]
            for i in range(1, int((n_azi_j - 1)/2)+1):
                rows_window += [rows-i, rows+i]

    rows_window, cols_window = np.concatenate(rows_window), np.concatenate(cols_window)
    if periodic in ('rows', 'both'):
        rows_window = np.mod(rows_window, shape[0])
    if periodic in ('cols', 'both'):
        cols_window = np.mod(cols_window, shape[1])
    
    if periodic in (None, 'rows'):
        cols_window[cols_window >= shape[1]] = shape[1]-1
    if periodic in (None, 'cols'):
        rows_window[rows_window >= shape[0]] = shape[0]-1
    return rows_window, cols_window