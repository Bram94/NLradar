# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:15:29 2019

@author: bramv
"""
import calendar
import time as pytime
import numpy as np



def get_datetimes_from_absolutetimes(absolutetimes,include_seconds=False):
    #Returns a list of integers if not include_seconds, and a list of strings otherwise, where the seconds are separated from the rest by a ':'
    absolutetimes1=[absolutetimes] if not isinstance(absolutetimes,(list,np.ndarray)) else absolutetimes
    structs=[pytime.gmtime(j) for j in absolutetimes1]
    if not include_seconds:
        datetimes=[int(str(j[0])+format(j[1], '02d')+format(j[2], '02d')+format(j[3], '02d')+format(j[4], '02d')) for j in structs]
    else:
        datetimes=[str(j[0], '02d')+format(j[1], '02d')+format(j[2], '02d')+format(j[3], '02d')+format(j[4], '02d')+':'+format(j[5], '02d') for j in structs]
    return datetimes[0] if not isinstance(absolutetimes,(list,np.ndarray)) else datetimes

def get_absolutetimes_from_datetimes(datetimes):
    #Does not work for seconds in the datetimes yet
    datetimes1=[datetimes] if not isinstance(datetimes,(list,np.ndarray)) else datetimes
    times_seconds=[calendar.timegm(tuple(map(int,(str(dt)[:4],str(dt)[4:6],str(dt)[6:8],str(dt)[8:10],str(dt)[10:12],30 if not ':' in str(dt) else dt[13:15],0,0,0)))) for dt in datetimes1]
    if isinstance(datetimes,np.ndarray): times_seconds=np.array(times_seconds,dtype='int64')
    return times_seconds[0] if not isinstance(datetimes,(list,np.ndarray)) else times_seconds

def get_datetimes_in_datetimerange(start_datetime, end_datetime, dt_minutes):
    start_abstime = get_absolutetimes_from_datetimes(start_datetime)
    end_abstime = get_absolutetimes_from_datetimes(end_datetime)
    
    return [str(j) for j in get_datetimes_from_absolutetimes(np.arange(start_abstime, end_abstime+dt_minutes, dt_minutes*60))]  