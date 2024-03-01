import requests
import json
import os
import numpy as np
import re
import time as pytime

import nlr_functions as ft



with open('us_states.json', 'r') as f:
    us_states = json.loads(f.read())
us_state_codes = [j['code'] for j in us_states]
print(us_state_codes)

with open('metar_stations.txt', 'r') as f:
    text = f.read()
data = ft.list_data(text, ' ')
state_station_map = {}
for state in us_state_codes:
    state_station_map[state] = []
    for j in data:
        if j[0] == state and not any(['88D' in i for i in j]):
            index = [i for i,k in enumerate(j) if k.startswith('K') and len(k) == 4 and j[i+1] == k[1:]]
            if not len(index):
                continue
            state_station_map[state] += [j[index[0]]]
print(state_station_map)
country_stations = state_station_map
state = 'CA'
country_stations = {state:country_stations[state]}
print(country_stations)
# 1/0

#%%
# country_stations = {'Poland':['EPSY'], 'Denmark':['EKYT'], 'France':['LFOE']}
# country_stations = {'AL':'TOI'}
# import json

# hdr = {"X-API-Key": "910f8b04ed2c4be99d3f9a88e9"}
# req = requests.get("https://api.checkwx.com/station/EPKT", headers=hdr)
# # print("Response from CheckWX.... \n")
# print(eval(req.text.replace('true', 'True')))
# 1/0

# try:
#     req.raise_for_status()
#     resp = json.loads(req.text)
#     print(json.dumps(resp, indent=1))

# except requests.exceptions.HTTPError as e:
#     print(e)

stations_exclude = ['EPMO']

string = ''
for country in country_stations:
    for request_station in country_stations[country]:
        print(request_station)
        params = {'TYPE':'sfregion', 'DATE':'current', 'HOUR':'current', 'UNITS':'M', 'STATION':request_station}
        text = requests.get('http://weather.uwyo.edu/cgi-bin/wyowx.fcgi', params=params).text
        indices = [i.end() for i in re.finditer('STATION=', text)]
        stations = np.unique([text[i:i+4].rstrip('>') for i in indices])
        print(country, stations)
        
        for station in stations:
            if len(station) == 3:
                station = 'K'+station
            if station in stations_exclude: continue
            if not station.startswith('K'): continue
        
            try:
                hdr = {"X-API-Key": "910f8b04ed2c4be99d3f9a88e9"}
                req = requests.get("https://api.checkwx.com/station/"+station, headers=hdr)
                req = eval(req.text.replace('true', 'True'))['data'][0]
                city = req['city']
                lon, lat = req['geometry']['coordinates']
                elev = req['elevation']['meters']
                string += f'{station}\t{lat:.6f}\t{lon:.6f}\t{elev:4}\t{city}\n'
            except Exception as e:
                print(station, e)
                pass
        if len(stations):
            break
string = string[:-1]
print(string)

#%%
# with open('D:/NLradar/Generated_files/sfc_obs/metar_stations.txt', 'w', encoding='utf-8') as f:
    # f.write(string)
# with open('D:/NLradar/Generated_files/sfc_obs/metar_stations.txt', 'a', encoding='utf-8') as f:
#     f.write('\n'+string)
with open('D:/NLradar/Generated_files/sfc_obs/metar_stations.txt', 'r', encoding='utf-8') as f:
    text = f.read()
with open('D:/NLradar/Generated_files/sfc_obs/metar_stations.txt', 'w', encoding='utf-8') as f:
    f.write(string+'\n'+text)