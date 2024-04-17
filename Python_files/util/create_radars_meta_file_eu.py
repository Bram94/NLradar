# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:33:34 2023

@author: -
"""
import os
import sys
sys.path.insert(0, '/'.join(__file__.split(os.sep)[:-2]))
import numpy as np
import pickle

import nlr_functions as ft

with open('D:/NLradar/Python_files/testing/french_radars.pkl', 'rb') as f:
    french_meta = pickle.load(f)
print(french_meta)


#When removing radars from a particular data source, then it is enough to remove them from the variables (lists) radars and data_sources_all.
radars_KNMI=['De Bilt','Den Helder','Herwijnen']
radars_TUDelft=['Cabauw']
radars_KMI=['Jabbeke','Wideumont']
radars_skeyes=['Zaventem']
radars_VMM=['Helchteren']
radars_DWD=['Borkum','Boostedt','Dresden','Eisberg','Emden','Essen','Feldberg','Flechtdorf','Hannover','Isen','Memmingen','Neuhaus','Neuheilenbach','Offenthal','Pr\u00F6tzel','Rostock','T\u00FCrkheim','Ummendorf']
radars_IMGW=['Brzuchania','Gda\u0144sk','Legionowo','Pastewnik','Pozna\u0144','Ram\u017ca','Rzesz\u00F3w','\u015Awidwin','U\u017cranki']
radars_DMI=['Juvre','Sindal','Bornholm','Stevns','Virring Skanderborg']
radars_MeteoFrance=list(french_meta)
radars=radars_KNMI+radars_TUDelft+radars_KMI+radars_skeyes+radars_VMM+radars_DWD+radars_IMGW+radars_DMI+radars_MeteoFrance

data_sources_all=['KNMI','KMI','skeyes','VMM','DWD','TU Delft','IMGW','DMI','Météo-France']
data_sources={}
for radar in radars:
    if radar in radars_KNMI: data_sources[radar]='KNMI'
    elif radar in radars_KMI: data_sources[radar]='KMI'
    elif radar in radars_skeyes: data_sources[radar]='skeyes'
    elif radar in radars_VMM: data_sources[radar]='VMM'
    elif radar in radars_DWD: data_sources[radar]='DWD'
    elif radar in radars_TUDelft: data_sources[radar]='TU Delft'
    elif radar in radars_IMGW: data_sources[radar]='IMGW'
    elif radar in radars_DMI: data_sources[radar]='DMI'
    elif radar in radars_MeteoFrance: data_sources[radar]='Météo-France'
    
radars_per_datasource={'KNMI':radars_KNMI,'KMI':radars_KMI,'skeyes':radars_skeyes,'VMM':radars_VMM,'DWD':radars_DWD,'TU Delft':radars_TUDelft,'IMGW':radars_IMGW,'DMI':radars_DMI,'Météo-France':radars_MeteoFrance}
for i,j in radars_per_datasource.items():
    print(i, len(j))
rplaces_to_ridentifiers={'De Bilt':'60','Den Helder':'61','Herwijnen':'62','Jabbeke':'bejab','Wideumont':'bewid','Zaventem':'bezav','Helchteren':'','Borkum':'asb','Boostedt':'boo','Dresden':'drs','Eisberg':'eis','Emden':'emd','Essen':'ess','Feldberg':'fbg','Flechtdorf':'fld','Hannover':'hnr','Isen':'isn','Memmingen':'mem','Neuhaus':'neu','Neuheilenbach':'nhb','Offenthal':'oft','Pr\u00F6tzel':'pro','Rostock':'ros','T\u00FCrkheim':'tur','Ummendorf':'umd','Cabauw':'IDRA','Brzuchania':'BRZ','Gda\u0144sk':'GDA','Legionowo':'LEG','Pastewnik':'PAS','Pozna\u0144':'POZ','Ram\u017ca':'RAM','Rzesz\u00F3w':'RZE','\u015Awidwin':'SWI','U\u017cranki':'UZR',
                         'Juvre':'dkrom','Sindal':'dksin','Bornholm':'dkbor','Stevns':'dkste','Virring Skanderborg':'dkvir'}
rplaces_to_ridentifiers.update({i:j['id'] for i,j in french_meta.items()})

radarcoords={'De Bilt':[52.10168,5.17834],'Den Helder':[52.95279,4.79061],'Herwijnen':[51.8369,5.1381],'Jabbeke':[51.1917,3.0642],'Wideumont':[49.9143,5.5056],'Zaventem':[50.9055,4.4550],'Helchteren':[51.0702, 5.4054],'Borkum':[53.564011,6.748292],'Boostedt':[54.00438,10.04687],'Dresden':[51.12465,13.76865],'Eisberg':[49.54066,12.40278],'Emden':[53.33872,7.02377],'Essen':[51.40563,6.96712],'Feldberg':[47.87361,8.00361],'Flechtdorf':[51.3112,8.802],'Hannover':[52.46008,9.69452],'Isen':[48.1747,12.10177],'Memmingen':[48.04214,10.21924],'Neuhaus':[50.50012,11.13504],'Neuheilenbach':[50.10965,6.54853],'Offenthal':[49.9847,8.71293],'Pr\u00F6tzel':[52.64867,13.85821],'Rostock':[54.17566,12.05808],'T\u00FCrkheim':[48.58528,9.78278],'Ummendorf':[52.16009,11.17609],'Cabauw':[51.97,4.926244],'Brzuchania':[50.394170,20.079720],'Gda\u0144sk':[54.384250,18.456310],'Legionowo':[52.405219,20.960911],'Pastewnik':[50.892000,16.039500],'Pozna\u0144':[52.413260,16.797060],'Ram\u017ca':[50.151670,18.726670],'Rzesz\u00F3w':[50.114090,22.037040],'\u015Awidwin':[53.790280,15.831110],'U\u017cranki':[53.856450,21.412140],'Juvre':[55.1725903,8.55052996],'Sindal':[57.48876226,10.13511376],'Bornholm':[55.11283297,14.8874575],'Stevns':[55.32561875,12.44817293],'Virring Skanderborg':[56.02386909,10.02516884]}
radarcoords.update({i:[j['lat'], j['lon']] for i,j in french_meta.items()})

# elevation+tower_height
radar_elevations_fromfile={'Wideumont':590,'Zaventem':90,'Jabbeke':50,'Helchteren':144,
                           'Boostedt':124,'Borkum':36,'Dresden':262,'Eisberg':798,'Emden':58,'Essen':185,'Feldberg':1516,
                           'Flechtdorf':627,'Hannover':97,'Isen':679,'Memmingen':725,'Neuhaus':879,'Neuheilenbach':585,
                           'Offenthal':245,'Pr\u00F6tzel':189,'Rostock':37,'T\u00FCrkheim':767,'Ummendorf':183,
                           'Brzuchania':453,'Gda\u0144sk':158,'Legionowo':119,'Pastewnik':688,'Pozna\u0144':130,
                           'Ram\u017ca':358,'Rzesz\u00F3w':235,'\u015Awidwin':146,'U\u017cranki':237,
                           'Juvre':15,'Sindal':109,'Bornholm':171,'Stevns':53,'Virring Skanderborg':142}
radar_elevations_fromfile.update({i:j['alt']+j['ant height'] for i,j in french_meta.items()})

# https://cdn.knmi.nl/knmi/pdf/bibliotheek/knmipubTR/TR293.pdf
# https://gemeenteraad.westbetuwe.nl/raadsinformatie/Overige-informatie/bijlage-emailverkeer-inwoners-Herwijnen-KNMI-Radar-data-sheet.pdf
# https://dataplatform.knmi.nl/dataset/cesar-idra-products-lb1-t00-v1-0
radar_elevations_manual = {'De Bilt':44,'Den Helder':52,'Herwijnen':22,'Cabauw':213}

radar_elevations = {**radar_elevations_fromfile, **radar_elevations_manual}

with open('radar_elevs_eu.txt', 'r', encoding='utf-8') as f:
    data = ft.list_data(f.read(), '\t')
    elevs = {i[0]:int(round(float(i[1]))) for i in data}
elevs.update({i:j['alt'] for i,j in french_meta.items()})

radar_bands = {j:'C' if j != 'Cabauw' else 'X' for j in radars}
radar_bands.update({i:j['band'] for i,j in french_meta.items()})



if __name__ == '__main__':    
    lats, lons = {i:j[0] for i,j in radarcoords.items()}, {i:j[1] for i,j in radarcoords.items()}
    attrs = [radars, rplaces_to_ridentifiers, lats, lons, radar_elevations, elevs, radar_bands]
    attrs = [i if isinstance(i, list) else [i[j] for j in radars] for i in attrs]
    lengths = [max(len(str(i)) for i in j) for j in attrs]
    
    source = None
    with open('D:/NLradar/Input_files/radars_eu.txt', 'w', encoding='utf-8') as f:
        for k,r in enumerate(radars):
            if data_sources[r] != source:
                if source:
                    f.write('\n')
                source = data_sources[r]
                f.write(source+'\n')
            for i,j in enumerate(attrs):
                s = str(j[k]) if j[k] != '' else '.'
                tabs = '\t'*int(np.ceil((np.ceil((lengths[i]+1)/8)*8-len(s))/8))
                f.write(s+tabs*(i+1 < len(attrs)))
            if not k == len(radars)-1:
                f.write('\n')