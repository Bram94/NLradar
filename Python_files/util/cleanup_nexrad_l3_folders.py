# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:38:31 2024

@author: bramv

Cleans up NEXRAD L3 directories that contain all available level 3 files, i.e. also many files that are not read by NLradar. These files are removed.
"""
import os


product_labels_l3 = {'z':{'N':['R','Z'], 'T':['Z','R']}, 'v':'V'}

def get_level(filename):
    return 3 if not filename[5].isdigit() else 2
def get_l3_type(filename):
    return filename[12] # Either 'N' for NEXRAD or 'T' for TDWR
def get_product_labels(l3_type, product):
    labels = product_labels_l3[product]
    if type(labels) == dict:
        labels = labels[l3_type]
    return labels #can be either a 1-character string or a list with > 1 element. Both are iterable, so no need to put string in list

basedir = 'D:/radar_data_NLradar/NWS/'
date_dirs = os.listdir(basedir)
for i in date_dirs:
    radar_dirs = os.listdir(basedir+i)
    for j in radar_dirs:
        directory = basedir+i+'/'+j+'/'
        
        entries = os.listdir(directory)
        if not entries or not get_level(entries[0]) == 3:
            continue
        
        # There are many kinds of L3 files. Here only those that contain the desired products are kept
        _entries = [j for j in entries if 'SDUS' in j]
        l3_type = get_l3_type(_entries[0])
        keep = []
        for p in product_labels_l3:
            plabels = get_product_labels(l3_type, p)
            for plabel in plabels:
                hits = [j for j in _entries if j[12] == l3_type and
                        ((j[13].isdigit() and j[14] == plabel) or (j[14].isdigit() and j[13] == plabel))]
                if hits:
                    keep += hits
                    break # Use only files for the first plabel for which files are available
                    
        for j in entries:
            if not j in keep:
                print(j)
                os.remove(directory+j)