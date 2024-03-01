# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:42:07 2021

@author: -
"""
import pickle
import os
import nlr_globalvars as gv
import nlr_functions as ft

gv.programdir = 'D:/NLradar'
cases_lists_filename = os.path.join(gv.programdir+'/Generated_files','cases_lists.pkl')
if os.path.exists(cases_lists_filename):
    with open(cases_lists_filename, 'rb') as f:
        cases_lists=pickle.load(f)
        
print(cases_lists)
for list_name, case_list in cases_lists.items():
    for case_dict in case_list:
        if 'stormmotion' in case_dict and not isinstance(case_dict['stormmotion'], dict):
            stormmotion = case_dict['stormmotion']
            case_dict['stormmotion'] = {'sm':stormmotion, 'radar':case_dict['radar']}
            
with open(cases_lists_filename,'wb') as f:
    pickle.dump(cases_lists,f)