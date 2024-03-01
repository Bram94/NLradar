# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 20:41:04 2023

@author: -
"""
import nlr_functions as ft

filename = "D:/NLradar/Input_files/Color_tables/Default/colortable_ET_default.csv"
with open(filename, 'r') as f:
    text = f.read()
    
content = ft.list_data(text, separator=',')
string = ''
for i,j in enumerate(content):
    jm1 = content[i-1]
    if len(j) == 7:
        string += ','.join(j[:1]+j[4:]+j[1:4])+'\n'
    elif len(j) == 4:
        if len(jm1) == 4:
            string += ','.join(jm1[:1]+j[1:])+'\n'+','.join(j[:1]+jm1[1:])+'\n'
    else:
        string += ' '.join(j)+'\n'
        
print(string)
filename = "D:/NLradar/Input_files/Color_tables/Default/colortable_ET_default.csv"
with open(filename, 'w') as f:
    f.write(string)