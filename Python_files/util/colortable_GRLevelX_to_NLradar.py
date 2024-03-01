# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 20:41:04 2023

@author: -
"""
import nlr_functions as ft

filename = "C:/Users/bramv/Downloads/SamBV.pal"
with open(filename, 'r') as f:
    text = f.read()
    
content = ft.list_data(text.lower(), separator=' ')

settings = [j for j in content if j[0] in (i+':' for i in ('step', 'units'))]
string = ''
for j in settings:
    string += j[0]+' '+j[1]+'\n'
string += '\n'

f = 1.60934/1.852
f = 1
colors = [[float(j[1])*f]+j[2:] for j in content if j[0] == 'color:']
colors = sorted(colors)
print(colors)


for j in colors:
    string += ','.join([str(ft.rifdot0(j[0]))]+j[1:])+'\n'
string = string[:-1]        
print(string)
filename = "D:/NLradar/Input_files/Color_tables/NWS/colortable_V_test.csv"
with open(filename, 'w') as f:
    f.write(string)