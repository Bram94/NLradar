import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))) #Add the Code directory to the path, to enable relative imports
import csv
import numpy as np

import nlr_functions as ft
import nlr_globalvars as gv

colortables_dirs_filenames = {'mesh': 'D:/NLradar/Input_files/Color_tables/Default/colortable_MESH_default.csv'}
products = ('mesh',)
productunits = {'mesh': 'cm'}


def set_colortables(colortables_dirs_filenames, products, productunits): #Is not ready yet. See return statement in nlradar_background.
    for j in products:
        datafile = open(colortables_dirs_filenames[j], 'r+')
        datareader = csv.reader(datafile)
        datareader_text=[row for row in datareader]
                                        
        data = np.zeros((0,7))
        for row in datareader_text:
            if (len(row)==4 or len(row)==7):
                test=[str(element)=='' or (not ft.to_number(str(element).strip()) is None and (element==row[0] or 0<=float(element)<=255)) for element in row]
                if all(test)==True:
                    if len(row)==4: appendarray=[-1,-1,-1]
                    else: appendarray=[]   
                    data=np.concatenate((data,[np.concatenate([row,appendarray])])) 
        datafile.close()
            
        cmapfilename_add='colortable_'+j+'_added.csv'
        np.savetxt('D:/NLradar/Generated_files/'+cmapfilename_add,data,fmt='%s',newline='\r\n',delimiter=',')

set_colortables(colortables_dirs_filenames, products, productunits)