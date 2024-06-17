# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import nlr_functions as ft
import nlr_globalvars as gv

import numpy as np
import os
opa=os.path.abspath
import re
import time as pytime
try:
    from pyshp import shapefile
except ImportError: import shapefile
import cv2
from PIL import ImageFont

import warnings
warnings.simplefilter("ignore", ResourceWarning) #Prevent ResourceWarnings for unclosed files and sockets from showing up



def import_shapefiles():
    shapefiles_latlon=[]; connect=[]; parts=[]
    #The reason for first opening the individual files and then creating the shapefile.Reader object is that it allows me to close the files, such that
    #no ResourceWarnings are raised anymore for unclosed files. Using warnings.simplefilter("ignore", ResourceWarning) does not work here unfortunately.
    shapefile_types = {'combined3':'countries','provinces_BE-NL(1)_Utrecht-correct':'provinces',\
                       'DEU_adm1(3)':'provinces','gadm34_FRA_2':'provinces','gadm36_POL_1':'provinces',\
                       'gadm36_CZE_1':'provinces','full_rivers_merge_nearNL':'rivers',
                       'cb_2018_us_state_500k':'countries','gadm41_CUB_0':'countries','gadm41_MEX_0':'countries',
                       'gadm41_CAN_0':'countries','gadm41_BHS_0':'countries','gadm41_DOM_0':'countries',
                       'gadm41_HTI_0':'countries','gadm41_FIN_0':'countries','gadm41_NOR_0':'countries',
                       'gadm41_SWE_0':'countries'}
    shapefiles = {}
    shapefile_readers = {}
    for j in shapefile_types:
        shapefiles[j] = {}
        shapefiles[j]['shp'] = open(opa(os.path.join(gv.programdir+'/Input_files/Shapefiles',j+'.shp')), "rb")
        shapefiles[j]['dbf'] = open(opa(os.path.join(gv.programdir+'/Input_files/Shapefiles',j+'.dbf')), "rb")
        shapefile_readers[j] = shapefile.Reader(shp = shapefiles[j]['shp'],dbf = shapefiles[j]['dbf'])
        
            
    shapefiles_latlon_combined={}
    shapefiles_connect_combined={}
    for i in shapefile_readers:
        s_type = shapefile_types[i]
        
        shapes=shapefile_readers[i].shapes()
        for j in range(len(shapes)):
            shape=shapes[j]
            parts=np.array(shape.parts)
            connect_add=np.ones(len(np.array(shape.points)), dtype='bool')
            connect_add[parts-1]=0 #The first number in the array parts is always zero, so using parts-1 causes the last number of connect_add to be set equal to zero.
            if j==0: 
                shapefiles_latlon=np.array(shape.points)
                connect=connect_add
            else: 
                shapefiles_latlon=np.concatenate((shapefiles_latlon,np.array(shape.points)),axis=0)
                connect=np.append(connect, connect_add)
        
        if not s_type in shapefiles_latlon_combined:
            shapefiles_latlon_combined[s_type]=np.transpose(np.transpose(shapefiles_latlon)[::-1])
            shapefiles_connect_combined[s_type]=connect
        else:
            shapefiles_latlon_combined[s_type]=np.concatenate([shapefiles_latlon_combined[s_type],np.transpose(np.transpose(shapefiles_latlon)[::-1])],axis=0)
            shapefiles_connect_combined[s_type]=np.append(shapefiles_connect_combined[s_type],connect)
            
    for j in shapefile_types:
        shapefiles[j]['shp'].close(); shapefiles[j]['dbf'].close()
    return shapefiles_latlon_combined, shapefiles_connect_combined



tickslim_modified={}; excluded_values_for_ticks={}; included_values_for_ticks={}; ticks_steps={}; content_before={}; last_modification_time_before={}
for j in gv.products_all:
    tickslim_modified[j]={'start':None,'end':None}
    excluded_values_for_ticks[j]=[]; included_values_for_ticks[j]=[]; ticks_steps[j]='-'
    content_before[j]={}
    last_modification_time_before[j]=0
def set_colortables(colortables_dirs_filenames,products,productunits): #Is not ready yet. See return statement in nlradar_background.
    global excluded_values_for_ticks, included_values_for_ticks, ticks_steps, content_before, last_modification_time_before, changed_colortables
    changed_colortables=[]; new_colortable={}; last_modification_time={}
    for j in products:
        use_black_colortable=0
        try:
            last_modification_time[j]=os.path.getmtime(colortables_dirs_filenames[j])
            new_colortable[j]=1 if last_modification_time_before[j]!=last_modification_time[j] else 0
            last_modification_time_before[j]=last_modification_time[j]
        except Exception as e:
            print(e,'set_colortables_1')
            if os.path.isfile(gv.colortables_dirs_filenames_Default[j]):
                colortables_dirs_filenames[j]=gv.colortables_dirs_filenames_Default[j]
                last_modification_time[j]=os.path.getmtime(colortables_dirs_filenames[j])
                new_colortable[j]=1 if last_modification_time_before[j]!=last_modification_time[j] else 0
                last_modification_time_before[j]=last_modification_time[j]
            else:
                use_black_colortable=1; new_colortable[j]=1; changed_colortables.append(j)
                         
        if new_colortable[j] and not use_black_colortable:
            try: 
                with open(colortables_dirs_filenames[j], 'r+') as f:
                    content = ft.list_data(f.read().lower())
                if content_before[j]==content:
                    new_colortable[j]=0
                else:
                    new_colortable[j]=1; changed_colortables.append(j)
                content_before[j]=content.copy()
                                
                if new_colortable[j]==1:
                    tickslim_modified[j]={'start':None,'end':None}
                    excluded_values_for_ticks[j]=[]; included_values_for_ticks[j]=[]; ticks_steps[j]='-'  
                    
                    data = np.zeros((0,7))
                    for i in range(len(content)):
                        row = content[i]
                        if len(row) in (4, 7):
                            test=[str(element)=='' or (not ft.to_number(str(element).strip()) is None and (element==row[0] or 0<=float(element)<=255)) for element in row]
                            if all(test):
                                if len(row)==4: appendarray=[-1,-1,-1]
                                else: appendarray=[]
                                data=np.concatenate((data,[np.concatenate([row,appendarray])])) 
                            else: 
                                use_black_colortable=1
                        elif len(row)>0 and row[0][:6]=='units:':
                            if j in ('v','w'):
                                if row[0][row[0].index(':')+1:].strip() in gv.scale_factors_velocities:
                                    productunits[j]=row[0][row[0].index(':')+1:].strip()
                                    if j=='v':
                                        productunits['s']=productunits['v']
                        elif len(row)>0 and row[0][:5]=='step:':
                            ticks_steps[j]=float(row[0][row[0].index(':')+1:].strip())
                        elif len(row)>0 and row[0][:18] in ('exclude for ticks:','include for ticks:'):
                            string=row[0][row[0].index(':')+1:].strip()
                            string_extra=''
                            for i in range(1,len(row)):
                                string_extra=string_extra+','+row[i]
                            string=string+string_extra
                            string_remainder=string
                            values_input=[]
                            while len(string_remainder)>0:
                                string_comma_pos=string_remainder.find(',')
                                if string_comma_pos!=-1:
                                    values_input.append(string_remainder[:string_remainder.find(',')])
                                    string_remainder=string_remainder[string_remainder.find(',')+1:]
                                else:
                                    values_input.append(string_remainder)
                                    string_remainder=''
                            if row[0][:18]=='exclude for ticks:':
                                excluded_values_for_ticks[j]=list(map(lambda x: ft.round_float(float(x)),ft.numbers_list(values_input)))
                            else:
                                included_values_for_ticks[j]=list(map(lambda x: ft.round_float(float(x)),ft.numbers_list(values_input)))
                        elif len(row) == 1 and not ft.to_number(str(row[0]).strip()) is None:
                            value = ft.to_number(str(row[0]).strip())
                            i_next = i+1
                            while i_next < len(content) and len(content[i_next]) == 0:
                                i_next += 1
                            i_prev = i-1
                            while i_prev >= 0 and len(content[i_prev]) == 0:
                                i_prev -= 1
                            key = 'end' if (i_next < len(content) and not ft.to_number(content[i_next][0]) is None\
                                and value < ft.to_number(content[i_next][0])) or (i_prev >= 0 and not\
                                ft.to_number(content[i_prev][0]) is None and value < ft.to_number(content[i_prev][0])) else 'start'
                            tickslim_modified[j][key] = value
                        elif len(row)>0:
                            use_black_colortable=1
            except Exception as e:
                print(e,'set_colortables2')
                use_black_colortable=1; new_colortable[j]=1; changed_colortables.append(j)
                tickslim_modified[j]={'start':None,'end':None}
                excluded_values_for_ticks[j]=[]; included_values_for_ticks[j]=[]; ticks_steps[j]='-'    
                                    
        if new_colortable[j]==1:                       
            if use_black_colortable==1:
                data=np.array([list(map(str, row)) for row in [[gv.products_maxrange[j][-1],0,0,0,-1,-1,-1],[gv.products_maxrange[j][0],0,0,0,-1,-1,-1]]])
            productvalues=data[:,0]
                    
            if j=='r': 
                for i in range(0,len(productvalues)):
                    if float(productvalues[i])<0.001: productvalues[i]='0.001'
                        
            if float(productvalues[0])<float(productvalues[-1]): data=data[::-1]

            cmapfilename_add='colortable_'+j+'_added.csv'
            np.savetxt(opa(os.path.join(gv.programdir+'/Generated_files',cmapfilename_add)),data,fmt='%s',newline='\r\n',delimiter=',')
            
    return tickslim_modified, ticks_steps, excluded_values_for_ticks, included_values_for_ticks, changed_colortables, productunits



"""cbars_pos_all gives for each number of panels and for each number of unique products the desired colorbar position for a given panel in 
which te product is displayed. The position ranges from 0 to 7, where 0-3 means a position on the left side, from top to bottom, and 4-7 means
a position on the right side, again from top to bottom. For a number of panels for which some panels are not located on the edge of the screen (in the
x direction), the colorbars could be positioned on either side of the screen. In this case, the colorbar position can be one of multiple 
positions, given in a list, and the one chosen depends on which positions are still available.
"""
cbars_pos_all={1:{1:{0:0}},
                     2:{1:{0:0,5:0},2:{0:0,5:4}},
                     3:{1:{0:0,1:0,2:0},2:{0:0,1:[0,4],2:4},3:{0:0,1:1,2:4}},
                     4:{1:{0:0,1:0,5:0,6:0},2:{0:0,1:4,5:0,6:4},3:{0:0,1:4,5:1,6:5},4:{0:0,1:4,5:1,6:5}},
                     6:{1:{0:0,1:0,2:0,5:0,6:0,7:0},2:{0:0,1:[0,4],2:4,5:0,6:[0,4],7:4},3:{0:0,1:[0,4,1],2:4,5:1,6:[1,5,0],7:5},4:{0:0,1:[0,4,1,5],2:4,5:1,6:[1,5,0,4],7:5},5:{0:0,1:1,2:4,5:2,6:5,7:6},6:{0:0,1:1,2:4,5:2,6:5,7:6}},
                     8:{1:{0:0,1:0,2:0,3:0,5:0,6:0,7:0,8:0},2:{0:0,1:[0,4],2:[4,0],3:4,5:0,6:[0,4],7:[4,0],8:4},
                        3:{0:0,1:[0,1,4],2:[4,5,0],3:4,5:1,6:[1,0,5],7:[5,4,1],8:5},4:{0:0,1:[0,1,4,5],2:[4,5,0,1],3:4,5:1,6:[1,0,5,4],7:[5,4,1,0],8:5},
                        5:{0:0,1:[0,1,2,4,5],2:[4,5,6,0,1],3:4,5:2,6:[2,1,0,6,5],7:[6,5,4,2,1],8:6},6:{0:0,1:[0,1,2,4,5,6],2:[4,5,6,0,1,2],3:4,5:2,6:[2,1,0,6,5,4],7:[6,5,4,2,1,0],8:6},
                        7:{0:0,1:1,2:5,3:4,5:3,6:2,7:6,8:7},8:{0:0,1:1,2:5,3:4,5:3,6:2,7:6,8:7}},
                     10:{1:{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0},2:{0:0,1:[0,4],2:[0,4],3:[4,0],4:4,5:0,6:[0,4],7:[4,0],8:[4,0],9:4},
                        3:{0:0,1:[0,1,4],2:[0,4,1],3:[4,5,0],4:4,5:1,6:[1,0,5],7:[5,1,4],8:[5,4,1],9:5},4:{0:0,1:[0,1,4,5],2:[0,4,1,5],3:[4,5,0,1],4:4,5:1,6:[1,0,5,4],7:[5,1,4,0],8:[5,4,1,0],9:5},
                        5:{0:0,1:[0,1,2,4,5],2:[0,4,1,5,2],3:[4,5,6,0,1],4:4,5:2,6:[2,1,0,6,5],7:[6,2,5,1,4],8:[6,5,4,2,1],9:6},6:{0:0,1:[0,1,2,4,5,6],2:[0,4,1,5,2,6],3:[4,5,6,0,1,2],4:4,5:2,6:[2,1,0,6,5,4],7:[6,2,5,1,4,0],8:[6,5,4,2,1,0],9:6},
                        7:{0:0,1:[0,1,2,3,4,5,6],2:[0,4,1,5,2,6,3],3:[4,5,6,7,0,1,2],4:4,5:3,6:[3,2,1,0,7,6,5],7:[7,6,5,4,3,2,1],8:[7,6,5,4,3,2,1],9:7},8:{0:0,1:[0,1,2,3,4,5,6,7],2:[0,4,1,5,2,6,3,7],3:[4,5,6,7,0,1,2,3],4:4,5:3,6:[3,2,1,0,7,6,5,4],7:[7,3,6,2,5,1,4,0],8:[7,6,5,4,3,2,1,0],9:7}}
                     }

products_colortables_before=[]; changed_colortables=[]
def determine_colortables(all_products,panels,panellist):
    global cbars_products, products_colortables_before, changed_colortables
    products=[all_products[j] for j in panellist]
    if len(products)==0 or len(panellist)==0: 
        products=['z']; panellist = [0] #Always show at least one color bar

    cbars_products, indices=np.unique(np.array(products),return_index=True)
    sorted_indices=np.sort(indices)
    cbars_products=list(np.array(products)[sorted_indices]) #np.unique doesn't preserve order
    cbars_nproducts_unique=len(cbars_products)
    cbars_nproducts=cbars_nproducts_unique if len(products)>1 else cbars_nproducts_unique+1
            
    panellist_indices=[panellist[j] for j in sorted_indices] #Gives for each product a panel in which it is displayed.
    cbars_pos={}; update_cbars_pos=[]; update_cbars_products=[]
    
    """First assign colorbar positions for products that are displayed in a panel that is located at the edge of the screen (cbars_pos_all
    does not contain a list).
    """
    for j in range(0,len(panellist_indices)):
        if not type(cbars_pos_all[panels][cbars_nproducts_unique][panellist_indices[j]])==list:
            cbars_pos_j_try=cbars_pos_all[panels][cbars_nproducts_unique][panellist_indices[j]]
            if cbars_pos_j_try not in cbars_pos.values():
                cbars_pos[j]=cbars_pos_j_try
            else:
                panellist_product_j=[i for i in panellist if all_products[i]==cbars_products[j]]
                for i in range(1,len(panellist_product_j)):
                    if not type(cbars_pos_all[panels][cbars_nproducts_unique][panellist_product_j[i]])==list:
                        cbars_pos_j_try=cbars_pos_all[panels][cbars_nproducts_unique][panellist_product_j[i]]
                        if cbars_pos_j_try not in cbars_pos.values():
                            cbars_pos[j]=cbars_pos_j_try      
    
    """It is possible that for some colorbar products no position could be assigned in the above loop, although 
    type(cbars_pos_all[panels][cbars_nproducts_unique][panellist_indices[j]]) is not a list. This occurs when all positions at which a particular
    colorbar can be positioned (given the panel(s) in which the product is located) are already occupied by others.
    For these products the position is determined here, and it is one of the positions that are not yet occupied.
    """
    for j in range(0,len(panellist_indices)):
        if not j in cbars_pos and not type(cbars_pos_all[panels][cbars_nproducts_unique][panellist_indices[j]])==list:       
            current_pos_list=[[i] if not type(i)==list else i for i in list(cbars_pos_all[panels][cbars_nproducts_unique].values())]
            current_pos=[]
            for i in current_pos_list:
                current_pos+=i
            current_pos_unique=np.unique(current_pos)
            cbars_pos[j]=[i for i in current_pos_unique if not i in cbars_pos.values()][0]
                                            
    """Next assign colorbar positions for products that are displayed in a panel that is not located at the edge of the screen (cbars_pos_all
    contains a list). This is done by iterating through each ith element in the lists with positions, for increasing values of i,
    in order to give each panel a fair chance at getting its desired cbar position. This requires that the lists with positions
    have the same length for each panel!!! When a position is not yet in cbars_pos.values(), then that the product in that panel
    gets assigned that cbar position.
    """
    remaining_panels = [j for j in panellist_indices if isinstance(cbars_pos_all[panels][cbars_nproducts_unique][j], list)]
    if len(remaining_panels) > 0:
        n = len(cbars_pos_all[panels][cbars_nproducts_unique][remaining_panels[0]])
        for i in range(n):
            for j in remaining_panels:
                index_j = panellist_indices.index(j)
                pos_list = cbars_pos_all[panels][cbars_nproducts_unique][j]
                if not index_j in cbars_pos and not pos_list[i] in cbars_pos.values():
                    cbars_pos[index_j] = pos_list[i]
                
                    if index_j>=len(products_colortables_before) or cbars_products[index_j]!=products_colortables_before[index_j]:
                        update_cbars_pos.append(cbars_pos[index_j])
                        update_cbars_products.append(cbars_products[index_j])
                
    """When only one product is shown, then it is deemed desired to display the color bar for this product on both sides of the radar panels, in order
    to have at least one color bar on each side.
    When the number of color bars is odd, then one of the positions as given in cbars_pos_all needs to be adjusted, in order to prevent that a color bar
    position is skipped. (e.g. positions 0 and 2 are available, but not 1. In that case 2 gets adjusted to 1)
    """
    if len(cbars_products) == 1:
        cbars_pos[1]=4
        cbars_products += [products[0]]
        
        if len(products_colortables_before)<2 or cbars_products[1]!=products_colortables_before[1]:
            update_cbars_pos.append(cbars_pos[1])
            update_cbars_products.append(cbars_products[1])
    elif cbars_nproducts%2 == 1:
        n_left = len([j for j in cbars_pos.values() if j < 4])
        n_right = cbars_nproducts - n_left
        #Don't use a dict comprehension here, since cbars_pos gets updated along the way,
        #and it is necessary to take this into account!!!
        minpos_left = min([j for j in cbars_pos.values() if j < 4])
        minpos_right = min([j for j in cbars_pos.values() if j > 3])
        for j in cbars_pos:
            if cbars_pos[j] < 4:
                cbars_pos[j] -= minpos_left
            else:
                cbars_pos[j] -= minpos_right-4
        posmax = cbars_nproducts//2
        while [j in (posmax, posmax+4) for j in cbars_pos.values()].count(True) == 2: 
            #One iteration of the procedure below might not be enough, so a loop is used.
            for j in cbars_pos:
                if n_left > n_right and cbars_pos[j]>4 and not cbars_pos[j]-1 in cbars_pos.values():
                    cbars_pos[j] -= 1
                elif n_right > n_left and 0<cbars_pos[j]<4 and not cbars_pos[j]-1 in cbars_pos.values():
                    cbars_pos[j] -= 1
        
    products_colortables_before=cbars_products
    return cbars_products, cbars_nproducts, cbars_pos, update_cbars_pos, update_cbars_products



def get_substrings_in_dir_string(dir_string):
    n_slash=dir_string.count('/')
    substrings=[]
    j_max=n_slash
    dir_string_copy=dir_string
    for j in range(0,j_max+1):
        if j<j_max:
            slash_index=dir_string_copy.index('/')
            substrings.append(dir_string_copy[:slash_index])
            dir_string_copy=dir_string_copy[slash_index+1:]
        else:
            substrings.append(dir_string_copy)
    return substrings

def check_correctness_dir_string(dir_string):
    """This function checks whether dir_string is correctly formatted, i.e. it checks whether variables have been included appropriately. 
    Returns True if this is the case, and False otherwise.
    """
    substrings=get_substrings_in_dir_string(dir_string)
    
    substrings_copy=substrings.copy()
    variables=[]
    substrings_indices=[]
    dir_string_novariables='' #Contains the part of the dir_string that does not contain variables (but it could contain $'s etc, as can
    #be the case when variables have been incorrectly included. This is checked below.)
    for j in range(0,len(substrings_copy)):
        while True:
            try:
                index1=substrings_copy[j].index('${')
                index2=substrings_copy[j].index('}')
                
                dir_string_novariables+=substrings_copy[j][:index1]
                variables.append(substrings_copy[j][index1+2:index2])
                substrings_indices.append(j)
                substrings_copy[j]=substrings_copy[j][index2+1:]
            except Exception: 
                dir_string_novariables+=substrings_copy[j]
                break
            
    #'\\' is the character used for a single backslash in Python, because '\' is used for escaping. Only forward slashes are allowed in
    #a dir_string.
    if any([j in dir_string_novariables for j in ('$','{','}','\\')]):
        return False
                
    allowed_minutes=[j for j in range(1,1441) if np.mod(1440/j,1)==0.]
    allowed_variables=['radarID','radar','date','time']+['time'+str(j) for j in allowed_minutes]+['datetime'+str(j) for j in allowed_minutes]
    allowed_variables_plusminus=['date','time']+['time'+str(j) for j in allowed_minutes]+['datetime'+str(j) for j in allowed_minutes]
    
    #Date and time variables are not allowed to appear in more than one substring, i.e. all date variables must be located in the same
    #substring, and the same holds for all time variables.
    #Further, when datetime variables are included, then no other date and time variables are allowed in other substrings.
    for j in ('date','time'):
        substrings_indices_j=[]
        for i in range(0,len(variables)):
            if j in variables[i]:
                substrings_indices_j.append(substrings_indices[i])
        if len(np.unique(substrings_indices_j))>1:
            return False
        
    if '${time}' in dir_string:
        #A time variable must include the number of minutes to which it should be floored, i.e. must have the form timeX.
        return False
        
    if '${date' in dir_string and '${time' in dir_string:
        #Time variables are not allowed to appear before date variables in dir_string.
        index_date=dir_string.index('${date')
        index_time=dir_string.index('${time')
        if index_time<index_date:
            return False
                        
    for j in range(0,len(variables)):
        v_j=variables[j]
        plus_count=v_j.count('+')
        plus=True if plus_count==1 else False
        j_plusminus=any([i in v_j and plus for i in allowed_variables_plusminus])

        if not v_j in allowed_variables and not j_plusminus:
            return False
        elif j_plusminus:
            try:                
                plus_index=v_j.index('+')
                if len(v_j)>plus_index+1:
                    #No characters are allowed after the plus/minus sign.
                    return False
                
                characters_before_plus=v_j[:plus_index]
                if not characters_before_plus in allowed_variables_plusminus:
                    return False
                #When j_plusminus, then characters_before_plus must also be in variables. 
                #Further, they must appear in the same substring, and the variable without the plus must appear before the variable with the plus.
                if characters_before_plus in variables:
                    index=variables.index(characters_before_plus)
        
                    if substrings_indices[index]!=substrings_indices[j]:
                        return False
                    if substrings[substrings_indices[j]].index(characters_before_plus)==substrings[substrings_indices[index]].index(v_j):
                        #This means that the variable without the plus is positioned after the variable with the plus, which is not allowed.
                        return False
                else: 
                    return False
            except Exception:
                return False
    return True
    
def dirstring_to_dirlist(dir_str):
    dir_list=[]
    
    common_part=''
    if ';;' in dir_str:
        index=dir_str.index(';;')
        common_part=dir_str[:index].strip()
        dir_str=dir_str[index+2:]
        
    index=dir_str.find(';')
    while index!=-1:
        dir_list.append((common_part+dir_str[:index].strip()).replace('//','/')) #Remove white spaces at the start and end, end remove a possible
        #double / at the transition between common_part and the new part of the directory string, which would be present when the new part starts with
        #a /.
        dir_str=dir_str[index+1:]
        index=dir_str.find(';')
        if all(j in (';', ' ') for j in dir_str[index:]):
            # In this case a ';' has been appended to the final dirstring, i.e. it isn't followed anymore by a new dirstring.
            dir_str = dir_str[:index]
            break
    dir_list.append((common_part+dir_str.strip()).replace('//','/'))
    return dir_list

def convert_dir_string_to_real_dir(dir_string,radar,date,time):
    """Converts a directory string with variables (${variable}) in it to a real directory.
    The supported variables are ${date}, ${date+}, ${timeX} ${timeX+}, ${datetimeX} and ${datetimeX+}.
    X denotes the number of minutes to which a time will be floored in order to get the time in the directory (e.g. with X=60 1442 becomes 1400).
    """
    dir_string = replace_radar_variables(dir_string,radar)
    
    real_dir=''
    while True:
        try:
            index1=dir_string.index('${')
            real_dir+=dir_string[:index1]
            dir_string=dir_string[index1+2:]
            
            index2=dir_string.index('}')
            var=dir_string[:index2]
            
            if var[:8]=='datetime':
                if not '+' in var:
                    n_minutes=eval(var[8:])
                    floored_datetime=date+ft.floor_time(time, n_minutes)
                    real_dir+=floored_datetime 
                else:
                    n_minutes=eval(var[8:-1])
                    floored_time=ft.floor_time(time, n_minutes)
                    real_dir+=''.join(ft.next_date_and_time(date,floored_time,n_minutes)) 
            elif var[:4]=='date':
                if not '+' in var:  
                    real_dir+=date
                else:
                    real_dir+=ft.next_date_and_time(date,'0000',1440)[0]
            elif var[:4]=='time':
                if not '+' in var:
                    n_minutes=eval(var[4:])
                    floored_time=ft.floor_time(time, n_minutes)
                    real_dir+=floored_time 
                else:
                    n_minutes=eval(var[4:-1])
                    floored_time=ft.floor_time(time, n_minutes)
                    real_dir+=ft.next_date_and_time('20000101',floored_time,n_minutes)[1]  
                    
            dir_string=dir_string[index2+1:]
            
        except Exception:
            real_dir+=dir_string
            break
    return real_dir

def get_date_and_time_from_dir(real_dir,dir_string,radar = None):
    """Determine the date and time from real_dir based on dir_string. Returns None if the date and/or time is not present in dir_string.
    """
    if '${radar}' in dir_string or '${radarID}' in dir_string:
        if radar is None:
            raise Exception('radar variables present in dir_string, without radar specified. bg.get_date_and_time_from_dir.')
        dir_string = replace_radar_variables(dir_string,radar)
    
    date=None; time=None
    index_add=0 #Index_add is added to the indices below to compensate for the difference in lengths between the date and time variables, and the 
    #corresponding dates and times.
    if '${date}' in dir_string:
        index=dir_string.index('${date}')+index_add
        date=real_dir[index:index+8]
        index_add+=1
    if '${time' in dir_string:
        index1=dir_string.index('${time')
        index2=index1+dir_string[index1:].index('}')
        
        index=index1+index_add
        time=real_dir[index:index+4]
        index_add-=1+index2-index1-4
    if '${datetime' in dir_string:
        index1=dir_string.index('${datetime')
        index2=index1+dir_string[index1:].index('}')
        
        index=index1+index_add
        datetime=real_dir[index:index+12]
        date=datetime[:8]; time=datetime[-4:]
        index_add-=1+index2-index1-12 
    return date, time

def get_datetime_bounds_dir_string(dir_string, date, time):
    """Returns the boundaries of the datetime interval that is represented by the directory for dir_string that contains the inputted date and time.
    The returned end datetime is actually the start datetime of the next directory.
    It is assumed that dir_string contains at least a $date variable, otherwise errors occur below.
    """
    i1 = [i.end() for i in re.finditer('\$\{', dir_string)]
    i2 = [i.start() for i in re.finditer('\}', dir_string)]
    variables = [dir_string[i1[j]:i2[j]] for j in range(len(i1))]
    for var in variables:
        plus_present = var[-1] == '+'
        if 'date' in var:
            start_datetime, end_datetime = date+'0000', ft.next_date(date, 1)+'0000'
        if 'time' in var:
            n_minutes = int(var[4:-1 if plus_present else None])
            start_datetime, end_datetime = ft.floor_datetime(date+time, n_minutes), ft.ceil_datetime(date+time, n_minutes)
    return start_datetime, end_datetime
        
correctness_direntries_per_substring={}
def check_correspondence_real_dir_to_dir_string(real_dir,dir_string):
    global correctness_direntries_per_substring
    """Checks whether a real directory has the format of dir_string. Returns True if this is the case, and False otherwise.
    """
    #First check whether it has already been determined before for this real_dir whether it is correct, and if so, return that result.
    #This is done because the repeated evaluation of this function appeared to be a bottleneck for speed.
    try:
        return correctness_direntries_per_substring[dir_string][real_dir]
    except Exception:
        pass
    
    date, time=get_date_and_time_from_dir(real_dir,dir_string)
    
    test_date=date if not date is None else '20010101'
    test_time=time if not time is None else '0000'
    if not ft.correct_datetimeinput(test_date,test_time) or test_date=='c' or test_time=='c':
        #In this case the directory name does not satisfy the format of dir_string, because the 'dates' and 'times' in real_dir where
        #apparently no dates/times.
        return False
                        
    radar=None #Not needed
    real_dir_string=convert_dir_string_to_real_dir(dir_string,radar,date,time)
    correct=real_dir_string==real_dir
    
    if not dir_string in correctness_direntries_per_substring:
        correctness_direntries_per_substring[dir_string]={}
    correctness_direntries_per_substring[dir_string][real_dir]=correct
    
    return correct
    
def check_dir_empty(dir_path):
    """Checks whether a directory is empty, and returns True/False if this is the case/not the case.
    It is checked by means of os.scandir, which returns an iterator. If there is at least one object in the iterator, then the directory is
    not empty. 
    This is much faster than using os.listdir when the directory is large, because os.scandir does not automatically return all files
    in the directory.
    """
        
    dir_empty=True
    iterator=os.scandir(dir_path)
    for j in iterator:
        dir_empty=False
        iterator.close()
        break
    return dir_empty

def check_dir_empty_radar(radar,dir_path,function_get_filenames_directory):
    """Checks whether a directory does not include files for a particular radar. Returns True if this is the case, and False otherwise.
    """
    filenames=function_get_filenames_directory(radar,dir_path)
    return len(filenames) == 0
    
def get_direntries(abs_path,compare_substring):
    direntries = []
    if os.path.exists(abs_path):
        # Earlier os.path.isdir was called to make sure that all dir entries are actually directories. But calling os.path.isdir appeared to
        # be quite slow on the first call, so it's now left out and hoped/assumed that the call of check_correspondence_real_dir_to_dir_string
        # filters out any non-directory entries.
        direntries = [j for j in os.listdir(abs_path) if check_correspondence_real_dir_to_dir_string(j, compare_substring)]
    return np.sort(direntries)
    
def replace_radar_variables(dir_string,radar):
    if '${radar}' in dir_string:
        #Replace ${radar} by the radar name without spaces.
        index=dir_string.index('${radar}')
        radar = gv.radars_nospecialchar_names[radar]
        dir_string=dir_string[:index]+radar.replace(' ','')+dir_string[index+len('${radar}'):]
    if '${radarID}' in dir_string:
        #Replace ${radarID} by the radar ID corresponding to the radar
        index=dir_string.index('${radarID}')
        dir_string=dir_string[:index]+gv.radar_ids[radar]+dir_string[index+len('${radarID}'):]
    return dir_string

def get_substringsindices_and_abspaths(substrings1,substrings2):
    """Returns absolute paths to a series of folders specified by a date/time variable. substrings1 should contain the variables, and substrings2
    should contain substrings for the relative path of the current directory. It can be set equal to substrings1 when you start with listing
    folders for the first absolute path (instead of the last, in which case a variable would remain in the absolute path when settings 
    substrings2=substrings1). This is because the last absolute path gets updated when having determined which folder to take from the series of
    folders in the first absolute path.
    When there is only one absolute path, then substrings2=substrings1 does also work.
    """
    abs_paths=[]
    substrings_indices=[]
    for j in range(0,len(substrings1)):
        if '${' in substrings1[j]:
            abs_paths.append(''.join([('' if i==0 else '/')+substrings2[i] for i in range(0,j)]))
            substrings_indices.append(j)
    return substrings_indices, abs_paths

def get_download_directory(dir_string):
    substrings=get_substrings_in_dir_string(dir_string)
    download_directory=''
    for j in range(0,len(substrings)):
        if not '${' in substrings[j]:
            download_directory+=('/' if j>0 else '')+substrings[j]
        else:
            break
    return opa(download_directory+'/Download')
    
def get_last_directory(dir_string,radar,function_get_filenames_directory,reverse=False):
    """Returns the absolute path of the last directory that contains data. This is done by first determining the last directory in
    the directory tree, and if that one is empty, then get_next_directory is called to obtain the nearest (in date and time) non-empty
    directory.
    Returns None if there are no directories that satisfy the format of dir_string.
    
    If reverse=True, then the first directory is returned.
    
    This function can handle empty directories, and skips them.
    """   
    dir_string=replace_radar_variables(dir_string,radar)
        
    if not '${' in dir_string:
        #In this case there are no date and/or time variables present in dir_string
        return opa(dir_string)
    
    index=0 if reverse else -1
    direction = 1 if reverse else -1    
    try:
        substrings=get_substrings_in_dir_string(dir_string)
        substrings_indices, abs_paths=get_substringsindices_and_abspaths(substrings,substrings)
        
        substring1=substrings[substrings_indices[0]]
        direntries1=get_direntries(opa(abs_paths[0]),substring1)
        while True:            
            last_directory1=direntries1[index]
                
            if len(abs_paths)==1:
                path_end=''.join(['/'+substrings[j] for j in range(substrings_indices[0]+1,len(substrings))])
                trial_last_directory_abspath=abs_paths[0]+'/'+last_directory1+path_end
                if not os.path.exists(trial_last_directory_abspath) and not index in (len(direntries1)-1, -len(direntries1)):
                    index += direction
                    continue
            else:
                path_end=''.join(['/'+substrings[j] for j in range(substrings_indices[0]+1,substrings_indices[-1])])
                abs_paths[-1]=abs_paths[0]+'/'+last_directory1+path_end
                if not os.path.exists(abs_paths[-1]) and not index in (len(direntries1)-1, -len(direntries1)):
                    index += direction
                    continue                
                                
                substring2=substrings[substrings_indices[-1]]
                direntries2=get_direntries(opa(abs_paths[-1]),substring2)
                last_directory2=direntries2[0 if reverse else -1]
                
                path_end=''.join(['/'+substrings[j] for j in range(substrings_indices[-1]+1,len(substrings))])
                trial_last_directory_abspath=abs_paths[-1]+'/'+last_directory2+path_end  
                               
            #opa is not used before, because using it could cause current_dir to contain '\' instead of '/'. 
            #Although this is not the case anymore, because backward slashes are replaced by forward slashes in current_dir in get_next_directory.
            if not check_dir_empty_radar(radar,opa(trial_last_directory_abspath),function_get_filenames_directory): #Skip empty directories
                return opa(trial_last_directory_abspath)
            else:
                return get_next_directory(dir_string,direction,radar,function_get_filenames_directory,current_dir=trial_last_directory_abspath)
    except Exception:
        #Can e.g. occur when there are no directories that satisfy the format.
        return None
    
def get_next_possible_dir_for_dir_string(dir_string, radar, date, time, direction):
    i1 = [i.end() for i in re.finditer('\$\{', dir_string)]
    i2 = [i.start() for i in re.finditer('\}', dir_string)]
    variables = [dir_string[i1[j]:i2[j]] for j in range(len(i1))]
    for var in variables[::-1]: # Run backward, to first check for times (to not break the for loop below too early)
        plus_present = var[-1] == '+'
        if 'time' in var:
            n_minutes = int(var[4:-1 if plus_present else None])
            date, time = ft.next_date_and_time(date, time, direction*n_minutes)
            break
        elif 'date' in var:
            date = ft.next_date(date, direction*1)
            break
    return convert_dir_string_to_real_dir(dir_string, radar, date, time)
    
def get_next_directory(dir_string,direction,radar,function_get_filenames_directory,current_dir=None,date=None,time=None):   
    """Given a particular dir_string, date and time, this function first determines the corresponding directory in which the program
    currently searches for files. The goal is then to determine the next directory in the direction given by direction (+-1), i.e. the 
    directory for the next combination of date and time for which files are available. 
    This is done by first determining at which positions in dir_string variables (with ${}) are located, because this determines where
    other directories can be found. Subsequently, directories are listed and sorted, such that the next one can be taken.
    
    If there are variables located in two substrings of dir_string (and with substrings I mean parts separated by a '/'), then it is e.g. 
    the case that there is a set of folders that contains the date as variable in its name, and inside these folders there are subfolders 
    that contain the time as variable in its name. In this case the function first searches for a possible next time subfolder, and if there
    is no next one, then it searches for the next date folder, and takes in this folder the first time subfolder.
    Empty directories are skipped.
    Date and time variables can be located in at maximum 2 parts of dir_string! Further, when a ${date+-x} variable is located in a part, then there
    must also be a ${date} variable, and the same holds for time variables! Finally, when there are time variables present in dir_string,
    then there also must be date variables.
    
    Returns the absolute path of the next directory if it exists, and returns the current directory if there is no next directory 
    (.e. the current subfolder is the last one in the directory tree).
    
    A '/' must be used as the separator in dir_string!
    current_dir is the path of the current directory.
    
    This function can handle empty directories, and skips them.
    """        
    dir_string=replace_radar_variables(dir_string,radar)
        
    if not '${' in dir_string:
        #In this case there are no date and/or time variables present in dir_string
        return opa(dir_string)
                
    if date:
        next_possible_dir = get_next_possible_dir_for_dir_string(dir_string, radar, date, time, direction)
        if os.path.exists(next_possible_dir) and not check_dir_empty_radar(radar,next_possible_dir,function_get_filenames_directory):
            return opa(next_possible_dir)
    
    try:
        if current_dir is None:
            current_dir=convert_dir_string_to_real_dir(dir_string,radar,date,time)
        else:
            current_dir=current_dir.replace('\\','/') #Ensure that '/' is used as the path separator.
                                         
        #substrings1 and substrings2 contain the different parts of dir_string and current_dir, that are separated by slashes.
        substrings1=get_substrings_in_dir_string(dir_string) #For dir_string
        substrings2=get_substrings_in_dir_string(current_dir) #For current_dir
                                    
        #There can be 1 or 2 parts of dir_string that contain variables, and to this correspond 1 or multiple series of directories.
        #abs_paths contains the absolute path that leads to a particular series of directories. If there are 2 series, than the path to the
        #inner (deeper) series of subfolders depends on the outer folder that is chosen, and if this outer folder changes, then the absolute
        #path to the inner series must also change. This is done below when necessary.
        substrings_indices, abs_paths=get_substringsindices_and_abspaths(substrings1,substrings2)
                        
        substring2=substrings1[substrings_indices[-1]]
        direntries2=get_direntries(abs_paths[-1],substring2)
        n_direntries2=len(direntries2)
        index2=np.where(direntries2==substrings2[substrings_indices[-1]])[0][0]
                
        next_directory_abspath=None
        if (direction==-1 and index2>0) or (direction==1 and index2<n_direntries2-1):
            #path_end contains the last part of the path name, i.e. the elements in substrings1 that come after the last element that contains
            #a $-variable.
            path_end=''.join(['/'+substrings1[j] for j in range(substrings_indices[-1]+1,len(substrings1))])
            new_index=index2
            while True:
                new_index+=direction
                if 0<=new_index<n_direntries2:
                    trial_next_directory_abspath=opa(abs_paths[-1]+'/'+direntries2[new_index]+path_end)
                    if not check_dir_empty_radar(radar,trial_next_directory_abspath,function_get_filenames_directory):
                        next_directory_abspath=trial_next_directory_abspath
                        break
                else:
                    break 
                
        if next_directory_abspath is None and len(abs_paths)>1:
            substring1=substrings1[substrings_indices[0]]
            direntries1=get_direntries(abs_paths[0],substring1)
            n_direntries1=len(direntries1)
            index1=np.where(direntries1==substrings2[substrings_indices[0]])[0][0]
            
            path_end1=''.join(['/'+substrings1[j] for j in range(substrings_indices[0]+1,substrings_indices[-1])])
            new_index1=index1
            while True:
                new_index1+=direction
                if next_directory_abspath is None and 0<=new_index1<n_direntries1:
                    abs_paths[-1]=opa(abs_paths[0]+'/'+direntries1[new_index1]+path_end1)
                    
                    substring2=substrings1[substrings_indices[-1]]
                    direntries2=get_direntries(abs_paths[-1],substring2)
                    n_direntries2=len(direntries2)
                    index2=n_direntries2 if direction==-1 else -1
                    
                    path_end2=''.join(['/'+substrings1[j] for j in range(substrings_indices[-1]+1,len(substrings1))])
                    new_index2=index2
                    while True:
                        new_index2+=direction
                        if 0<=new_index2<n_direntries2:
                            trial_next_directory_abspath=abs_paths[-1]+'/'+direntries2[new_index2]+path_end2
                            if not check_dir_empty_radar(radar,opa(trial_next_directory_abspath),function_get_filenames_directory): 
                                #Skip empty directories
                                next_directory_abspath=trial_next_directory_abspath
                                break
                        else:
                            break
                else:
                    break
                
                index1+=direction
                
        if next_directory_abspath is None:
            #Return the current directory
            return opa(current_dir)
        else:
            return opa(next_directory_abspath)   
    except Exception as e:
        print(e,'get_next_directory')
        #Can e.g. occur when there are no directories that satisfy the format of dir_string.
        return opa(current_dir)
    
def determine_nearest_dir(radar,trial_dir1,trial_dir2,desired_datetime,function_get_filenames_directory,function_get_datetimes_from_files):
    """Determines for 2 directories which of both contains the file whose datetime is closest to the desired datetime, and returns the absolute
    path of that directory.
    If one of the directories is empty, then the other is returned. If they are both empty, then still one of them gets returned.
    """
    filenames1=function_get_filenames_directory(radar,trial_dir1)
    filenames2=function_get_filenames_directory(radar,trial_dir2)
    datetimes1=function_get_datetimes_from_files(radar,filenames1,trial_dir1)
    datetimes2=function_get_datetimes_from_files(radar,filenames2,trial_dir2)
    
    if len(datetimes1)==0:
        return trial_dir2
    elif len(datetimes2)==0:
        return trial_dir1
    
    if np.abs(ft.datetimediff_s(datetimes1[-1],desired_datetime))<np.abs(ft.datetimediff_s(datetimes2[0],desired_datetime)):
        nearest_dir=trial_dir1
    else: 
        nearest_dir=trial_dir2
    return nearest_dir
    
def get_nearest_directory(dir_string,radar,date,time,function_get_filenames_directory,function_get_datetimes_from_files):
    """
    For a particular combination of radar, date and time, this function finds the directory that contains the file whose date and time are closest
    to the input date and time.
    
    function_get_filenames_directory should be a function that returns filenames in a directory, for input consisting of radar and directory path.
    function_get_datetimes_from_files should be a function that returns datetimes from filenames, for input consisting of radar and filenames.
    
    In order to find the nearest subfolder, first sorted lists of subfolders are created. It is then checked at which position the desired subfolder 
    is positioned. If this is the first position, then the first subfolder in the sorted list is selected. When this is the last position, then a
    similar treatment follows. If it is not the first or last position, then the files in the 2 surrounding subfolders are listed, and the subfolder
    is selected that contains the file whose datetime is closest to the input date and time.
    
    The above description is for the situation in which date/time variables are located in only 1 substring. If they are located in 2 substrings,
    then a deeper search is required, but the method does not differ much.
    
    This function does not necessarily skip empty directories, because of the difficulty of implementing methods to do this!!! It does include some
    effort to skip them however.
    """
    dir_string=replace_radar_variables(dir_string,radar)
        
    if not '${' in dir_string:
        #In this case there are no date and/or time variables present in dir_string
        return opa(dir_string)
    
    try:
        desired_datetime=date+time
        desired_dir=convert_dir_string_to_real_dir(dir_string,radar,date,time)
        
        if os.path.exists(desired_dir) and not check_dir_empty_radar(radar,desired_dir,function_get_filenames_directory):
            return opa(desired_dir)
        
        #substrings1 and substrings2 contain the different parts of dir_string and desired_dir, that are separated by slashes.
        substrings1=get_substrings_in_dir_string(dir_string) #For dir_string
        substrings2=get_substrings_in_dir_string(desired_dir) #For desired_dir
                    
        #There can be 1 or 2 parts of dir_string that contain variables, and to this correspond 1 or multiple series of directories.
        #abs_paths contains the absolute path that leads to a particular series of directories. If there are 2 series, then the path to the
        #inner (deeper) series of subfolders depends on the outer folder that is chosen, and if this outer folder changes, then the absolute
        #path to the inner series must also change. This is done below when necessary.
        substrings_indices, abs_paths=get_substringsindices_and_abspaths(substrings1,substrings2)
        
        if len(abs_paths)==1:
            path_end1=''.join(['/'+substrings1[j] for j in range(substrings_indices[0]+1,len(substrings1))])
        else:
            path_end1=''.join(['/'+substrings1[j] for j in range(substrings_indices[0]+1,substrings_indices[-1])])
            
        substring1=substrings1[substrings_indices[0]]
        desired_dir_substring1=substrings2[substrings_indices[0]]
        direntries1=get_direntries(opa(abs_paths[0]), substring1)
        #Check existence path_end1, to prevent that subdirectories are chosen that certainly do not contain data for the radar.
        direntries1 = [j for j in direntries1 if os.path.exists(opa(abs_paths[0]+'/'+j+path_end1))]
        direntries1_plus_desired_dir=np.sort(direntries1+[desired_dir_substring1])
        n_direntries=len(direntries1_plus_desired_dir)
        index=np.where(direntries1_plus_desired_dir==desired_dir_substring1)[0][0]
            
        if len(abs_paths)==1:
            if index==0:
                nearest_dir=abs_paths[0]+'/'+direntries1_plus_desired_dir[1]+path_end1
            elif index==n_direntries-1:
                nearest_dir=abs_paths[0]+'/'+direntries1_plus_desired_dir[-2]+path_end1
            else:
                trial_dir1=abs_paths[0]+'/'+direntries1_plus_desired_dir[index-1]+path_end1
                trial_dir2=abs_paths[0]+'/'+direntries1_plus_desired_dir[index+1]+path_end1
                nearest_dir=determine_nearest_dir(radar,trial_dir1,trial_dir2,desired_datetime,function_get_filenames_directory,function_get_datetimes_from_files)
        else:   
            """It is possible that the path in direntries1 that gets selected actually does not contain data for radar, which can occur when the 
            structure of the directory path is the same for multiple radars, at least until direntries2 are reached. In this case it is possible 
            that the first path that gets selected contains data for another radar. 
            In order to handle this, the following while loop is used, where in the case of an exception (which occurs always when the selected
            path in direntries1 does not contain data for radar) the problematic path is removed from direntries1_plus_desired_dir, whereafter
            the process is repeated.
            The while loop stops when a directory that satisfies the format of dir_string is found, or when all paths in direntries1 have been 
            checked.
            """
            while True:
                try:
                    if desired_dir_substring1 in np.delete(direntries1_plus_desired_dir,index):
                        direntries1_plus_desired_dir_indices=[index]
                        
                        abs_paths[-1]=abs_paths[0]+'/'+desired_dir_substring1+path_end1
        
                        path_end2=''.join(['/'+substrings1[j] for j in range(substrings_indices[-1]+1,len(substrings1))])
                        
                        substring2=substrings1[substrings_indices[-1]]
                        desired_dir_substring2=substrings2[substrings_indices[-1]]
                        direntries2=get_direntries(opa(abs_paths[-1]), substring2)
                        #Check existence path_end2, to prevent that subdirectories are chosen that certainly do not contain data for the radar.
                        direntries2 = [j for j in direntries2 if os.path.exists(opa(abs_paths[-1]+'/'+j+path_end2))]
                        direntries2_plus_desired_dir=np.sort(direntries2+[desired_dir_substring2])
                        n_direntries=len(direntries2_plus_desired_dir)
                        index=np.where(direntries2_plus_desired_dir==desired_dir_substring2)[0][0]
                        
                        if index==0:
                            nearest_dir=abs_paths[-1]+'/'+direntries2_plus_desired_dir[1]+path_end2
                        elif index==n_direntries-1:
                            nearest_dir=abs_paths[-1]+'/'+direntries2_plus_desired_dir[-2]+path_end2
                        else:
                            trial_dir1=abs_paths[-1]+'/'+direntries2_plus_desired_dir[index-1]+path_end2
                            trial_dir2=abs_paths[-1]+'/'+direntries2_plus_desired_dir[index+1]+path_end2
                            nearest_dir=determine_nearest_dir(radar,trial_dir1,trial_dir2,desired_datetime,function_get_filenames_directory,function_get_datetimes_from_files)
                    else:
                        if index==0:
                            direntries1_plus_desired_dir_indices=[1]
                            
                            nearest_dir_substring=direntries1_plus_desired_dir[1]
                            abs_paths[-1]=abs_paths[0]+'/'+nearest_dir_substring+path_end1
                    
                            substring2=substrings1[substrings_indices[-1]]
                            direntries2=get_direntries(opa(abs_paths[-1]),substring2)
                            path_end2=''.join(['/'+substrings1[j] for j in range(substrings_indices[-1]+1,len(substrings1))])
                            nearest_dir=abs_paths[-1]+'/'+direntries2[0]+path_end2
                        elif index==n_direntries-1:
                            direntries1_plus_desired_dir_indices=[n_direntries-2]
                            
                            nearest_dir_substring=direntries1_plus_desired_dir[-2]
                            abs_paths[-1]=abs_paths[0]+'/'+nearest_dir_substring+path_end1
                    
                            substring2=substrings1[substrings_indices[-1]]
                            direntries2=get_direntries(opa(abs_paths[-1]),substring2)
                            path_end2=''.join(['/'+substrings1[j] for j in range(substrings_indices[-1]+1,len(substrings1))])
                            nearest_dir=abs_paths[-1]+'/'+direntries2[-1]+path_end2 
                        else:               
                            direntries1_plus_desired_dir_indices=[index-1,index+1]
                            
                            trial_dir1=abs_paths[0]+'/'+direntries1_plus_desired_dir[index-1]+path_end1
                            trial_dir2=abs_paths[0]+'/'+direntries1_plus_desired_dir[index+1]+path_end1
                            substring2=substrings1[substrings_indices[-1]]
                            direntries21=get_direntries(trial_dir1,substring2)
                            direntries22=get_direntries(trial_dir2,substring2)
                            
                            path_end2=''.join(['/'+substrings1[j] for j in range(substrings_indices[-1]+1,len(substrings1))])
                            trial_dir1=trial_dir1+'/'+direntries21[-1]+path_end2
                            trial_dir2=trial_dir2+'/'+direntries22[0]+path_end2
                            nearest_dir=determine_nearest_dir(radar,trial_dir1,trial_dir2,desired_datetime,function_get_filenames_directory,function_get_datetimes_from_files)
                    #break while loop if no exception is raised
                    break
                except Exception:
                    direntries1_plus_desired_dir=np.delete(direntries1_plus_desired_dir,direntries1_plus_desired_dir_indices)
                    n_direntries=len(direntries1_plus_desired_dir)
                    if n_direntries<=1: 
                        #If n_direntries==1, then only the desired directory is present.
                        break
                    index=np.where(direntries1_plus_desired_dir==desired_dir_substring1)[0][0]
                
        if check_dir_empty_radar(radar,nearest_dir,function_get_filenames_directory):
            previous_dir=get_next_directory(dir_string,-1,radar,function_get_filenames_directory,current_dir=nearest_dir)
            next_dir=get_next_directory(dir_string,1,radar,function_get_filenames_directory,current_dir=nearest_dir)
            nearest_dir=determine_nearest_dir(radar,previous_dir,next_dir,desired_datetime,function_get_filenames_directory,function_get_datetimes_from_files)
        return opa(nearest_dir)
    except Exception as e:
        print(e,'get_nearest_directory')
        #Can e.g. occur when there are no directories that satisfy the format.
        return None
    
def get_abspaths_directories_in_datetime_range(dir_string,radar,startdatetime=None,enddatetime=None):
    """Determines which directories contain files for datetimes between startdatetime and enddatetime. When startdatetime=None, then
    there is no lower bound for the datetime, and when enddatetime=None, then there is no upper bound for the datetime.
    Returns the absolute paths of these directories, and returns the bool dirs_filtered, which is True when it was possible to filter dates
    based on startdatetime and enddatetime, and False otherwise. It is not possible to filter dates when there is no ${date} variable in 
    dir_string.
    """
    if not startdatetime is None:
        startdate=int(startdatetime[:8]); starttime=int(startdatetime[-4:]); startdatetime=int(startdatetime)
    if not enddatetime is None:
        enddate=int(enddatetime[:8]); endtime=int(enddatetime[-4:]); enddatetime=int(enddatetime)
    
    dir_string=replace_radar_variables(dir_string,radar)
        
    if not '${' in dir_string:
        #In this case there are no date and/or time variables present in dir_string
        return [opa(dir_string)], False
    
    try:
        substrings=get_substrings_in_dir_string(dir_string)
        substrings_indices, abs_paths=get_substringsindices_and_abspaths(substrings,substrings)
        
        substring1=substrings[substrings_indices[0]]
        direntries1=np.array(get_direntries(opa(abs_paths[0]),substring1))
        if len(direntries1)==0: return [], False
        
        if '${date}' in dir_string or '${datetime' in dir_string:
            dirs_filtered=True
            if '${date}' in dir_string:
                dates=np.array([get_date_and_time_from_dir(j,substring1)[0] for j in direntries1],dtype='int')
            else:
                datetimes=np.array([''.join(get_date_and_time_from_dir(j,substring1)) for j in direntries1],dtype='int64')
                
            if not startdatetime is None and not enddatetime is None:
                requested=(dates>=startdate) & (dates<=enddate) if '${date}' in dir_string else (datetimes>=startdatetime) & (datetimes<=enddatetime)
            elif not startdatetime is None:
                requested=dates>=startdate if '${date}' in dir_string else datetimes>=startdatetime
            elif not enddatetime is None:
                requested=dates<=enddate if '${date}' in dir_string else datetimes<=enddatetime
            else:
                requested=np.ones(len(direntries1),dtype='bool')
                
            if '${date}' in dir_string:
                dates_requested=dates[requested]
        else:
            dirs_filtered=False
            requested=np.ones(len(direntries1),dtype='bool')
        direntries1_requested=direntries1[requested]
            
        if len(abs_paths)==1:
            path_end=''.join(['/'+substrings[j] for j in range(substrings_indices[0]+1,len(substrings))])
            direntries1_abspaths=[opa(abs_paths[0]+'/'+j+path_end) for j in direntries1_requested]
            return [j for j in direntries1_abspaths if os.path.exists(j)], dirs_filtered
        else:
            #In this case there must be a ${date} variable.
            path_end1=''.join(['/'+substrings[j] for j in range(substrings_indices[0]+1,substrings_indices[-1])])
                        
            requested_dirs_abspaths=[]
            for i in range(0,len(direntries1_requested)):
                abs_paths[-1]=abs_paths[0]+'/'+direntries1_requested[i]+path_end1
                if not os.path.exists(abs_paths[-1]): continue
                
                substring2=substrings[substrings_indices[-1]]
                direntries2=np.array(get_direntries(opa(abs_paths[-1]),substring2))
                if len(direntries2)==0: continue
                
                if (not startdatetime is None and dates_requested[i]==startdate) or (not enddatetime is None and dates_requested[i]==enddate):
                    times=np.array([get_date_and_time_from_dir(j,substring2)[1] for j in direntries2],dtype='int')
                    if not startdatetime is None and not enddatetime is None and startdate==enddate:
                        requested=(times>=starttime) & (times<=endtime)
                    elif not startdatetime is None and dates_requested[i]==startdate:
                        requested=times>=starttime
                    elif not enddatetime is None and dates_requested[i]==enddate:
                        requested=times<=endtime
                    else:
                        requested=np.ones(len(direntries2),dtype='bool')                        
                else:
                    requested=np.ones(len(direntries2),dtype='bool')
                    
                direntries2_requested=direntries2[requested]
                    
                path_end2=''.join(['/'+substrings[j] for j in range(substrings_indices[-1]+1,len(substrings))])
                for j in direntries2_requested:
                    dir_abspath_complete=abs_paths[-1]+'/'+j+path_end2  
                    if os.path.exists(opa(dir_abspath_complete)):
                        requested_dirs_abspaths.append(dir_abspath_complete)
                        
            return requested_dirs_abspaths, dirs_filtered
    except Exception:
        return [], False
            
def get_dates_with_archived_data(dir_string,radar):
    """Determines for which dates there is archived data present for the input radar, based on directory names. It does not use filenames, because
    listing large directories can be very slow. 
    Because this function uses directory names, it can be unable to determine the correct list with dates with archived data when there are empty
    directories, or when files for multiple radars are put in the same directory. It is also unable to determine the correct list when files for
    all dates are put in the same directory, but it seems very unlikely that such a case will be encountered.
    """
    dir_string=replace_radar_variables(dir_string,radar)
        
    if not '${date' in dir_string:
        #In this case there are no dates in the directory name, and because this function does not use filenames to determine for which files
        #there is data, it cannot determine the dates for which archived data is present, and an empty list is returned.
        return []
        
    try:
        substrings=get_substrings_in_dir_string(dir_string)
        substrings_indices, abs_paths=get_substringsindices_and_abspaths(substrings,substrings)
        
        substring1=substrings[substrings_indices[0]]
        direntries1=get_direntries(opa(abs_paths[0]),substring1)
            
        if len(abs_paths)==1:
            path_end=''.join(['/'+substrings[j] for j in range(substrings_indices[0]+1,len(substrings))])
            direntries1_nonempty=[abs_paths[0]+'/'+j+path_end for j in direntries1 if os.path.exists(opa(abs_paths[0]+'/'+j+path_end)) and not check_dir_empty(opa(abs_paths[0]+'/'+j+path_end))]
            archived_dates=[get_date_and_time_from_dir(j,dir_string)[0] for j in direntries1_nonempty]
        else:
            path_end1=''.join(['/'+substrings[j] for j in range(substrings_indices[0]+1,substrings_indices[-1])])
            
            archived_dates=[]
            for i in direntries1:
                abs_paths[-1]=abs_paths[0]+'/'+i+path_end1
                if not os.path.exists(abs_paths[-1]): continue
                
                substring2=substrings[substrings_indices[-1]]
                direntries2=get_direntries(opa(abs_paths[-1]),substring2)
                
                path_end2=''.join(['/'+substrings[j] for j in range(substrings_indices[-1]+1,len(substrings))])
                for j in direntries2:
                    dir_abspath_complete=abs_paths[-1]+'/'+j+path_end2  
                    if os.path.exists(opa(dir_abspath_complete)) and not check_dir_empty(opa(dir_abspath_complete)):
                        archived_dates.append(get_date_and_time_from_dir(dir_abspath_complete,dir_string)[0])
                        break
                    
        return archived_dates
    except Exception:
        return []

def sort_volume_attributes(scanangles_all,radial_bins_all,radial_res_all,extra_attrs=[],start_scan=1):
    scannumbers_all = {}
    
    scans = list(scanangles_all)
    attrs = [scanangles_all,radial_bins_all,radial_res_all]+extra_attrs
    
    if len(scans) == 1:
        scannumbers_all[start_scan] = [scans[0]]
    else:
        inverse_radial_range = {j:1/(radial_bins_all[j]*radial_res_all[j]) for j in scans}
        sort_attrs = (scanangles_all,inverse_radial_range)
        variables = list(zip(*[attr.values() for attr in sort_attrs]+[scans]))
        variables_sorted = sorted(variables)
        
        duplicates = []
        variables_sorted_withoutduplicates = []
        for j, v in enumerate(variables_sorted):
            v_previous = variables_sorted[j-1] if j > 0 else None
            if j > 0 and v_previous[:-1] == v[:-1]:
                duplicates.append(j)
                if not j-1 in duplicates:
                    scannumbers_all[j+start_scan-len(duplicates)] = [v_previous[-1], v[-1]]
                else:
                    scannumbers_all[j+start_scan-len(duplicates)].append(v[-1])
            else:
                variables_sorted_withoutduplicates.append(v)
                scannumbers_all[j+start_scan-len(duplicates)] = [v[-1]]
                
    for i,attr in enumerate(attrs):
        attr_copy = attrs[i].copy()
        attr.clear()
        for j,k in scannumbers_all.items():
            attr[j] = attr_copy[k[0]]
    
    return scannumbers_all

    
# Calculate widths of strings in pixels. These are used to determine the number of spaces that can be put between panel titles.
# This is done as little as possible, because calculating it is relatively expensive
font = ImageFont.truetype('_data/OpenSans-Bold.ttf', 12)
char_sizes = {}
def char_size(char):
    if not char in char_sizes:
        bbox = font.getbbox(char)
        char_sizes[char] = bbox[2]-bbox[0]
    return char_sizes[char]
def string_size(string):
    # The size of a complete string is not necessarily equal to the sum of the sizes of individual characters. The difference is
    # determined by which characters are positioned next to each other, hence calculate the actual string size in the following way
    s = 0
    for i in range(len(string)-1):
        s += char_size(string[i:i+2])
    return s-sum([char_size(char) for char in string[1:-1]])

def get_titles(relwidth,fontsizes_main_titles,radar,panels,panellist,panelnumber_to_plotnumber,plotnumber_to_panelnumber,data_empty,using_olddata,products,date,scantimes,scanangles,using_unfilteredproduct,using_verticalpolarization,apply_dealiasing,productunits,stormmotion,PP_parameter_values,PP_parameters_panels,show_vvp):
    if all([data_empty[j] for j in panellist]):
        #No title_top in this case
        return '','',{}
    date_formatted = ft.format_date(date,'YYYYMMDD->YYYY-MM-DD')
    title_top=radar+'  '+date_formatted
    title_bottom = gv.radars_different_actual_source.get(radar, gv.data_sources[radar])
    
    if (stormmotion[1] != 0. or 's' in products.values()) and not show_vvp:
        #storm motion is always shown when 's' in products, and otherwise only when it is nonzero.
        title_bottom+='   SM: '+str(int(round(stormmotion[0])))+u'\u00b0'+str(int(round(stormmotion[1]*gv.scale_factors_velocities[productunits['s']])))+' '+productunits['s']
        
    def paneltitle(panels,panelnumber):
        product = products[panelnumber]
        paneltitle=''
        
        if using_unfilteredproduct[panelnumber]: 
            paneltitle += 'u'
        paneltitle += gv.productnames[product]
        if using_verticalpolarization[panelnumber]: 
            paneltitle+='v'
        if product in ('v','s') and apply_dealiasing[panelnumber]: 
            paneltitle += '*'
        if product not in gv.plain_products:
            scanangle_show="%.1f" % scanangles[panelnumber]
            paneltitle+=' '+scanangle_show+u'\u00b0'
        else:
            if product in PP_parameter_values:
                param_value = PP_parameter_values[product][PP_parameters_panels[panelnumber]]
                
            if product=='e':
                paneltitle+=' ('+str(ft.rifdot0(param_value))+' dBZ)'
            elif product=='r':
                paneltitle+=' ('+str(gv.CAPPI_height_R)+' km)'
            elif product=='a':
                paneltitle+=' ('+str(ft.rifdot0(param_value))+' km)'
            elif product == 'm':
                if PP_parameter_values[product][PP_parameters_panels[panelnumber]] > 0.:
                    paneltitle+=' ('+str(ft.rifdot0(param_value))+'+ km)'
            elif product=='h':
                paneltitle+=' ('+('cap' if param_value else 'no cap')+')'
            elif product=='l':
                paneltitle+=' ('+('cap' if param_value[0] else 'no cap')+\
                            (', '+str(ft.rifdot0(param_value[1]))+'+ km' if param_value[1] else '')+')'
                            
        paneltitle += '  '
        scantime = scantimes[panelnumber]
        if '-' in scantime:
            scantime = ft.get_avg_scantime([scantime])+'Z  '+ft.scantimerange_formatted(scantime)
        else:
            scantime += 'Z'
        paneltitle += scantime
        if using_olddata[panelnumber]:
            paneltitle+=' (OLD)'
        return paneltitle
    
    paneltitles_raw = {j: '' if data_empty[j] else paneltitle(panels,j) for j in panellist}
    n_cols = panels if panels<4 else panels//2
    center = (n_cols-1)/2
    if n_cols%2 == 1:
        center = int(center)
        n = 8 if panels < 10 else 3
        paneltitles_raw[center] = title_top+' '*n+paneltitles_raw[center]
        title_top = ''
        if n_cols < panels:
            paneltitles_raw[center+5] = title_bottom+' '*n+paneltitles_raw[center+5]
            title_bottom = ''
    paneltitles = paneltitles_raw.copy()
    
    sizes = {j:string_size(paneltitles[j]) for j in panellist}
    size_top = string_size(title_top)
    size_bottom = string_size(title_bottom)
    
    """Add extra spaces to the panel titles if that is needed in order to not have them overlap with other text.
    The number of extra spaces is calculated based on estimates of the relative width of the different strings
    """
    # Scale factor for converting string length to relative width. Factor is based on assumption that string contains only spaces.
    # A correction factor is needed if that is not the case (which is given by the object f below).
    scale_fac = (775*relwidth*10/fontsizes_main_titles)
    for j in panellist:
        plot = panelnumber_to_plotnumber[panels][j]
        size = size_top if plot//n_cols == 0 else size_bottom     
        
        # Determine the number of spaces that should be put between panel titles in order to span the full half-width of
        # the main GUI widget. The actual number that will be added is taken to be at most and preferentially 8, and at least 3.
        s = size if n_cols%2 == 0 else sizes[center]
        i_end = int(np.floor(center-0.5))
        for i in range(0,i_end+1):
            add_cols = n_cols if plot>=n_cols else 0
            plot_i = i+add_cols if plot%n_cols < center else n_cols-1-i+add_cols
            panel = plotnumber_to_panelnumber[panels][plot_i]
            s += sizes[panel]*2
        b = s/(char_sizes[' ']*scale_fac)
        n = int(round((0.5-b)*scale_fac/max([n_cols//2,1])))
        # print(j,b,n)
        n = min([max([n,3]),8])
        
        # Calculate where the center of the panel title should be positioned in order to not have it overlap with other strings
        # The resulting relative position (between 0 and 1) is stored in the object b
        s = size+sizes[j] if n_cols%2 == 0 else sizes[j]
        i_start = 1+(plot%n_cols if plot%n_cols < center else (n_cols-1-plot)%n_cols)
        i_end = int(np.floor(center))
        for i in range(i_start,i_end+1):
            add_cols = n_cols if j>=n_cols else 0
            plot_i = i+add_cols if plot%n_cols < center else n_cols-1-i+add_cols
            panel = plotnumber_to_panelnumber[panels][plot_i]
            s += sizes[panel]
            if ((i <= i_end) if n_cols%2 == 0 else (i < i_end)):
                # In this case the full string should be counted instead of half of the string. So add it twice.
                # In this case also the extra spaces should be counted
                s += char_sizes[' ']*n+sizes[panel]
        b = s/(char_sizes[' ']*scale_fac)
        
        # a contains the actual relative position of the panel title center
        a = abs(0.5-(plot%n_cols+0.5)/n_cols)
        # print(j,a,b)
        
        # Add extra spaces if required/desired
        # The ' '*int(round((b-a)*scale_fac)) part adds the minimum number of spaces that is needed to prevent overlap.
        # And the +n part adds extra spaces to arrive at the desired spacing between panel titles
        if a < b+0.02:
            if (plot+0.5)/n_cols%1 < 0.5:
                paneltitles[j] += ' '*int(round((b-a)*scale_fac+n))
            elif (plot+0.5)/n_cols%1 > 0.5: # Not if equal to 0.5, because then the title has been added to the panel title
                paneltitles[j] = ' '*int(round((b-a)*scale_fac+n))+paneltitles[j]
    return title_top, title_bottom, paneltitles
    
    
    
def determine_gridpos(physical_size_cm_main,rel_xdim,corners,text_fontsize,panels,panellist,nrows,ncolumns,show_vwp):    
    xdim, ydim = corners[0][-1]-corners[0][1]
    # grid spacing options (which can be multiplied by a power of 10)
    gs_arr = np.array([100,90,80,70,60,50,40,30,25,20,15,12], dtype='float64')
    N = xdim/gs_arr[0]
    min_N, max_N = 2, 7/ncolumns*rel_xdim
    gs = gs_try_before = gs_arr[0]
    if N < max_N:
        while N < max_N and gs_arr.min() > 0:
            for gs_try in gs_arr:
                N = xdim/gs_try
                if N >= max_N: break
                gs_try_before = gs_try
            gs_arr /= 10.
        gs = gs_try_before if xdim/gs_try_before >= min_N else gs_try
    elif N > max_N:
        while N > max_N:
            gs_arr *= 10.
            for gs_try in gs_arr[::-1]:
                N = xdim/gs_try
                if N <= max_N: break
                gs_try_before = gs_try
        gs = gs_try if N >= min_N else gs_try_before
     
    gridlines_vertices_panels, gridlines_connect_panels = {}, {}
    gridlines_text_hor_pos_panels, gridlines_text_hor_panels = {j:[] for j in panellist}, {j:[] for j in panellist}
    gridlines_text_vert_pos_panels, gridlines_text_vert_panels = {j:[] for j in panellist}, {j:[] for j in panellist}
    for j in panellist:
        xmin, xmax = corners[j][(0, -1), 0]
        ymin, ymax = corners[j][(1, -1), 1]
    
        xstart, xend = np.ceil(xmin/gs)*gs, np.floor(xmax/gs)*gs
        ystart, yend = np.ceil(ymin/gs)*gs, np.floor(ymax/gs)*gs
        xvalues, yvalues = np.arange(xstart, xend+0.1*gs, gs), np.arange(ystart, yend+0.1*gs, gs) 
                
        #20.53 is the reference physical y dimension
        text_offset = text_fontsize/11.5*20.53/physical_size_cm_main[1]*6*ydim*nrows/1e3
            
        ygridlines_vertices, ygridlines_connect, ygridlines_textpos, ygridlines_text = [], [], [], []
        if len(yvalues):
            ygridlines_vertices = np.concatenate([[[xmin, y],[xmax, y]] for y in yvalues])
            ygridlines_connect = np.array([True, False]*len(yvalues))
            ygridlines_textpos = np.array([[xmin+text_offset, y] for y in yvalues])
            ygridlines_text = [str(ft.round_float(y)) for y in yvalues]
            
        xgridlines_vertices, xgridlines_connect, xgridlines_textpos, xgridlines_text = [], [], [], []
        if len(xvalues):
            xgridlines_vertices = np.concatenate([[[x, ymin],[x, ymax]] for x in xvalues])
            xgridlines_connect = np.array([True, False]*len(xvalues))
            xgridlines_textpos = np.array([[x, ymin+text_offset] for x in xvalues])
            xgridlines_text = [str(ft.round_float(x)) for x in xvalues]
            
            if xstart-xmin < 37*ydim*nrows/1e3 and ystart-ymin < 37*ydim*nrows/1e3:
                #This is to prevent that the first horizontal and vertical tick are on top of each other.
                xgridlines_textpos, xgridlines_text = xgridlines_textpos[1:], xgridlines_text[1:]
        
        gridlines_vertices, gridlines_connect = [], []
        if len(xvalues) and len(yvalues):
            gridlines_vertices = np.concatenate([xgridlines_vertices, ygridlines_vertices])
            gridlines_connect = np.append(xgridlines_connect, ygridlines_connect)
        elif len(xvalues) or len(yvalues):
            gridlines_vertices = xgridlines_vertices if len(xvalues) else ygridlines_vertices
            gridlines_connect = xgridlines_connect if len(xvalues) else ygridlines_connect
    
        gridlines_vertices_panels[j] = gridlines_vertices
        gridlines_connect_panels[j] = gridlines_connect
        if ((j < 5 and panels < 4) or (j > 4 and (panels == 2 or panels > 3))) and len(xgridlines_text):
            gridlines_text_hor_pos_panels[j] = xgridlines_textpos
            gridlines_text_hor_panels[j] = xgridlines_text
        if (j == 0 or (j == 5 and panels > 3)) and len(ygridlines_text):
            gridlines_text_vert_pos_panels[j] = ygridlines_textpos
            gridlines_text_vert_panels[j] = ygridlines_text
    return (gridlines_vertices_panels, gridlines_connect_panels, gridlines_text_hor_pos_panels, gridlines_text_hor_panels,
            gridlines_text_vert_pos_panels, gridlines_text_vert_panels)
            
            
            
heights={}; hranges={}
def determine_heightrings(rel_xdim,corners,ncolumns,panellist,scanangles,use_previous_hrange=False):
    # With use_previous_hrange=True the locations (ranges) of the height rings are not changed, only height values will be updated (with 1-decimal precision).
    # In the case of a moving panel center (like with storm-moving view), additional height rings will be added when needed, and out-of-view rings 
    # will be removed.
    global heights, hranges, previous_center_dist, previous_scanangles
    
    xdim_full = ncolumns*(corners[0][-1,0]-corners[0][0,0])
    dr_desired = xdim_full/(5*rel_xdim)
    
    heights_old = {p: heights[p] for p in heights}
    hranges_old = {p: hranges[p] for p in hranges}
    center_dist = {}
    for j, p in enumerate(panellist):
        p = panellist[j]
        min_r, max_r = ft.mindist_maxdist_maxangle(corners[p])[:2]
        center_dist[p] = np.linalg.norm(np.mean(corners[p], axis=0))
        
        dmin = 0.035*xdim_full
        h_min_text = ft.c1dec(ft.var1_to_var2(min_r+dmin, scanangles[j], 'gr+theta->h'))
        h_max_text = ft.f1dec(ft.var1_to_var2(max_r-dmin, scanangles[j], 'gr+theta->h'))
        r_min_text = bool(min_r > 0.)*ft.var1_to_var2(h_min_text, scanangles[j], 'h+theta->gr')
        r_max_text = ft.var1_to_var2(h_max_text, scanangles[j], 'h+theta->gr')

        no_heightrings = scanangles[j] == 90. or scanangles[j] < 0. or h_max_text > 100
        if no_heightrings:
            heights[p] = hranges[p] = np.array([])
            continue
                
        N_min = 2 if h_max_text-h_min_text < 4 else 3
        N = min([6, max([N_min, int(round((r_max_text-r_min_text)/dr_desired))+int(min_r > 0.)])])
        r_text = np.linspace(r_max_text, r_min_text, N, endpoint=min_r > 0.)[::-1]
        dr = min([dr_desired, r_text[1]-r_text[0]])
        if dr == 0.: # Quick solution, better needed
            dr = dr_desired
            
        old_present = p in heights_old and len(heights_old[p])
        use_old_hranges = use_previous_hrange and old_present and abs(scanangles[j]-previous_scanangles.get(p, 999)) <= 0.3
        if use_old_hranges:
            # Recalculate the height from the old range, since the scanangle might have changed.
            r_old = hranges_old[p]
            h_old = ft.r1dec(ft.var1_to_var2(r_old, scanangles[j], 'gr+theta->h'))
            
            # retain is None implies retaining all previous values. This is always done when there's no change in view, in order to prevent 
            # that the lowest/highest height is removed due to small changes in r_min_text/r_max_text caused by changes in scanangle
            retain = None if center_dist[p] == previous_center_dist[p] else ((r_old >= r_min_text) & (r_old <= r_max_text))
            h_old, r_old = h_old[retain], r_old[retain]
            if len(h_old):
                s = np.sign(center_dist[p]-previous_center_dist[p])
                r = r_old[-1 if s > 0 else 0]
                r_text = []
                # s might be 0
                while s and min([0.1, r_min_text]) < r < r_max_text:
                    r_text += [r]
                    # Use dr_desired instead of dr
                    r_try = max(r_text[-1]+s*dr_desired, 0.)
                    h = ft.r1dec(ft.var1_to_var2(r_try, scanangles[j], 'gr+theta->h'))
                    r = ft.var1_to_var2(h, scanangles[j], 'h+theta->gr')
                    if abs(r-r_text[-1]) < 0.85*dr:
                        if s > 0 or h > 0.1:
                            h += s*0.1
                            r = ft.var1_to_var2(h, scanangles[j], 'h+theta->gr')
                        else:
                            break
                    if r/xdim_full < 0.025:
                        # Don't position height rings too close to the origin
                        break
                r_text = np.sort(np.array(r_text[1:])) # First element is from old height rings 
            else:
                use_old_hranges = False
        
        heights[p] = [j for j in ft.r1dec(ft.var1_to_var2(r_text, scanangles[j],'gr+theta->h')) if not j == 0.]
        heights[p], indices = np.unique(heights[p], return_index=True)
        r_text = r_text[indices]
        
        # Round heights to integers when this doesn't change the corresponding radius too much
        for i, r in enumerate(r_text):
            h_rounded = round(heights[p][i])
            r_new = ft.var1_to_var2(h_rounded, scanangles[j], 'h+theta->gr')
            if np.abs(r_new-r)/dr < 0.15 and (i > 0 or r_new-min_r >= dmin) and (i < len(r_text)-1 or max_r-r_new >= dmin):
                heights[p][i] = h_rounded
        hranges[p] = ft.var1_to_var2(heights[p], scanangles[j], 'h+theta->gr')
                
        # Check whether some height rings are undesirably close to each other. In that case either change 1 of the heights, if that doesn't let it 
        # get too close to the other heights, or delete 1 of the heights otherwise.
        d = 0.667
        too_close = (hranges[p][1:]-hranges[p][:-1])/dr < d
        n = 0
        while too_close.any() and len(heights[p]) > 2 and n < 5: # Without a maximum, it can happen that this loop continues indefinitely
            n += 1
            i = np.nonzero(too_close)[0][0]
            r12 = hranges[p][i:i+2]
            h12 = heights[p][i:i+2]
            h12_test = ft.r1dec(h12+0.1*np.array([-1, 1]))
            r12_test = ft.var1_to_var2(h12_test, scanangles[j], 'h+theta->gr')
            if 0 < i < len(too_close)-1 or (i == 0 and min_r == 0.):
                r_before, r_after = hranges[p][i-1] if not i == 0 else 0., hranges[p][i+2]
                dr12_test = np.array([r12_test[0]-r_before, r_after-r12_test[1]])/dr
                i_max_dr, max_dr = dr12_test.argmax(), dr12_test.max()
                if max_dr > d:
                    heights[p][i+i_max_dr] = h12_test[i_max_dr]
                else:
                    dr12 = np.array([r12[1]-hranges[p][i-1], hranges[p][i+2]-r12[0]])/dr
                    i_min_dr = dr12.argmin()
                    heights[p] = np.delete(heights[p], i if i_min_dr == 0 else i+1)
            else:
                k = 1 if i == 0 else 0
                l = 2 if i == 0 else -3
                if np.abs(r12_test[k]-hranges[p][l])/dr > d:
                    heights[p][l-np.sign(l)] = h12_test[k]
                else:
                    heights[p] = np.delete(heights[p], l-np.sign(l))
            hranges[p] = ft.var1_to_var2(heights[p], scanangles[j], 'h+theta->gr')
            too_close = (hranges[p][1:]-hranges[p][:-1])/dr < 0.5
            
        if use_old_hranges:
            heights[p] = np.sort(np.append(heights[p], h_old))
            hranges[p] = np.sort(np.append(hranges[p], r_old))
            
    previous_center_dist = center_dist.copy()
    previous_scanangles = {p:scanangles[i] for i,p in enumerate(panellist)}
    return heights, hranges
    

xfactor={1:1,2:0.5,3:1./3.,4:0.5,6:1./3.,8:0.25,10:0.2}
yfactor={1:1,2:1,3:1./3.,4:0.5,6:0.5,8:0.5,10:0.5}
def determine_textangles(corners,panels,panellist,products,hranges):
    ydim = corners[0][-1,1]-corners[0][1,1]
    
    textangles = {}
    for j in panellist:
        min_r, max_r, angle_min_r, angle_max_r = ft.mindist_maxdist_maxangle(corners[j])
        current_xlim, current_ylim = corners[j][(0, -1), 0], corners[j][(1, -1), 1]
        
        textangles[j] = np.zeros(len(hranges[j]))
        
        angle_perturbation_step = 1
        n_perturbations = 0
        for i, r in enumerate(hranges[j][::-1]):
            if min_r==0.:
                textangles[j][i] = angle_max_r
            else:
                try:
                    textangles[j][i] = ft.av_angle_circle_in_rectangle(current_xlim,current_ylim,r)
                except Exception:
                    # Happens when the circle is not within the current view
                    textangles[j][i] = 0.
                
            circle_point = r*np.array([np.sin(textangles[j][i]), np.cos(textangles[j][i])])
            if ft.distance_to_rectangle_point_inside(corners[j], circle_point)*yfactor[panels]*4e2/ydim < 15:
                # This can happen only for the first and last circle
                if i == 0:
                    cornerx,cornery = max_r*np.array([np.sin(angle_max_r), np.cos(angle_max_r)])
                    cornervector = [np.sign(np.sin(angle_max_r)), np.sign(np.cos(angle_max_r))]
                    textangles[j][i] = ft.angle_intersection_circle_with_corner_bisector(cornerx,cornery,cornervector,r)
                elif not angle_min_r is None:
                    cornerx,cornery = min_r*np.array([np.sin(angle_min_r), np.cos(angle_min_r)])
                    cornervector = [np.sign(np.sin(angle_min_r)), np.sign(np.cos(angle_min_r))]
                    textangles[j][i] = ft.angle_intersection_circle_with_corner_bisector(cornerx,cornery,cornervector,r)
                    
            if products[j] in gv.plain_products:
                if n_perturbations % 3 == 0 or r/max_r > 0.9:
                    angle_perturbation = 0.
                else:
                    angle_perturbation = angle_perturbation_step*10*np.pi/180*ydim/6e2
                    angle_perturbation_step *= -1
                n_perturbations += 1
                if abs(angle_perturbation*150/r) < np.pi/4: angle_perturbation *= 150/r
                else: angle_perturbation = np.pi/4
                textangles[j][i] += angle_perturbation
        textangles[j] = textangles[j][::-1]
    return textangles