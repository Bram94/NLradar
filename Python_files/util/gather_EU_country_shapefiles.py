# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:56:50 2020

@author: bramv
"""
import os

basedir = 'C:/users/bramv/Downloads/Countries_Europe'
content = os.listdir(basedir+'/All')
for c in content:
    if c[-5] == '0':
        try:
            os.rename(basedir+'/All/'+c, basedir+'/'+c)
        except Exception as e:
            print(e)