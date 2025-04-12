# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:16:09 2024

@author: bramv
"""

from setuptools import setup

setup(name='NLradar',
      version='0.1',
      description='',
      author="Bram van 't Veen",
      packages=[],
      install_requires=['numpy','pyproj','pyshp','opencv-python','pillow','imageio','pyqt5','scipy','requests','gpxpy','netcdf4==1.6','pyopengl','matplotlib','numpy-bufr @ git+https://github.com/Bram94/numpy_bufr.git','xmltodict','boto3','pytz','av','tensorflow==2.10']
     )