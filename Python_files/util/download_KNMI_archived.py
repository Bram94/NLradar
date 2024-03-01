# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:56:34 2020

@author: bramv
"""
import os
import requests
import tarfile



api_key = ""
dataset = "https://api.dataplatform.knmi.nl/open-data/datasets/radar_tar_vol_full_herwijnen/versions/1.0/files"
# dataset = "https://api.dataplatform.knmi.nl/open-data/datasets/radar_tar_volume_denhelder/versions/1.0/files"
# dataset = "https://api.dataplatform.knmi.nl/open-data/datasets/radar_tar_volume_debilt/versions/1.0/files"
output = requests.get(dataset, headers={"Authorization": api_key}, params = {"maxKeys": 1, "startAfterFilename": "RAD62_OPER_O___TARVOL__L2__20190507T000000_20190508T000000_0001.tar"})

list_files = output.json()
dataset_files = list_files.get("files")
print(dataset_files)
for file in dataset_files:
    file_name = file.get("filename")
    print(file_name)
    get_file_response = requests.get(dataset+'/'+file_name+'/url', headers={"Authorization": api_key})
    download_url = get_file_response.json().get("temporaryDownloadUrl")
    directory = 'H:/radar_data_NLradar/KNMI'
    os.makedirs(directory, exist_ok = True)
    filepath = directory+'/'+file_name
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1000000000): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
                
    radar_dir = directory+'/'+file_name[:file_name.index('.')]
    with tarfile.open(directory+'/'+file_name, 'r') as f:
        f.extractall(radar_dir)
        
    os.remove(directory+'/'+file_name)