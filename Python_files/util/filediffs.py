# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:58:26 2024

@author: -
"""

with open('C:/Users/bramv/AppData/Local/GR2Analyst_2/kpoe/KPOE_20240112_1135', "rb") as f1, open('D:/radar_data_NLradar/NWS/20240112/KPOE/KPOE20240112_1135', "rb") as f2:
    line = 0
    while True:
        line += 1
        s1 = f1.readline()
        s2 = f2.readline()
        if len(s1) == 0:
            # file1 is at end
            if len(s2) != 0:
                # file2 is not at the end
                print(f"Line {line}:")
                print(s2)
            break
        if len(s2) == 0:
          # file1 has data but file2 is at end
          # print(f"Line {line}:")
          # print(s2)
          break
        if s1 == s2:
            continue
            print("IDENTICAL ")
        else:
            print(f"Line {line}:")
            print('start: \n', s1)
            print('start: \n', s2)
            break
