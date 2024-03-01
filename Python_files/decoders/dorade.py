# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

# https://www.eol.ucar.edu/sites/default/files/files_live/private/files/field_project/EMEX/DoradeDoc.pdf

import struct
import re
import numpy as np
import time as pytime



class DORADEFile():
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            self.content = f.read()
            
        self.i_ryib = [i.start() for i in re.finditer(b'RYIB', self.content)]
        self.i_parm = [i.start() for i in re.finditer(b'PARM', self.content)]
        self.i_rdat = [i.start() for i in re.finditer(b'RDAT', self.content)]
        self.i_asib = [i.start() for i in re.finditer(b'ASIB', self.content)] if b'ASIB' in self.content else None
        self.i_radd = self.content.index(b'RADD')
        self.i_celv = self.content.find(b'CELV') # might not exist
        
        i = self.i_radd
        self.radar_name = self.content[i+8:i+16].decode().replace('\x00', '').strip()
        print(self.radar_name, self.content[i+8:i+16])
        
        i = self.i_ryib[0]
        nbytes = struct.unpack('i', self.content[i+4:i+8])[0]
        byte_order = 'little' if abs(nbytes) > 44 else 'big'
        self.bo = '!' if byte_order == 'little' else ''
        self.dt_int16, self.dt_float32 = np.dtype('int16'), np.dtype('float32')
        self.dt_int16 = self.dt_int16.newbyteorder('>' if byte_order == 'little' else '<')
        self.dt_float32 = self.dt_float32.newbyteorder('>' if byte_order == 'little' else '<')
        
        self.n_params = len(self.i_parm)
        self.params = [''.join([j for j in self.content[i+8:i+16].decode() if j.isalpha()]) for i in self.i_parm]
        print(self.params)
        
        self.n_azi = len(self.i_ryib)
        if self.i_celv != -1:
            i = self.i_celv
            self.n_rad = struct.unpack(self.bo+'i', self.content[i+8:i+12])[0]
        else:
            i = self.i_parm[0]
            self.n_rad = struct.unpack(self.bo+'i', self.content[i+200:i+204])[0]
            
        self.compression = struct.unpack(self.bo+'h', self.content[i+68:i+70])[0]
            
    def get_meta(self):
        if self.i_celv != -1:
            i = self.i_celv
            ranges = np.frombuffer(self.content[i+12:i+1512], dtype=self.dt_float32)
            first_gate = ranges[0]
            dr = ranges[1]-ranges[0]
        else:
            i = self.i_parm[0]
            first_gate = struct.unpack(self.bo+'f', self.content[i+204:i+208])[0]
            dr = struct.unpack(self.bo+'f', self.content[i+208:i+212])[0]
        
        azimuths, elevations, times = [], [], []
        for i, j in enumerate(self.i_ryib):
            heading = 0.
            if self.i_asib:
                k = self.i_asib[i]
                # heading = struct.unpack(self.bo+'f', self.content[k+36:k+40])[0]
                if i == 0: print(struct.unpack(self.bo+'f', self.content[k+36:k+40])[0], 'heading')
            azimuth = (heading + struct.unpack(self.bo+'f', self.content[j+24:j+28])[0]) % 360
            azimuths.append(azimuth)
            elevations.append(struct.unpack(self.bo+'f', self.content[j+28:j+32])[0])
            hour = struct.unpack(self.bo+'h', self.content[j+16:j+18])[0]
            minute = struct.unpack(self.bo+'h', self.content[j+18:j+20])[0]
            second = struct.unpack(self.bo+'h', self.content[j+20:j+22])[0]
            times.append(':'.join([format(k, '02d') for k in (hour, minute, second)]))
        azimuths, elevations, times = np.array(azimuths), np.array(elevations), np.array(times)
        
        i = self.i_radd
        v_nyquist = struct.unpack(self.bo+'f', self.content[i+92:i+96])[0]        
        lon, lat, altitude = [struct.unpack(self.bo+'f', self.content[i+80+j*4:i+84+j*4])[0] for j in range(3)]
        altitude *= 1e3 # km to m
            
        meta = {'n_azi':self.n_azi, 'n_rad':self.n_rad, 'dr':dr, 'first_gate':first_gate, 
                'azimuths':azimuths, 'elevations':elevations, 'times':times, 'v_nyquist':v_nyquist,
                'lon':lon, 'lat':lat, 'altitude':altitude}
        return meta

    def get_data(self, param_name, mask_value=np.nan): 
        # param_name can be either a string or a list of strings with possible names that refer to the same parameter
        if not isinstance(param_name, list):
            param_name = [param_name]
        i_param = self.params.index([p for p in param_name if p in self.params][0])
        
        self.data = np.empty((self.n_azi, self.n_rad), dtype='int16')
        for i, j in enumerate(self.i_ryib):
            j = self.i_rdat[i*self.n_params+i_param]
            nbytes = struct.unpack(self.bo+'i', self.content[j+4:j+8])[0]
            data = np.frombuffer(self.content[j+16:j+nbytes], dtype=self.dt_int16)
            if self.compression:
                self._rle_decode(data, i)
            else:
                self.data[i] = data
        
        i = self.i_parm[i_param]
        scale = struct.unpack(self.bo+'f', self.content[i+92:i+96])[0]
        bias = struct.unpack(self.bo+'f', self.content[i+96:i+100])[0]
        print(scale, bias)
        print(self.data.max())
            
        data_mask = self.data <= -32767
        self.data = self.data.astype('float32') / scale + bias
        self.data[data_mask] = mask_value
        
        return self.data        
    
    def _rle_decode(self, data_comp, i_row):
        i = j = 0
        while i < len(data_comp):
            val = data_comp[i]
            n = val & 32767
            if val & -32768:
                self.data[i_row, j:j+n] = data_comp[i+1:i+1+n]
                i += 1+n
                j += n
            else:
                self.data[i_row, j:j+n] = -32768
                i += 1
                j += n
        
        
    


if __name__ == '__main__':
    filename = 'D:/radar_data_NLradar/ARRC/20090605/swp.1090605214326.MWR_05XP.0.1.0_PPI_v2'
    filename = 'D:/radar_data_NLradar/ARRC/20090605/swp.1090605214341.MWR_05XP.0.1.0_PPI_v16'
    # filename = 'D:/radar_data_NLradar/ARRC/20090605/swp.1090605214350.MWR_05XP.0.1.0_PPI_v30'
    # filename = 'D:/radar_data_NLradar/ARRC/20090605/swp.1090605220338.MWR_05XP.0.1.0_PPI_v506'
    # filename = 'D:/radar_data_NLradar/ARRC/20090605/swp.1090605220453.MWR_05XP.0.1.0_PPI_v128'
    filename = 'D:/radar_data_NLradar/ARRC/20090605/Radar_1/swp.1090605220639.MWR_05XP.0.2.5_PPI_v367'
    filename = 'D:/radar_data_NLradar/ARRC/20100510/Radar_1/swp.1100510224506.OU-PRIME.0.1.0_SUR_v019'
    # filename = 'D:/radar_data_NLradar/ARRC/20100611/20100611232937/swp.1100611233006.UMass-XP.0.02.9_PPI_v2'
    filename = 'D:/radar_data_NLradar/ARRC/20100611/Radar_1/20100611232937/swp.1100611233124.UMass-XP.0.13.9_PPI_v2'
    # filename = 'D:/radar_data_NLradar/ARRC/20090605/Radar_2/swp.1090605214056.DOW7.546.1.0_PPI_v90'
    filename = 'D:/radar_data_NLradar/ARRC/20090605/Radar_2/swp.1090605220651.DOW7.109.0.4_PPI_v297'
    # filename = 'D:/radar_data_NLradar/ARRC/20090605/Radar_1/swp.1090605220223.MWR_05XP.0.1.0_PPI_v338'
    filename = 'D:/radar_data_NLradar/ARRC/20090605/Radar_3/swp.1090605214848.DOW6.59.0.4_PPI_v68'
    filename = 'D:/radar_data_NLradar/ARRC/20100525/Radar_2/dep2/swp/swp.1100525232446.UMass-WD.0.00.7_PPI_v2'
    
    d = DORADEFile(filename)
    out = d.get_data(['CR', 'DBZ', 'ZH', 'DZ'], 0)
    out = d.get_data(['VE'], 0)
    meta = d.get_meta()
    print(out, out.min(), out.max())
    # print(meta)