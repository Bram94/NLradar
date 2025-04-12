# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:13:56 2024

@author: bramv
"""
import os
import gzip
import struct
import numpy as np
import time as pytime


VOLUME_HEADER = {'identification':
                 [(0, 4, 1, 'magic number'),
                 (4, 4, 1, 'bit-byte ordering'),
                 (8, 2, 2, 'version number'),
                 (10, 2, 2, 'mode of operation'),
                 (12, 12, 2, 'creation time and date'),
                 (24, 12, 2, 'volume start time and date'),
                 (36, 12, 2, 'volume stop time and date')],
                 'site information':
                 [(48, 2, 2, 'WMO country code'),
                  (50, 2, 2, 'WMO site number'),
                  (52, 6, 2, 'radar site longitude'),
                  (58, 6, 2, 'radar site latitude'),
                  (64, 2, 2, 'local site number'),
                  (66, 2, 2, 'local grid easting'),
                  (68, 2, 2, 'local grid northing'),
                  (70, 2, 2, 'sensor height')],
                 'hardware configuration':
                 [(72, 2, 2, 'hardware type'),
                  (74, 2, 2, 'software type'),
                  (76, 2, 2, 'number of channels'),
                  (80, 2, 2, 'channel 1 polarisation mode'),
                  (82, 2, 2, 'channel 2 polarisation mode'),
                  (84, 2, 2, 'channel 1 radar constant'),
                  (86, 2, 2, 'channel 2 radar constant'),
                  (88, 2, 2, 'channel 1 wavelength'),
                  (90, 2, 2, 'channel 2 wavelength'),
                  (92, 2, 2, 'channel 1 beam width'),
                  (94, 2, 2, 'channel 2 beam width')],
                 'processor configuration':
                 [(108, 2, 2, 'type of scan'),
                  (110, 2, 2, 'number of scans in volume'),
                  (112, 2, 2, 'number of rays per scan'),
                  (114, 2, 2, 'number of bins per ray'),
                  (116, 2, 2, 'processed range bin length'),
                  (118, 2, 2, 'pulse length'),
                  (120, 2, 2, 'antenna rotation rate'),
                  (122, 2, 2, 'number of samples'),
                  (124, 2, 2, 'primary PRF'),
                  (126, 2, 2, 'secondary PRF'),
                  (128, 2, 2, 'unambiguous range'),
                  (130, 2, 2, 'unambiguous velocity')],
                 'data processing':
                 [(144, 2, 2, 'processor flags'),
                  (146, 2, 2, 'clutter filter type'),
                  (148, 2, 2, 'clutter filter response'),
                  (150, 2, 2, 'clutter power threshold'),
                  (152, 2, 2, 'noise threshold'),
                  (154, 2, 2, 'SQI threshold')],
                 'data storage':
                 [(168, 2, 2, 'derived data type'),
                  (170, 2, 2, 'source data type'),
                  (172, 2, 2, 'number of bytes per element'),
                  (174, 2, 2, 'number of values per element'),
                  (176, 2, 2, 'compression')],
                 'diagnostics section':
                 [(192, 2, 2, 'diagnostic flags')]
                 }
    
SCAN_HEADER = [(0, 2, 2, 'scan index'),
               (2, 2, 2, 'number of rays in scan'),
               (4, 2, 2, 'number of bins per ray'),
               (6, 2, 2, 'range to first bin'),
               (8, 2, 2, 'scan start seconds'),
               (10, 2, 2, 'scan stop seconds'),
               (12, 2, 2, 'scan start azimuth'),
               (14, 2, 2, 'scan stop azimuth'),
               (16, 2, 2, 'scan requested elevation'),
               (18, 2, 2, 'scan average elevation'),
               (20, 2, 2, 'beam number'),
               (22, 2, 2, 'number of elevations in scan'),
               (24, 2, 2, 'RHI start elevation'),
               (26, 2, 2, 'RHI stop elevation'),
               (28, 4, 4, 'transmitting frequency'),
               (32, 2, 2, 'channel 1 noise sample'),
               (34, 2, 2, 'channel 2 noise sample'),
               (36, 2, 2, 'channel 1 ambient noise'),
               (38, 2, 2, 'channel 2 ambient noise'),
               (40, 2, 2, 'channel 1 RX calibration p_offset'),
               (42, 2, 2, 'channel 2 RX calibration p_offset'),
               (44, 2, 2, 'channel 1 receiver saturation'),
               (46, 2, 2, 'channel 2 receiver saturation'),
               (48, 2, 2, 'channel 1 injected noise power'),
               (50, 2, 2, 'channel 2 injected noise power'),
               (52, 2, 2, 'channel 1 receiver noise power'),
               (54, 2, 2, 'channel 2 receiver noise power'),
               (56, 2, 2, 'transmitter pulse power'),
               (58, 2, 2, 'radar constant'),
               (60, 4, 4, 'NCO frequency')]

RAY_HEADER = [(0, 2, 2, 'V channel noise'),
              (2, 2, 2, 'azimuth center angle'),
              (4, 2, 2, 'elevation angle'),
              (6, 2, 2, 'H channel noise'),
              (8, 2, 2, 'number of ray per degree')]

# Product bit range per multi-parameter data type
p_bits = {2111:{'REF':[0, 12], 'CI':[12, 16]},
          2122:{'REF':[0, 12], 'CI':[12, 16], 'VEL':[16, 28], 'SQI':[28, 32]},
          2179:{'REF':[0, 10], 'CPA':[10, 16]}, #This data type is not defined in the docs, so unknown what last 2 bytes contain
          2213:{'REF':[0, 12], 'CI':[12, 16], 'VEL':[16, 28], 'SQI':[28, 32], 'SW':[32, 40], 'ZDR':[40, 48], 'RHO':[48, 64], 'PHI':[64, 76]}}
p_offset = {'REF':-32., 'CI':0, 'VEL':np.pi, 'SQI':0., 'SW':0., 'ZDR':-8., 'RHO':-20, 'PHI':-180, 'CPA':0.}
p_gain = {'REF':0.1, 'CI':1, 'VEL':-np.pi/2**11, 'SQI':1/15, 'SW':np.pi/256, 'ZDR':0.0625, 'RHO':0.002, 'PHI':180/2**11, 'CPA':1/2**6}
p_lookup_table = {'CI':np.array([0, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 5, 5.5, 6, 7])}


def bytes_to_array(data, endianness, datadepth=8):
    datawidth = int(datadepth / 8)

    datatype = endianness + 'u' + str(datawidth)

    return np.ndarray((int(len(data) / datawidth),),
                  dtype=datatype, buffer=data)

def bits_to_n(bits, endianness, dtype, signed=False):
    """Convert a sequence of bits to a number, assuming that the most significant bits are placed first (big endianness style)
    If signed=True, then it is assumed that the first bit represents the sign of the number, which is 1 when the first bit
    is 0, and -1 otherwise.
    """
    s = np.s_[:] if endianness == '>' else np.s_[..., ::-1]
    bits = bits[s]    
    if type(bits) is np.ndarray:
        if signed:
            return -2*(bits[..., 0]-0.5)*np.sum(bits[..., 1:]*np.array([2**j for j in reversed(range(bits.shape[-1]-1))], dtype), axis=-1)
        else:
            return np.sum(bits*np.array([2**j for j in reversed(range(bits.shape[-1]))], dtype), axis=-1)      
    else:
        if signed:
            return int(-2*(bits[0]-0.5)*np.sum(bits[1:]*np.array([2**j for j in reversed(range(len(bits)-1))])))
        else:
            return int(np.sum(bits*np.array([2**j for j in reversed(range(len(bits)))])))
        
fmts = {1:'b', 2:'h', 4:'l'}
def unpack_struct(data, endianness, length):
    fmt = endianness+fmts[length]
    if len(data) == length:
        return struct.unpack(fmt, data)[0]
    else:
        return [struct.unpack(fmt, data[i:i+length])[0] for i in range(0, len(data), length)]





class UKMOPolarFile():
    def __init__(self):
        self.filepath = None
        
    
    def __call__(self, filepath, products='all'):
        products = list(p_bits) if products == 'all' else products
            
        if filepath != self.filepath or (products and self.data_bits is None):
            self.read_file(filepath, obtain_products=bool(products))
            self.filepath = filepath
            self.data = {}
            
        self.data_type_id = self.volume_header['derived data type']
        for p in products:
            if p in p_bits[self.data_type_id] and not p in self.data:
                self.data[p] = self.read_product(p)
        
        if products:
            return self.data, self.volume_header, self.scan_header, self.ray_headers
        else:
            return self.volume_header, self.scan_header
        
    def read_file(self, filepath, obtain_products=True):
        with gzip.open(filepath, 'rb') as f:
            _volume_header = f.read(256)
            self.volume_header = {}
            for section, items in VOLUME_HEADER.items():
                for item in items:
                    start, full_length, length, name = item
                    self.volume_header[name] = _volume_header[start:start+full_length]
                    if name == 'magic number':
                        self.endianness = '<' if self.volume_header[name][0] == 65 else '>'
                    else:
                        self.volume_header[name] = unpack_struct(self.volume_header[name], self.endianness, length)
    
            _scan_header = f.read(64)
            self.scan_header = {}
            for item in SCAN_HEADER:
                start, full_length, length, name = item
                self.scan_header[name] = _scan_header[start:start+full_length]
                self.scan_header[name] = unpack_struct(self.scan_header[name], self.endianness, length)
            
            self.data_bits = None
            if obtain_products:
                scan_content = f.read()
                
                nrays = self.scan_header['number of rays in scan']
                ngates = self.scan_header['number of bins per ray']
                bytes_per_bin = self.volume_header['number of bytes per element']
                header_bytes = 10
                ray_bytes = ngates*bytes_per_bin+header_bytes
                
                self.ray_headers = {}
                for i in range(nrays):
                    _ray_header = scan_content[i*ray_bytes:i*ray_bytes+header_bytes]
                    for item in RAY_HEADER:
                        start, full_length, length, name = item
                        if not name in self.ray_headers:
                            self.ray_headers[name] = []
                        self.ray_headers[name].append(_ray_header[start:start+full_length])
                        self.ray_headers[name][-1] = unpack_struct(self.ray_headers[name][-1], self.endianness, length)
                                                     
                scan_data_uints = bytes_to_array(scan_content, self.endianness).reshape(nrays, ray_bytes)[:, header_bytes:].reshape(nrays, ngates, bytes_per_bin)
                self.data_bits = np.unpackbits(scan_data_uints, axis=-1, bitorder='big' if self.endianness == '>' else 'little')
                        
    def read_product(self, p):
        bits, offset, gain = p_bits[self.data_type_id][p], p_offset[p], p_gain[p]
        dtype = 'int16' if p in p_lookup_table else 'float32'
        data = offset + gain*bits_to_n(self.data_bits[..., bits[0]:bits[1]], self.endianness, dtype)
        if p in p_lookup_table:
            data = p_lookup_table[p][data]
        if p in ('VEL', 'SW'):
            vn = self.volume_header['unambiguous velocity']/100
            data *= vn/np.pi
        return data         





if __name__ == '__main__':
    with open('D:/NLradar/NLradar_private/Python_files/util/radar_elevs_eu.txt') as f:
        content = f.read()
    radar_elevs = [j.split('\t') for j in content.split('\n') if j]
    radar_elevs = {j[0]:j[1] for j in radar_elevs}
    
    reader = UKMOPolarFile()
    
    t = pytime.time()
    base_dir = 'H:/radar_data_NLradar/UKMO'
    dates = os.listdir(base_dir)
    for radar in radar_elevs:
        for date in dates:
            if date != '20210625':
                continue
            directory = base_dir+f'/{date}'
            radar_dir = directory+'/'+radar.replace(' ', '')+'_V'
            if not os.path.exists(radar_dir):
                radar_dir = directory+'/'+radar.replace(' ', '')+'_Z'
            if os.path.exists(radar_dir):
                filepath = radar_dir+'/'+os.listdir(radar_dir)[0]
                data, volume_header, scan_header, ray_headers = reader(filepath, ['REF'])
                s = radar+'\t\t.\t'
                for attr in ('lat', 'long'):
                    _ = volume_header[f'radar site {attr}itude']
                    s += format(_[0]+_[1]/60+_[2]/3600, '.5f')+'\t'
                s += str(volume_header['sensor height'])+'\t'
                s += radar_elevs[radar]+'\t'
                s += 'C'
                print(s)
                break