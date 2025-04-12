#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append('src/')

import numpy as np
import time as pytime
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # for gpu in gpus:
        # tf.config.experimental.set_memory_growth(gpu, True)
    # Restrict TensorFlow to only allocate x GB of memory on the first GPU
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)]) # Notice here

from dealiasing.unet_vda.src.dealias import VelocityDealiaser
from dealiasing.unet_vda.src.feature_extraction import create_downsampler, create_upsampler

from nlr_functions import angle_diff


# tf.config.run_functions_eagerly(True)


class Unet_VDA():
    
    
    def load_model(self):
        start_neurons_az = 16 # sn=16
        self.data = tf.keras.Input(shape=(None,None,1))
                
        # Down portion of unet
        down = create_downsampler(inp=self.data, start_neurons=start_neurons_az, input_channels=1)
        # Up portion of unet        
        up = create_upsampler(n_inputs=1, start_neurons=start_neurons_az, n_outputs=6)
        
        # Full model
        self.vda = VelocityDealiaser(down, up)
        
        # Load weights
        self.vda.load_weights('D:/NLradar/Python_files/dealiasing/unet_vda/models/dealias_sn16_csi9764.SavedModel')
        
        self.run_only_once_for_da_gt_1 = True
        
        self.n_rads = []


    def __call__(self, data, vn, azis=None, da=None, limit_shape_changes=False):
        """
        Parameters
        ----------
        data : np.ndarray (2D)
            Array with velocity data. Incorrect values or empty bins should be set to np.nan.
        vn : float or np.ndarray (1D)
            Nyquist velocity. Either a single value for the whole scan, or a 1D array with a value for each azimuth. The latter is required when
            different Nyquist velocities are used for different azimuthal sectors of the scan. This is sometimes the case with NEXRAD scans.
        azis : np.ndarray (1D), optional
            The default assumption is that data spans (approximately) 360 degrees, with any data gap filled with nans. If that's not the case, 
            then it's possible to provide an array of azimuths corresponding to the data rows. This information will then be used to create a new
            data array, with data gaps filled with nans.
        da : float, optional
            Should also be provided when providing azis. Represents the average angle between azimuths.
        limit_shape_changes : bool, optional
            Tensorflow creates a new computational graph when data with a new shape is passed to certain operations. This takes some time,
            and it could therefore be desired to limit these shape changes. Since the number of rows (azimuths) passed to the model gets fixed at 360,
            it is the number of columns (radial gates) that determines whether a new graph is created. When setting limit_shape_changes=True, a formerly
            used radial dimension is reused when the difference is within a certain margin. This reduces recalculation of graphs, but can increase 
            the execution time of the algorithm to some extent, due to the extra data columns.

        Returns
        -------
        np.ndarray (2D)
            Array with dealiased velocity data.
        """
        self.limit_shape_changes = limit_shape_changes
        if not hasattr(self, 'vda'):
            self.load_model()
            
        indices = np.s_[:]
        if not azis is None:
            data, vn, indices = self.expand_data_to_360deg(data, vn, azis, da)
        
        vn_unique = np.unique(np.asarray(vn))
        if len(vn_unique) == 1:
            return self.prepare_and_run_model(data, vn_unique[0])[indices]
        else:
            vn = np.tile(vn, (data.shape[1], 1)).T
            return self.prepare_and_run_model(data, vn)[indices]
            
    def expand_data_to_360deg(self, data, vn, azis, da):
        n_azi, n_rad = data.shape
        
        _data, indices = [], []
        j = 0
        for i,a in enumerate(azis):
            n = angle_diff(a, azis[(i+1)%n_azi], between_0_360=True)/da
            if n > 3:
                k = sum(map(len, _data))
                _data += [data[j:i+1], np.full((round(n)-1, n_rad), np.nan, data.dtype)]
                indices += list(range(k, k+i+1-j))
                j = i+1
        k = sum(map(len, _data))
        n_expected = round(360/da)
        n_extra = max(0, k-n_expected)
        _data += [data[j:], np.full((n_extra, n_rad), np.nan, data.dtype)]
        indices += list(range(k, k+n_azi-j))
        _data = np.concatenate([j for j in _data if len(j)])
        
        shift = round(azis[0]/da)
        _data = np.roll(_data, shift, axis=0)
        indices = (np.array(indices)+shift) % len(_data)
        
        _vn = np.full(len(_data), np.nan, data.dtype)
        _vn[indices] = vn
        
        return _data, _vn, indices
    
    def prepare_and_run_model(self, data, vn):        
        """Note: the model has been trained on data with an azimuthal resolution of 1°, implying that input data needs to consist
        of 360 azimuthal bins. If the actual number is different, then measures are needed.
        """
        n_azi, n_rad = data.shape
        
        remap_data = n_azi%360 != 0
        if remap_data:
            orig_data, orig_n_azi = data, n_azi
            n_azi = round(n_azi/360)*360
            indices = (np.arange(0.5, n_azi)*orig_n_azi/n_azi).astype('uint16')
            data = data[indices]
            if type(vn) == np.ndarray:
                orig_vn = vn
                vn = vn[indices]
        da = round(n_azi/360)
        
        self.data = np.empty(data.shape, data.dtype)
        if da == 1:
            self.data = self.run_unet_vda(data, vn)
        elif self.run_only_once_for_da_gt_1:    
            data_mask = np.isnan(data)
            
            phi = np.pi*data/vn
            sin_phi, cos_phi = np.sin(phi), np.cos(phi)
            sin_phi[data_mask] = cos_phi[data_mask] = 0.
            def reshape(arr):
                return arr.T.reshape((n_rad, n_azi//da, da)).transpose((1,0,2))
            sin_phi_sum, cos_phi_sum = reshape(sin_phi).sum(axis=-1), reshape(cos_phi).sum(axis=-1)
            phi_mean = np.arctan2(sin_phi_sum, cos_phi_sum)
            
            unmasked = reshape(~data_mask).astype('uint8').sum(axis=-1)
            if type(vn) == np.ndarray:
                vn[data_mask] = 0
                vn_mean = reshape(vn).sum(axis=-1)/unmasked
            else:
                vn_mean = vn
            v_mean = phi_mean*vn_mean/np.pi
            v_mean[unmasked == 0] = np.nan
            
            dealiased_vel = self.run_unet_vda(v_mean, vn_mean)
            
            for i in range(da):
                vn_ref = vn[i::da] if type(vn) == np.ndarray else vn 
                n = np.round((dealiased_vel-data[i::da])/(2*vn_ref))
                self.data[i::da] = data[i::da]+2*n*vn_ref
        else:
            for i in range(da):
                self.data[i::da] = self.run_unet_vda(data[i::da], vn[i::da] if type(vn) == np.ndarray else vn)        
                
        if remap_data:
            copy = self.data
            self.data = np.empty((orig_n_azi, copy.shape[1]), copy.dtype)
            self.data[indices] = copy
            for i in range(orig_n_azi):
                 if not i in indices:
                     v_ref = 0.5*(self.data[i-1]+self.data[(i+1)%orig_n_azi])
                     vn_ref = orig_vn[i] if type(vn) == np.ndarray else vn
                     n = np.round((v_ref-orig_data[i])/(2*vn_ref))
                     self.data[i] = orig_data[i]+2*n*vn_ref
                
        return self.data
    
    def run_unet_vda(self, vel, vn):
        n_azi, n_rad = vel.shape
        
        # The number of radial bins needs to be an integer multiple of 64
        n = 64
        n_rad_extra = n-n_rad%n
        if self.limit_shape_changes and self.n_rads:
            diffs = np.array(self.n_rads)-(n_rad+n_rad_extra)
            diffs = diffs[(diffs >= 0) & (diffs <= 256)]
            if len(diffs):
                n_rad_extra = diffs.min()+n_rad_extra
        vel = np.concatenate([vel, np.full((n_azi, n_rad_extra), np.nan, dtype=vel.dtype)], axis=1)
        if not vel.shape[1] in self.n_rads:
            self.n_rads.append(vel.shape[1])
        
        ##  Prep data for UNet
        # shape (batch, n_frames, Naz, Nrng, 1)
        vel = vel[None, None, :, :, None]
        
        # Pad data 12 degrees on either side with periodic boundary conditions
        pad_deg = 12
        vel = np.concatenate((vel[:,:,-pad_deg:,:,:], vel, vel[:,:,:pad_deg,:,:]), axis=2)
        if type(vn) == np.ndarray:
            # Assumes that the input shapes of vel and vn are equal
            vn = np.concatenate([vn, np.full((n_azi, n_rad_extra), np.nan, dtype=vn.dtype)], axis=1)
            vn = vn[None, None, :, :, None]
            vn = np.concatenate((vn[:,:,-pad_deg:,:,:], vn, vn[:,:,:pad_deg,:,:]), axis=2)
        else: 
            # Assumes that vn is a single number
            # shape (batch, n_frames, 1)
            vn = np.array([[[vn]]])
        
        # Run UNet
        inp = {'vel':vel, 'nyq':vn}
        out = self.vda(inp)
        return out['dealiased_vel'].numpy()[0,pad_deg:-pad_deg,:n_rad,0]