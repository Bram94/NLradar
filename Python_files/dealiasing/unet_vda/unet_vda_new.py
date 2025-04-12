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
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

from dealiasing.unet_vda.src.dealias import VelocityDealiaser
from dealiasing.unet_vda.src.feature_extraction import create_downsampler, create_upsampler

from nlr_functions import angle_diff


# tf.config.run_functions_eagerly(True)


class Unet_VDA():
    def __init__(self):
        self.run_only_once_for_na_gt_1 = True
        self.n_rads = []
        
    
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
            #Add rows with nans in case that data doesn't cover the full 360 degrees
            data, vn, indices = self.expand_data_to_360deg(data, vn, azis, da)
        
        vn_unique = np.unique(np.asarray(vn, dtype=data.dtype))
        vn = vn_unique[0] if len(vn_unique) == 1 else np.tile(vn, (data.shape[1], 1)).T
        
        orig_n_rad = data.shape[1]
        # The number of radial bins needs to be an integer multiple of 64
        data, vn = self.check_rad_dim(data, vn)
        # The number of azimuthal bins needs to be an integer multiple of 360
        data, vn = self.check_azi_dim(data, vn)
        
        n_azi, n_rad = data.shape        
        na = round(n_azi/360)
        vn_dims = len(vn.shape)
        data = self.run_model(data, vn, data.shape, vn_dims).numpy()
        #restore number of azimuthal bins to value before calling self.check_azi_dim, and then remove any extra nan rows by using indices
        return self.restore_azi_dim(data, vn)[indices, :orig_n_rad]
            
    def expand_data_to_360deg(self, data, vn, azis, da):
        """Add rows of nans in case of data gaps. An array of indices is created that contains for each data row the row in which it appears
        in the expanded data array. These indices are used to obtain back the original data array after dealiasing.
        """
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
    
    def check_rad_dim(self, data, vn):
        """The number of radial bins needs to be an integer multiple of 64. Here the radial dimension is expanded if needed.
        Also, a formerly used radial dimension can be reused when self.limit_shape_changes = True. This limits data shape variability,
        which triggers recreation of computational graphs, an expensive operation.
        """
        n_azi, n_rad = data.shape
        
        n = 64
        n_rad_extra = n-n_rad%n
        if self.limit_shape_changes and self.n_rads:
            diffs = np.array(self.n_rads)-(n_rad+n_rad_extra)
            diffs = diffs[(diffs >= 0) & (diffs <= 256)]
            if len(diffs):
                n_rad_extra = diffs.min()+n_rad_extra
        data = np.concatenate([data, np.full((n_azi, n_rad_extra), np.nan, dtype=data.dtype)], axis=1)
        if not data.shape[1] in self.n_rads:
            self.n_rads.append(data.shape[1])
        
        if type(vn) == np.ndarray:
            # Assumes that the input shapes of data and vn are equal
            vn = np.concatenate([vn, np.full((n_azi, n_rad_extra), np.nan, dtype=vn.dtype)], axis=1)
        return data, vn 
    
    def check_azi_dim(self, data, vn):
        """Azimuthal dimension should be an integer multiple of 360. If not, then certain rows are either repeated or skipped, in order
        to arrive at the desired dimension.
        """
        n_azi = data.shape[0]
        self.remap_data = n_azi%360 != 0
        if self.remap_data:
            self.orig_data, self.orig_n_azi = data, n_azi
            n_azi = round(n_azi/360)*360
            self.remap_indices = (np.arange(0.5, n_azi)*self.orig_n_azi/n_azi).astype('uint16')
            data = data[self.remap_indices]
            if type(vn) == np.ndarray:
                self.orig_vn = vn
                vn = vn[self.remap_indices]
        return data, vn
                
    def restore_azi_dim(self, data, vn):
        """Restore original azimuthal dimension, now that velocity is dealiased. In the case that the number of rows had to be reduced to
        arrive at the desired dimension, some data rows will not have been dealiased yet. For these a correction factor is obtained by
        comparing the potentially aliased velocity with the average dealiased velocity in the neighbouring rows.
        """
        if self.remap_data:
            copy = data
            data = np.empty((self.orig_n_azi, copy.shape[1]), copy.dtype)
            data[self.remap_indices] = copy
            for i in range(self.orig_n_azi):
                 if not i in self.remap_indices:
                     v_ref = 0.5*(data[i-1]+data[(i+1)%self.orig_n_azi])
                     vn_ref = self.orig_vn[i] if type(vn) == np.ndarray else vn
                     n = np.round((v_ref-self.orig_data[i])/(2*vn_ref))
                     data[i] = self.orig_data[i]+2*n*vn_ref
        return data
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.float32),
                                  tf.TensorSpec(shape=[2], dtype=tf.int32), tf.TensorSpec(shape=None, dtype=tf.int32)))
    def run_model(self, data, vn, data_shape, vn_dims):        
        """The model has been trained on data with an azimuthal resolution of 1°, implying that input data needs to consist
        of 360 azimuthal bins. Former steps have ensured that azimuthal dimension is integer multiple of 360. When this
        integer multiple is 2 (or more), then the model can't be applied to the full dataset. In this case one can run it twice on
        alternating rows (to get 1° seperation), or run it on a reduced dataset, with 2-row averaged (aliased) velocities.
        The latter is more computationally efficient, and is the default (self.run_only_once_for_na_gt_1 = True). In the latter
        case correction factors for the actual data rows are obtained obtained by comparing their potentially aliased velocities
        with the dealiased velocities from the model.
        """
        n_azi, n_rad = data_shape[0], data_shape[1]
        na = tf.cast(tf.round(n_azi/360), 'int32')
        
        if na == 1:
            self.data = self.run_vda(data, vn, vn_dims)
        elif self.run_only_once_for_na_gt_1:
            data_mask = tf.math.is_nan(data)
            
            # Reduce azimuthal dimension to 360, by averaging neigbhouring data rows. This is not a normal average, since that leads
            # to problems when averaging over both aliased and non-aliased velocities. The average is calculated by first converting
            # velocities to phases, then calculating a circular mean phase, and then converting this phase back to a velocity.
            phi = np.pi*data/vn
            sin_phi = tf.where(data_mask, 0., tf.sin(phi))
            cos_phi = tf.where(data_mask, 0., tf.cos(phi))
            def reshape(arr):
                return tf.transpose(tf.reshape(tf.transpose(arr), (n_rad, n_azi//na, na)), (1,0,2))
            sin_phi_sum = tf.reduce_sum(reshape(sin_phi), axis=-1)
            cos_phi_sum = tf.reduce_sum(reshape(cos_phi), axis=-1)
            phi_mean = tf.atan2(sin_phi_sum, cos_phi_sum)
            
            unmasked = tf.reduce_sum(tf.cast(reshape(~data_mask), 'float32'), axis=-1)
            if vn_dims == 2:
                vn = tf.where(data_mask, 0., vn)
                vn_mean = tf.reduce_sum(reshape(vn), axis=-1)/unmasked
            else: 
                vn_mean = vn
            v_mean = tf.where(unmasked == 0., np.nan, phi_mean*vn_mean/np.pi)
            
            dealiased_vel = self.run_vda(v_mean, vn_mean, vn_dims)
            
            self.data = []
            vn_ref = vn[0::na] if vn_dims == 2 else vn 
            n = tf.round((dealiased_vel-data[0::na])/(2*vn_ref))
            self.data += [data[0::na]+2*n*vn_ref]
            vn_ref = vn[1::na] if vn_dims == 2 else vn 
            n = tf.round((dealiased_vel-data[1::na])/(2*vn_ref))
            self.data += [data[1::na]+2*n*vn_ref]
            self.data = tf.reshape(tf.concat([tf.expand_dims(arr, 1) for arr in self.data], axis=1), data_shape)
        else:
            self.data = []
            for i in range(na):
                self.data += [self.run_vda(data[i::na], vn[i::na] if vn_dims == 2 else vn, vn_dims)]
            self.data = tf.reshape(tf.concat([tf.expand_dims(arr, 1) for arr in self.data], axis=1), data_shape)
               
        return self.data
    
    def run_vda(self, vel, vn, vn_dims):
        ##  Prep data for UNet
        # shape (batch, n_frames, Naz, Nrng, 1)
        vel = vel[None, None, :, :, None]
         
        # Pad data 12 degrees on either side with periodic boundary conditions
        pad_deg = 12
        vel = tf.concat((vel[:,:,-pad_deg:,:,:], vel, vel[:,:,:pad_deg,:,:]), axis=2)
        if vn_dims == 2:
            vn = vn[None, None, :, :, None]
            vn = tf.concat((vn[:,:,-pad_deg:,:,:], vn, vn[:,:,:pad_deg,:,:]), axis=2)
        else: 
            # Assumes that vn is a single number
            # shape (batch, n_frames, 1)
            vn = tf.reshape(vn, (1,1,1))
        
        # Run UNet
        inp = {'vel':vel, 'nyq':vn}
        out = self.vda(inp)
        return out['dealiased_vel'][0,pad_deg:-pad_deg,:,0]