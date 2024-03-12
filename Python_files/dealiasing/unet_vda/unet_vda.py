# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import sys
sys.path.append('src/')

import numpy as np
import time as pytime
import tensorflow as tf
tf.debugging.disable_traceback_filtering()

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
print('detected GPUs:', gpus)
if gpus:
    # for gpu in gpus:
        # tf.config.experimental.set_memory_growth(gpu, True)
    # Restrict TensorFlow to only allocate x GB of memory on the first GPU
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=384)])

from dealiasing.unet_vda.src.dealias import VelocityDealiaser
from dealiasing.unet_vda.src.feature_extraction import create_downsampler, create_upsampler

import nlr_globalvars as gv
from nlr_functions import angle_diff


# tf.config.run_functions_eagerly(True)


class Unet_VDA():
    def __init__(self):
        self.run_only_once_for_na_gt_1 = True
        
    
    def load_model(self):
        start_neurons_az = 16 # sn=16
        self.data = tf.keras.Input(shape=(None,None,1))
                
        # Down portion of unet
        down = create_downsampler(inp=self.data, start_neurons=start_neurons_az, input_channels=1)
        # Up portion of unet        
        up = create_upsampler(n_inputs=1, start_neurons=start_neurons_az, n_outputs=6)
        
        # Full model
        self.vda = VelocityDealiaser(down, up)
        
        # Load weights. Append variables/variables to model path, since without it some users report an error that prevents loading the model.
        self.vda.load_weights(gv.programdir+'/Python_files/dealiasing/unet_vda/models/dealias_sn16_csi9764.SavedModel/variables/variables')


    def __call__(self, data, vn, azis=None, da=None, extra_dealias=True):
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

        Returns
        -------
        np.ndarray (2D)
            Array with dealiased velocity data.
        """
        if not hasattr(self, 'vda'):
            self.load_model()
            
        indices = np.s_[:]
        if not azis is None:
            #Add rows with nans in case that data doesn't cover the full 360 degrees
            data, vn, indices = self.expand_data_to_360deg(data, vn, azis, da)
                
        vn = vn if type(vn) in (list, np.ndarray) else np.repeat(vn, len(data))
        
        data = self.run_model(data, vn[:, None], list(data.shape), extra_dealias).numpy()
        # Remove any extra nan rows by using indices
        return data[indices]
            
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
    
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32), tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                                  tf.TensorSpec(shape=[2], dtype=tf.int32), tf.TensorSpec(shape=[], dtype=tf.bool)))
    def run_model(self, data, vn, data_shape, extra_dealias):        
        """The model has been trained on data with an azimuthal resolution of 1°, implying that input data needs to consist
        of 360 azimuthal bins. First step below ensures that azimuthal dimension is integer multiple of 360. When this
        integer multiple is 2 (or more), then the model can't be applied to the full dataset. In this case one can run it twice on
        alternating rows (to get 1° seperation), or run it on a reduced dataset, with 2-row averaged (aliased) velocities.
        The latter is more computationally efficient, and is the default (self.run_only_once_for_na_gt_1 = True). In the latter
        case correction factors for the actual data rows are obtained obtained by comparing their potentially aliased velocities
        with the dealiased velocities from the model.
        """
        remap_data = data_shape[0]%360 != 0
        # Azimuthal dimension should be an integer multiple of 360. If not, then certain rows are either repeated or skipped, in order
        # to arrive at the desired dimension.
        _data_shape, remap_indices = data_shape, tf.constant(0)
        if remap_data:
            remap_indices, data, vn, _data_shape = self.remap_azi_dim(data, vn, data_shape)
            
        n_azi, n_rad = _data_shape[0], _data_shape[1]
        na = tf.cast(tf.round(n_azi/360), 'int32')
        
        if self.run_only_once_for_na_gt_1:
            # Reduce azimuthal dimension to 360, by averaging neigbhouring data rows. This is not a normal average, since that leads
            # to problems when averaging over both aliased and non-aliased velocities. The average is calculated by first converting
            # velocities to phases, then calculating a circular mean phase, and then converting this phase back to a velocity.
            _data = tf.reshape(data/vn, (n_azi//na, na, n_rad))
            data_mask = tf.math.is_nan(_data)
            
            phi = np.pi*_data
            sin_phi_sum = tf.reduce_sum(tf.where(data_mask, 0., tf.sin(phi)), axis=1)
            cos_phi_sum = tf.reduce_sum(tf.where(data_mask, 0., tf.cos(phi)), axis=1)
            phi_mean = tf.atan2(sin_phi_sum, cos_phi_sum)
            
            vn_mean = tf.reduce_mean(tf.reshape(vn, (360, na, 1)), axis=1)
            unmasked = tf.reduce_sum(tf.cast(~data_mask, 'uint8'), axis=1)
            v_mean = tf.where(unmasked == 0, np.nan, phi_mean*vn_mean/np.pi)
            
            dealiased_vel = self.run_vda(v_mean, vn_mean, _data_shape)
            
            n = tf.round((tf.repeat(dealiased_vel, na, axis=0)-data)/(2*vn))
            self.data = data+2*n*vn
        else:
            self.data = [self.run_vda(data[::na], vn[::na], _data_shape),
                         self.run_vda(data[1::na], vn[1::na], _data_shape)]
            self.data = tf.reshape(tf.stack(self.data, axis=1), _data_shape)
            
        # Restore original azimuthal dimension, now that velocity is dealiased. In the case that the number of rows had to be reduced to
        # arrive at the desired dimension, some data rows will not have been dealiased yet. For these a correction factor is obtained by
        # comparing the potentially aliased velocity with a reference average dealiased velocity over the neighbouring rows.
        # This is done before performing extra dealiasing, since using the reference velocity might not work well in regions of strong azimuthal
        # shear. Resulting errors can then be corrected by the extra dealiasing procedure.
        if remap_data:
            self.data, vn = self.restore_azi_dim(remap_indices, data_shape)
               
        if extra_dealias:
            self.data = self.perform_extra_dealiasing(vn, data_shape)
            
        return self.data
    
    def remap_azi_dim(self, data, vn, data_shape):
        self.orig_data, self.orig_vn, orig_n_azi = data, vn, tf.cast(data_shape[0], 'float32')
        n_azi = tf.cast(tf.round(data_shape[0]/360)*360, 'float32')
        remap_indices = tf.cast(tf.range(0.5, n_azi, dtype='float32')*orig_n_azi/n_azi, 'int32')[:, None]
        data = tf.gather_nd(data, remap_indices)
        vn = tf.gather_nd(vn, remap_indices)
        _data_shape = tf.stack((tf.cast(n_azi, 'int32'), data_shape[1]))
        return remap_indices, data, vn, _data_shape
    
    def restore_azi_dim(self, remap_indices, data_shape):
        data = tf.scatter_nd(remap_indices, self.data, data_shape)
        if len(remap_indices) > data_shape[0]:
            indices_counts = tf.cast(tf.unique_with_counts(remap_indices[:,0])[2], 'float32')
            # Without division by indices_counts, velocity values will be doubled when remap_indices contains a repeated index
            data /= indices_counts[:,None]
        vn = self.orig_vn
        
        select = tf.concat((remap_indices[1:,0]-remap_indices[:-1,0] == 2, [False]), axis=0)
        i = tf.boolean_mask(remap_indices, select)+1
        im1, ip1 = i-1, (i+1) % data_shape[0]
        
        v_ref = 0.5*(tf.gather_nd(data, im1)+tf.gather_nd(data, ip1))
        v_i, vn_i = tf.gather_nd(self.orig_data, i), tf.gather_nd(vn, i)
        n = tf.round((v_ref-v_i)/(2*vn_i))
        update = v_i+2*n*vn_i
        data = tf.tensor_scatter_nd_update(data, i, update)
        return data, vn

    def run_vda(self, vel, vn, data_shape):
        ##  Prep data for UNet
        n_rad = data_shape[1]
        # The number of radial bins needs to be an integer multiple of 64. Here the radial dimension is expanded if needed.
        n = 64
        n_rad_extra = n-n_rad%n
        vel = tf.pad(vel, [[0, 0], [0, n_rad_extra]], 'CONSTANT', constant_values=np.nan)
        
        # Pad data 12 degrees on either side with periodic boundary conditions
        pad_deg = 12
        vel = tf.concat((vel[-pad_deg:,:], vel, vel[:pad_deg,:]), axis=0)
        vn = tf.concat((vn[-pad_deg:,:], vn, vn[:pad_deg,:]), axis=0)
        
        # vel shape (batch, n_frames, Naz, Nrng, 1)
        vel = vel[None, None, :, :, None]
        # vn shape (batch, n_frames, Naz, 1)
        vn = vn[None, None, :, :]
        
        # Run UNet
        inp = {'vel':vel, 'nyq':vn}
        out = self.vda(inp)
        return out['dealiased_vel'][0,pad_deg:-pad_deg,:n_rad,0]
    
    def perform_extra_dealiasing(self, vn, data_shape):
        n_azi, n_rad = data_shape[0], data_shape[1]
        rows = tf.transpose(tf.tile([tf.range(n_azi)], (n_rad, 1)))
        
        mask = ~tf.math.is_nan(self.data)
        mask.set_shape((None, None))
        mask_indices = tf.cast(tf.where(mask), 'int32')
        _data = tf.boolean_mask(self.data, mask)
        _rows = tf.boolean_mask(rows, mask)
        
        diff_1d = tf.concat(([0], tf.where(_rows[1:] == _rows[:-1], _data[1:]-_data[:-1], 0.)), axis=0)
        diff = tf.scatter_nd(mask_indices, diff_1d, data_shape)
        corr, valid_data = self.calculate_correction_ints(diff, mask, vn)
        
        diff_1d = -tf.roll(diff_1d, -1, axis=0)
        diff = tf.scatter_nd(mask_indices, diff_1d, data_shape)
        corr2, valid_data2 = self.calculate_correction_ints(diff, mask, vn, -1)
                    
        corr = tf.where(valid_data & valid_data2 & (corr != corr2), 0., tf.where(valid_data, corr, corr2))
        return self.data - 2*vn*corr
    
    def calculate_correction_ints(self, diff, mask, vn, direction=1):
        s = np.s_[:] if direction == 1 else np.s_[:,::-1]
        diff, mask = diff[s], mask[s]
        
        ratio = tf.round(diff/(2*vn))
        cs = tf.where(mask, tf.math.cumsum(ratio, axis=1), 0.)
        ccs = tf.math.cumsum(tf.abs(tf.sign(cs)), axis=1)
        keep = ccs <= 20
        ratio = tf.where(keep, ratio, 0.)
        beyond_last_cs0 = tf.math.cumsum(tf.cast(tf.math.cumsum(ratio, axis=1) == 0., 'uint8')[:,::-1], axis=1)[:,::-1] == 0
        valid_data = keep[:,-1,None] | ~beyond_last_cs0
        ratio = tf.where(valid_data, ratio, 0.)
        cs = tf.math.cumsum(ratio, axis=1)
        
        return cs[s], valid_data[s]