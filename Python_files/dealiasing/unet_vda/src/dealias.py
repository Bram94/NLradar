"""
dealias model

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2022 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

"""

import tensorflow as tf
from packaging import version
import time as pytime

class VelocityDealiaser(tf.keras.Model):
    """
    Main veloicty dealiasing class

    This model takes as input a dict
      {
        'vel':    Aliased velocity of shape [batch, n_times, n_az, n_rad, 1]
        'nyq':    nyquist velocitys used for degradation w/shape [batch, n_times, n_az, 1]
      }

    """
    def __init__(self,
                 extractor,
                 upsampler):
        super(VelocityDealiaser, self).__init__()
        self.extractor=extractor
        self.upsampler=upsampler
        self.alias_loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        if version.parse(tf.__version__) >= version.parse("2.11.0"):
            self.optimizer = tf.keras.optimizers.legacy.Adam()
        else:
            self.optimizer = tf.keras.optimizers.Adam()

    def call(self, inputs):
        """
        Inputs is a dict:
           {
               'vel':    aliased velocity of shape [batch, n_times, n_az, n_rad, 1]
               'nyq':    nyquist velocitys used for degradation w/shape [batch, n_times, n_az, 1]
           }
           Note that it is assumed that -nyq < vel < nyq,   no checks are done on this
        """
        #vel_in=inputs['degraded_N0U']
        vel_in=inputs['vel']
        nyq=inputs['nyq'][:,:,:,None,:] # Add the radial dimension
        
        # Normalize velocity by nyquist
        vel = vel_in / nyq #[N,n_f,L,W,1]
        vel,bad_mask=make_velocity_mask(vel,
                         fill_val=tf.cast(-3.0,vel.dtype)) 
        # run feature extraction
        x=tf.transpose(vel,(0,2,3,1,4))
        xshp = tf.shape(x)
        x=tf.reshape(x,(xshp[0],xshp[1],xshp[2],xshp[3]*xshp[4]))

        f=self.extractor(x)
        out=self.upsampler(f) # [N,L,W,6] one-hot pred
        out=tf.clip_by_value(out,-50.0,50.0) # improved training stability
        
        # Dealias velocity
        # take last frame only
        vel_in = vel_in[:,-1] # [batch,L,W,1]
        nyq = nyq[:,-1] # [batch,L,W,1]
        vel_pred = self.dealias_vel(vel_in,out,nyq)
            
        out = tf.cast(out,tf.float32)
        vel_pred=tf.cast(vel_pred,tf.float32)
        # print(vel_pred)
        
        return {'alias_mask':out,'dealiased_vel':vel_pred}

    def dealias_vel(self,vel,alias_onehot,nyq):
        """
        Uses nyquist velocity and output of segmentation model to correct aliased regions

        vel            = [B,L,W,1]
        alias_onehot   = [B,L,W,6]
        nyq            = [B,L,W,1]
        """
        #nyq=tf.cast(nyq,tf.float32)
        cat=tf.argmax(alias_onehot,axis=-1) # [B,L,W]
        cat=cat[:,:,:,None]                 # [B,L,W,1]
        vel=self.apply_correction(cat==1,vel,-4*nyq)
        vel=self.apply_correction(cat==2,vel,-2*nyq)
        vel=self.apply_correction(cat==4,vel, 2*nyq)
        vel=self.apply_correction(cat==5,vel, 4*nyq)
        return vel

    def apply_correction(self,mask,vel,correction):
        vel=tf.where(mask,vel+correction,vel)
        return vel 




def make_velocity_mask(vel,fill_val=0):
    """
    Splits velocity into pair of
    vel,bad_mask where bad_mask denotes nil pixels,
    and vel has nil pixels replaced with fill_val
    """
    bad_mask = tf.math.is_nan(vel)
    vel = tf.where(bad_mask,fill_val,vel)
    return vel,bad_mask