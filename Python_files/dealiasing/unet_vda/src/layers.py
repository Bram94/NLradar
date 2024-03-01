"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2022 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

"""

import tensorflow as tf

class PartialConv2D(tf.keras.layers.Conv2D):
    """
    A variation of the traditional Conv2D layer that accounts
    for masked pixels.   
    
    NOTE:   This only works if stride=1 and padding='valid'
    
    Given a batch of image tensors I  with shape [N,L,W,C] 
    and a binary mask M with shape [N,L,W,C], this layer computes the
    following masked image convolution [*M] with weights W defined as
    
    I [*M] W =  ( (IxM) * W ) x M  + bias
    
    where * is traditional convolution and x is element-wise multiplication.
    
    If the setting normalize_with_masked_weights is set to True, this
    layer also normalizes the above by the convolution of M and W:
    
    (I [*M] W) / (M * W) + bias
    
    
    """
    def __init__(self,filters,
                      kernel_size,
                      normalize_with_masked_weights,
                      eps=1e-6,
                      **kwargs):
        #TODO enforce strides=(1, 1)?
        super(PartialConv2D, self).__init__(filters,kernel_size,**kwargs)
        self.normalize_with_masked_weights=normalize_with_masked_weights
        self.eps=eps
        
    def call(self, inputs, masks):
        """
        inputs is [N, L, W, C] tensor
        masks is [N, L, W, 1] tensor representing the [L,W]-mask denoting
              valid pixels (masks==1) or invalid pixels (masks==0)
        """
        masked_inputs = inputs*masks # [N, L, W, C]
        conv1 = self.convolution_op(masked_inputs,self.kernel)
        #conv1 = conv1*masks # sets masks outputs to 0
        
        if self.normalize_with_masked_weights:
            shp=tf.shape(inputs)
            norm=self.convolution_op(masks,self.kernel)
            conv1=tf.where(norm!=0,conv1/(norm+self.eps),0)
        
        if self.use_bias:
            conv1 = conv1 + self.bias
        
        return conv1


