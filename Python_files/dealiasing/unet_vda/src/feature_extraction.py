"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2022 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

"""

import tensorflow as tf

_POOL=[ 
    (1,2), 
    (2,4),
    (4,2),
    (4,4),
]

# Make arch for a processing layer
def create_downsampler(inp=None,
                           input_channels: int=1,
                           start_neurons: int=32):
    """
    Input of block should have shape (batch, naz, nrad, input_channels)
    
    Output will be list of size n_layers of decreasing resolution
    
    output[0] : [batch, naz/2 , nrad/2, start_neurons]
    output[1] : [batch, naz/4 , nrad/8, start_neurons*2]
    output[2] : [batch, naz/16 , nrad/16, start_neurons*4]
    output[3] : [batch, naz/64 , nrad/64, start_neurons*8]
    
    """
    if inp is None:
        inp = tf.keras.Input(shape=(None,None,input_channels))
    
    inp_norm=inp # Assumed normalized prior to being passed in!

    x0=polar_block(inp_norm,
                   input_channels=input_channels,
                   pool_size=_POOL[0],
                   kernel_size=(7,7),
                   output_channels=start_neurons)
    x1=polar_block(x0,
                   input_channels=start_neurons, 
                   pool_size=_POOL[1],
                   kernel_size=(5,5),
                   output_channels=2*start_neurons)
    x2=polar_block(x1,
                   input_channels=2*start_neurons,
                   pool_size=_POOL[2],
                   kernel_size=(3,3),
                   output_channels=4*start_neurons)
    x3=polar_block(x2,
                   input_channels=4*start_neurons,
                   pool_size=_POOL[3],
                   kernel_size=(3,3),
                   output_channels=8*start_neurons)
    
    return tf.keras.Model(inputs=inp,outputs=[x0,x1,x2,x3],name='PolarExtractor')
    
    
def polar_block(x, 
                input_channels, 
                kernel_size, 
                pool_size, 
                output_channels,
                alpha=0.1):
    """
    Block used in feature extraction
    
    x->Conv2D(kernel)->LRelu->Conv2D(kernel)->LRelu->Pool(pool_size) -> Add()
    |
    ----------------------> Conv2d(1) ---------------------------------^
    """
    if input_channels != output_channels:
        # Resize the channel dimension
        x1 = tf.keras.layers.Conv2D(output_channels, 1, strides=(1, 1), padding='same')(x)
    else:
        x1=x
    x1=tf.keras.layers.AveragePooling2D(pool_size)(x1)
    
    x2 = tf.keras.layers.Conv2D(output_channels,kernel_size,padding='same')(x)
    x2 = tf.keras.layers.LeakyReLU(alpha)(x2)
    x2 = tf.keras.layers.Conv2D(output_channels,kernel_size,padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU(alpha)(x2)
    x2 = tf.keras.layers.AveragePooling2D(pool_size)(x2)
    
    return tf.keras.layers.Add()([x1,x2])

def up_block(x,output_channels,kernel_size,upsample_factor,x_skip=None,):
    
    if x_skip is not None:
        x=tf.keras.layers.concatenate([x_skip, x], axis=-1)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)
    x=tf.keras.layers.UpSampling2D(size=upsample_factor)(x)
    x=tf.keras.layers.Conv2D(output_channels, kernel_size, padding='same')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)
    x=tf.keras.layers.Conv2D(output_channels, kernel_size, padding='same')(x)
    x=tf.keras.layers.Conv2D(output_channels, kernel_size, padding='same')(x)
    return x
    
    
def create_upsampler(n_inputs=2,start_neurons=32,n_outputs=4):
    """
    Output will be list of size n_layers of decreasing resolution
    
    input[0] : [batch, naz/2 , nrad/2, start_neurons]
    input[1] : [batch, naz/4 , nrad/8, start_neurons*2]
    input[2] : [batch, naz/4 , nrad/16, start_neurons*4]
    input[3] : [batch, naz/4 , nrad/64, start_neurons*8]
    """
    i0=tf.keras.Input(shape=(None,None,n_inputs*start_neurons))
    i1=tf.keras.Input(shape=(None,None,n_inputs*start_neurons*2))
    i2=tf.keras.Input(shape=(None,None,n_inputs*start_neurons*4))
    i3=tf.keras.Input(shape=(None,None,n_inputs*start_neurons*8))
    
    # run lowest conv2D at multiple kernel sizes and concatenate
    x=tf.keras.layers.Conv2D(start_neurons*8,3,padding='same')(i3) 
    x1=tf.keras.layers.Conv2D(start_neurons*8,5,padding='same')(i3) 
    x2=tf.keras.layers.Conv2D(start_neurons*8,7,padding='same')(i3) 
    x=tf.keras.layers.Concatenate()([x,x1,x2])
    # ABLATION skip convolutions at lowest level
    #x=i3

    x=up_block(x,start_neurons*8,(3,3),_POOL[3],x_skip=None) 
    x=up_block(x,start_neurons*4,(3,3),_POOL[2],x_skip=i2)   
    x=up_block(x,start_neurons*2,(3,3),_POOL[1],x_skip=i1)   
    x=up_block(x,start_neurons,(3,3),(1,2),x_skip=i0)   # naz/2 , nrad
    x=tf.keras.layers.Conv2D(n_outputs, (3,3), padding='same')(x)
    return tf.keras.Model(inputs=[i0,i1,i2,i3],outputs=x)
 
