import pandas as pd
import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_vision_transformer import swin_layers
from keras_vision_transformer import transformer_layers
from keras_vision_transformer import utils




@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  #input_image, real_image = resize(input_image, real_image, IMG_HEIGHT*2, IMG_WIDTH*2)

  # Random cropping back to 256x256
  #input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
    
  if tf.random.uniform(()) > 0.5:
     rot_angle = tf.random.uniform(()).numpy()*5
     input_image = tfa.image.rotate(images=input_image, angles=rot_angle, fill_mode='reflect')
     real_image = tfa.image.rotate(images=real_image, angles=rot_angle, fill_mode='reflect')
  return input_image, real_image

@tf.function()
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMG_HEIGHT, IMG_WIDTH])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMG_HEIGHT, IMG_WIDTH])
        image = tf.image.adjust_gamma(image, gamma=1/0.75, gain=0.9)
        image = image / 127.5 - 1
    return image

def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    #image, mask = tf.py_function(func=remove_clutter, inp=[image, mask], Tout=tf.uint8)
    #image, mask = random_jitter(image, mask)
    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        
    if tf.random.uniform(()) > 0.5:
         rot_angle = tf.random.uniform(())*5
         image = tfa.image.rotate(images=image, angles=rot_angle, fill_mode='reflect')
         mask = tfa.image.rotate(images=mask, angles=rot_angle, fill_mode='reflect')
            
    #if one_hot:
    #    mask = tf.one_hot(tf.cast(mask, tf.int32), NUM_CLASSES)
    #    mask = tf.cast(mask, tf.float32)
    #    mask = tf.reshape(mask, (IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES))
    return image, mask
def load_test_data(image_list):
    image = read_image(image_list)
    return image
def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset
def test_data_generator(image_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list))
    dataset = dataset.map(load_test_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

def convolution_block(
            block_input,
            num_filters=256,
            kernel_size=3,
            dilation_rate=1,
            padding="same",
            use_bias=False,
        ):
            x = layers.Conv2D(
                num_filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding="same",
                use_bias=use_bias,
                kernel_initializer=keras.initializers.HeNormal(),
            )(block_input)
            x = layers.BatchNormalization()(x)
            return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output
def DeeplabV3Plus(image_size, num_classes, weights):
    model_input = keras.Input(shape=(image_size[0], image_size[1], 3))
    if weights:
        resnet50 = keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_tensor=model_input
        )
    else:
        resnet50 = keras.applications.ResNet50(
            weights=None, include_top=False, input_tensor=model_input
        )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)
    input_a = layers.UpSampling2D(
        size=(image_size[0] // 4 // x.shape[1], image_size[1] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size[0] // x.shape[1], image_size[1] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def UNet_Generator(image_size, num_classes):
  inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    #downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(num_classes, 
                                         kernel_size=(1, 1),
                                         padding='same',
                                         kernel_initializer=initializer)  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, name=''):
    '''
    Stacked Swin Transformers that share the same token size.
    
    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''
    # Turn-off dropouts
    mlp_drop_rate = 0 # Droupout after each MLP layer
    attn_drop_rate = 0 # Dropout after Swin-Attention
    proj_drop_rate = 0 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    drop_path_rate = 0 # Drop-path within skip-connections
    
    qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
    
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0
    
    for i in range(stack_num):
    
        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = swin_layers.SwinTransformerBlock(dim=embed_dim, 
                                             num_patch=num_patch, 
                                             num_heads=num_heads, 
                                             window_size=window_size, 
                                             shift_size=shift_size_temp, 
                                             num_mlp=num_mlp, 
                                             qkv_bias=qkv_bias, 
                                             qk_scale=qk_scale,
                                             mlp_drop=mlp_drop_rate, 
                                             attn_drop=attn_drop_rate, 
                                             proj_drop=proj_drop_rate, 
                                             drop_path_prob=drop_path_rate, 
                                             name='name{}'.format(i))(X)
    return X
def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, shift_window=True, name='swin_unet'):
    '''
    The base of Swin-UNET.
    
    The general structure:
    
    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head
    
    '''
    # Compute number be patches to be embeded
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]
    
    # Number of Embedded dimensions
    embed_dim = filter_num_begin
    
    depth_ = depth
    
    X_skip = []

    X = input_tensor
    
    # Patch extraction
    X = transformer_layers.patch_extract(patch_size)(X)

    # Embed patches to tokens
    X = transformer_layers.patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)
    
    # The first Swin Transformer stack
    X = swin_transformer_stack(X, 
                               stack_num=stack_num_down, 
                               embed_dim=embed_dim, 
                               num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], 
                               window_size=window_size[0], 
                               num_mlp=num_mlp, 
                               shift_window=shift_window, 
                               name='{}_swin_down0'.format(name))
    X_skip.append(X)
    
    # Downsampling blocks
    for i in range(depth_-1):
        
        # Patch merging
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
        
        # update token shape info
        embed_dim = embed_dim*2
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, 
                                   stack_num=stack_num_down, 
                                   embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i+1], 
                                   window_size=window_size[i+1], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   name='{}_swin_down{}'.format(name, i+1))
        
        # Store tensors for concat
        X_skip.append(X)
        
    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    
    depth_decode = len(X_decode)
    
    for i in range(depth_decode):
        
        # Patch expanding
        X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                               embed_dim=embed_dim, 
                                               upsample_rate=2, 
                                               return_vector=True)(X)
        

        # update token shape info
        embed_dim = embed_dim//2
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2
        
        # Concatenation and linear projection
        X =  tf.keras.layers.concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X =  tf.keras.layers.Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, 
                                   stack_num=stack_num_up, 
                                   embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i], 
                                   window_size=window_size[i], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   name='{}_swin_up{}'.format(name, i))
        
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    
    X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                           embed_dim=embed_dim, 
                                           upsample_rate=patch_size[0], 
                                           return_vector=False)(X)
    
    return X


def SwimT_Generator(image_size, num_classes):
    filter_num_begin = image_size[0]     # number of channels in the first downsampling block; it is also the number of embedded dimensions
    depth = 4                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
    stack_num_down = 2         # number of Swin Transformers per downsampling level
    stack_num_up = 2           # number of Swin Transformers per upsampling level
    patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
    num_heads = [4, 8, 8, 8]   # number of attention heads per down/upsampling level
    window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
    num_mlp = 512              # number of MLP nodes within the Transformer
    shift_window=True          # Apply window shifting, i.e., Swin-MSA
    input_size = (image_size[0] , image_size[1] , 3)
    initializer = tf.random_normal_initializer(0., 0.02)
    IN =  tf.keras.layers.Input(input_size)
    
    # Base architecture
    X = swin_unet_2d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up, 
                          patch_size, num_heads, window_size, num_mlp, 
                          shift_window=shift_window, name='swin_unet')
    # Output section                                                   
                                              
    # (OUTPUT_CHANNELS, kernel_size=1, use_bias=False, activation='softmax')(X)
    OUT = tf.keras.layers.Conv2D(num_classes, 
                                 kernel_size=(1, 1), 
                                 kernel_initializer=initializer, 
                                 use_bias=False, 
                                 padding='same')(X)
    
    # Model configuration
    model = Model(inputs=[IN,], outputs=[OUT,])
    return model

def residual_block(inputs, filters, strides=1):    
    bn1 = tf.keras.layers.BatchNormalization()(inputs)
    bn1 = tf.keras.layers.Activation("relu")(bn1)
    conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=strides)(bn1)
    
    bn2 = tf.keras.layers.BatchNormalization()(conv1)
    bn2 = tf.keras.layers.Activation("relu")(bn2)
    conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', strides=1)(bn2)
    
    s=tf.keras.layers.Conv2D(filters,(1, 1),padding='same',strides=strides)(inputs)
    s= tf.keras.layers.BatchNormalization()(s)
    
    output = tf.keras.layers.Add()([s,conv2])
    return output

def ResUNet_Generator(image_size, num_classes):

    initializer = tf.random_normal_initializer(0., 0.02)

    f = [16, 32, 64, 128, 256]
    inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)) #inputs = Input(input_shape)
    
    ## Encoder
    conv1_1 = tf.keras.layers.Conv2D(16,(3,3),padding='same',strides=1)(inputs)
    conv1_bn = tf.keras.layers.BatchNormalization()(conv1_1)
    conv1_bn = tf.keras.layers.Activation("relu")(conv1_bn)
    conv1_2 = tf.keras.layers.Conv2D(16,(3,3), padding='same', strides=1)(conv1_bn)
    
    s1 = tf.keras.layers.Conv2D(16, kernel_size=(1,1), padding='same', strides=1)(inputs)
    s1 = tf.keras.layers.BatchNormalization()(s1)
    
    encoder1 = tf.keras.layers.Add()([conv1_2,s1])
    encoder2 = residual_block(encoder1, 32, strides=2)
    encoder3= residual_block(encoder2, 64, strides=2)
    encoder4 = residual_block(encoder3, 128, strides=2)
    encoder5 = residual_block(encoder4, 256, strides=2)
    
    ## Bridge
    b0_bn = tf.keras.layers.BatchNormalization()(encoder5)
    b0_bn = tf.keras.layers.Activation("relu")(b0_bn)
    b0_conv = tf.keras.layers.Conv2D(256,(3,3),padding='same',strides=1)(b0_bn)
    
    b1_bn = tf.keras.layers.BatchNormalization()(b0_conv)
    b1_bn = tf.keras.layers.Activation("relu")(b1_bn)
    b1_conv = tf.keras.layers.Conv2D(256,(3,3),padding='same',strides=1)(b1_bn)

    # Decoder
    up1 = tf.keras.layers.UpSampling2D((2, 2))(b1_conv)
    c1 = tf.keras.layers.Concatenate()([up1, encoder4])
    d1 = residual_block(c1, 256)
    
    up2 = tf.keras.layers.UpSampling2D((2, 2))(d1)
    c2 = tf.keras.layers.Concatenate()([up2, encoder3])
    d2 = residual_block(c2, 128)
    
    up3 = tf.keras.layers.UpSampling2D((2, 2))(d2)
    c3 = tf.keras.layers.Concatenate()([up3, encoder2])
    d3 = residual_block(c3, 64)

    up4 = tf.keras.layers.UpSampling2D((2, 2))(d3)
    c4 = tf.keras.layers.Concatenate()([up4, encoder1])
    d4 = residual_block(c4, 32)
    
    #outputs = Conv2D(OUTPUT_CHANNELS, strides=2, padding="same", kernel_initializer=initializer, activation="tanh")(d4)
    #Transpose
    outputs = tf.keras.layers.Conv2D(num_classes, (1,1),padding='same',kernel_initializer=initializer)(d4)
    model = Model(inputs, outputs)
    return model#tf.keras.Model(inputs=inputs, outputs=x)

def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)

def attention_block(x,g,inter_shape):      
    #input shapes
    x_shape = K.int_shape(x)
    g_shape = K.int_shape(g)

    # x vector input and processing
    theta_x = tf.keras.layers.Conv2D(inter_shape,(2,2), strides=2, padding='same', kernel_initializer='he_normal', activation=None)(x)
    shape_theta_x = K.int_shape(theta_x)

    # gating signal ""
    phi_g = tf.keras.layers.Conv2D(inter_shape, kernel_size = 1, strides = 1, padding='same', kernel_initializer='he_normal', activation=None)(g)
    shape_phi_g = K.int_shape(phi_g)
    upsample_phi_g = tf.keras.layers.Conv2DTranspose(inter_shape,kernel_size=(3, 3),strides=(shape_theta_x[1] // g_shape[1], shape_theta_x[2] // g_shape[2]),padding='same')(phi_g)

    # Add components
    concat_xg = tf.keras.layers.add([upsample_phi_g, theta_x])
    act_xg = tf.keras.layers.Activation('relu')(concat_xg)

    # Apply convolution
    psi = tf.keras.layers.Conv2D(1, kernel_size = 1, strides = 1, padding='same', kernel_initializer='he_normal', activation=None)(act_xg)

    # Apply sigmoid activation
    sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)

    # UpSample and resample to correct size
    upsample_psi = tf.keras.layers.UpSampling2D(interpolation='bilinear', size=(x_shape[1] // shape_sigmoid[1], x_shape[2] // shape_sigmoid[2]))(sigmoid_xg)
    upsample_psi = expend_as(upsample_psi,x_shape[3])
    
    y = multiply([upsample_psi, x])


    res= tf.keras.layers.Conv2D(x_shape[3],(1, 1),padding='same')(y)
    res_norm = tf.keras.layers.BatchNormalization()(res)
    return res_norm
          
def Attention_UNet_Generator(image_size, num_classes):
    dropout_rate=0.1     
    initializer = tf.random_normal_initializer(0., 0.02)

    f = [16, 32, 64, 128, 256]
    inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)) #inputs = Input(input_shape)
          
    #Contraction path
    conv1 =  tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    conv1 =  tf.keras.layers.Dropout(dropout_rate)(conv1)
    conv1 =  tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 =  tf.keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 =  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 =  tf.keras.layers.Dropout(dropout_rate)(conv2)
    conv2 =  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 =  tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 =  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 =  tf.keras.layers.Dropout(dropout_rate)(conv3)
    conv3 =  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 =  tf.keras.layers.MaxPooling2D((2, 2))(conv3)

    conv4 =  tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 =  tf.keras.layers.Dropout(dropout_rate)(conv4)
    conv4 =  tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    pool4 =  tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 =  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 =  tf.keras.layers.Dropout(dropout_rate)(conv5)
    conv5 =  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)

    #Expansive path 
    u6 =  tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    #Attention
    merge6 = attention_block(conv4, conv5, 128) 
    u6 = concatenate([u6, merge6])
    conv6 =  tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    conv6 =  tf.keras.layers.Dropout(dropout_rate)(conv6)
    conv6 =  tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

    u7 =  tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = attention_block(conv3, conv6, 64) 
    u7 = concatenate([u7, merge7])
    conv7 =  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    conv7 =  tf.keras.layers.Dropout(dropout_rate)(conv7)
    conv7 =  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)

    u8 =  tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = attention_block(conv2, conv7, 32) 
    u8 = concatenate([u8, merge8])
    conv8 =  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    conv8 =  tf.keras.layers.Dropout(dropout_rate)(conv8)
    conv8 =  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)

    u9 =  tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = attention_block(conv1, conv8, 16) 
    u9 = concatenate([u9, merge9], axis=3)
    conv9 =  tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    conv9 =  tf.keras.layers.Dropout(dropout_rate)(conv9)
    conv9 =  tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)

    #outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    outputs = tf.keras.layers.Conv2D(num_classes,kernel_size=(1, 1),padding='same',kernel_initializer=initializer)(conv9) #tanh activation='tanh'

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model      
          
def FCN_8_Generator(image_size, num_classes):
    dropout_rate=0.1     
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3)) #inputs = Input(input_shape)
    IMAGE_ORDERING = 'channels_last'
    #ENCODER
    #BLOCK 1 
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation='relu', name='block1_conv1', data_format=IMAGE_ORDERING)(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation='relu', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    f1=x
    #BLOCK 2
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    f2=x
    #BLOCK 3
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu', name='block3_conv1', data_format=IMAGE_ORDERING)(x)    
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu', name='block3_conv2', data_format=IMAGE_ORDERING)(x) 
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu', name='block3_conv3', data_format=IMAGE_ORDERING)(x) 
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    pool3=x
    #BLOCK 4
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)
    pool4=x
    #BLOCK 5
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu', name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)
    pool5=x
    #DECODER
    conv6 = (tf.keras.layers.Conv2D(4096, (7, 7), padding = 'same',activation='relu', kernel_initializer = 'he_normal', name = "conv6"))(pool5)
    conv6 = tf.keras.layers.Dropout(dropout_rate)(conv6)
    conv7 = (tf.keras.layers.Conv2D(4096, (1, 1), padding = 'same',activation='relu', kernel_initializer = 'he_normal', name = "conv7"))(conv6)
    conv6 = tf.keras.layers.Dropout(dropout_rate)(conv6)
    pool4_n = tf.keras.layers.Conv2D(OUTPUT_CHANNELS,(1,1), padding='same', activation='relu')(pool4)
    up_2 = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, kernel_size=(2,2), strides=(2,2), padding='same')(conv7)
    up_2_skip = tf.keras.layers.Add()([pool4_n,up_2])
    pool3_n = tf.keras.layers.Conv2D(OUTPUT_CHANNELS,(1,1), padding='same', activation='relu')(pool3)
    up_4 = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, kernel_size=(2,2), strides=(2,2), padding='same')(up_2_skip)
    up_4_skip = tf.keras.layers.Add()([pool3_n,up_4])
    #output = Conv2DTranspose(num_classes, kernel_size=(8,8), strides=(8,8), padding='same', activation='softmax')(up_4_skip)
    outputs = tf.keras.layers.Conv2DTranspose(num_classes,kernel_size=(1,1),padding='same',kernel_initializer=initializer)(up_4_skip)

    model = tf.keras.Model(inputs, outputs)
    model.model_name = "fcn_8"
    return model      

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay

def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()

def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, NUM_CLASSES)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], figsize=(18, 14)
        )

def plot_test_results(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        tf.keras.utils.save_img(f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/prediction_mask/{image_file.split("/")[-1].split(".")[0]}_prediction_mask.png', np.expand_dims(prediction_mask, 2), data_format=None, file_format=None, scale=True)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, NUM_CLASSES)
        tf.keras.utils.save_img(f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/prediction_colormap/{image_file.split("/")[-1].split(".")[0]}_prediction_colormap.png', prediction_colormap, data_format=None, file_format=None, scale=True)
        overlay = get_overlay(image_tensor, prediction_colormap)
        tf.keras.utils.save_img(f'../mammoSEG/{model_type}_3class_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/prediction_overlay/{image_file.split("/")[-1].split(".")[0]}_prediction_overlay.png', overlay, data_format=None, file_format=None, scale=True)

def plot_test_end_epoch(images_list, test_masks, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file)
        test_masks = read_image(test_masks)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        #tf.keras.utils.save_img(f'../mammoSEG/{model_type}_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/prediction_mask/{image_file.split("/")[-1].split(".")[0]}_prediction_mask.png', np.expand_dims(prediction_mask, 2), data_format=None, file_format=None, scale=True)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, NUM_CLASSES)
        #tf.keras.utils.save_img(f'../mammoSEG/{model_type}_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/prediction_colormap/{image_file.split("/")[-1].split(".")[0]}_prediction_colormap.png', prediction_colormap, data_format=None, file_format=None, scale=True)
        #overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib([image_tensor, test_masks, prediction_colormap], figsize=(18, 14))
        #tf.keras.utils.save_img(f'../mammoSEG/{model_type}_segmentation_model/logs{IMG_HEIGHT}x{IMG_WIDTH}/test_results/prediction_overlay/{image_file.split("/")[-1].split(".")[0]}_prediction_overlay.png', overlay, data_format=None, file_format=None, scale=True)

class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%10 == 0:
            print(f'Epoch: {epoch}')
            plot_test_end_epoch(test_images[:1], test_masks[0], colormap, model)

def add_sample_weights(image, label):
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    class_weights = tf.constant([1.0, 10.0, 3.0, 2.0])
    class_weights = class_weights/tf.reduce_sum(class_weights)
    
    # Create an image of `sample_weights` by using the label at each pixel as an 
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    
    return image, label, sample_weights
      