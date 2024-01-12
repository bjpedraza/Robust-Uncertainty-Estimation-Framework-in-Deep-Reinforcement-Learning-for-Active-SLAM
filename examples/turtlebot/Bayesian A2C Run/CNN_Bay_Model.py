import tensorflow as tf
from tensorflow import keras
import os
import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
import wandb
import xlsxwriter
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline
#os.environ["WANDB_API_KEY"] = " "
#plt.ioff()
cifar10 = tf.keras.datasets.cifar10
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def x_Sigma_w_x_T(x, W_Sigma):
    batch_sz = x.shape[0]
    xx_t = tf.reduce_sum(tf.multiply(x, x), axis=1, keepdims=True)
    xx_t_e = tf.expand_dims(xx_t, axis=2)
    return tf.multiply(xx_t_e, W_Sigma)

def w_t_Sigma_i_w(w_mu, in_Sigma):
    Sigma_1_1 = tf.matmul(tf.transpose(w_mu), in_Sigma)
    Sigma_1_2 = tf.matmul(Sigma_1_1, w_mu)
    return Sigma_1_2

def tr_Sigma_w_Sigma_in(in_Sigma, W_Sigma):
    Sigma_3_1 = tf.linalg.trace(in_Sigma)
    Sigma_3_2 = tf.expand_dims(Sigma_3_1, axis=1)
    Sigma_3_3 = tf.expand_dims(Sigma_3_2, axis=1)
    return tf.multiply(Sigma_3_3, W_Sigma)

def activation_Sigma(gradi, Sigma_in):
    grad1 = tf.expand_dims(gradi, axis=2)
    grad2 = tf.expand_dims(gradi, axis=1)
    return tf.multiply(Sigma_in, tf.matmul(grad1, grad2))

def activation_function_Sigma(gradi, Sigma_in):
    batch_size = gradi.shape[0]
    '''
    if(gradi.shape[0] == None):
        batch_size = 0
    else:
        batch_size = gradi.shape[0]
    '''
    channels = gradi.shape[-1]
    gradient_matrix = tf.reshape(gradi, [batch_size, -1,   channels])  # shape =[batch_size, image_size*image_size, channels]
    grad1 = tf.expand_dims(tf.transpose(gradient_matrix, [0, 2, 1]),  3)  # shape =[batch_size, channels, image_size*image_size, 1]
    grad_square = tf.matmul(grad1, tf.transpose(grad1, [0, 1, 3,2]))  # shape =[batch_size, channels, image_size*image_size, image_size*image_size]
    grad_square = tf.transpose(grad_square, [0, 2, 3,  1])  # shape =[ batch_size, image_size*image_size, image_size*image_size, channels]
    sigma_out = tf.multiply(Sigma_in, grad_square)
    return sigma_out

def Hadamard_sigma(sigma1, sigma2, mu1, mu2):
    sigma_1 = tf.multiply(sigma1, sigma2)
    sigma_2 = tf.matmul(tf.matmul(tf.linalg.diag(mu1), sigma2), tf.linalg.diag(mu1))
    sigma_3 = tf.matmul(tf.matmul(tf.linalg.diag(mu2), sigma1), tf.linalg.diag(mu2))
    return sigma_1 + sigma_2 + sigma_3

def grad_sigmoid(mu_in):
    with tf.GradientTape() as g:
        g.watch(mu_in)
        out = tf.sigmoid(mu_in)
    gradi = g.gradient(out, mu_in)
    return gradi

def grad_tanh(mu_in):
    with tf.GradientTape() as g:
        g.watch(mu_in)
        out = tf.tanh(mu_in)
    gradi = g.gradient(out, mu_in)
    return gradi

def mu_muT(mu1, mu2):
    mu11 = tf.expand_dims(mu1, axis=2)
    mu22 = tf.expand_dims(mu2, axis=1)
    return tf.matmul(mu11, mu22)

def sigma_regularizer(x):
    alpha = 1.
    f_s = tf.math.softplus(x)  # tf.math.log(1. + tf.math.exp(x))
    return - alpha*tf.reduce_mean(1. + tf.math.log(f_s) - f_s, axis=-1)

class VDP_first_Conv(keras.layers.Layer):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding="VALID"):
        super(VDP_first_Conv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding

    def build(self, input_shape):
        ini_sigma = -6.9
        # min_sigma = -4.5
        tau = 1. #/ (self.kernel_size * self.kernel_size * input_shape[-1])
        self.w_mu = self.add_weight(name='w_mu',
                                    shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.kernel_num),
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
                                    regularizer=tf.keras.regularizers.l2(tau),  #l1_l2(l1=tau, l2=tau)
                                    trainable=True,
                                    )
        self.w_sigma = self.add_weight(name='w_sigma',
                                       shape=(self.kernel_num,),
                                       initializer=tf.constant_initializer(ini_sigma),  # tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,  seed=None),
                                       regularizer=sigma_regularizer,  
                                       trainable=True,
                                       )

    def call(self, mu_in):
        batch_size = mu_in.shape[0]
        '''
        if(mu_in.shape[0] == None):
            batch_size = 0
        else:
            batch_size = mu_in.shape[0]
        '''

        num_channel = mu_in.shape[-1]
        mu_out = tf.nn.conv2d(mu_in, self.w_mu, strides=[1, self.kernel_stride, self.kernel_stride, 1],
                              padding=self.padding, data_format='NHWC')
        x_train_patches = tf.image.extract_patches(mu_in, sizes=[1, self.kernel_size, self.kernel_size, 1],
                                                   strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                                   rates=[1, 1, 1, 1],
                                                   padding=self.padding)  # shape=[batch_size, image_size, image_size, kernel_size*kernel_size*num_channel]
        print(x_train_patches)
        x_train_matrix = tf.reshape(x_train_patches, [batch_size, -1,
                                                      self.kernel_size * self.kernel_size * num_channel])  # shape=[batch_size, image_size*image_size, patch_size*patch_size*num_channel]
        print(x_train_matrix)
        X_XTranspose = tf.matmul(x_train_matrix, tf.transpose(x_train_matrix, [0, 2, 1]))  # shape=[batch_size,image_size*image_size, image_size*image_size ] 
        X_XTranspose = tf.ones([1, 1, 1, self.kernel_num]) * tf.expand_dims(X_XTranspose, axis=-1)
        Sigma_out = tf.multiply(tf.math.log(1. + tf.math.exp(self.w_sigma)),  X_XTranspose)  # shape=[batch_size,image_size*image_size, image_size*image_size, kernel_num]  
#        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))
        return mu_out, Sigma_out

class VDP_intermediate_Conv(keras.layers.Layer):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding="VALID"):
        super(VDP_intermediate_Conv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding

    def build(self, input_shape):
        ini_sigma = -6.9
        # min_sigma = -4.5
        tau = 1. #/ (self.kernel_size * self.kernel_size * input_shape[-1])
        self.w_mu = self.add_weight(name='w_mu',
                                    shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.kernel_num),
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
                                    regularizer=tf.keras.regularizers.l2(tau),#l1_l2(l1=tau, l2=tau)
                                    trainable=True,
                                    )
        self.w_sigma = self.add_weight(name='w_sigma',
                                       shape=(self.kernel_num,),
                                       initializer=tf.constant_initializer(ini_sigma),  # tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,  seed=None),
                                       regularizer=sigma_regularizer,  
                                       trainable=True,
                                       )

    def call(self, mu_in, Sigma_in):
        batch_size = mu_in.shape[0]
        '''
        if(mu_in.shape[0] == None):
            batch_size = 0
        else:
            batch_size = mu_in.shape[0]
        '''

        num_channel = mu_in.shape[-1]  # shape=[batch_size, im_size, im_size, num_channel]
        mu_out = tf.nn.conv2d(mu_in, self.w_mu, strides=[1, self.kernel_stride, self.kernel_stride, 1],
                              padding=self.padding, data_format='NHWC')
        Sigma_in1 = tf.transpose(Sigma_in, [0, 3, 1, 2])
        diag_sigma = tf.linalg.diag_part(Sigma_in1)  # shape=[batch_size, num_channel,im_size*im_size]
        diag_sigma = tf.transpose(diag_sigma, [0, 2, 1])  # shape=[batch_size, im_size*im_size,num_channel]
        diag_sigma = tf.reshape(diag_sigma, [batch_size, mu_in.shape[1], mu_in.shape[2],
                                             num_channel])  # shape=[batch_size, im_size,im_size,num_channel]
        diag_sigma_patches = tf.image.extract_patches(diag_sigma, sizes=[1, self.kernel_size, self.kernel_size, 1],
                                                      strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                                      rates=[1, 1, 1, 1], padding=self.padding)
        # shape=[batch_size, new_im_size, new_im_size, kernel_size*kernel_size*num_channel]
        diag_sigma_g = tf.reshape(diag_sigma_patches,
                                  [batch_size, -1, self.kernel_size * self.kernel_size * num_channel])
        # shape=[batch_size, new_im_size*new_im_size,   self.kernel_size*self.kernel_size*num_channel ]
        mu_cov_square = tf.reshape(tf.multiply(self.w_mu, self.w_mu),
                                   [self.kernel_size * self.kernel_size * num_channel, self.kernel_num])
        # shape[ kernel_size*kernel_size*num_channel,   kernel_num]
        mu_wT_sigmags_mu_w1 = tf.matmul(diag_sigma_g, mu_cov_square)  # shape=[batch_size, new_im_size*new_im_size , kernel_num   ]
        mu_wT_sigmags_mu_w = tf.linalg.diag(tf.transpose(mu_wT_sigmags_mu_w1, [0, 2, 1]))  # shape=[batch_size, kernel_num, new_im_size*new_im_size, new_im_size*new_im_size]
        trace = tf.math.reduce_sum(diag_sigma_g, 2, keepdims=True)  # shape=[batch_size,  new_im_size* new_im_size, 1]
        trace = tf.ones([1, 1, self.kernel_num]) * trace  # shape=[batch_size,  new_im_size*new_im_size, kernel_num]
        trace = tf.transpose(tf.multiply(tf.math.log(1. + tf.math.exp(self.w_sigma)), trace),  [0, 2, 1])  # shape=[batch_size, kernel_num, new_im_size*new_im_size]
        trace1 = tf.linalg.diag( trace)  # shape=[batch_size, kernel_num, new_im_size*new_im_size, new_im_size*new_im_size]
        mu_in_patches = tf.reshape(tf.image.extract_patches(mu_in, sizes=[1, self.kernel_size, self.kernel_size, 1],
                                                            strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                                            rates=[1, 1, 1, 1], padding=self.padding),
                                   [batch_size, -1, self.kernel_size * self.kernel_size * num_channel])
        # shape=[batch_size, new_im_size*new_im_size, self.kernel_size*self.kernel_size*num_channel]
        mu_gT_mu_g = tf.matmul(mu_in_patches, tf.transpose(mu_in_patches, [0, 2,  1]))  # shape=[batch_size, new_im_size*new_im_size,new_im_size*new_im_size]
        mu_gT_mu_g1 = tf.ones([1, 1, 1, self.kernel_num]) * tf.expand_dims(mu_gT_mu_g, axis=-1)
        # shape=[batch_size, new_im_size*new_im_size, new_im_size*new_im_size, kernel_num]
        sigmaw_mu_gT_mu_g = tf.transpose(tf.multiply(tf.math.log(1. + tf.math.exp(self.w_sigma)), mu_gT_mu_g1),  [0, 3, 1, 2])
        # shape=[batch_size, kernel_num, new_im_size*new_im_size, new_im_size*new_im_size]
        Sigma_out = trace1 + mu_wT_sigmags_mu_w + sigmaw_mu_gT_mu_g  # shape=[batch_size, kernel_num, new_im_size*new_im_size, new_im_size*new_im_size]
        Sigma_out = tf.transpose(Sigma_out, [0, 2, 3, 1])
        return mu_out, Sigma_out

class VDP_MaxPooling(keras.layers.Layer):
    """VDP_MaxPooling"""
    def __init__(self, pooling_size=2, pooling_stride=2, pooling_pad='SAME'):
        super(VDP_MaxPooling, self).__init__()
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad

    def call(self, mu_in, Sigma_in):
        batch_size = mu_in.shape[0]
        '''
        if(mu_in.shape[0] == None):
            batch_size = 0
        else:
            batch_size = mu_in.shape[0]
        '''

        hw_in = mu_in.shape[1]
        num_channel = mu_in.shape[-1]
        mu_out, argmax_out = tf.nn.max_pool_with_argmax(mu_in, ksize=[1, self.pooling_size, self.pooling_size, 1],
                                                        strides=[1, self.pooling_stride, self.pooling_stride, 1],
                                                        padding=self.pooling_pad)  # shape=[batch_zise, new_size,new_size,num_channel]
        hw_out = mu_out.shape[1]
        argmax1 = tf.transpose(argmax_out, [0, 3, 1, 2])
        argmax2 = tf.reshape(argmax1, [batch_size, num_channel,
                                       -1])  # shape=[batch_size, num_channel, new_size*new_size]
        x_index = tf.math.floormod(tf.compat.v1.floor_div(argmax2, tf.constant(num_channel,
                                                                               shape=[batch_size, num_channel,
                                                                                      hw_out * hw_out], dtype='int64')),
                                   tf.constant(hw_in, shape=[batch_size, num_channel, hw_out * hw_out], dtype='int64'))
        aux = tf.compat.v1.floor_div(tf.compat.v1.floor_div(argmax2, tf.constant(num_channel,
                                                                                 shape=[batch_size, num_channel,
                                                                                        hw_out * hw_out],
                                                                                 dtype='int64')),
                                     tf.constant(hw_in, shape=[batch_size, num_channel, hw_out * hw_out],
                                                 dtype='int64'))
        y_index = tf.math.floormod(aux,  tf.constant(hw_in, shape=[batch_size, num_channel, hw_out * hw_out], dtype='int64'))
        index = tf.multiply(y_index, hw_in) + x_index  # shape=[batch_size, num_channel,new_size*new_size]        
        Sigma_in1 = tf.transpose(Sigma_in,     [0, 3, 1, 2])  # shape=[batch_size,num_channel,im_size*im_size, im_size*im_size]
        gath1 = tf.gather(Sigma_in1, index, batch_dims=2, axis=2)
        Sigma_out = tf.gather(gath1, index, batch_dims=2,     axis=-1)  # shape=[batch_size,num_channel,new_size*new_size, new_size*new_size]
        Sigma_out = tf.transpose(Sigma_out,    [0, 2, 3, 1])  # shape=[batch_size,new_size*new_size, new_size*new_size, num_channel]
        return mu_out, Sigma_out


class VDP_Flatten_and_FC(keras.layers.Layer):   
    def __init__(self, units):
        super(VDP_Flatten_and_FC, self).__init__()
        self.units = units                
    def build(self, input_shape):
        ini_sigma = -6.9
        #min_sigma = -4.5
        tau = 1.#/(input_shape[1]*input_shape[2]*input_shape[-1] )      
        self.w_mu = self.add_weight(name = 'w_mu', shape=(input_shape[1]*input_shape[2]*input_shape[-1], self.units),
            initializer=tf.random_normal_initializer( mean=0.0, stddev=0.05, seed=None), regularizer=tf.keras.regularizers.l2(tau),#l1_l2(l1=tau, l2=tau), 
            trainable=True,
        )
        self.w_sigma = self.add_weight(name = 'w_sigma',
            shape=(self.units,),
            initializer= tf.constant_initializer(ini_sigma) , regularizer=sigma_regularizer, 
            trainable=True, #tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None) 
        )    
    def call(self, mu_in, Sigma_in): 
        #batch_size = 64 #mu_in.shape[0] #shape=[batch_size, im_size, im_size, num_channel]
        batch_size = mu_in.shape[0]

        hw_in = mu_in.shape[1]
        num_channel = mu_in.shape[-1]   
        mu_flatt = tf.reshape(mu_in, [batch_size, -1]) #shape=[batch_size, im_size*im_size*num_channel]           
        mu_out = tf.matmul(mu_flatt, self.w_mu)

        fc_weight_mu1 = tf.reshape(self.w_mu, [num_channel, hw_in*hw_in ,self.units]) #shape=[num_channel, new_size*new_size, units]
        fc_weight_mu1T = tf.transpose(fc_weight_mu1,[0,2,1]) #shape=[num_channel,units,new_size*new_size]
        sigma_in1 = tf.transpose(Sigma_in, [0, 3, 1, 2]) #shape=[batch_size, num_channel, new_size*new_size, new_size*new_size]
        Sigma_1 = tf.matmul(tf.matmul(fc_weight_mu1T, sigma_in1), fc_weight_mu1 )#shape=[batch_size, num_channel, units, units]
        Sigma_1 = tf.math.reduce_sum(Sigma_1, axis=1) #shape=[batch_size, units, units]
        diag_elements = tf.linalg.trace(sigma_in1) #shape=[batch_size, num_channel]     
        tr_sigma_b =tf.math.reduce_sum(diag_elements,axis=1, keepdims=True) #shape=[batch_size, 1]
        tr_sigma_h_sigma_b = tf.multiply(tf.math.log(1. + tf.math.exp(self.w_sigma)), tr_sigma_b ) # shape=[batch_size, units] 
        Sigma_2 = tf.linalg.diag(tr_sigma_h_sigma_b)# shape=[batch_size, units, units]
        mu_bT_mu_b = tf.math.reduce_sum(tf.multiply(mu_flatt, mu_flatt),axis=1, keepdims=True)  #shape=[batch_size, 1]
        mu_bT_sigma_h_mu_b = tf.multiply(tf.math.log(1. + tf.math.exp(self.w_sigma)), mu_bT_mu_b) # shape=[batch_size, units] 
        Sigma_3 = tf.linalg.diag(mu_bT_sigma_h_mu_b)# shape=[batch_size, units, units]     
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3

        return mu_out, Sigma_out


class VDP_FC(keras.layers.Layer):
    """y = w.x + b"""
    def __init__(self, units):
        super(VDP_FC, self).__init__()
        self.units = units                
    def build(self, input_shape):
        ini_sigma = -4.6       
        tau = -1. #/input_shape[-1]         
        self.w_mu = self.add_weight(name = 'w_mu', shape=(input_shape[-1], self.units),
            initializer=tf.random_normal_initializer( mean=0.0, stddev=0.05, seed=None), regularizer=tf.keras.regularizers.l1_l2(l1=tau, l2=tau),#tau/self.units), #tf.keras.regularizers.l2(0.5*0.001),
            trainable=True,
        )
        self.w_sigma = self.add_weight(name = 'w_sigma',
            shape=(self.units,),
            initializer= tf.constant_initializer(ini_sigma),#tf.random_uniform_initializer(minval= min_sigma, maxval=ini_sigma, seed=None) , 
            regularizer=sigma_regularizer, 
            trainable=True,
        )   
    def call(self, mu_in, Sigma_in):
     
        mu_out = tf.matmul(mu_in, self.w_mu) #+ self.b_mu

        W_Sigma = tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.w_sigma)))       
        Sigma_1 = w_t_Sigma_i_w (self.w_mu, Sigma_in)
        Sigma_2 = x_Sigma_w_x_T(mu_in, W_Sigma)                                   
        Sigma_3 = tr_Sigma_w_Sigma_in (Sigma_in, W_Sigma)
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3 #+ tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.b_sigma)))  
        
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)  
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))

        return mu_out, Sigma_out 


class VDP_Flatten(keras.layers.Layer): #Custom
    def __init__(self, units):
        super(VDP_Flatten, self).__init__()
        self.units = units            
    def build(self, input_shape):
        ini_sigma = -6.9
        #min_sigma = -4.5     
   
    def call(self, mu_in, Sigma_in): 
        #batch_size = mu_in.shape[0] #shape=[batch_size, im_size, im_size, num_channel]
        batch_size = mu_in.shape[0]
        '''
        if(mu_in.shape[0] == None):
            batch_size = 0
        else:
            batch_size = mu_in.shape[0]
        '''
    
        mu_flatt = tf.reshape(mu_in, [batch_size, -1]) #shape=[batch_size, im_size*im_size*num_channel]           
        sigma_in1 = tf.transpose(Sigma_in, [0, 3, 1, 2]) #shape=[batch_size, num_channel, new_size*new_size, new_size*new_size]
        diag_elements = tf.linalg.trace(sigma_in1)

        return mu_flatt, diag_elements 


class mysoftmax(keras.layers.Layer):
    def __init__(self):
        super(mysoftmax, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.softmax(mu_in)
        pp1 = tf.expand_dims(mu_out, axis=2)
        pp2 = tf.expand_dims(mu_out, axis=1)
        ppT = tf.matmul(pp1, pp2)
        p_diag = tf.linalg.diag(mu_out)
        grad = p_diag - ppT
        Sigma_out = tf.matmul(grad, tf.matmul(Sigma_in, tf.transpose(grad, perm=[0, 2, 1])))
#        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
#        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.linalg.set_diag(Sigma_out, tf.abs(tf.linalg.diag_part(Sigma_out)))
        return mu_out, Sigma_out

class VDPReLU(keras.layers.Layer):
    def __init__(self):
        super(VDPReLU, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.relu(mu_in)
        with tf.GradientTape() as g:
            g.watch(mu_in)
            out = tf.nn.relu(mu_in) 
        gradi = g.gradient(out, mu_in)
        Sigma_out = activation_function_Sigma(gradi, Sigma_in)
        return mu_out, Sigma_out

class VDPELU(keras.layers.Layer):
    def __init__(self):
        super(VDPELU, self).__init__()
    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.elu(mu_in)
        with tf.GradientTape() as g:
            g.watch(mu_in)
            out = tf.nn.elu(mu_in)
        gradi = g.gradient(out, mu_in)
        Sigma_out = activation_function_Sigma(gradi, Sigma_in)
        return mu_out, Sigma_out

class VDPDropout(keras.layers.Layer):
    def __init__(self, drop_prop):
        super(VDPDropout, self).__init__()
        self.drop_prop = drop_prop

    def call(self, mu_in, Sigma_in, Training=True):
        '''
        batch_size = 0#mu_in.shape[0]  # shape=[batch_size, im_size, im_size, num_channel]
        '''	
        batch_size = mu_in.shape[0]
        '''
        if(mu_in.shape[0] == None):
            batch_size = 0
        else:
            batch_size = mu_in.shape[0]
        '''
        

        hw_in = mu_in.shape[1]
        num_channel = mu_in.shape[-1]
        scale_sigma = 1.0 / (1 - self.drop_prop)
        
        if Training:        
           mu_out = tf.nn.dropout(mu_in, rate=self.drop_prop)
           mu_out1 = tf.reshape(tf.transpose(mu_out, [0, 3, 1, 2]), [batch_size, num_channel, -1])
   
           non_zero = tf.not_equal(mu_out1, tf.zeros_like(mu_out1))
           non_zero_sigma1 = tf.tile(tf.expand_dims(non_zero, -1), [1, 1, 1, hw_in * hw_in])
           non_zero_sigma2 = tf.tile(tf.expand_dims(non_zero, 2), [1, 1, hw_in * hw_in, 1])
           non_zero_sigma = tf.math.logical_and(non_zero_sigma1, non_zero_sigma2)
           Sigma_in1 = tf.transpose(Sigma_in, [0, 3, 1, 2])
           non_zero_sigma_mask = tf.boolean_mask(Sigma_in1, non_zero_sigma)
           idx_sigma = tf.dtypes.cast(tf.where(non_zero_sigma), tf.int32)
           Sigma_out = (scale_sigma ** 2) * tf.scatter_nd(idx_sigma, non_zero_sigma_mask, tf.shape(non_zero_sigma))
           Sigma_out = tf.transpose(Sigma_out, [0, 2, 3, 1])
        else:
           mu_out = mu_in
           Sigma_out = Sigma_in         
        return mu_out, Sigma_out

class VDPBatch_Normalization(keras.layers.Layer):
    def __init__(self, var_epsilon):
        super(VDPBatch_Normalization, self).__init__()
        self.var_epsilon = var_epsilon

    def call(self, mu_in, Sigma_in):
        mean, variance = tf.nn.moments(mu_in, [0, 1, 2])
        mu_out = tf.nn.batch_normalization(mu_in, mean, variance, offset=None, scale=None,
                                           variance_epsilon=self.var_epsilon)
        Sigma_out = tf.multiply(Sigma_in, 1 / (variance + self.var_epsilon))
        return mu_out, Sigma_out        
        
class Density_prop_CNN(tf.keras.Model):
    def __init__(self, kernel_size, num_kernel, pooling_size, pooling_stride, pooling_pad, units, drop_prop=0.2,
                  var_epsilon=1e-4, name=None):#
        super(Density_prop_CNN, self).__init__()
        self.kernel_size = kernel_size
        self.num_kernel = num_kernel
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad
        self.units = units
        self.drop_prop = drop_prop
        self.var_epsilon = var_epsilon

        self.conv_1 = VDP_first_Conv(kernel_size=self.kernel_size[0], kernel_num=self.num_kernel[0], padding='VALID')
        self.conv_2 = VDP_intermediate_Conv(kernel_size=self.kernel_size[1], kernel_num=self.num_kernel[1],   padding='SAME')
        self.conv_3 = VDP_intermediate_Conv(kernel_size=self.kernel_size[2], kernel_num=self.num_kernel[2],   padding='SAME')
        self.conv_4 = VDP_intermediate_Conv(kernel_size=self.kernel_size[3], kernel_num=self.num_kernel[3],   padding='SAME')
        self.conv_5 = VDP_intermediate_Conv(kernel_size=self.kernel_size[4], kernel_num=self.num_kernel[4],   padding='SAME')
        self.conv_6 = VDP_intermediate_Conv(kernel_size=self.kernel_size[5], kernel_num=self.num_kernel[5],  padding='SAME')
        self.conv_7 = VDP_intermediate_Conv(kernel_size=self.kernel_size[6], kernel_num=self.num_kernel[6],   padding='SAME')
        self.conv_8 = VDP_intermediate_Conv(kernel_size=self.kernel_size[7], kernel_num=self.num_kernel[7],   padding='SAME')
        self.conv_9 = VDP_intermediate_Conv(kernel_size=self.kernel_size[8], kernel_num=self.num_kernel[8],   padding='SAME')
        self.conv_10 = VDP_intermediate_Conv(kernel_size=self.kernel_size[9], kernel_num=self.num_kernel[9],   padding='SAME')

        self.elu_1 = VDPELU()
        self.maxpooling_1 = VDP_MaxPooling(pooling_size=self.pooling_size[0], pooling_stride=self.pooling_stride[0],   pooling_pad=self.pooling_pad)
        self.maxpooling_2 = VDP_MaxPooling(pooling_size=self.pooling_size[1], pooling_stride=self.pooling_stride[1],   pooling_pad=self.pooling_pad)
        self.maxpooling_3 = VDP_MaxPooling(pooling_size=self.pooling_size[2], pooling_stride=self.pooling_stride[2],   pooling_pad=self.pooling_pad)
        self.maxpooling_4 = VDP_MaxPooling(pooling_size=self.pooling_size[3], pooling_stride=self.pooling_stride[3],   pooling_pad=self.pooling_pad)
        self.maxpooling_5 = VDP_MaxPooling(pooling_size=self.pooling_size[4], pooling_stride=self.pooling_stride[4],   pooling_pad=self.pooling_pad)

        self.dropout_1 = VDPDropout(self.drop_prop)        
        self.batch_norm = VDPBatch_Normalization(self.var_epsilon)
        self.fc_1 = VDP_Flatten_and_FC(self.units)
        self.mysoftma = mysoftmax()

    def call(self, inputs, training=True):
        mu, sigma = self.conv_1(inputs)       
        mu, sigma = self.elu_1(mu, sigma)
        mu, sigma = self.batch_norm(mu, sigma)
        mu, sigma = self.maxpooling_1(mu, sigma) 
      
        mu, sigma = self.conv_2(mu, sigma)        
        mu, sigma = self.elu_1(mu, sigma)
        mu, sigma = self.batch_norm(mu, sigma)
        mu, sigma = self.conv_3(mu, sigma)        
        mu, sigma = self.elu_1(mu, sigma)
        mu, sigma = self.batch_norm(mu, sigma)
        mu, sigma = self.maxpooling_2(mu, sigma) 
        mu, sigma = self.dropout_1(mu, sigma, Training=training)           
        
        mu, sigma = self.conv_4(mu, sigma)       
        mu, sigma = self.elu_1(mu, sigma) 
        mu, sigma = self.batch_norm(mu, sigma)
        mu, sigma = self.conv_5(mu, sigma)       
        mu, sigma = self.elu_1(mu, sigma)
        mu, sigma = self.batch_norm(mu, sigma)
        mu, sigma = self.maxpooling_3(mu, sigma)
        mu, sigma = self.dropout_1(mu, sigma, Training=training)   
        
        mu, sigma = self.conv_6(mu, sigma)      
        mu, sigma = self.elu_1(mu, sigma)
        mu, sigma = self.batch_norm(mu, sigma)
        mu, sigma = self.conv_7(mu, sigma)      
        mu, sigma = self.elu_1(mu, sigma) 
        mu, sigma = self.batch_norm(mu, sigma)
        mu, sigma = self.maxpooling_4(mu, sigma)
        mu, sigma = self.dropout_1(mu, sigma, Training=training)   
              
        mu, sigma = self.conv_8(mu, sigma)      
        mu, sigma = self.elu_1(mu, sigma)
        mu, sigma = self.batch_norm(mu, sigma)
        mu, sigma = self.conv_9(mu, sigma)      
        mu, sigma = self.elu_1(mu, sigma)
        mu, sigma = self.batch_norm(mu, sigma)
        mu, sigma = self.maxpooling_5(mu, sigma)
        mu, sigma = self.dropout_1(mu, sigma, Training=training)   
        
        mu, sigma = self.conv_10(mu, sigma)       
        mu, sigma = self.elu_1(mu, sigma)
        mu, sigma = self.batch_norm(mu, sigma)        
              
        mu, sigma = self.fc_1(mu, sigma)
        mu_out, Sigma_out = self.mysoftma(mu, sigma)
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        return mu_out, Sigma_out


def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels, batch_size):     
    y_pred_sd_ns = y_pred_sd 
    s, u, v = tf.linalg.svd(y_pred_sd_ns, full_matrices=True, compute_uv=True)	
    s_ = s + 1.0e-3
    s_inv = tf.linalg.diag(tf.math.divide_no_nan(1., s_) )    
    y_pred_sd_inv = tf.matmul(tf.matmul(v, s_inv), tf.transpose(u, [0, 2,1])) 
    mu_ = y_test - y_pred_mean 
    mu_sigma = tf.matmul( tf.expand_dims(mu_, axis=1)  ,  y_pred_sd_inv)     
    loss1 =  tf.squeeze(tf.matmul(mu_sigma ,  tf.expand_dims(mu_, axis=2) ))   
    loss2 =  tf.math.reduce_mean(tf.math.reduce_sum(tf.math.log(s_), axis =-1) )        
    loss = tf.math.reduce_mean(tf.math.add(loss1,loss2))
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)   
    return loss           
 
def main_function(input_dim=32, n_channel=3, num_kernels=[32, 32, 32, 32, 64, 64, 64, 128, 128, 128],
                  kernels_size=[5, 3, 3, 3, 3, 3, 3, 3, 3, 1], maxpooling_size=[2, 2, 2, 2, 2],
                  maxpooling_stride=[2, 2, 2, 2, 2], maxpooling_pad='SAME', class_num=10,
                  batch_size=50, epochs=350, lr=0.0002, lr_end = 0.0001, kl_factor=0.00001, 
                  Random_noise=False, Adversarial_noise=False, HCV=0.01,   Black_box_attack=False, PGDBlack_box_attack=True, 
                  adversary_target_cls=3, Targeted=True, PGD_Adversarial_noise=False, stepSize=1, maxAdvStep=40,
                  Training=False, Testing = False, continue_training=False, saved_model_epochs=300):
    PATH = './latest_run/saved_models/VDP_cnn_epoch_{}/'.format(epochs)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() 
    x_train, x_test = x_train / 255.0, x_test / 255.0 
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    one_hot_y_train = tf.one_hot(np.squeeze(y_train).astype(np.float32), depth=class_num)
    one_hot_y_test = tf.one_hot(np.squeeze(y_test).astype(np.float32), depth=class_num)
    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, one_hot_y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)

    cnn_model = Density_prop_CNN(kernel_size=kernels_size, num_kernel=num_kernels, pooling_size=maxpooling_size,
                                 pooling_stride=maxpooling_stride, pooling_pad=maxpooling_pad, units=class_num,
                                 name='vdp_cnn')
    num_train_steps = epochs * int(x_train.shape[0] / batch_size)
    #    step = min(step, decay_steps)
    #    ((initial_learning_rate - end_learning_rate) * (1 - step / decay_steps) ^ (power) ) + end_learning_rate
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr, decay_steps=num_train_steps,  end_learning_rate=lr_end, power=40.)    
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)  # , clipnorm=1.0)
    
    @tf.function  # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            mu_out, sigma = cnn_model(x, training=True)  
            cnn_model.trainable = True         
            loss_final = nll_gaussian(y, mu_out,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+5),
                                   clip_value_max=tf.constant(1e+5)), class_num , batch_size)
            regularization_loss=tf.math.add_n(cnn_model.losses)             
            loss = 0.5 * (loss_final + kl_factor*regularization_loss )           
        gradients = tape.gradient(loss, cnn_model.trainable_weights)   
        gradients = [(tf.where(tf.math.is_nan(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        gradients = [(tf.where(tf.math.is_inf(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        optimizer.apply_gradients(zip(gradients, cnn_model.trainable_weights))       
        return loss, mu_out, sigma, gradients
        
    @tf.function
    def validation_on_batch(x, y):                     
        mu_out, sigma = cnn_model(x, training=False) 
        cnn_model.trainable = False              
        vloss = nll_gaussian(y, mu_out,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+3),
                                           clip_value_max=tf.constant(1e+3)), class_num , batch_size)                                           
        regularization_loss=tf.math.add_n(cnn_model.losses)
        total_vloss = 0.5 *(vloss + kl_factor*regularization_loss)    
        return total_vloss, mu_out, sigma
    @tf.function
    def test_on_batch(x, y):  
        cnn_model.trainable = False                    
        mu_out, sigma = cnn_model(x, training=False)            
        return mu_out, sigma
    @tf.function    
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            cnn_model.trainable = False 
            prediction, sigma = cnn_model(input_image) 
            loss_final = nll_gaussian(input_label, prediction,  tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+3),
                                   clip_value_max=tf.constant(1e+3)), class_num , batch_size)                         
            loss = 0.5 * loss_final 
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
          gradient = tf.where(tf.math.is_nan(gradient),  tf.constant(1.0e-6, shape=gradient.shape), gradient)
          gradient = tf.where(tf.math.is_inf(gradient),  tf.constant(1.0e-6, shape=gradient.shape), gradient)          
          # Get the sign of the gradients to create the perturbation
          signed_grad = tf.sign(gradient)
          return signed_grad  
    if Training: 
       # wandb.init(entity = "dimah", project="VDP_CNN_Cifar10_11layers_epochs_{}_lr_{}_github".format(epochs, lr)) 
        if continue_training:
            saved_model_path = './latest_run/saved_models/VDP_cnn_epoch_{}/'.format(saved_model_epochs)
            cnn_model.load_weights(saved_model_path + 'vdp_cnn_model')
        train_acc = np.zeros(epochs)
        valid_acc = np.zeros(epochs)        
        train_err = np.zeros(epochs)       
        valid_err = np.zeros(epochs)
        start = timeit.default_timer()
        for epoch in range(epochs): 
            print('Epoch: ', epoch + 1, '/', epochs)            
            tr_no_steps = 0
            va_no_steps = 0
            # -------------Training--------------------
            acc_training = np.zeros(int(x_train.shape[0] / (batch_size)))
            err_training = np.zeros(int(x_train.shape[0] / (batch_size)))
            for step, (x, y) in enumerate(tr_dataset):
                update_progress(step / int(x_train.shape[0] / (batch_size)))
                loss, mu_out, sigma, gradients = train_on_batch(x, y)               
                err_training[tr_no_steps] = loss.numpy()
                corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
                accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))                
                acc_training[tr_no_steps] = accuracy.numpy()
                if step % 1000 == 0:
                    print('\n gradient', np.mean(gradients[0].numpy()))
                    print('\n Matrix Norm', np.mean(sigma))   
                    print("\n Step:", step, "Loss:", np.mean(np.amin(err_training)) )                    
                    print("Training accuracy so far: %.4f" % float(np.mean(np.amax(acc_training)) ))                                      
                tr_no_steps += 1                 
            
            train_acc[epoch] = np.mean(acc_training)
            train_err[epoch] = np.mean(err_training)
            print('Training Acc  ', train_acc[epoch])
            print('Training loss  ', train_err[epoch])                    
            # ---------------Validation----------------------
            acc_validation = np.zeros(int(x_test.shape[0] / (batch_size)))
            err_validation = np.zeros(int(x_test.shape[0] / (batch_size)))            
            for step, (x, y) in enumerate(val_dataset):
                update_progress(step / int(x_test.shape[0] / (batch_size)))
                total_vloss, mu_out, sigma  = validation_on_batch(x, y)                
                err_validation[va_no_steps] = total_vloss.numpy()
                corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
                va_accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))                
                acc_validation[va_no_steps] = va_accuracy.numpy()
                if step % 1000 == 0:
                    print("Step:", step, "Loss:", float(np.mean(np.amin(err_validation))))
                    print("validation accuracy so far: %.4f" % np.mean(np.amax(acc_validation)) )
                va_no_steps += 1                
            
            valid_acc[epoch] = np.mean(acc_validation)
            valid_err[epoch] = np.mean(err_validation)
            stop = timeit.default_timer() 
            cnn_model.save_weights(PATH + 'vdp_cnn_model')                   
##            wandb.log({"Training Loss":  train_err[epoch],
##                       "Training Accuracy": train_acc[epoch],    
##                       "Validation Loss": valid_err[epoch],
##                       "Validation Accuracy": valid_acc[epoch],
##                        'epoch': epoch
##                       }) 
            
            print('Total Training Time: ', stop - start)
            print('Training Acc   ', train_acc[epoch])            
            print('Validation Acc ', valid_acc[epoch])            
            print('------------------------------------')
            print('Training error   ', train_err[epoch])           
            print('Validation error',  valid_err[epoch])
            # -----------------End Training--------------------------
        cnn_model.save_weights(PATH + 'vdp_cnn_model')
        if (epochs > 1):
            xnew = np.linspace(0, epochs-1, 20) 
            train_spl = make_interp_spline(np.array(range(0, epochs)), train_acc)
            train_acc1 = train_spl(xnew)
            valid_spl = make_interp_spline(np.array(range(0, epochs)), valid_acc)
            valid_acc1 = valid_spl(xnew)             
            fig = plt.figure(figsize=(15, 7))            
            plt.plot(xnew, train_acc1, 'b', label='Training acc')
            plt.plot(xnew, valid_acc1, 'r', label='Validation acc')
            plt.ylim(0, 1.1)
            plt.title("Density Propagation CNN on CIFAR-10 Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'VDP_CNN_on_CIFAR10_Data_acc.png')
            plt.close(fig) 
            
            
            train_spl = make_interp_spline(np.array(range(0, epochs)), train_err)
            train_err1 = train_spl(xnew)
            valid_spl = make_interp_spline(np.array(range(0, epochs)), valid_err)
            valid_err1 = valid_spl(xnew)  
            fig = plt.figure(figsize=(15, 7))
            plt.plot(xnew, train_err1, 'b', label='Training loss')
            plt.plot(xnew, valid_err1, 'r', label='Validation loss')            
            plt.title("Density Propagation CNN on CIFAR-10 Data")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VDP_CNN_on_CIFAR10_Data_error.png')
            plt.close(fig)
                
        f1 = open(PATH + 'training_validation_acc_error.pkl', 'wb')
        pickle.dump([train_acc, valid_acc, train_err, valid_err], f1)
        f1.close()

        textfile = open(PATH + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(input_dim))
        textfile.write('\n No of Kernels : ' + str(num_kernels))
        textfile.write('\n Number of Classes : ' + str(class_num))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))
        textfile.write('\n kernels Size : ' + str(kernels_size))
        textfile.write('\n Max pooling Size : ' + str(maxpooling_size))
        textfile.write('\n Max pooling stride : ' + str(maxpooling_stride))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        if Training:
            textfile.write('\n Total run time in sec : ' + str(stop - start))
            if (epochs == 1):
                textfile.write("\n Training Accuracy : " + str(train_acc))                
                textfile.write("\n Validation Accuracy : " + str(valid_acc))                
                textfile.write("\n Training error : " + str(train_err))                 
                textfile.write("\n Validation error : " + str(valid_err))
            else:
                textfile.write("\n Training Accuracy : " + str(np.mean(train_acc[epoch])))                
                textfile.write("\n Validation Accuracy : " + str(np.mean(valid_acc[epoch])))                
                textfile.write("\n Training error : " + str(np.mean(train_err[epoch])))                
                textfile.write("\n Validation error : " + str(np.mean(valid_err[epoch])))
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.close()
    # -------------------------Testing-----------------------------
    elif (Testing):
        test_path = 'test_results/'
        if Random_noise:
            test_path = 'test_results_random_noise_{}/'.format(HCV)  
            gaussain_noise_std = HCV/3    
        
        cnn_model.load_weights(PATH + 'vdp_cnn_model')
        test_no_steps = 0
       
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, n_channel])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num, class_num])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))        
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)))
            true_x[test_no_steps, :, :, :] = x
            true_y[test_no_steps, :, :] = y
            if Random_noise:
                noise = tf.random.normal(shape=[batch_size, input_dim, input_dim, 1], mean=0.0,
                                         stddev=gaussain_noise_std, dtype=x.dtype)
                x = x + noise
            mu_out, sigma   = test_on_batch(x, y)            
            mu_out_[test_no_steps, :, :] = mu_out
            sigma_[test_no_steps, :, :, :] = sigma
            corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
            accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            acc_test[test_no_steps] = accuracy.numpy()
            if step % 50 == 0:
                print("Total running accuracy so far: %.4f" % acc_test[test_no_steps])
            test_no_steps += 1
            
        test_acc = np.mean(acc_test)        
        print('Test accuracy : ', test_acc)
        print('Best Test accuracy : ', np.amax(acc_test) )

        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')
        pickle.dump([mu_out_, sigma_, true_x, true_y, test_acc], pf)
        pf.close()
        
        
        var = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        snr_signal = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        for i in range(int(x_test.shape[0] / (batch_size))):
            for j in range(batch_size):
                if Random_noise:
                    noise = tf.random.normal(shape = [input_dim, input_dim, 1], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype ) 
                    snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:,:, :]))/np.sum( np.square(noise) ))
                predicted_out = np.argmax(mu_out_[i,j,:])
                var[i,j] = sigma_[i,j, int(predicted_out), int(predicted_out)]
                
         
        print('Output Variance', np.mean(var))
        if Random_noise:
            print('SNR', np.mean(snr_signal)) 

        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(input_dim))
        textfile.write('\n No of Kernels : ' + str(num_kernels))
        textfile.write('\n Number of Classes : ' + str(class_num))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))
        textfile.write('\n kernels Size : ' + str(kernels_size))
        textfile.write('\n Max pooling Size : ' + str(maxpooling_size))
        textfile.write('\n Max pooling stride : ' + str(maxpooling_stride))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : " + str(test_acc))
        textfile.write("\n Output Variance: "+ str(np.mean(var))) 
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: ' + str(gaussain_noise_std))
            textfile.write('\n Random Noise HCV: ' + str(HCV))
            textfile.write("\n SNR: "+ str(np.mean(snr_signal))) 
        textfile.write("\n---------------------------------")
        textfile.close()
        
    if (Adversarial_noise):
        if Targeted:
            test_path = 'test_results_targeted_adversarial_noise_{}/'.format(HCV)
        else:
            test_path = 'test_results_non_targeted_adversarial_noise_{}/'.format(HCV)
        cnn_model.load_weights(PATH + 'vdp_cnn_model')
        cnn_model.trainable = False         

        test_no_steps = 0        
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, n_channel])
        adv_perturbations = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim,n_channel])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num, class_num])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        epsilon = HCV/3
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)))
            true_x[test_no_steps, :, :, :] = x
            true_y[test_no_steps, :, :] = y

            if Targeted:
                y_true_batch = np.zeros_like(y)
                y_true_batch[:, adversary_target_cls] = 1.0
                adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(x, y_true_batch)
            else:
                adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(x, y)
            adv_x = x + epsilon * adv_perturbations[test_no_steps, :, :, :]
            adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
            
            mu_out, sigma   = test_on_batch(adv_x, y)            
            mu_out_[test_no_steps, :, :] = mu_out
            sigma_[test_no_steps, :, :, :] = sigma
            corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
            accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            acc_test[test_no_steps]=accuracy.numpy()            
            if step % 50 == 0:
                print("Total running accuracy so far: %.4f" % acc_test[test_no_steps])
            test_no_steps += 1
        test_acc = np.mean(acc_test)            
        print('Test accuracy : ', test_acc)
        print('Best Test accuracy : ', np.amax(acc_test) )

        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')
        pickle.dump([mu_out_, sigma_, true_x, true_y, adv_perturbations, test_acc], pf)
        pf.close()
        
        
        var = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
        snr_signal = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
        for i in range(int(x_test.shape[0] /batch_size)):
            for j in range(batch_size):               
                predicted_out = np.argmax(mu_out_[i,j,:])
                var[i,j] = sigma_[i,j, int(predicted_out), int(predicted_out)]
                snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :,:]))/np.sum( np.square(epsilon*adv_perturbations[i, j, :, :, :]  ) ))
         
        print('Output Variance', np.mean(var))
        print('SNR', np.mean(snr_signal)) 
        
##        var1 = np.reshape(var, int(x_test.shape[0]/(batch_size))* batch_size)  
##        #print(var1)              
##        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
##        df = pd.DataFrame(np.abs(var1) )   
##        # Write your DataFrame to a file   
##        df.to_excel(writer, "Sheet")            
##        writer.save()

        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(input_dim))
        textfile.write('\n No of Kernels : ' + str(num_kernels))
        textfile.write('\n Number of Classes : ' + str(class_num))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))
        textfile.write('\n kernels Size : ' + str(kernels_size))
        textfile.write('\n Max pooling Size : ' + str(maxpooling_size))
        textfile.write('\n Max pooling stride : ' + str(maxpooling_stride))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : " + str(test_acc))
        textfile.write("\n Output Variance: "+ str(np.mean(var)))
        textfile.write("\n---------------------------------")
        if Adversarial_noise:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))
            else:
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: ' + str(epsilon))
            textfile.write('\n Adversarial Noise HCV: ' + str(HCV))
            textfile.write("\n SNR: "+ str(np.mean(snr_signal))) 
        textfile.write("\n---------------------------------")
        textfile.close()
    
    if (Black_box_attack):
        if Targeted:
            test_path = 'test_results_targeted_Black_box_attack_{}/'.format(HCV)
        else:
            test_path = 'test_results_non_targeted_Black_box_attack_{}/'.format(HCV)
        cnn_model.load_weights(PATH + 'vdp_cnn_model')
        cnn_model.trainable = False         

              
        
        pf = open(PATH + test_path + 'info_{}.pkl'.format(HCV), 'rb')
        DCNNout_, DCNNtrue_x, DCNNtrue_y, adv_perturbations, DCNNtest_acc = pickle.load(pf)
        pf.close()
        
        test_samples = adv_perturbations.shape[0]
        test_no_steps = 0
        mu_out_ = np.zeros([int(test_samples), batch_size, class_num])
        sigma_ = np.zeros([int(test_samples), batch_size, class_num, class_num])
        acc_test = np.zeros(int(test_samples))
        epsilon = HCV/3 
        
        for step, (x, y) in enumerate(val_dataset):
            if(test_no_steps < test_samples): 
               update_progress(step / int(x_test.shape[0] / (batch_size)))

               adv_x = x + epsilon * adv_perturbations[test_no_steps, :, :, :]
               adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
               
               mu_out, sigma   = test_on_batch(adv_x, y)            
               mu_out_[test_no_steps, :, :] = mu_out
               sigma_[test_no_steps, :, :, :] = sigma
               corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
               accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
               acc_test[test_no_steps]=accuracy.numpy()            
               if step % 50 == 0:
                   print("Total running accuracy so far: %.4f" % acc_test[test_no_steps])
               test_no_steps += 1
        test_acc = np.mean(acc_test)            
        print('Test accuracy : ', test_acc)
        print('Best Test accuracy : ', np.amax(acc_test) )

        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')
        pickle.dump([mu_out_, sigma_,  test_acc], pf)
        pf.close()
        
        
        var = np.zeros([int(test_samples) ,batch_size])      
        for i in range(int(test_samples)):
            for j in range(batch_size):               
                predicted_out = np.argmax(mu_out_[i,j,:])
                var[i,j] = sigma_[i,j, int(predicted_out), int(predicted_out)]               
         
        print('Output Variance', np.mean(var))
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(input_dim))
        textfile.write('\n No of Kernels : ' + str(num_kernels))
        textfile.write('\n Number of Classes : ' + str(class_num))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))
        textfile.write('\n kernels Size : ' + str(kernels_size))
        textfile.write('\n Max pooling Size : ' + str(maxpooling_size))
        textfile.write('\n Max pooling stride : ' + str(maxpooling_stride))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : " + str(test_acc))
        textfile.write("\n Output Variance: "+ str(np.mean(var)))
        textfile.write("\n---------------------------------")
        if Black_box_attack:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))
            else:
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: ' + str(epsilon))
            textfile.write('\n Adversarial Noise HCV: ' + str(HCV))         
        textfile.write("\n---------------------------------")
        textfile.close()
        
        
        
    if (PGDBlack_box_attack):
        if Targeted:
            test_path = 'test_results_targeted_PGDBlack_box_attack_{}/'.format(HCV)
        else:
            test_path = 'test_results_non_targeted_PGDBlack_box_attack_{}/'.format(HCV)
        cnn_model.load_weights(PATH + 'vdp_cnn_model')
        cnn_model.trainable = False         
              
        
        pf = open(PATH + test_path + 'info_{}.pkl'.format(HCV), 'rb')
        DCNNout_, adv_perturbations, DCNNtest_acc = pickle.load(pf)
        pf.close()
        
        test_samples = adv_perturbations.shape[0]
        test_no_steps = 0
        mu_out_ = np.zeros([int(test_samples), batch_size, class_num])
        sigma_ = np.zeros([int(test_samples), batch_size, class_num, class_num])
        acc_test = np.zeros(int(test_samples))
        epsilon = HCV/3 
        
        for step, (x, y) in enumerate(val_dataset):
            if(test_no_steps < test_samples): 
               update_progress(step / int(x_test.shape[0] / (batch_size)))
               adv_x = x + adv_perturbations[test_no_steps, :, :, :]
               adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)              

               mu_out, sigma   = test_on_batch(adv_x, y)            
               mu_out_[test_no_steps, :, :] = mu_out
               sigma_[test_no_steps, :, :, :] = sigma
               corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
               accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
               acc_test[test_no_steps]=accuracy.numpy()            
               if step % 50 == 0:
                   print("Total running accuracy so far: %.4f" % acc_test[test_no_steps])
               test_no_steps += 1
        test_acc = np.mean(acc_test)            
        print('Test accuracy : ', test_acc)
        print('Best Test accuracy : ', np.amax(acc_test) )

        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')
        pickle.dump([mu_out_, sigma_,  test_acc], pf)
        pf.close()
        
        
        var = np.zeros([int(test_samples) ,batch_size])      
        for i in range(int(test_samples)):
            for j in range(batch_size):               
                predicted_out = np.argmax(mu_out_[i,j,:])
                var[i,j] = sigma_[i,j, int(predicted_out), int(predicted_out)]               
         
        print('Output Variance', np.mean(var))       

        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(input_dim))
        textfile.write('\n No of Kernels : ' + str(num_kernels))
        textfile.write('\n Number of Classes : ' + str(class_num))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))
        textfile.write('\n kernels Size : ' + str(kernels_size))
        textfile.write('\n Max pooling Size : ' + str(maxpooling_size))
        textfile.write('\n Max pooling stride : ' + str(maxpooling_stride))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : " + str(test_acc))
        textfile.write("\n Output Variance: "+ str(np.mean(var)))
        textfile.write("\n---------------------------------")
        if Black_box_attack:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))
            else:
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: ' + str(epsilon))
            textfile.write('\n Adversarial Noise HCV: ' + str(HCV))        
        textfile.write("\n---------------------------------")
        textfile.close()       
            
            
    if (PGD_Adversarial_noise):
        if Targeted:
            test_path = 'test_results_targeted_PGDadversarial_noise_{}_max_iter_{}_{}/'.format(HCV, maxAdvStep, stepSize)
        else:
            test_path = 'test_results_non_targeted_PGDadversarial_noise_{}_max_iter_{}_{}/'.format(HCV, maxAdvStep, stepSize)
        cnn_model.load_weights(PATH + 'vdp_cnn_model')
        cnn_model.trainable = False         

        test_no_steps = 0        
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, n_channel])
        adv_perturbations = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim,n_channel])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num, class_num])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        epsilon = HCV/3
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)))
            true_x[test_no_steps, :, :, :] = x
            true_y[test_no_steps, :, :] = y            
            
            adv_x = x + tf.random.uniform(x.shape, minval=-epsilon, maxval=epsilon)
            adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
            for advStep in range(maxAdvStep):
                if Targeted:
                    y_true_batch = np.zeros_like(y)
                    y_true_batch[:, adversary_target_cls] = 1.0
                    adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(adv_x, y_true_batch)
                else:
                    adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(adv_x, y)
                adv_x = adv_x + stepSize * adv_perturbations[test_no_steps, :, :, :]
                adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)                          
            
            mu_out, sigma   = test_on_batch(adv_x, y)            
            mu_out_[test_no_steps,  :, :] = mu_out
            sigma_[test_no_steps, :, :, :] = sigma
            corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
            accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            acc_test[test_no_steps]=accuracy.numpy()            
            if step % 50 == 0:
                print("Total running accuracy so far: %.4f" % acc_test[test_no_steps] )
            test_no_steps += 1
        test_acc = np.mean(acc_test)            
        print('Test accuracy : ', test_acc)
        print('Best Test accuracy : ', np.amax(acc_test) )

        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')
        pickle.dump([mu_out_, sigma_, true_x, true_y, test_acc], pf)
        pf.close()
        
        
        var = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])        
        for i in range(int(x_test.shape[0] /batch_size)):
            for j in range(batch_size):               
                predicted_out = np.argmax(mu_out_[i,j,:])
                var[i,j] = sigma_[i,j, int(predicted_out), int(predicted_out)]              
         
        print('Output Variance', np.mean(var))     
        
##        var1 = np.reshape(var, int(x_test.shape[0]/(batch_size))* batch_size)                  
##        writer = pd.ExcelWriter(PATH + test_path + 'variance.xlsx', engine='xlsxwriter')
##        df = pd.DataFrame(np.abs(var1) )   
##        # Write your DataFrame to a file   
##        df.to_excel(writer, "Sheet")            
##        writer.save()

        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(input_dim))
        textfile.write('\n No of Kernels : ' + str(num_kernels))
        textfile.write('\n Number of Classes : ' + str(class_num))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' +str(lr_end))
        textfile.write('\n kernels Size : ' + str(kernels_size))
        textfile.write('\n Max pooling Size : ' + str(maxpooling_size))
        textfile.write('\n Max pooling stride : ' + str(maxpooling_stride))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : " + str(test_acc))
        textfile.write("\n Output Variance: "+ str(np.mean(var)))
        textfile.write("\n---------------------------------")
        if PGD_Adversarial_noise:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))
            else:
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: ' + str(epsilon))
            textfile.write('\n Adversarial Noise HCV: ' + str(HCV))
          #  textfile.write("\n SNR: "+ str(np.mean(snr_signal))) 
            textfile.write("\n stepSize: "+ str(stepSize)) 
            textfile.write("\n Maximum number of iterations: "+ str(maxAdvStep))
        textfile.write("\n---------------------------------")
        textfile.close()    
        


if __name__ == '__main__':
    main_function()
