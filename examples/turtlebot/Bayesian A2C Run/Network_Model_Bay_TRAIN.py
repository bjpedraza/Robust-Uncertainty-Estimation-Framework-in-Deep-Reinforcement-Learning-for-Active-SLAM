#!/usr/bin/env python

import logging
import argparse

import numpy as np
import os
import random

import time
from distutils.dir_util import copy_tree
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko


from tensorflow.keras.models import Sequential, load_model, Model
from keras.initializers import normal
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Flatten, ZeroPadding2D
from tensorflow.keras.layers import Dense, Dropout, Activation, InputLayer
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD , Adam , RMSprop
import memory
from tensorflow.keras import backend as K

import VDP_CNN_CIFAR10_11layers as vdp


#K.set_image_dim_ordering('th')

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-n', '--num_updates', type=int, default=250)
parser.add_argument('-lr', '--learning_rate', type=float, default=7e-3)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)
args = parser.parse_args()


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Actor_VDP(tf.keras.Model):
    def __init__(self, kernel_size, num_kernel, pooling_size, pooling_stride, pooling_pad, units, drop_prop=0.2,
                  var_epsilon=1e-4, name=None):

        super(Actor_VDP,self).__init__('mlp_policy')

        self.kernel_size = kernel_size
        self.num_kernel = num_kernel
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad
        self.units = units
        self.drop_prop = drop_prop
        self.var_epsilon = var_epsilon

        #self.input = Input((480,480,1), name='policy_input')

        # -------- CNN Part --------
        self.conv_1 = vdp.VDP_first_Conv(kernel_size=self.kernel_size[0], kernel_num=self.num_kernel[0], padding='VALID')
        self.conv_2 = vdp.VDP_intermediate_Conv(kernel_size=self.kernel_size[1], kernel_num=self.num_kernel[1],   padding='SAME')
        self.conv_3 = vdp.VDP_intermediate_Conv(kernel_size=self.kernel_size[2], kernel_num=self.num_kernel[2],   padding='SAME')
        self.conv_4 = vdp.VDP_intermediate_Conv(kernel_size=self.kernel_size[3], kernel_num=self.num_kernel[3],   padding='SAME')
        self.conv_5 = vdp.VDP_intermediate_Conv(kernel_size=self.kernel_size[4], kernel_num=self.num_kernel[4],   padding='SAME')
        self.conv_6 = vdp.VDP_intermediate_Conv(kernel_size=self.kernel_size[5], kernel_num=self.num_kernel[5],  padding='SAME')
        self.conv_7 = vdp.VDP_intermediate_Conv(kernel_size=self.kernel_size[6], kernel_num=self.num_kernel[6],   padding='SAME')
        self.conv_8 = vdp.VDP_intermediate_Conv(kernel_size=self.kernel_size[7], kernel_num=self.num_kernel[7],   padding='SAME')
        self.conv_9 = vdp.VDP_intermediate_Conv(kernel_size=self.kernel_size[8], kernel_num=self.num_kernel[8],   padding='SAME')
        self.conv_10 = vdp.VDP_intermediate_Conv(kernel_size=self.kernel_size[9], kernel_num=self.num_kernel[9],   padding='SAME')

        self.elu_1 = vdp.VDPELU()

        self.maxpooling_1 = vdp.VDP_MaxPooling(pooling_size=self.pooling_size[0], pooling_stride=self.pooling_stride[0],   pooling_pad=self.pooling_pad)
        self.maxpooling_2 = vdp.VDP_MaxPooling(pooling_size=self.pooling_size[1], pooling_stride=self.pooling_stride[1],   pooling_pad=self.pooling_pad)
        self.maxpooling_3 = vdp.VDP_MaxPooling(pooling_size=self.pooling_size[2], pooling_stride=self.pooling_stride[2],   pooling_pad=self.pooling_pad)
        self.maxpooling_4 = vdp.VDP_MaxPooling(pooling_size=self.pooling_size[3], pooling_stride=self.pooling_stride[3],   pooling_pad=self.pooling_pad)
        self.maxpooling_5 = vdp.VDP_MaxPooling(pooling_size=self.pooling_size[4], pooling_stride=self.pooling_stride[4],   pooling_pad=self.pooling_pad)

        self.dropout_1 = vdp.VDPDropout(self.drop_prop)
        self.batch_norm = vdp.VDPBatch_Normalization(self.var_epsilon)


        self.flat_TF = Flatten() # TensorFlow flatten
        self.fc_linear_actor = vdp.VDP_Flatten_and_FC(self.units)
        self.mysoftma = vdp.mysoftmax()

        self.dist = ProbabilityDistribution()

        #self.Critic_model = critic_model

        #  For Critic Value:
        #self.hidden1_Cri = kl.Dense(400, activation='relu')
        #self.hidden2_Cri = kl.Dense(200, activation='relu')
        #self.valueOut_Cri = kl.Dense(1, name='value')


    def call(self, inputs, training):
        # Inputs is a numpy array, convert to a tensor.
        #x = tf.convert_to_tensor(inputs)
        #print(kwargs.get("training") == True)

        # ---------- CNN Part ---------------
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
        #mu, sigma = self.batch_norm(mu, sigma)
        #mu, sigma = self.maxpooling_5(mu, sigma)


        # ---------- Critic Part ---------------
        #val_out_cri = self.flat_TF(mu)
        #val_out_cri = self.hidden1_Cri(val_out_cri)
        #val_out_cri = self.hidden2_Cri(val_out_cri)
        #val_out_cri = self.valueOut_Cri(val_out_cri)
                # OR ----------- Critic Part --------------- OR
        #val_out_cri = self.Critic_model(mu)


        # ----------- Actor Part --------------
        mu_linear_out_act, Sigma_linear_out_act = self.fc_linear_actor(mu, sigma)
        mu_linear_out_act, Sigma_linear_out_act = self.mysoftma(mu_linear_out_act, Sigma_linear_out_act)
        

        # Set Range for actions (Linear and Angle)
        #mu_linear_out_act = self.flat_TF(mu_linear_out_act)
        #mu_linear_out_act = self.mu_input(mu_linear_out_act)
        #mu_linear_out_act = self.lin_act(mu_linear_out_act)
        #mu_linear_out_act = self.linear_output(mu_linear_out_act)

        Sigma_linear_out_act = tf.where(tf.math.is_nan(Sigma_linear_out_act), tf.zeros_like(Sigma_linear_out_act), Sigma_linear_out_act)
        Sigma_linear_out_act = tf.where(tf.math.is_inf(Sigma_linear_out_act), tf.zeros_like(Sigma_linear_out_act), Sigma_linear_out_act)

        return mu_linear_out_act, Sigma_linear_out_act


class Critic_NoCNN(tf.keras.Model):
    def __init__(self, state_dim):
        super(Critic_NoCNN,self).__init__('mlp_policyCri')
        self.state_dim = state_dim
        #self.model = self.create_model()
        #self.opt = tf.keras.optimizers.Adam(args.critic_lr)

        #self.flat = Flatten()
        self.hidden1_Cri = kl.Dense(400, activation='relu')
        self.hidden2_Cri = kl.Dense(200, activation='relu')
        self.valueOut_Cri = kl.Dense(1, name='value')

    def call(self, inputs):
        #val_out_cri = self.flat(inputs)
        val_out_cri = self.hidden1_Cri(inputs)
        val_out_cri = self.hidden2_Cri(val_out_cri)
        val_out_cri = self.valueOut_Cri(val_out_cri)

        return val_out_cri



class Critic_CNN(tf.keras.Model):
    def __init__(self, state_dim):
        super(Critic_CNN,self).__init__('mlp_policyCri')
        self.state_dim = state_dim
        self.inputLay = InputLayer((1, 32,32,1), name='policy_input')
        '''
        # ----- CNN Part -----
        self.conv1 = Conv2D(16, (3,3), strides=(2,2))
        self.act1 = Activation('relu')
        self.zero_pad1 = ZeroPadding2D((1, 1))
        self.conv2 = Conv2D(16, (3,3), strides=(2,2))
        self.act2 = Activation('relu')
        self.max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2,2))
        '''
        # ----- CNN Part (Optional) -----
        self.conv1 = Conv2D(32, (4,4), strides=(2,2), activation='relu')
        self.conv2 = Conv2D(64, (4,4), strides=(2,2), activation='relu')
        #self.zero_pad1 = ZeroPadding2D((1, 1))
        self.conv3 = Conv2D(64, (3,3), strides=(2,2), activation='relu')
        self.max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2,2))
        self.drop1 = Dropout(0.2)
        
        # ---- Values ------
        self.flat = Flatten()
        self.hidden1 = kl.Dense(64, activation='relu')
        self.hidden2 = kl.Dense(400, activation='relu')
        self.hidden3 = kl.Dense(200, activation='relu')
        self.valueOut = kl.Dense(1, name='value')

    def call(self, inputs):
        hidLay = self.inputLay(inputs)
        hidLay = self.conv1(hidLay)
        hidLay = self.conv2(hidLay)

        #hidLay = self.zero_pad1(hidLay)

        hidLay = self.conv3(hidLay)
        hidLay = self.max_pool(hidLay)
        hidLay = self.drop1(hidLay)

        hidLay = self.flat(hidLay)
        hidLay = self.hidden1(hidLay)
        hidLay = self.hidden2(hidLay)
        hidLay = self.hidden3(hidLay)
        #hidLay = self.hidden4(hidLay)
        value = self.valueOut(hidLay)

        return value
'''
#From 2021 article
class Critic_CNN(tf.keras.Model):
    def __init__(self, state_dim):
        super(Critic_CNN,self).__init__('mlp_policyCri')
        self.state_dim = state_dim
        self.inputLay = InputLayer((1, 32,32,1), name='policy_input')

        # ----- CNN Part -----
        self.conv1 = Conv2D(64, (4,4), activation='relu')
        self.conv2 = Conv2D(64, (4,4), activation='relu')
        self.max_pool1 = MaxPooling2D(pool_size=(2, 2))
        self.drop1 = Dropout(0.4)

        self.conv3 = Conv2D(128, (4,4), activation='relu')
        self.conv4 = Conv2D(128, (4,4), activation='relu')
        self.max_pool2 = MaxPooling2D(pool_size=(2, 2))
        self.drop2 = Dropout(0.4)

        # ---- Values ------
        self.flat = Flatten()
        self.hidden1 = kl.Dense(1152, activation='relu')
        self.hidden2 = kl.Dense(400, activation='relu')
        self.hidden3 = kl.Dense(200, activation='relu')
        self.valueOut = kl.Dense(1, name='value')

    def call(self, inputs):
        hidLay = self.inputLay(inputs)
        hidLay = self.conv1(hidLay)
        hidLay = self.conv2(hidLay)
        hidLay = self.max_pool1(hidLay)
        hidLay = self.drop1(hidLay)

        hidLay = self.conv3(hidLay)
        hidLay = self.conv4(hidLay)
        hidLay = self.max_pool2(hidLay)
        hidLay = self.drop2(hidLay)   

        hidLay = self.flat(hidLay)
        hidLay = self.hidden1(hidLay)
        hidLay = self.hidden2(hidLay)
        hidLay = self.hidden3(hidLay)
        value = self.valueOut(hidLay)

        return value
'''

class A2CAgent:
    def __init__(self, model_Act, model_Cri, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c

        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.0002, decay_steps=100000,  end_learning_rate=0.0001, power=40.)
        self.optimizer_actor = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn) # , clipnorm=1.0)

        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr)
        #self.optimizer_critic = tf.keras.optimizers.RMSprop(learning_rate=lr)
        #self.optimizer_critic = tf.keras.optimizers.SGD(learning_rate=lr)
        #self.optimizer_critic = ko.RMSprop(lr)
        
        self.model_Act = model_Act
        self.model_Cri = model_Cri

        self.old_neg_log_prob = 0

        #self.model.compile(
        #optimizer=ko.RMSprop(lr=lr),
        # Define separate losses for policy logits and value estimate.
        #loss=[self._logits_loss_mu, self._logits_loss_sigma, self._value_loss])


    def action_value(self, obs):
        # Executes `call()` under the hood.
        mu_action, sig_action = self.model_Act.predict_on_batch(obs)
        #action = self.model_Act.dist.predict_on_batch(mu_action)[0]
        value = self.model_Cri.predict_on_batch(obs)


        print("yayyy")
        print(mu_action)
        print(sig_action)
        print(value)
        print("yayyy22")
        
        act_mu = mu_action[0]
        #act_sig = sig_action[0]

        uncert_sig = np.diag( sig_action[0] )
        #uncert_sig_cpy = np.arange(3)
        certain = uncert_sig.argmin()
        #uncertain = uncert_sig.argmax()

        action = act_mu.argmax()
        
        # Threshold for Uncertainty for Action
        if uncert_sig[action] > 0.3:

            #second_high_mu = np.argpartition(act_mu.flatten(), -2)[-2]

            #if uncert_sig[actionMax] < 0.3:
            #    action = actionMax
            #if uncert_sig[second_high_mu] < 0.3: # and second_high_mu == certain:
            #    action = second_high_mu
            if uncert_sig[certain] < 0.3:  #and act_mu[certain] > 0.05:
                action = certain

            #if (action != actionMax): #and (not check_bool_uncert):
            #    if uncert_sig[actionMax] < 0.3:
            #        action = actionMax

        #print(uncert_sig)
        #print(uncert_sig_cpy)
        print(action)

        return action, value[0][0]

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), [next_value], axis=-1)
        #OR returns = np.array(rewards + [next_value[0]])
        #returns = np.zeros_like(rewards)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case). TD Formula
        advantages = returns - values

        return returns, advantages


    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kls.mean_squared_error(returns, value)


    def _logits_loss_mu(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        # ------ Edit Here For 2 Actions: ----------
        #actions = actions_and_advantages[:,0:1]
        #advantages = actions_and_advantages[:,1]
        #print(  actions[:,0])

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True, reduction=kls.Reduction.SUM)
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.float32)
        policy_loss1 = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.

        return (policy_loss1) - self.entropy_c * entropy_loss


    def _logits_loss_sigma(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        #actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        # ------ Edit Here For 3  : ----------
        actions = actions_and_advantages[:,0:3]
        advantages = actions_and_advantages[:,3]
        #print(  actions[:,0])
        #print(logits)

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True, reduction=kls.Reduction.SUM)
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.float32)
        policy_loss1 = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss1 - self.entropy_c * entropy_loss


    @tf.function  # Make it fast.
    def train_on_batch_actor(self, x, actions, advs):

        #actions, advantages = tf.split(y, 2, axis=-1)
        actions = tf.cast(actions, tf.int32)
        encodeActions = tf.one_hot(actions,3)
        advs = tf.cast(advs, tf.float32)
        encodeActions = tf.cast(encodeActions, tf.int32)


        with tf.GradientTape() as tape: # Actor
            mu_out, sigma = self.model_Act(x, training=True)
            self.model_Act.trainable = True

            # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
            # `from_logits` argument ensures transformation into normalized probabilities.
            weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=False) # reduction=kls.Reduction.SUM)


            policy_loss1 = weighted_sparse_ce(actions, mu_out, sample_weight=advs)
            #policy_loss2 = weighted_sparse_ce(encodeActions, sigma, sample_weight=advs)
            #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mu_out,labels=actions)
            #neg_log_prob2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sigma,labels=encodeActions)
            #ratio = tf.exp(neg_log_prob - self.old_neg_log_prob)
            #surrogate_loss = tf.reduce_mean(-ratio*advs )
            #surrogate_loss = tf.reduce_mean(neg_log_prob*advs)

            #action_probs = tf.math.log(mu_out)
            #actor_loss = -tf.math.reduce_sum(action_probs*advs)

            #probs = tf.nn.softmax(mu_out)
            #entropy_loss = kls.categorical_crossentropy(probs, probs)

            loss_final = nll_gaussian(encodeActions, mu_out, tf.clip_by_value(t=sigma, clip_value_min=tf.constant(-1e+5),
                                   clip_value_max=tf.constant(1e+5)), num_labels=3 , batch_size=16, advs=advs)
            regularization_loss=tf.math.add_n(self.model_Act.losses)
            kl_factor = 10e-4
            loss =  0.5*(loss_final + kl_factor*regularization_loss) + tf.math.reduce_mean( policy_loss1 )#+ tf.math.reduce_mean(self.entropy_c * entropy_loss)


        # ---- For Actor weights ----
        gradients = tape.gradient(loss, self.model_Act.trainable_weights)
        #gradients, _ = tf.clip_by_global_norm(tape.gradient(loss, self.model_Act.trainable_weights), 2.0)

        #gradients, _ = tf.clip_by_value(tape.gradient(loss, self.model_Act.trainable_weights), clip_value_min=-1.0, clip_value_max=1.0)
        #gradients = [(tf.clip_by_value(grad, clip_value_min=-2.0, clip_value_max=2.0)) for grad in gradients]
        #gradients = [(tf.clip_by_norm(grad, clip_norm=2.0)) for grad in gradients]

        gradients = [(tf.where(tf.math.is_nan(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        gradients = [(tf.where(tf.math.is_inf(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]

        self.optimizer_actor.apply_gradients(zip(gradients, self.model_Act.trainable_weights)) 

        return loss #, mu_out, sigma, gradients



    @tf.function  # Make it fast.
    def train_on_batch_critic(self, x, returns):
        with tf.GradientTape() as tape:
            val = self.model_Cri(x, training=True)
            #print(self.model.Critic_model.summary())

            self.model_Cri.trainable = True 
            loss = self._value_loss(returns, val) #tf.stop_gradient(returns), val)

        gradients = tape.gradient(loss, self.model_Cri.trainable_weights)
        #gradients = [(tf.clip_by_value(grad, clip_value_min=-1.2, clip_value_max=1.2)) for grad in gradients]

        self.optimizer_critic.apply_gradients(zip(gradients, self.model_Cri.trainable_weights))

        return loss #, val, gradients



def nll_gaussian(y_test, y_pred_mean, y_pred_sd, num_labels, batch_size, advs):
    y_pred_sd_ns = tf.cast(y_pred_sd, tf.float32)
    y_test = tf.cast(y_test, tf.float32)
    y_pred_mean = tf.cast(y_pred_mean, tf.float32)

    s, u, v = tf.linalg.svd(y_pred_sd_ns, full_matrices=True, compute_uv=True)  
    s_ = s + 1.0e-3
    s_inv = tf.linalg.diag(tf.math.divide_no_nan(1., s_) )
    y_pred_sd_inv = tf.matmul(tf.matmul(v, s_inv), tf.transpose(u, [0, 2,1]))
    mu_ = y_test - y_pred_mean
    mu_sigma = tf.matmul( tf.expand_dims(mu_, axis=1),  y_pred_sd_inv)

    loss1 = tf.squeeze(tf.matmul(mu_sigma,tf.expand_dims(mu_, axis=2)))  #, tf.expand_dims(tf.expand_dims(0.80*advs,axis=1),axis=1) )) 
    loss2 = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.log(s_), axis =-1))
    loss = tf.math.reduce_mean(tf.math.add(loss1,loss2))
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
    loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
    return loss 
