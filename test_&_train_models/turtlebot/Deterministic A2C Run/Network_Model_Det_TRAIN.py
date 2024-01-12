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
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, Flatten, ZeroPadding2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD , Adam
import memory
from keras import backend as K
#K.set_image_dim_ordering('th')


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-n', '--num_updates', type=int, default=250)
parser.add_argument('-lr', '--learning_rate', type=float, default=7e-3)
parser.add_argument('-r', '--render_test', action='store_true', default=False)
parser.add_argument('-p', '--plot_results', action='store_true', default=False)


class Model_CNN:
    def __init__(self, outputs, memorySize, discountFactor, learningRate, learnStart, img_rows, img_cols, img_channels):
        """
        Parameters:
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels

        self.network_outputs = 3



    def initNetworks(self):
        model = self.createModel()
        self.model = model

    def createModel(self):
        # CNN Network structure must be directly changed here.
        '''
        model = Sequential()
        model.add(InputLayer((32,32,1)))
        model.add(Convolution2D(16, (3,3), strides=(2,2))) #input_shape=(32,32,1)))
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(16, (3,3), strides=(2,2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Flatten())


        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(self.network_outputs))
        #adam = Adam(lr=self.learningRate)
        #model.compile(loss='mse',optimizer=adam)
        '''
        #model_CNN.compile(RMSprop(lr=self.learningRate), 'MSE')
        #model.summary()

        model = Sequential()
        model.add(InputLayer((32,32,1)))
        model.add(Convolution2D(16, (3,3), strides=(2,2), activation='relu')) #input_shape=(32,32,1)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(16, (3,3), strides=(2,2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(Flatten())



        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print("layer ",i,": ",weights)
            i += 1

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)


    # predict Q values for all the actions
    def getFlatObs(self, state):
        return self.model.predict(state)


    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state)
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else :
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)

            # ----- MAYBE BE a ISSUE HERE ---------------
            X_batch = np.empty((1,self.img_rows,self.img_cols,self.img_channels), dtype = np.float64)
            Y_batch = np.empty((1,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)
                X_batch = np.append(X_batch, state.copy(), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, newState.copy(), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, validation_split=0.2, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        os.unlink(file)



class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model_A2C(tf.keras.Model):
    def __init__(self, num_actions):
        super(Model_A2C,self).__init__('mlp_policy')
        self.value_c = 0.5
        self.gamma = 0.99
        self.entropy_c = 1e-4

        input = Input((32,32,1), name='policy_input')
        '''
        # ------ CNN Part ------
        conv_1 = Conv2D(64, (3,3), strides=(2,2), activation='relu') (input)
        zeroPad = ZeroPadding2D((1, 1)) (conv_1)
        conv_2 = Conv2D(64, (3,3), strides=(2,2), activation='relu') (zeroPad)
        maxPool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2)) (conv_2)
        flat = Flatten() (maxPool_1)
        '''
        # ----- CNN Part (Optional) 1st run -----
        conv1 = Conv2D(32, (4,4), strides=(2,2), activation='relu') (input)
        conv2 = Conv2D(64, (4,4), strides=(2,2), activation='relu') (conv1)
        #zero_pad1 = ZeroPadding2D((1, 1)) (conv2)
        conv3 = Conv2D(64, (3,3), strides=(2,2), activation='relu') (conv2)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2,2)) (conv3)

        '''
        # ----- CNN Part (Optional) 2nd/3ird run -----
        conv1 = Conv2D(32, (4,4), strides=(2,2), activation='relu') (input)
        conv2 = Conv2D(64, (4,4), strides=(2,2), activation='relu') (conv1)
        conv3 = Conv2D(64, (2,2), strides=(2,2), activation='relu') (conv2)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2,2)) (conv3)
        '''

        drop1 = Dropout(0.2) (max_pool)
        flat = Flatten() (drop1)
        
        # ------- A2C Part -------
        # Note: no tf.get_variable(), just simple Keras API!
        #input_X = kl.Dense(480, activation='relu') (flat) #For Actor
        #input_Cri = kl.Dense(240,  kernel_initializer='he_uniform',activation='relu') (flat) #For Critic

        #  For Actor Action:
        hidden1_Act = kl.Dense(64, activation='relu') (flat)
        hidden2_Act = kl.Dense(400, activation='relu') (hidden1_Act)
        hidden3_Act = kl.Dense(200, activation='relu') (hidden2_Act)
        # Logits are unnormalized log probabilities.
        out_action = kl.Dense(num_actions, name='policy_logits1') (hidden3_Act)

        #  For Critic Value:
        hidden1_Cri = kl.Dense(64, activation='relu') (flat)
        hidden2_Cri = kl.Dense(400, activation='relu') (hidden1_Cri)
        hidden3_Cri = kl.Dense(200, activation='relu') (hidden2_Cri)
        valueOut_Cri = kl.Dense(1, name='value') (hidden3_Cri)


        # ------ Create Network Model ------
        self.network = Model(inputs=input, outputs=[out_action, valueOut_Cri]) # for value
        #self.network.compile( optimizer=ko.RMSprop(lr=7e-3),loss=self._value_loss)
        #self.network2 = Model(inputs=input, outputs=[mu_linear_output, mu_angle_output, std_output]) # for continous action
        #self.network2.compile( optimizer=ko.RMSprop(lr=7e-3),loss=[self._logits_loss, 'MSE', 'MSE'])

        self.dist = ProbabilityDistribution()


    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        #x = tf.convert_to_tensor(inputs)
        #print(kwargs.get("training") == True)
        return self.network(inputs)



class A2CAgent:
    def __init__(self, model, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
        # `gamma` is the discount factor; coefficients are used for the loss terms.
        self.gamma = gamma
        self.value_c = value_c
        self.entropy_c = entropy_c

        self.model = model

        self.model.compile(
        optimizer=ko.RMSprop(lr=lr),
        # Define separate losses for policy logits and value estimate.
        loss=[self._logits_loss, self._value_loss])


    def action_value(self, obs):
        # Executes `call()` under the hood.
        #self.obs_temp[0] = obs

        logits, value = self.model.predict_on_batch(obs)

        action = self.model.dist.predict_on_batch(logits)

        # Another way to sample actions:
        #   action = tf.random.categorical(logits, 1)
        # Will become clearer later why we don't use it.
        return action, value[0]


    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        #OR returns = np.array(rewards + [next_value[0]])
        #returns = np.zeros_like(rewards)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        returns = returns[:-1]


        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kls.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        # ------ Edit Here For 2 Actions: ----------
        #actions = actions_and_advantages[:,0:2]
        #advantages = actions_and_advantages[:,2]
        #print(  actions[:,0])
        #print(logits)

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True) # reduction=kls.Reduction.SUM)
        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages) # For Both Actions Now

        # ------------------ REVIEW OVER ----------------
        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)
        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return (policy_loss) - self.entropy_c * entropy_loss



    def _logits_loss_ppo(self, old_logits, logits, actions, advs, n_actions):
        actions_oh = tf.one_hot(actions, n_actions)
        actions_oh = tf.reshape(actions_oh, [-1, n_actions])
        actions_oh = tf.cast(actions_oh, tf.float32)
        actions_oh = tf.stop_gradient(actions_oh)

        new_policy = tf.nn.log_softmax(logits)
        old_policy = tf.nn.log_softmax(old_logits)
        old_policy = tf.stop_gradient(old_policy)

        old_log_p = tf.reduce_sum(old_policy * actions_oh, axis=1)
        log_p = tf.reduce_sum(new_policy * actions_oh, axis=1)
        ratio = tf.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(
            ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        advs = tf.stop_gradient(advs)
        advs = tf.cast(advs, tf.float32)
        surrogate = tf.minimum(ratio * advs, clipped_ratio * advs)
        return -tf.reduce_mean(surrogate) - self.entropy_c * kloss.categorical_crossentropy(new_policy, new_policy)
