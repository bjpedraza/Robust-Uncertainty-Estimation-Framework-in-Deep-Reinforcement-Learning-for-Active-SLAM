#!/usr/bin/env python

import gym
from gym import wrappers
import gym_gazebo
import os
import time
import numpy as np
import random
import time
import Network_Model_Bay_TEST as net
import liveplot
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import wandb

from distutils.dir_util import copy_tree
import json

#os.environ["WANDB_API_KEY"] = " Insert Your Own wandb ID "

def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)

if __name__ == '__main__':
    #modelTes = tf.keras.models.Sequential()
    #modelTes.add(tf.keras.layers.Input(1))
    #modelTes.add(tf.keras.layers.Dense(1))

    #VDP parameters
    input_dim=32
    n_channel=3
    num_kernels=[32, 32, 32, 32, 64, 64, 64, 128, 128, 128]
    kernels_size=[5, 3, 3, 3, 3, 3, 3, 3, 3, 1]
    maxpooling_size=[2, 2, 2, 2, 2]
    maxpooling_stride=[2, 2, 2, 2, 2]
    maxpooling_pad='SAME'
    class_num = 3 #<---------- Specify output neurons
    batch_size=50
    epochs=350
    lr=0.0002
    lr_end = 0.0001
    kl_factor=0.00001
    Random_noise=False
    Adversarial_noise=False
    HCV=0.01
    Black_box_attack=False
    PGDBlack_box_attack=True
    adversary_target_cls=3
    Targeted=True
    PGD_Adversarial_noise=False
    stepSize=1
    maxAdvStep=40
    Training=False
    Testing = False
    continue_training=False
    saved_model_epochs=300


    args = net.parser.parse_args()
    net.logging.getLogger().setLevel(net.logging.INFO)

    #env = gym.make('GazeboCircuitTurtlebotLidar-v0')
    #env = gym.make('GazeboCircuit2cTurtlebotCameraNnEnv-v0')
    env = gym.make('GazeboCircuit2TurtlebotLidar-v0')
    outdir = '/tmp/gazebo_gym_experiments'

    #env = gym.wrappers.Monitor(env, outdir, force=True)
    plotter = liveplot.LivePlot(outdir)
    last_time_steps = np.ndarray(0)


    # For CNN parameters
    minibatch_size = 32
    learningRate = 1e-3#1e6
    discountFactor = 0.95
    network_outputs = 3
    memorySize = 100000
    learnStart = 10000 # timesteps to observe before training
    EXPLORE = memorySize # frames over which to anneal epsilon
    INITIAL_EPSILON = 1 # starting value of epsilon
    FINAL_EPSILON = 0.01 # final value of epsilon
    explorationRate = INITIAL_EPSILON
    current_epoch = 0
    stepCounter = 0
    loadsim_seconds = 0
    img_rows, img_cols, img_channels = env.img_rows, env.img_cols, env.img_channels


    #----------------------- Load Weights for Testing: -----------------
    model_Critic = net.Critic_CNN(state_dim=3)
    model_Actor = net.Actor_VDP(kernel_size=kernels_size, num_kernel=num_kernels, pooling_size=maxpooling_size,
                                 pooling_stride=maxpooling_stride, pooling_pad=maxpooling_pad, units=class_num,
                                 name='vdp_cnn')

    model_Actor.load_weights("/home/bryan/Documents/Bay A2C GOOD/turtle_c2c_Actor_ep350_train")
    #model_Actor.dist.load_weights("/home/bryan/Documents/Bay A2C GOOD/turtle_Dist350_train")
    model_Critic.load_weights("/home/bryan/Documents/Bay A2C GOOD/turtle_c2c_Critic_ep350_train")
    agent = net.A2CAgent(model_Actor, model_Critic, learningRate)

    #env = gym.wrappers.Monitor(env, outdir, force=True)

    # FOR CNN reward
    last100Rewards = [0] * 100
    last100RewardsIndex = 0
    last100Filled = False
    myRewardList = []
    x_plot_avg = []
    y_plot_avg = []
    steps = 1500




    total_episodes = 10000
    highest_reward = 0
    epsilon_discount = 0.999 # 1098 eps to reach 0.1

    # For Done/Dynamic
    addRow = np.zeros((1,32,32,1))
    #addRow_action = np.zeros((1,2))
    batch_sz = 16
    ep_rewards = [0.0]
    wandb.init(entity = "bpedraz4", project="TEST_Bay_Framework_{}_lr_{}_kfold".format(total_episodes, learningRate))

    start_time = time.time()
    observation = env.reset()

    for x in range(1, total_episodes, 1):
        done = False
        cumulated_reward = 0

        mem_observation = np.zeros(( batch_sz, 32,32,1))
        actions = np.zeros(batch_sz)
        values = np.zeros(batch_sz)
        rewards = np.zeros(batch_sz)
        dones = np.zeros(batch_sz)
        batch_step = 0
        #temp_obs = np.empty((1,5))

        #act = np.zeros(3)
        #sig = np.zeros(3)
        myAvgCertain = []
        myAvgUncertain = []
        
        #table_mean = wandb.Table(columns=["1", "2", "3"])
        #table_sig = wandb.Table(columns=["1", "2", "3"])

        #while not done:
        for i in range(steps): #<------ steps=1500


            actions[batch_step], values[batch_step], snr_signal, uncertain, certain, var = agent.action_value(observation)
            newObservation, rewards[batch_step], dones[batch_step], info = env.step(actions[batch_step])

            
            myAvgUncertain.append(uncertain)
            myAvgCertain.append(certain)

            
            wandb.log({'Uncertain': uncertain,
                       'Certain': certain
                      })
            


            mem_observation[batch_step] = observation
            observation = newObservation
            cumulated_reward += rewards[batch_step]
            #ep_rewards[-1] += rewards[i]
            #env._flush(force=True)

            '''
            print('below')
            print(rewards[batch_step])
            print(var)
            time.sleep(8)
            print('above')
            '''


            if (batch_step == 15):
                batch_step = -1
                #mem_observation = np.zeros(( batch_sz, 32,32,1))
                #actions = np.zeros(batch_sz)
                #values = np.zeros(batch_sz)
                #rewards = np.zeros(batch_sz)
                #dones = np.zeros(batch_sz)
            '''
            if (i>900): #save 100 action_mean, sigma, snr_signal
                if (i==901):
                    textfile = open('Parameters.txt', 'w')
                    textfile.write(' Mean: ' + str(act))
                    textfile.write(', Uncertain: ' + str(sig))
                    textfile.write(', SNR_signal: ' + str(snr_signal))

                textfile.write('\n Mean: ' + str(act))
                textfile.write(', Uncertain: ' + str(sig))
                textfile.write(', SNR_signal: ' + str(snr_signal))

                if (i==1000):
                    textfile.close()
                    print("---- DONEEEEEE ----")
            '''


            if (dones[batch_step]):
                last_time_steps = np.append(last_time_steps, [int(i + 1)])
                #ep_rewards.append(0.0)
                #net.logging.info("Episode: %03d, Reward: %03d" % (len(ep_rewards) - 1, ep_rewards[-2]))

                last100Rewards[last100RewardsIndex] = cumulated_reward
                last100RewardsIndex += 1
                if last100RewardsIndex >= 100:
                    last100Filled = True
                    last100RewardsIndex = 0
                m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                h, m = divmod(m, 60)
                if not last100Filled:
                    print ("EP "+str(x)+" - {} steps".format(i+1)+" - CReward: "+str(round(cumulated_reward, 2))+"  Eps="+str(round(explorationRate, 2))+"  Time: %d:%02d:%02d" % (h, m, s))
                else :
                    print ("EP "+str(x)+" - {} steps".format(i+1)+" - last100 C_Rewards : "+str(int((sum(last100Rewards)/len(last100Rewards))))+" - CReward: "+str(round(cumulated_reward, 2))+"  Eps="+str(round(explorationRate, 2))+"  Time: %d:%02d:%02d" % (h, m, s))
        

                '''
                mem_observation = np.append(mem_observation, addRow, axis=0)

                values = np.append(values, 0)
                actions = np.append(actions, addRow_action, axis=0)
                lin_act = np.append(lin_act, addRow_action, axis=0)
                ang_act = np.append(ang_act, addRow_action, axis=0)

                dones = np.append(dones, 0)
                rewards = np.append(rewards, 0)
                '''

                observation = env.reset()
                #break
            '''
            else :
                mem_observation = np.append(mem_observation, addRow, axis=0)

                values = np.append(values, 0)
                actions = np.append(actions, addRow_action, axis=0)
                lin_act = np.append(lin_act, addRow_action, axis=0)
                ang_act = np.append(ang_act, addRow_action, axis=0)

                dones = np.append(dones, 0)
                rewards = np.append(rewards, 0)

                #state = ''.join(map(str, next_obs))
                #state = nextState
            '''

            if i % 2500 == 0:
                print("Frames = "+str(i))

            batch_step += 1
            



        myRewardList.append(cumulated_reward)
        if x % 50 == 0:
            print("PLOTTTTTTT")
            x_plot = range(x) #list( range(100) ) #for total number of episodes/epochs
            x_plot_avg.append(x)
            y_plot_avg.append( sum(myRewardList[x-50:x]) /50 )
            
            plt.plot(x_plot, myRewardList, color='blue')
            plt.plot(x_plot_avg, y_plot_avg , color='red')

            plt.pause(0.001)

            #Save Values:
            #np_x_plot = np.asarray(x_plot)
            np_rewards = np.asarray(myRewardList)
            np_x_plot_avg = np.asarray(x_plot_avg)
            np_rewards_avg = np.asarray(y_plot_avg)

            np.save('x_plot.npy', x_plot)
            np.save('rewards.npy', np_rewards)
            np.save('x_plot_avg.npy', np_x_plot_avg)
            np.save('reward_avg.npy', np_rewards_avg)

            wandb.log({"Cumulative Rewards": sum(myRewardList[x-50:x]) /50 })


        wandb.log({'Rewards': myRewardList[x-1],
                           'epoch': x,
                           'SNR_signal': snr_signal,
                           'Var': var,
                           'Avg Certain': sum(myAvgCertain)/steps,
                           'Avg Uncertain': sum(myAvgUncertain)/steps
                           })



        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        #print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))


        #net.logging.debug("[%d/%d] Losses: %s" % (x + 1, total_episodes, losses_actor))


    #Github table content
    #print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    #print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()

