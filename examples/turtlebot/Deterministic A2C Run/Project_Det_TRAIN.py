#!/usr/bin/env python3

import gym
from gym import wrappers
import gym_gazebo
import os
import time
import numpy as np
import random
import liveplot
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import wandb

from distutils.dir_util import copy_tree
import json

import Network_Model_Det_TRAIN as net

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

    args = net.parser.parse_args()
    net.logging.getLogger().setLevel(net.logging.INFO)

    #env = gym.make('GazeboCircuitTurtlebotLidar-v0')
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
    loadsim_seconds = 0
    img_rows, img_cols, img_channels = env.img_rows, env.img_cols, env.img_channels


    model_A2C = net.Model_A2C(num_actions=3)
    agent = net.A2CAgent(model_A2C, learningRate)


    # FOR CNN reward
    last100Rewards = [0] * 100
    last100RewardsIndex = 0
    last100Filled = False
    myRewardList = []
    x_plot_avg = []
    y_plot_avg = []
    steps = 1000

    total_episodes = 10000
    highest_reward = 0
    epsilon_discount = 0.999 # 1098 eps to reach 0.1

    # For Done/Dynamic
    addRow = np.zeros((1,32,32,1))
    #addRow_action = np.zeros((1,2))
    batch_sz = 16
    ep_rewards = [0.0]

    #state = ''.join(map(str, next_obs[0]))
    angleRange = np.pi/3
    #print(next_obs)
    #print(len(next_obs[0]))
    #print(len(next_obs[0][0]))
    #print(type(next_obs))

    wandb.init(entity = "bpedraz4", project="Train_Tradional_A2C_{}_lr_{}_kfold".format(total_episodes, learningRate))

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


        #while not done:
        for i in range(steps): # <----- # of steps

            actions[batch_step], values[batch_step] = agent.action_value(observation)
            newObservation, rewards[batch_step], dones[batch_step], info = env.step(actions[batch_step])

            mem_observation[batch_step] = observation
            observation = newObservation


            cumulated_reward += rewards[batch_step]
            #ep_rewards[-1] += rewards[i]

            #env._flush(force=True)


            if (batch_step == 15): #<---- batch size: learn after 250 steps
                if(i>250):
                    #add noise here during testing phase:

                    _, next_value = agent.action_value( observation )
                    returns, advs = agent._returns_advantages(rewards, dones, values, next_value)
                    acts_and_advs = np.concatenate([actions[:,None], advs[:, None]], axis=-1)
                    # A trick to input actions and advantages through same API.
                    #acts_and_advs1 = np.concatenate([actions[:,0][:,None], advs[:, None]], axis=-1)
                    #acts_and_advs2 = np.concatenate([actions[:,1][:,None], advs[:, None]], axis=-1)
                    # Performs a full training step on the collected batch.
                    # Note: no need to mess around with gradients, Keras API handles it.
                    # combine the actions and advantages into a combined array for passing to
                    # actor_loss function
                    losses = agent.model.train_on_batch(mem_observation, [acts_and_advs, returns])
                    wandb.log({'loss': np.asarray(losses[0])})

                batch_step = -1
                mem_observation = np.zeros(( batch_sz, 32,32,1))
                actions = np.zeros(batch_sz)
                values = np.zeros(batch_sz)
                rewards = np.zeros(batch_sz)
                dones = np.zeros(batch_sz)


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



        #SAVE AND PLOT DATA
        myRewardList.append(cumulated_reward)
        if x % 50 == 0:
            #SAVE model weights and monitoring data every 50 epochs.
            agent.model.save_weights('/tmp/turtle_c2c_Actor_ep'+str(x)+"_train")
            
            #copy_tree(outdir,'/tmp/turtle_c2c_dqn_ep'+str(x))
            #save simulation parameters.
            parameter_keys = ['explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_outputs','current_epoch','stepCounter','EXPLORE','INITIAL_EPSILON','FINAL_EPSILON','loadsim_seconds']
            parameter_values = [explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_outputs, x, i, EXPLORE, INITIAL_EPSILON, FINAL_EPSILON,s]
            parameter_dictionary = dict(zip(parameter_keys, parameter_values))
            with open('/tmp/turtle_c2c_dqn_ep'+str(x)+'.json', 'w') as outfile:
                json.dump(parameter_dictionary, outfile)


            #PLOT
            x_plot = range(x) #list( range(50) ) #for total number of episodes/epochs
            x_plot_avg.append(x)
            y_plot_avg.append( sum(myRewardList[x-50:x]) /50 )

            '''
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
            '''

            wandb.log({"Cumulative Rewards": sum(myRewardList[x-50:x]) /50 })


        wandb.log({'Rewards': np.asarray(myRewardList[x-1]), #<-- numpy type
                           'epoch': x
                           })


        #wandb.log({'Rewards': myRewardList[x-1]})


        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        
        net.logging.debug("[%d/%d] Losses: %s" % (x + 1, total_episodes, losses))
        


        if (x == 200):
            total_time = ( time.time() - start_time )

            wandb.log({'Total Training Time': total_time })
            np.save('Total_Test_Time_DET.npy', total_time)

            #break;


    #Github table content
    #print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    #print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
