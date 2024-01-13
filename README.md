# Robust-Uncertainty-Estimation-Framework-in-Deep-Reinforcement-Learning-for-Active-SLAM
This repository contains work on deep reinforcement learning for a TurtleBot robot using a custom-built architecture called Bayesian A2C. The project used the Gazebo simulator and an open-source OpenAI gym extension named [gym-gazebo](https://github.com/erlerobot/gym-gazebo). It was constructed using the Linux OS, TensorFlow, and the Robot Operating System (ROS).

## About
Using simultaneous localization and mapping (SLAM) the project introduce a new approach to address robust navigation and mapping of robot actions using Bayesian Actor-Critic (A2C) reinforcement learning that generates predicted actions and quantifies the uncertainty associated with these actions. The approach incorporates Bayesian inference and optimizes the variational posterior distribution over the unknown model parameters using the evidence lower bound (ELBO) objective. The first-order Taylor series approximation was used to estimate the mean and covariance of the variational distribution when passed through non-linear functions in the A2C model. The propagated covariance estimates the robot’s action uncertainty at the output of the actor-network.

The project used the Gazebo simulator as the environment and an open-source OpenAI gym extension called gym-gazebo. The Turtlebot3 robot was used as the agent for simulations to train the Bayesian network model and a deterministic model. After the simulation phase or after training, both models are deployed into a test environment and compared to each other in terms of their performances, robustness, convergence, and stability through the rewards received. This technique demonstrated superior performance, remarkable results and stability within various noisy simulated scenarios interrupted by Gaussian and adversarial noise.

This demonstrated the superior robustness of the proposed Bayesian A2C model when exploring environments with high levels of noise compared to deterministic alternatives. Furthermore, the proposed framework has the potential for various applications where robustness and uncertainty quantification are crucial.


## Journal Article
This project's **full** paper is currently in the process of being submitted for the public by ProQuest, as well as another publisher. 

While a shorter version of the project's idea was already submitted through the IEEE Conference on Artificial Intelligence (CAI) 2023, the **extended** version is set to be published publicly later this year in **2024**.

However, for now the shorter version can be found and read on the ieeexplore website and can be accessed at [Robust Active Simultaneous Localization and Mapping Based on Bayesian Actor-Critic Reinforcement Learning](https://ieeexplore.ieee.org/document/10195002).



## Software and Components
Linux OS

TensorFlow 2

NVIDIA GPU Drivers

Robot Operating System (ROS) Melodic

[Gym-Gazebo](https://github.com/erlerobot/gym-gazebo) Repository: an extension of OpenAI gym 

Deep Reinforement Learning

Python

CUDA Toolkit

cuDNN SDK

Simulated Depth-Camera

Simulated LiDAR

Slam_gmapping (edited)

OpenCV

Cv_bridge that bridges between ROS messages and OpenCV

Gazebo Simulator

wandb


