import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2
import math
from std_msgs.msg import Bool, Float64

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from cv_bridge import CvBridge, CvBridgeError

from gym.utils import seeding


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

class GazeboCircuit2TurtlebotLidarNnEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuit2TurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = spaces.Discrete(3) #F,L,R
        self.observation_space = spaces.Discrete(3)
        self.reward_range = (-np.inf, np.inf)

        self._seed()
        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 1
        self.angleRange = 0.5

        self.resSLAM = rospy.Publisher('resetSLAM', Bool, queue_size=1)
        self.rate = rospy.Rate(1000)
        self.entropy = rospy.Subscriber('entropy', Float64, callback)
    '''
    def calculate_observation(self,data):
        min_range = 0.2
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return data.ranges,done
    '''
    def calculate_observation(self,data):
        min_range = 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        entropyVal = rospy.wait_for_message('/slam_gmapping/entropy', Float64)
        print(entropyVal.data)

        '''
        max_ang_speed = 0.3
        ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)
        '''
        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.10
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 0.2
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -0.2
            self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        done = self.calculate_observation(data)
        image_data = None
        success=False
        cv_image = None
        while image_data is None or success is False:
            try:
                
                #image_data = rospy.wait_for_message('/camera/depth/points', PointCloud2, timeout=5)
                image_data = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data,  desired_encoding="passthrough")

                '''
                #temporal fix, check image is not corrupted
                if np.isnan(cv_image).any():
                    #pass
                    cv_image = np.nan_to_num(cv_image)
                else:
                    success = True
                '''

                #if np.isnan(cv_image).all():
                #    continue;
                if np.isnan(cv_image).any():
                    cv_image = np.nan_to_num(cv_image)
                if not (cv_image[h//2,w//2]==178):
                    success = True
                else:
                    print("/camera/depth/image_raw ERROR, retrying")
                    pass

                    #print("/camera/rgb/image_raw ERROR, retrying")
            except CvBridgeError:
                print(CvBridgeError)
                pass
                '''
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                
                #temporal fix, check image is not corrupted
                if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                    success = True
                else:
                    pass
                    #print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass
                '''

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")


        '''
        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
        else:
            reward = -200
        '''
        #------- REWARD Function: ---------
        if not done:
            if action == 0: #go forward
                reward = 2 + math.tanh(0.01 / entropyVal.data)
            else:
                reward = -0.05 + math.tanh(0.01 / entropyVal.data)
        else:
            reward = -100

        #cv_image = np.array(cv_image, dtype=np.float32)
        cv_image = cv_image.astype('float32')
        #print(cv_image)
        #print("step")

        #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))

        '''
        #ADD Noise to Image:
        gauss = np.random.normal(0, 0.15, cv_image.shape)
        gauss = gauss.reshape(cv_image.shape)
        cv_image = cv_image + gauss
        '''

        state = cv_image.reshape(1, cv_image.shape[0], cv_image.shape[1], 1)
        return state, reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        
        #rospy.wait_for_message('/slam_gmapping/entropy', Float64)        
        try:
            self.resSLAM.publish(True)
            self.rate.sleep()
        except rospy.ROSInterruptException as e:
            print ("/slam_gmapping/entropy service call failed")

        
        image_data = None
        success=False
        cv_image = None
        while image_data is None or success is False:
            try:
                
                #image_data = rospy.wait_for_message('/camera/depth/points', PointCloud2, timeout=5)
                image_data = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, desired_encoding="passthrough")

                #temporal fix, check image is not corrupted
                if np.isnan(cv_image).any():
                    #cv_image = np.nan_to_num(cv_image)
                    continue;
                if not (cv_image[h//2,w//2]==178):
                    success = True
                else:
                    print("/camera/depth/image_raw ERROR, retrying")
                    pass

                    #print("/camera/rgb/image_raw ERROR, retrying")
            except CvBridgeError:
                print(CvBridgeError)
                pass

                '''
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                
                #temporal fix, check image is not corrupted
                if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                    success = True
                else:
                    print("/camera/rgb/image_raw ERROR, retrying")
                    pass

            except:
                pass
                '''


        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")


        #cv_image = np.array(cv_image, dtype=np.float32)
        cv_image = cv_image.astype('float32')
        #print(cv_image)
        #print("rest")

        #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))


        #imgplot = plt.imshow(cv_image)
        #plt.show()
        #print("random:")
        

        '''
        #ADD Noise to Image:
        gauss = np.random.normal(0, 0.15, cv_image.shape)
        gauss = gauss.reshape(cv_image.shape)
        cv_image = cv_image + gauss
        '''

        #imgplot = plt.imshow(cv_image)
        #plt.show()

        state = cv_image.reshape(1, cv_image.shape[0], cv_image.shape[1], 1)
        return state
