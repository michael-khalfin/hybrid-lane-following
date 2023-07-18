#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneHoughConfig
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import String
import numpy as np
from math import log, sin, cos, atan, exp
import statistics
import time
import pandas as pd
import os
import tensorflow as tf
import sys
from reg_model import RegModel

class FollowLine:

    def __init__(self):
        rospy.loginfo("Follow line initialized")
        rospy.loginfo(os.path.abspath('follow_line_hough.py'))
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        self.vel_msg.angular.z = 0
        self.empty = Empty()
        self.twist = TwistStamped()
        self.rate = rospy.Rate(20)
        self.prev_right = 500
        self.prev_left = 0
        
        self.mean_list=[0]*7

        self.cols = 0
        self.rows = 0
        
        # red detect
        self.drive_pub = rospy.Publisher('/drive_enabled', Bool, queue_size=1)
        rospy.Subscriber('/red_detect_topic', String, self.red_callback)

        self.enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
        self.velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)
        self.median_pub = rospy.Publisher('median', Float64, queue_size=1)
        self.slope_pub = rospy.Publisher('slope', Float64, queue_size=1)
        self.wslope_pub = rospy.Publisher('wslope', Float64, queue_size=1)
        self.twist_pub = rospy.Publisher('twist',Float64,queue_size=1)


        self.config = None
        self.srv = Server(FollowLaneHoughConfig, self.dyn_rcfg_cb)

        self.config.canny_thresh_l = 20
        self.config.canny_thresh_u = 120

        # self.model = RegModel('../actor_ws/src/follow_lane_pkg/scripts/2023-07-05-15-23-41.bag', model_name=0)
        self.model = RegModel('/home/reu-actor/actor_ws/src/follow_lane_pkg/scripts/2023-07-06-13-55-27.bag', model_name=2)

        # while( not rospy.is_shutdown() ):
        rospy.Subscriber('/vehicle/twist', TwistStamped, self.vel_callback)
        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)

        self.rate.sleep()
        

    def dyn_rcfg_cb(self, config, level):
        self.config = config
        bool_val = Bool()
        bool_val.data = self.config.enable_drive
        
        self.drive_pub.publish(bool_val)
        return config

    def red_callback(self, msg):
        if len(msg.data) > 0:
            print(msg.data)
        with open("/home/reu-actor/actor_ws/src/follow_lane_pkg/scripts/data.txt", "a") as f:
            f.write(msg.data + "\n")

    def vel_callback(self, msg: TwistStamped):
        self.twist = msg.twist
        self.enable_car.publish(Empty())
        self.velocity_pub.publish(self.vel_msg)

    def camera_callback(self, msg: Image):
        #rospy.loginfo("Got image")
        if not self.config:
            rospy.logwarn("Waiting for config...")
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        #Resize the image before preprocessing
        (rows, cols, channels) = image.shape
        self.cols = cols
        self.rows=rows
        image = image[900:]
        (rows, cols, channels) = image.shape
        self.cols = cols
        self.rows=rows
        #Process the image to allow for hough lines to be drawn
        proc_image= self.preprocess(image)
        proc_image = cv2.resize(proc_image, (500,500))
        proc_image = proc_image[:,15:]

        image = cv2.resize(image, (500,500))
        #image=image[:,15:]
        (rows, cols, channels) = image.shape
        self.cols = cols
        self.rows=rows

        #Theta is set to 1 degree = pi/180 radians = 0.01745329251
        #threshold=100
        #rho=1
        #minLineLength=70
        #maxLineGap=4
        lines=[]
        lines = cv2.HoughLinesP(proc_image, 
                               rho=self.config.lines_rho, 
                               theta=0.01745329251, 
                               threshold=self.config.lines_thresh,
                               minLineLength=self.config.minLineLength,
                               maxLineGap=self.config.maxLineGap
                               )
        
        slopes=[]
        lengths=[]
        if lines is not None:
            lines=[l[0] for l in lines]
            
            for l in lines:
                slope=0
                try:
                    slope=(l[1]-l[3])/(l[0]-l[2])
                    length = ((l[1]-l[3])**2 + (l[0]-l[2])**2)**.5
                except:
                    rospy.logwarn("Divided by zero in slopes")
                    continue
                if abs(slope)<0.25 or abs(slope)>100:
                    continue
                
                if (((l[0]+l[2])/2<self.cols/2) and ((l[1]-l[3])**2 + (l[0]-l[2])**2)**.5>140):
                    cv2.line(image,(l[0],l[1]),(l[2],l[3]),(255,255,0),2)
                elif ((l[1]-l[3])**2 + (l[0]-l[2])**2)**.5<180:
                    cv2.line(image,(l[0],l[1]),(l[2],l[3]),(255,0,0),2)
                else:
                    cv2.line(image,(l[0],l[1]),(l[2],l[3]),(255,0,255),2)
                if isinstance(slope, np.float64) and not np.isnan(slope) \
                and isinstance(length, np.float64) and not np.isnan(length):
                    slopes.append(slope)
                    lengths.append(length)

        image=self.drive_2_follow_line(lines,image,slopes,lengths)

        cv2.imshow("My Image Window", image)
        cv2.imshow("BW Image", proc_image / 255.)
        cv2.waitKey(1)

    def preprocess(self,orig_image):
        """
        Inputs:
            orig_image: original bgr8 image before preprocessing
        Outputs:
            bw_image: black-white image after preprocessing
        """

        #blur_image = cv2.medianBlur(orig_image,self.config.blur_kernal)

        tf_img = np.copy(orig_image)
        tf_img = cv2.resize(tf_img, (200,200))

        tf_img = np.expand_dims(tf_img, axis=0)

        predicted_img = artist.predict(tf_img / 255.)[0]

        (rows, cols, channels) = predicted_img.shape
        self.cols = cols
        self.rows=rows

        predicted_img=255*predicted_img
        predicted_img=tf.keras.preprocessing.image.img_to_array(predicted_img,dtype='uint8')

        return predicted_img

    def drive_2_follow_line(self, lines,image,slopes,lengths):
        """
        Inputs:
            lines: list of Hough lines in form of [x1,y1,x2,y2]
        Outputs:
            Image for the purposes of labelling
        Description:
            Self drive algorithm to follow lane by rotating wheels to steer
            toward center of the lane
        """
        
        mid = self.cols // 2 
        

        if self.config.enable_drive:
            self.vel_msg.linear.x = 1
            center = mid

            if lines is not None:
                left=[(l[0]+l[2])/2 for l in lines if (((l[0]+l[2])/2<self.cols/2))]#and (l[1]-l[3])**2 + (l[0]-l[2])**2)**.5>140
                right=[(l[0]+l[2])/2 for l in lines if (((l[0]+l[2])/2>self.cols/2) and ((l[1]-l[3])**2 + (l[0]-l[2])**2)**.5>180)]


            new_left=0
            new_right = 0

            if len(left)!=0:
                new_left = sum(left)/len(left)
                if self.prev_left ==0 or abs(new_left - self.prev_left)<80:
                    self.prev_left=new_left


            if len(right)!=0:
                new_right = sum(right)/len(right)
                if self.prev_right ==500 or abs(new_right - self.prev_right)<80:
                    self.prev_right=new_right


            center = (self.prev_left+self.prev_right)//2
    

            cv2.line(image,(int(self.prev_left),1),(int(self.prev_left),self.rows),(255,255,0),2)

            cv2.line(image,(int(self.prev_right),1),(int(self.prev_right),self.rows),(0,255,255),2)

            cv2.line(image,(int(center),1),(int(center),self.rows),(0,255,0),2)

            cv2.line(image,(mid,1),(mid,self.rows),(0,0,255),2)


            ratio=self.vel_msg.angular.z = 0.9*(mid-center)/(mid) 
            self.mean_list.pop(0)
            self.mean_list.append(ratio)
            mean = statistics.mean(self.mean_list)


            if center<mid-7:
                # rospy.loginfo("Turn left!")
                self.vel_msg.angular.z = mean
            elif center>mid+7:
                # rospy.loginfo("Turn right!")
                self.vel_msg.angular.z = mean
            else:
                # rospy.logwarn("Go straight!")
                self.vel_msg.angular.z=0
            
        else:
            #rospy.logwarn(f"else state")
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0

        self.velocity_pub.publish(self.vel_msg)

        return image

if __name__ == '__main__':
    rospy.init_node('follow_line', anonymous=True)
    os.chdir("/home/reu-actor/actor_ws/src/jacks_pkg/scripts")
    artist = tf.keras.models.load_model("DL/artist")
    FollowLine()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
