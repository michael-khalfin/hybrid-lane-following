#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneLukeConfig
from geometry_msgs.msg import Twist
import numpy as np
from math import log


class FollowLine:

    def __init__(self):
        rospy.loginfo("Follow line initialized")
        self.bridge = CvBridge()
        self.vel_msg = Twist()
        self.empty = Empty()

        self.cols = 0 # set later
        self.rows = 0

        self.enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
        self.velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)

        self.config = None
        self.srv = Server(FollowLaneLukeConfig, self.dyn_rcfg_cb)

        rospy.Subscriber('/camera/image_raw', Image, self.camera_callback)

    def dyn_rcfg_cb(self, config, level):
        rospy.logwarn("Got config")
        self.config = config
        return config

    def camera_callback(self, msg: Image):
        #rospy.loginfo("Got image")
        if not self.config:
            rospy.logwarn("Waiting for config...")
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        #REsize the image before preprocessing
        image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        
        image = image[504:]
        (rows, cols, channels) = image.shape
        self.cols = cols
        self.rows=rows
        #Process the image to allow for hough lines to be drawn
        proc_image = self.preprocess(image, 210)
        # left_image = proc_image[:self.rows,:self.cols//2]
        # right_image = proc_image[:self.rows,self.cols//2:]
        contours,hierarchy = cv2.findContours(proc_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cx1=0
        cy1=0
        cx2=0
        cy2=0
        if contours:
            #contours = [print(c) for c in contours]
            max_c=0
            max2_c=0
            max_area = 0
            max_area2=0
            for c in contours:
                #print(cv2.contourArea(c))
                M = cv2.moments(c)
                area = cv2.contourArea(c)
                if area > max_area:
                    temp_x,temp_y = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    if ((temp_x-cx1)**2+(temp_y-cy1)**2)**0.5>500:
                        cx2,cy2=cx1,cy1
                        if M['m00'] != 0:
                            cx1,cy1 = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                        
                        max_area2=max_area
                        max2_c=max_c
                        max_area = area
                        max_c = c
                    else:
                        
                        if M['m00'] != 0:
                            cx1,cy1 = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                        max_area = area
                        max_c = c
                elif area>max_area2:
                    temp_x,temp_y = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    if ((temp_x-cx1)**2+(temp_y-cy1)**2)**0.5>500:
                        max2_c=c
                        max_area2=area
                        if M['m00'] != 0:
                            cx2,cy2 = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    else:
                        pass
            
            cx=(cx1+cx2)//2
            cy=(cy1+cy2)//2
            #draw the obtained contour lines(or the set of coordinates forming a line) on the original image
            cv2.drawContours(image, max_c, -1, (0,0,255), 10)
            if max2_c is not None:
                cv2.drawContours(image, max2_c, -1, (0,0,255), 10)
            cv2.circle(image, (cx,cy), 10, (0,255,0), -1) # -1 fill the circle
            cv2.circle(image, (cx1,cy1), 10, (0,255,0), -1) # -1 fill the circle
            cv2.circle(image, (cx2,cy2), 10, (0,255,0), -1) # -1 fill the circle

        # left_contours,hierarchy = cv2.findContours(left_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # right_contours,hierarchy=cv2.findContours(right_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # if left_contours and right_contours:

        #     left_max_c=0
        #     left_max_area = 0
        #     for c in left_contours:
        #         #print(cv2.contourArea(c))
        #         M = cv2.moments(c)
        #         area = cv2.contourArea(c)
        #         if area > left_max_area:
        #             if M['m00'] != 0:
        #                 lcx,lcy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        #             left_max_area = area
        #             left_max_c = c

        #     right_max_c=0
        #     right_max_area = 0
        #     for c in right_contours:
        #         #print(cv2.contourArea(c))
        #         M = cv2.moments(c)
        #         area = cv2.contourArea(c)
        #         if area > right_max_area:
        #             if M['m00'] != 0:
        #                 rcx,rcy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        #             right_max_area = area
        #             right_max_c = c
        #     #draw the obtained contour lines(or the set of coordinates forming a line) on the original image
        #     cv2.drawContours(image, left_max_c, -1, (0,0,255), 10)
        #     cv2.drawContours(image, right_max_c, -1, (255,0,0), 10)

        #     rospy.logwarn("Reached this point")
            
        #     cx=int((lcx+rcx)/2)
        #     cy=int((lcy+rcy)/2)
        #    cv2.circle(image, (cx,cy), 10, (0,255,0), -1) # -1 fill the circle
            image = self.drive_2_follow_line(cx,image)
                

        
        cv2.imshow("My Image Window", image)
        cv2.imshow("BW_Image", proc_image)

        # cv2.imshow("Left Image", left_image)
        # cv2.imshow("Right Image", right_image)

        cv2.waitKey(1)

    # def preprocess(self, orig_image):
    #     """
    #     Inputs:
    #         orig_image: original bgr8 image before preprocessing
    #     Outputs:
    #         bw_image: black-white image after preprocessing
    #     """

    #     blur_image = cv2.medianBlur(orig_image,self.config.blur_kernal)
        

    #     (rows, cols, channels) = blur_image.shape
    #     self.cols = cols
    #     self.rows=rows
    #     blur_image=cv2.cvtColor(blur_image,cv2.COLOR_BGR2GRAY)
        
    #     canny_image = cv2.Canny(blur_image,self.config.canny_thresh_l,self.config.canny_thresh_u,apertureSize=3)

    #     blob_size=self.config.dilation_base
    #     dilation_size=(2*blob_size+1,2*blob_size+1)
    #     dilation_anchor=(blob_size,blob_size)
    #     dilate_element=cv2.getStructuringElement(cv2.MORPH_RECT,dilation_size,dilation_anchor)
    #     bw_image=cv2.dilate(canny_image,dilate_element)
    #     return bw_image

    def preprocess(self, orig_image, thresh) -> 'blackwhite_image':
        """
        Inputs:
            orig_image: original bgr8 image before preprocessing
        Outputs:
            bw_image: black-white image after preprocessing
        """

        orig_image = cv2.medianBlur(orig_image,9)

        (rows, cols, channels) = orig_image.shape
        self.cols = cols

        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        ret, bw_image = cv2.threshold(gray_image, # input image
                                        thresh,     # threshold value
                                        255,        # max value in image
                                        cv2.THRESH_BINARY) # threshold type

        num_white_pix = cv2.countNonZero(bw_image)
        total_pix = rows * cols
        percent_white = num_white_pix / total_pix * 100

        thresh_max = 248
        thresh_min = 0
        change = 64

        while (percent_white > self.config.percent_white_max) or \
        (percent_white < self.config.percent_white_min):
            if percent_white > self.config.percent_white_max:
                thresh += change
                if thresh > thresh_max:
                    thresh = thresh_max
            elif percent_white < self.config.percent_white_min:
                thresh -= change
                if thresh < thresh_min:
                    thresh = thresh_min
            else:
                break
            ret, bw_image = cv2.threshold(gray_image, # input image
                                            thresh,     # threshold value,
                                            255,        # max value in image
                                            cv2.THRESH_BINARY) # threshold type
            num_white_pix = cv2.countNonZero(bw_image)
            percent_white = num_white_pix / total_pix * 100
            change /= 2
            if change < 2:
                break
        blob_size=10#self.config.dilation_base
        dilation_size=(2*blob_size+1,2*blob_size+1)
        dilation_anchor=(blob_size,blob_size)
        dilate_element=cv2.getStructuringElement(cv2.MORPH_RECT,dilation_size,dilation_anchor)
        bw_image=cv2.dilate(bw_image,dilate_element)
        rospy.loginfo(f"The percent white is: {percent_white}%")
        rospy.loginfo(f"The Threshold is: {thresh}")
        
        return bw_image
    

    def drive_2_follow_line(self, cx,image):
        """
        Inputs:
            lines: list of Hough lines in form of [x1,y1,x2,y2]
        Outputs:
            Image for the purposes of labelling
        Description:
            Self drive algorithm to follow lane by rotating wheels to steer
            toward center of the lane
        """

        mid = self.cols / 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.enable_car.publish(self.empty)
        

        if self.config.enable_drive:
            self.vel_msg.linear.x=self.config.speed
            if self.config.speed<2:
                strength_ratio = 0.8*(mid-cx)/mid
            else:
            #turn harder at faster speeds
                strength_ratio = 0.8*(mid-cx)/mid
            if cx<mid-100:
                cv2.putText(image,f"Turn Left",(10,self.rows-10), font, 1,(125,125,125),2,cv2.LINE_AA)
                self.vel_msg.angular.z=strength_ratio
                self.velocity_pub.publish(self.vel_msg)
            elif cx>mid+100:
                cv2.putText(image,f"Turn Right",(10,self.rows-10), font, 1,(125,125,125),2,cv2.LINE_AA)
                self.vel_msg.angular.z=strength_ratio
                self.velocity_pub.publish(self.vel_msg)
            else: 
                cv2.putText(image,f"Go Staight",(10,self.rows-10), font, 1,(125,125,125),2,cv2.LINE_AA)
                self.vel_msg.angular.z=0
                self.velocity_pub.publish(self.vel_msg)

                

        else:
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0

        self.velocity_pub.publish(self.vel_msg)
        return image

if __name__ == '__main__':
    rospy.init_node('follow_line', anonymous=True)
    FollowLine()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass