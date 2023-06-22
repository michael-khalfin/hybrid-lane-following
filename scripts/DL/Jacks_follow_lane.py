#!/usr/bin/env python3

import rospy
import cv2
from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import FollowLaneConfig
from geometry_msgs.msg import Twist
from dl_path_class import DL_Path
import os

empty = Empty()
bridge = CvBridge()
it = 0

def camera_callback(ros_image):
    global bridge, velocity_pub, dl_lane_finder, enable_car, it
    '''if it != 3:
        it += 1
        enable_car.publish(empty)
        return'''
    it = 0
    twist_msg = Twist()
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)

    twist_msg = dl_lane_finder.predict(cv_image)

    rospy.loginfo(twist_msg)
    
    enable_car.publish(empty)
    velocity_pub.publish(twist_msg)

if __name__ == '__main__':
    rospy.loginfo("Follow line initialized")
    rospy.init_node('Jack_follow_lane', anonymous=True)
    rospy.Subscriber('/camera/image_raw', Image, camera_callback)
    enable_car = rospy.Publisher('/vehicle/enable', Empty, queue_size=1)
    velocity_pub = rospy.Publisher('/vehicle/cmd_vel', Twist, queue_size=1)
    os.chdir("../../../home/reu-actor/actor_ws/src/follow_lane_pkg/scripts/DL")
    print(os.getcwd())
    dl_lane_finder = DL_Path()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass