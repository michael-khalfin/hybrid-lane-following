# Authors: Jack Volgren
# Jun 13
# The purpose of this class is to recieve images and return a angular z value for other classes to publish to /cmd_vel

import rospy
from geometry_msgs.msg import Twist
import tensorflow as tf
import numpy as np
import cv2 as cv
import os

class DL_Path:
    
    def __init__(self):
        rospy.loginfo("MODEL INITIALIZED...")
        self.model = tf.keras.models.load_model('path_finder')
        self.min_thresh = 10
        self.yaw=0

    def predict(self, img):
        
        SCALAR = 1
        img = img[900:]
        img = cv.resize(img, (50, 50))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kernel = np.ones((3,3),np.uint8)
        img = cv.dilate(img, kernel)
        _, img = cv.threshold(img, self.min_thresh, 255, cv.THRESH_BINARY)
        img = cv.dilate(img, kernel)
        disp = img
        img = img/255
        img = np.expand_dims(img, axis=0)
        pct_white = np.sum(disp >= self.min_thresh) / (50 * 50)
        if pct_white >= 0.18:
            rospy.loginfo("TOO BRIGHT")
            rospy.loginfo(str(pct_white))
            self.min_thresh += 5 if self.min_thresh < 250 else 2
            #return Twist()
        elif pct_white <= 0.05:
            rospy.loginfo("TOO DARK")
            rospy.loginfo(str(pct_white))
            self.min_thresh -= 5 if self.min_thresh > 5 else 2
            #return Twist()
        else:
            # twist_msg = Twist()
            self.yaw = self.model.predict(img)[0][0]

        cv.imshow("Window", disp)
        cv.waitKey(1)

        twist_msg = Twist()
        twist_msg.angular.z = self.yaw
        twist_msg.linear.x = 2.2


        return twist_msg
    
    def decode_prediction(self, px, py):
        if px < 50:
            pxm50 = -(abs(px-50)**(1/3))
        else:
            pxm50 = (px-50)**(1/3)

        pvel = -((py-95)/15)
        pyaw = -pxm50/(6*pvel)

        return pvel, pyaw

if __name__ == "__main__":
    # Another test
    img = cv.imread("../bags/img/training/ds02/-0.089884452521801_656.png")
    direction_model = DL_Direction("direction_finder_model")
    print(direction_model.predict(img))