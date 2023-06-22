# Authors: Jack Volgren
# Jun 13
# The purpose of this class is to recieve images and return a angular z value for other classes to publish to /cmd_vel

import rospy
from geometry_msgs.msg import Twist
import tensorflow as tf
import numpy as np
import cv2 as cv
import os

class DL_Direction:
    
    def __init__(self):
        rospy.loginfo("MODEL INITIALIZED...")
        self.model = tf.keras.models.load_model('angular_model')

    def predict(self, img):
        
        SCALAR = 1

        unstandardize = lambda zScore: (zScore * 0.18732622265815735) - 0.0658792182803154

        img = img[0:][720:][0:]
        img = cv.resize(img, (100, 100))
        img = img.astype("float32")
        img /= 255
        img = np.expand_dims(img, axis=0)

        twist_msg = Twist()
        prediction = self.model.predict(img)[0][0]
        prediction = unstandardize(prediction)
        twist_msg.angular.z = prediction * SCALAR

        # Remove this in the future, purely testing purposes
        twist_msg.linear.x = 2

        return twist_msg
    
if __name__ == "__main__":
    # Another test
    img = cv.imread("../bags/img/training/ds02/-0.089884452521801_656.png")
    direction_model = DL_Direction("direction_finder_model")
    print(direction_model.predict(img))