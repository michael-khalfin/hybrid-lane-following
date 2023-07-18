#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from std_msgs.msg import String 
from dynamic_reconfigure.server import Server
from follow_lane_pkg.cfg import DetectRedConfig
from geometry_msgs.msg import Twist, TwistStamped
import time

vel_msg = Twist()
red_msg = String()

bridge = CvBridge() 
state = 'NO_RED'
global cf

n_laps = 0
tot_frames = 0
sum_steer_err = 0
avg_steer_err = 0
t0 = 0

drive = False

# for comfortability
# comf_count = 0
prev_velocity = 0.0
prev_time = 0.0
comf_frames = 1
comf_sum = 0
count = 0
comf_avg = 0 
    

def dyn_rcfg_cb(config, level):
  global cf, perimeter
  cf = config

  perimeter = 71.43 if cf.inner_lane else 86.32 if cf.outer_lane else 86.32 
  print(f"Perimeter is {perimeter}")
  return config # this is required


def drive_cb(msg):
    global drive, t0
    if drive == False and msg.data == True:
        drive = True
        t0 = rospy.Time.now().to_sec()
    elif drive == True and msg.data == False: 
        drive = False
    


def image_callback(ros_image): #get contour image

    global bridge, cols, state, n_laps, tot_frames, t0, perimeter,sum_steer_err, comf_avg, comf_frames, comf_sum

    pix_dist_to_center = 380
    try: #convert ros_image into an opencv-compatible imageadi
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8") #get image from the camera, and turn it into a CV Accesible image
    except CvBridgeError as e:
        print(e)

    cv_image = cv2.resize(cv_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    (rows,cols,channels) = cv_image.shape
    # print(f"original shape: {cv_image.shape}")
    cv2.imshow("RGB image", cv_image)
    hsv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    # print(f"normal shape: {cv_image.shape}")

  
    tcolLower = (cf.hue_l, cf.sat_l, cf.val_l) # Lower bounds of H, S, V for the target color
    tcolUpper =  (cf.val_h, cf.sat_h, cf.val_h)                              # Upper bounds of H, S, V for the target color <====
    mask4y = cv2.inRange(hsv_img, tcolLower, tcolUpper) # <=====
    mask4y = mask4y[500:, 400:]
    cv2.imshow("Red Mask", mask4y)
    num_white_pix = cv2.countNonZero(mask4y)                # <====
    white_pct = (100 * num_white_pix) / (rows * cols)

    tot_frames+=1
    # cv2.imshow("CV IMAGE", cv_image)
    # steer_err_cb(cv_image)
    # cropped_img = cv_image
    # cropped_img = cv_image[500:, 400:]
    cropped_img = cv_image[500:,:]
    # print(f"cropped_img shape: {cropped_img.shape}")
    gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) #turns image to grey scale --> no more color
    ret, bw_image = cv2.threshold(gray_image, # input image
                                cf.thresh,     # threshol_value, make dynamic
                                255,        # max value in image
                                cv2.THRESH_BINARY) # threshold type
    #thresh determined by the dynamic reconfig ... This takes the thresh values and turns the image to black and white based on that threshold.

    #find the contour 
    contours, heirarchy = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #find contours ---> contours are the boundaries of the part of the image we want
    x = 0
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area > max_area:
            max_area = area
            max_c = c

    if x > 600:

        cv2.drawContours(cropped_img, max_c, -1, (0,0,255), 10)

    # finding centroids of max contour and draw a circle there
    # https://www.geeksforgeeks.org/python-opencv-find-center-of-contour/
    # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        M = cv2.moments(max_c)
        try:
            cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    # https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
            cv2.circle(cropped_img, (cx,cy), 10, (0,0,0), -1) # -1 fill the circle

    #show the image
            

            center_loc = cx - pix_dist_to_center
            cv2.circle(cropped_img, (center_loc,cy), 10, (0,250,0), -1) # -1 fill the circle
            vehicle_center = int(cols / 2)
            # print(f"vehicle_center is {vehicle_center}")
            # print(f"countour - vehicle_center is {cx - vehicle_center}")
            cv2.line(cropped_img, (vehicle_center, 0), (vehicle_center, rows), (0,0,0), 3)
            cv2.imshow('Cropped with circle', cropped_img)
            # print(f"cropped shape: {cropped_img.shape}")

            mid = cols / 2
        # steer_err = mid - true_middle 
        # cv2.imshow("RGB Cropped Image", cropped_image)
            steer_err = abs((cropped_img.shape[1] / 2) - center_loc)
            sum_steer_err += steer_err
        except:
            pass

    # comfortability


    if(state == 'NO_RED' and white_pct >= 0.15):

        state = 'RED'
        print("RED DETECTED")
        rospy.loginfo(f"Red Detected, Percentage: {white_pct}%")

    elif(state == 'RED' and white_pct <= 0.1):
        state = "NO_RED"
        
        
        avg_steer_err = sum_steer_err / tot_frames #sum_steer_err/tot_frames
        
        t1 = rospy.Time.now().to_sec()
        dt = t1 - t0
        s = perimeter/dt # meter per second
        km_h = s*3.6 # 3,600m seconds / 1,000 meters (1km)
        miles_h = km_h * 0.62137119223733
        comf_avg = comf_sum / comf_frames
        red_msg.data = f"** Lap#{n_laps}, t taken: {dt:.2f} seconds\n    Avg speed: {s:.2f} m/s, {km_h:.2f} km/h, {miles_h: .2f} miles/h\n    Avg steer centering err = {avg_steer_err}\n  Comfortability: {comf_avg}\n"
        n_laps+=1           
        red_detect_pub.publish(red_msg)
        red_msg.data = ''

        t0 = rospy.Time.now().to_sec()
        sum_steer_err = 0
        tot_frames = 0
        comf_frames = 0
        comf_sum = 0
    cv2.waitKey(3)
    
def jerk_cb(msg):
    global prev_velocity, prev_time, prev_velocity_derivative, count, comf_frames, comf_sum
    # Get current time
    current_time = rospy.Time.now().to_sec()

    # Calculate time difference
    dt = (current_time - prev_time)

    # Calculate current velocity

    current_velocity = msg.twist.angular.z

    count+=1
    if count >= 60:
        # Calculate velocity derivative
        ang_z_deriv = (current_velocity - prev_velocity) / dt
        
        # Update previous values
        prev_velocity = current_velocity
        prev_time = current_time
        count = 0
        comf_frames+=1 
        comf_sum += abs(ang_z_deriv)

if __name__ == '__main__':

    rospy.init_node('red_detect', anonymous=True)  #makes the rospy node, calls it follow_lane

    rospy.Subscriber('/drive_enabled', Bool, drive_cb) #subscribe to topic to see if enable_drive is true
    rospy.Subscriber('/camera/image_raw', Image, image_callback)

    rospy.Subscriber('/vehicle/twist', TwistStamped, jerk_cb)

    red_detect_pub = rospy.Publisher('/red_detect_topic', String, queue_size=1) #publish red detect messages to the follow_lane node

    srv = Server(DetectRedConfig, dyn_rcfg_cb)
    try:
        rospy.spin() #loop code until it is shutdown by the user
    except rospy.ROSInterruptException:
        pass