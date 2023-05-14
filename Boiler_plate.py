#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

class FollowingNode:

    def __init__(self):
        rospy.init_node('following_node')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.target_depth_pub = rospy.Publisher('/target_depth', Float32, queue_size=1)
        kp = 0.5  # replace tuned values -Adi
        ki = 0.01 # replace tuned values -Adi
        kd = 0.2  # replace tuned values -Adi
        self.pid_controller = PIDController(kp=kp, ki=ki, kd=kd)
        

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Apply YOLO model to get the segmentation mask of the human
        mask = yolo_model.predict(cv_image)
        # Convert the mask to a binary image
        mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
        # Get the centroid of the mask
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
        # Publish the target depth based on the depth image
        depth_image = rospy.wait_for_message('/camera/depth/image_rect_raw', Image)
        depth_array = np.array(self.bridge.imgmsg_to_cv2(depth_image), dtype=np.float32)
        target_depth = depth_array[cy, cx]
        self.target_depth_pub.publish(target_depth)

    def depth_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth_array = np.array(depth_image, dtype=np.float32)
        # Compute the error between the target depth and the actual depth
        target_depth = rospy.wait_for_message('/target_depth', Float32)
        error = target_depth.data - depth_array[cy, cx]
        # Compute the PID output
        output = self.pid_controller.compute(error)
        # Publish the velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2
        cmd_vel.angular.z = output
        self.cmd_vel_pub.publish(cmd_vel)

class PIDController:

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0
        self.prev_error = 0

    def compute(self, error):
        self.error_sum += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.error_sum + self.kd * derivative
        self.prev_error = error
        return output

if __name__ == '__main__':
    try:
        node = FollowingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass