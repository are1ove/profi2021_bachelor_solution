#!/usr/bin/env python

import time
from math import sin, cos

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from hector_uav_msgs.srv import EnableMotors

import cv2
from cv_bridge import CvBridge, CvBridgeError


class SimpleMover():

    def __init__(self):
        rospy.init_node('simple_mover', anonymous=True)

        if rospy.has_param('/profi2021_bachelor_solution/altitude_desired'):
            self.altitude_desired = rospy.get_param(
                '/profi2021_bachelor_solution/altitude_desired')  # in solution.launch
        else:
            rospy.logerr("Failed to get param '/profi2021_bachelor_solution/altitude_desired'")

        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        rospy.Subscriber("cam_1/camera/image", Image, self.camera_cb)
        self.rate = rospy.Rate(30)

        self.cv_bridge = CvBridge()

        rospy.on_shutdown(self.shutdown)

    def camera_cb(self, msg):

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        except CvBridgeError, e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        self.show_image(cv_image)

    def show_image(self, img):
        cv2.imshow("Camera 1 from Robot", img)
        cv2.waitKey(3)

    def enable_motors(self):

        try:
            rospy.wait_for_service('enable_motors', 2)
            call_service = rospy.ServiceProxy('enable_motors', EnableMotors)
            response = call_service(True)
        except Exception as e:
            print("Error while try to enable motors: ")
            print(e)

    def take_off(self):

        self.enable_motors()

        start_time = time.time()
        end_time = start_time + 3
        twist_msg = Twist()
        twist_msg.linear.z = self.altitude_desired

        while (time.time() < end_time) and (not rospy.is_shutdown()):
            self.cmd_vel_pub.publish(twist_msg)
            self.rate.sleep()

    def spin(self):

        self.take_off()

        start_time = time.time()
        while not rospy.is_shutdown():
            twist_msg = Twist()
            t = time.time() - start_time
            twist_msg.linear.z = 0.8 * cos(1.2 * t)
            twist_msg.linear.y = 0.8 * sin(0.6 * t)
            self.cmd_vel_pub.publish(twist_msg)
            self.rate.sleep()

    def shutdown(self):
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)


simple_mover = SimpleMover()
simple_mover.spin()
