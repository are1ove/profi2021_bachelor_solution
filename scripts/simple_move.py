#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2
import numpy as np
import time
from math import sin, cos

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from std_msgs.msg import Int16  # For error/angle plot publishing

from hector_uav_msgs.srv import EnableMotors

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
        rospy.Subscriber("cam_1/camera/image", Image, self.line_detect)
        self.rate = rospy.Rate(30)
        self.pub_error = rospy.Publisher('error', Int16, queue_size=10)
        self.pub_angle = rospy.Publisher('angle', Int16, queue_size=10)

        self.cv_bridge = CvBridge()
        self.Kp = 0.112  # Ku=0.14 T=6. PID: p=0.084,i=0.028,d=0.063. PD: p=0.112, d=0.084/1. P: p=0.07
        self.Ki = 0
        self.kd = 1
        self.integral = 0
        self.derivative = 0
        self.last_error = 0
        self.Kp_ang = 0.01  # Ku=0.04 T=2. PID: p=0.024,i=0.024,d=0.006. PD: p=0.032, d=0.008. P: p=0.02/0.01
        self.Ki_ang = 0
        self.kd_ang = 0
        self.integral_ang = 0
        self.derivative_ang = 0
        self.last_ang = 0
        self.was_line = 0
        self.line_side = 0
        self.battery = 0
        self.line_back = 1
        self.landed = 0
        self.takeoffed = 0
        self.error = []
        self.angle = []
        self.fly_time = 0.0
        self.start = 0.0
        self.stop = 0.0
        self.velocity = 0.5

        rospy.on_shutdown(self.shutdown)

    # def camera_cb(self, msg):
    #
    #     try:
    #         cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    #
    #     except CvBridgeError, e:
    #         rospy.logerr("CvBridge Error: {0}".format(e))
    #
    #     self.show_image(cv_image)

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

    def zoom(self, cv_image, scale):
        height, width, _ = cv_image.shape
        # print(width, 'x', height)
        # prepare the crop
        centerX, centerY = int(height / 2), int(width / 2)
        radiusX, radiusY = int(scale * height / 100), int(scale * width / 100)

        minX, maxX = centerX - radiusX, centerX + radiusX
        minY, maxY = centerY - radiusY, centerY + radiusY

        cv_image = cv_image[minX:maxX, minY:maxY]
        cv_image = cv2.resize(cv_image, (width, height))

        return cv_image

    def line_detect(self, msg):
        # Create a mask
        # cv_image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        if int(self.altitude_desired) >= 5 or int(self.altitude_desired) <= 2.4:
            cv_image = self.zoom(cv_image, scale=20)
        else:
            cv_image = self.zoom(cv_image, scale=35)
        # cv_image = cv2.add(cv_image, np.array([-50.0]))
        mask = cv2.inRange(cv_image, (20, 20, 20), (130, 130, 130))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=5)
        mask = cv2.dilate(mask, kernel, iterations=9)
        _, contours_blk, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blk.sort(key=cv2.minAreaRect)

        if len(contours_blk) > 0 and cv2.contourArea(contours_blk[0]) > 500:
            self.was_line = 1
            if int(self.altitude_desired) > 2.4:
                blackbox_left = cv2.minAreaRect(contours_blk[0])
                blackbox_right = cv2.minAreaRect(contours_blk[-1])
                (x_left, y_left), (w_left, h_left), angle_left = blackbox_left
                (x_right, y_right), (w_right, h_right), angle_right = blackbox_right
                x_min, y_min, w_min, h_min, angle = (x_left + x_right) / 2, (y_left + y_right) / 2, (
                        w_left + w_right) / 2, (h_left + h_right) / 2, (angle_left + angle_right) / 2
            else:
                blackbox = cv2.minAreaRect(contours_blk[0])
                (x_min, y_min), (w_min, h_min), angle = blackbox

            if angle < -45:
                angle = 90 + angle
            if w_min < h_min and angle > 0:
                angle = (90 - angle) * -1
            if w_min > h_min and angle < 0:
                angle = 90 + angle

            setpoint = cv_image.shape[1] / 2
            error = int(x_min - setpoint)
            self.error.append(error)
            self.angle.append(angle)
            normal_error = float(error) / setpoint

            if error > 0:
                self.line_side = 1  # line in right
            elif error <= 0:
                self.line_side = -1  # line in left

            self.integral = float(self.integral + normal_error)
            self.derivative = normal_error - self.last_error
            self.last_error = normal_error

            error_corr = -1 * (
                    self.Kp * normal_error + self.Ki * self.integral + self.kd * self.derivative)  # PID controler
            # print("error_corr:  ", error_corr, "\nP", normal_error * self.Kp, "\nI", self.integral* self.Ki, "\nD", self.kd * self.derivative)

            angle = int(angle)
            normal_ang = float(angle) / 90

            self.integral_ang = float(self.integral_ang + angle)
            self.derivative_ang = angle - self.last_ang
            self.last_ang = angle

            ang_corr = -1 * (
                    self.Kp_ang * angle + self.Ki_ang * self.integral_ang + self.kd_ang * self.derivative_ang)  # PID controler
            if int(self.altitude_desired) > 2.4:
                box_left = cv2.boxPoints(blackbox_left)
                box_left = np.int0(box_left)

                cv2.drawContours(cv_image, [box_left], 0, (0, 0, 255), 3)

                box_right = cv2.boxPoints(blackbox_right)
                box_right = np.int0(box_right)

                cv2.drawContours(cv_image, [box_right], 0, (0, 0, 255), 3)

            else:
                box = cv2.boxPoints(blackbox)
                box = np.int0(box)

                cv2.drawContours(cv_image, [box], 0, (0, 0, 255), 3)

            cv2.putText(cv_image, "Angle: " + str(angle), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)

            cv2.putText(cv_image, "Error: " + str(error), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.line(cv_image, (int(x_min), 200), (int(x_min), 250), (255, 0, 0), 3)

            twist = Twist()
            twist.linear.x = self.velocity
            twist.linear.y = error_corr
            # twist.linear.z = 0
            # twist.angular.x = 0
            # twist.angular.y = 0
            twist.angular.z = ang_corr
            self.cmd_vel_pub.publish(twist)
            # print("angVal: ", twist.angular.z)

            ang = Int16()
            ang.data = angle
            self.pub_angle.publish(ang)

            err = Int16()
            err.data = error
            self.pub_error.publish(err)

        if len(contours_blk) == 0 and self.was_line == 1 and self.line_back == 1:
            twist = Twist()
            if self.line_side == 1:  # line at the right
                twist.linear.y = -0.05
                self.cmd_vel_pub.publish(twist)
            if self.line_side == -1:  # line at the left
                twist.linear.y = 0.05
                self.cmd_vel_pub.publish(twist)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1) & 0xFF
        # cv2.imshow("mask", mask)
        # cv2.waitKey(1) & 0xFF

    def spin(self):

        self.take_off()

        start_time = time.time()

        while not rospy.is_shutdown():
            twist_msg = Twist()
            t = time.time() - start_time

            self.rate.sleep()

    def shutdown(self):
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)


simple_mover = SimpleMover()
simple_mover.spin()
