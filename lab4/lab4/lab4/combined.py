#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import time
import numpy as np
import cv2 
import mediapipe as mp

class GotoGoalNode(Node):
    def _init_(self):
        super()._init_("move_robot")
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.subscriber = self.create_subscription(Odometry, "odom", self.control_loop, 10)
        self.gesture_control = GestureControl()
               
    def control_loop(self, msg):
        i = self.gesture_control.detect_gesture()
        vel = Twist()

        if i == "f":
            vel.linear.x = 1.0
        elif i == "b":
            vel.linear.x = -1.0
        elif i == "l":
            vel.linear.x = 1.0
            vel.angular.z = 20.0
        elif i == "r":
            vel.linear.x = 1.0
            vel.angular.z = -1.0
        elif i == "s":
            vel.linear.x = 0.0
            vel.angular.z = 0.0

        print('speed : {}'.format(vel))    
        self.publisher.publish(vel)

class GestureControl:
    def _init_(self):
        self.cap = cv2.VideoCapture(0)
        self.w, self.h = 1200, 1000
        self.cap.set(3, self.w)  # width
        self.cap.set(4, self.h)  # height

        self.hands = mp.solutions.hands.Hands(False, 2, 0.5, 0.5)

    def detect_gesture(self):
        _, frame = self.cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                return self.interpret_gesture(hand_landmarks.landmark)
        return None

    def interpret_gesture(self, landmarks):
        joint_pixel = [(int(self.w * l.x), int(self.h * l.y), l.z) for l in landmarks]
        jointdata = [np.transpose(joint_pixel[1:5]),  # thumb
                     np.transpose(joint_pixel[5:9]),  # pointer
                     np.transpose(joint_pixel[9:13]), # middle
                     np.transpose(joint_pixel[13:17]),# ring
                     np.transpose(joint_pixel[17:])]  # pinky
        finger = [int(self.ordered(list(l[1]))) for l in jointdata]
        gestures = {
            (1, 1, 1, 1, 1): "f",
            (0, 0, 0, 0, 0): "s",
            (0, 1, 1, 0, 0): "r",
            (0, 1, 0, 0, 0): "l",
            (0, 0, 1, 0, 0): "b"
        }
        return gestures.get(tuple(finger), None)

    @staticmethod
    def ordered(l):
        t = sorted(l)
        return l == t or l[::-1] == t

def main(args=None):
    rclpy.init(args=args)
    node = GotoGoalNode()
    rclpy.spin(node)
    node.gesture_control.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "_main_":
    main()