#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import transforms3d
import math
from os import system
import time
import numpy as np
import cv2 
import mediapipe
from std_msgs.msg import String
i= ''

class GotoGoalNode(Node):
    def __init__(self):
        super().__init__("move_robot")
        self.target_x = 2
        self.target_y = 2
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.subscriber = self.create_subscription(String, "ges", self.control_loop, 10)

    def control_loop(self, msg):
        i = msg.data
        vel = Twist()


        if i == "f":
            vel.linear.x=1.0
        elif i == "b":
            vel.linear.x=-1.0
        elif i == "l":
            vel.linear.x=1.0
            vel.angular.z=5.0
        elif i == "r":
            vel.linear.x=1.0
            vel.angular.z=-5.0
        elif i == "s":
            vel.linear.x=0.0
            vel.angular.z=0.0
        
        
        print('speed : {}'.format(vel))    
        self.publisher.publish(vel)
        
     
def main(args=None):
    rclpy.init(args=args)
    node = GotoGoalNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
       
if __name__ == "_main_":
	main()