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

  
def ordered(l):
	t = sorted(l)
	return l == t or l[::-1] == t

direction = 0
class lychee():

	cap = cv2.VideoCapture(0)
	w,h = 1200,1000
	cap.set(3,w) #width
	cap.set(4,h) #height


	mediahands = mediapipe.solutions.hands
	hands = mediahands.Hands(False,2) # idk , max hand count , min detection confidance , min tracking confidance
	drawLMS = lambda frame, landmark : [
		mediapipe.solutions.drawing_utils.draw_landmarks(frame,landmark,lychee.mediahands.HAND_CONNECTIONS)][0]
	process = lambda image : lychee.hands.process(image)

	def findhandjoints(image):
		rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		joints = lychee.process(rgb).multi_hand_landmarks
		h, w, _ = image.shape
		if joints:
			joint = joints[0].landmark
			return joints
		return None

	def getjoints(image):
		rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		joints = lychee.process(rgb).multi_hand_landmarks
		h, w, _ = image.shape
		if joints:
			joint = joints[0].landmark
			return [(l.x, l.y) for l in joint]
		return None

	def findhand(image):
		rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		joints = lychee.process(rgb).multi_hand_landmarks
		h,w,_ = image.shape
		lychee.h = h 
		lychee.w = w
		if joints:
			joints = joints[0].landmark
			joint_pixel = [(int(w*l.x) , int(h*l.y), l.z) for l in joints]
			return joint_pixel
		return None
	

	def getjointdata(joint_pixel):
		thumb = np.transpose(joint_pixel[1:5])
		pointer = np.transpose(joint_pixel[5:9])
		middle = np.transpose(joint_pixel[9:13])
		ring = np.transpose(joint_pixel[13:17])
		pinky = np.transpose(joint_pixel[17:])
		return  [thumb[::-1] , pointer , middle , ring , pinky]
	

	def countfingers_lol(jointdata):
		# Convert jointdata to a list of integers representing finger states
		finger = [int(ordered(list(l[1]))) for l in jointdata ]
		# Define the gestures based on finger_states
		gestures = {
			(1, 1, 1, 1, 1): "f",
			(0, 0, 0, 0, 0): "s" ,
			(0, 1, 1, 0, 0): "r",
			(0, 1, 0, 0, 0): "l",
			(0, 0, 1, 0, 0): "b"

			
		}

		return gestures.get(tuple(finger), None)	
	
	def countfingers(jointdata):
		finger = [int(ordered(list(l[1]))) for l in jointdata ]
		if finger == [1,1,1,1,1]:
			return("1")
		elif finger == [0,0,0,0,0]:
			return("2")
		elif finger == [0,1,1,0,0]:
			return("3")
		elif finger == [0,1,0,0,0]:
			return("4")
		elif finger == [0,0,1,0,0]:
			return("5")
		else:
			return("6")

		

	def detect_motion(sensitivity=(1/20),showjoint=False,joint_index=8):
		global direction
		x,y,_ = 0,0,0
		thehell = False
		t0 = time.time()
		while cv2.waitKey(1) != ord('q'):
			_ ,frame = lychee.cap.read()

			if showjoint:
				if thehell:
					cv2.circle(frame,(x,y),3,(0,0,255),cv2.FILLED)
				cv2.imshow('spice',frame)

			try:
				nx,ny,_ = lychee.findhand(frame)[joint_index]
				thehell = True
			except TypeError:
				thehell = False
				continue
		
			t = time.time()
			fps = int(1/(t-t0))
			t0 = t 
	
			if nx - x > 1/sensitivity:
				direction = -1

			elif x - nx > 1/sensitivity:
				direction = 1

			if ny - y > 1/sensitivity:
				direction = -1j

			elif y - ny > 1/sensitivity:
				direction = 1j

			x,y = nx,ny
			print(direction)

	def render():
		t0 = time.time()
		while cv2.waitKey(1) != ord('q'):
			_ , frame = lychee.cap.read()
			landmarks = lychee.findhandjoints(frame)
			if landmarks:
				for landmark in landmarks:
					lychee.drawLMS(frame,landmark)
			fps = int(1/(time.time() - t0))
			t0 = time.time()
			cv2.imshow('yo',frame)

	def get(joints):
		t = []
		for y in joints:
			t += [y[0] , y[1]]
		return t

			
	def __init__(self):
		while cv2.waitKey(1) != ord('q'):
			_ , frame = lychee.cap.read()
			joints = lychee.getjoints(frame)
			landmarks = lychee.findhandjoints(frame)
			if landmarks:
				for landmark in landmarks:
					lychee.drawLMS(frame,landmark)
			cv2.imshow('robolab_gestures', frame)
			if joints:
				r = lychee.getjointdata(joints)
				i=lychee.countfingers(r)	
				print(lychee.countfingers(r), end = '\r')
				
			
			

lychee()
   
     
class GotoGoalNode(Node):
    def __init__(self):
        super().__init__("move_robot")
        self.target_x = 2
        self.target_y = 2
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.subscriber = self.create_subscription(Odometry, "odom", self.control_loop, 10)
               
    def control_loop(self, msg):
        
        while(1):

            vel = Twist()
			

            if i == "1":
                vel.linear.x=1.0
            elif i == "b":
                vel.linear.x=-1.0
            elif i == "l":
                vel.linear.x=1.0
                vel.angular.z=20.0
            elif i == "r":
                vel.linear.x=1.0
                vel.angular.z=-1.0
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
    
       
if __name__ == "__main__":
	main()
