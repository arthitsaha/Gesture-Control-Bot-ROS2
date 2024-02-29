#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import transforms3d
import math
     
     
class GotoGoalNode(Node):
    def __init__(self):
        super().__init__("move_robot")
        self.target_x = 2
        self.target_y = 2
        self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        self.subscriber = self.create_subscription(Odometry, "odom", self.control_loop, 10)
               
    def control_loop(self, msg):
        
        vel = Twist()
        print('speed : {}'.format(vel))    
        self.publisher.publish(vel)
    def findhandjoints(self , image):
        rgb = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
        joints = lychee.process(rgb).multi_hand_landmarks
        h, w, _ = image.shape
        if joints:
            joint = joints[0].landmark
            return joints
        return None
    def getjoints(self,image):
        rgb = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
        joints = lychee.process(rgb).multi_hand_landmarks
        h, w, _ = image.shape
        if joints:
            joint = joints[0].landmark
            return [(l.x, l.y) for l in joint]
        return None
    def findhand(self , image):
        rgb = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
        joints = lychee.process(rgb).multi_hand_landmarks
        h,w,_ = image.shape
        lychee.h = h 
        lychee.w = w
        if joints:
            joints = joints[0].landmark
            joint_pixel = [(int(w*l.x) , int(h*l.y), l.z) for l in joints]
            return joint_pixel
        return None      
    def getjointdata(self , joint_pixel):
        thumb = np.transpose(self.joint_pixel[1:5])
        pointer = np.transpose(self.joint_pixel[5:9])
        middle = np.transpose(self.joint_pixel[9:13])
        ring = np.transpose(self.joint_pixel[13:17])
        pinky = np.transpose(self.joint_pixel[17:])
        return  [thumb[::-1] , pointer , middle , ring , pinky]

    def countfingers_lol(self , jointdata):
		# Convert jointdata to a list of integers representing finger states
        finger = [int(ordered(list(l[1]))) for l in self.jointdata ]
		# Define the gestures based on finger_states
        gestures = {
			(1, 1, 1, 1, 1): "f",
			(0, 0, 0, 0, 0): "s" ,
			(0, 1, 1, 0, 0): "r",
			(0, 1, 0, 0, 0): "l",
			(0, 0, 1, 0, 0): "b"

			
		}

        return gestures.get(tuple(finger), None)	
    def countfingers(self,jointdata):
        finger = [int(ordered(list(l[1]))) for l in self.jointdata ]
        if finger == [1,1,1,1,1]:
            return("f")
        elif finger == [0,0,0,0,0]:
            return("s")
        elif finger == [0,1,1,0,0]:     
            return("r")
        elif finger == [0,1,0,0,0]:
            return("l")
        elif finger == [0,0,1,0,0]:
            return("b")
        else:
            return("-")

		

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

    def get(self, joints):
        t = []
        for y in self.joints:
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
                print(lychee.countfingers(r), end = '\r')

    
def main(args=None):
    rclpy.init(args=args)
    node = GotoGoalNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
       
if __name__ == "__main__":
	main()
