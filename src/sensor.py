from math import sqrt
import numpy as np 

class VisionSensor:

	def __init__(self,radius,angle):
		self.radius = radius
		self.angle = angle

	def in_radius(self,eye_position,obj_position):
		if self.distance(eye_position,obj_position) <= self.radius:
			return True
		return False

	def distance(self,eye_position,obj_position):
		x = (eye_position[0],obj_position[0])
		y = (eye_position[1],obj_position[1])
		return sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)

	def in_angle(self,eye_position,direction,obj_position):
		xt = obj_position[0] - eye_position[0]
		yt = obj_position[1] - eye_position[1]

		x = np.cos(direction)*xt + np.sin(direction)*yt
		y = -np.sin(direction)*xt + np.cos(direction)*yt

		if -(self.angle/2)-0.0001 <= np.arctan2(y,x) <= (self.angle/2)+0.0001 :
			return True
		return False

	def copy(self):
		copy_sensor = VisionSensor(self.radius,self.angle)
		return copy_sensor
