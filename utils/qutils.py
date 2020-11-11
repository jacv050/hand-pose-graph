import quaternion
import numpy as np

def rotX(angle):
	return quaternion.quaternion(np.cos(angle/2., dtype=np.float64), np.sin(angle/2., dtype=np.float64), 0., 0.)

def rotY(angle):
	return quaternion.quaternion(np.cos(angle/2., dtype=np.float64), 0., np.sin(angle/2., dtype=np.float64), 0.)

def rotZ(angle):
	return quaternion.quaternion(np.cos(angle/2., dtype=np.float64), 0., 0., np.sin(angle/2., dtype=np.float64))

#angles [x,y,z]
def euler2quaternion(angles, order=['X', 'Y', 'Z']):
	q = quaternion.quaternion(1., 0., 0., 0.)
	
	x = rotX(angles[0])
	y = rotY(angles[1])
	z = rotZ(angles[2])

	for c in order:
		if c == 'X':
			q *= x
			#q = x * q
		elif c == 'Y':
			q *= y
			#q = y * q
		elif c == 'Z':
			q *= z
			#q = z * q

	return q
    	
def eulerXYZquaternion(angles):
	return rotX(angles[0])*rotY(angles[1])*rotZ(angles[2])