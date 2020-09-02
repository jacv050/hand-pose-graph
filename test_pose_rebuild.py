import argparse
#import pytorch3d
import utils.qutils as qutils
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

#%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

def plot_pose(hands, cloud, title=""):
    #for hand in hands:
    #  for
    u = []
    v = []
    w = []
    x = []
    y = []
    z = []
    print(len(hands))
    for hand in hands:
      iterator = iter(hand)
      #root hand
      rjoint = next(iterator)
      #print("rhand")
      #print(rjoint)
      x.append(0)
      y.append(0)
      z.append(0)
      u.append(rjoint[0])
      v.append(rjoint[1])
      w.append(rjoint[2])
      #print(rjoint)
      #finger
      for finger in iterator:
        #print("Finger")
        #print(finger)
        aux = rjoint
        for joint in finger:
          #print("FJoint")
          x.append(aux[0])
          y.append(aux[1])
          z.append(aux[2])
          #print(aux)
          aux = aux+joint
          u.append(joint[0])
          v.append(joint[1])
          w.append(joint[2])
          #print(aux)

      # Sample points uniformly from the surface of the mesh.
      fig = plt.figure(figsize=(10, 10))
      ax = Axes3D(fig)
      ax.scatter3D(np.array(x), np.array(y), np.array(z))
      #ax.scatter3D(cloud[0], cloud[1], cloud[2])
      ax.quiver(np.array(x), np.array(y), np.array(z), np.array(u), np.array(v), np.array(w))
      #ax.quiver(x,y,z,u,v,w)
      ax.set_xlim([-0.7, 0.2])
      ax.set_ylim([-0.5, 0.5])
      ax.set_zlim([-0.25, 0.6])
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('z')
      ax.set_title(title)
      #ax.view_init(190, 30)
      ax.view_init(0, 0)
      plt.show()

def new_pose(pose_reference, pose_prediction, hand):
    #print(len(pose_reference["right_hand"]))
    output_pose = []
    ijoint = 0
    iterator = iter(pose_reference[hand])

    #Root rotation
    order_axi = [0,1,2] #201 UE
    root   = next(iterator) #pos = 0 orien = 1 (p,y,r)
    #order_rot = ['Z','X','Y'] #Really XYZ #YXZ PRY
    #order_rot = ['Y','X','Z'] #Really ZYX #YXZ PRY
    order_rot = ['Z','X','Y'] #Really XYZ -> Order real ZYX -> 
    
    #order_rot = ['X','Y','Z']
    #order_rot = ['Z','Y','X']
    print(root[1])
    root_rot     = qutils.euler2quaternion(np.radians(root[1])[order_axi], order_rot) #FROM UNREAL
    pos = np.array(root[0])[order_axi]
    root_pos     = qutils.quaternion.from_float_array([0, pos[0], pos[1], pos[2]]) #Test y-zx

    root_rotated = root_pos

    output_pose.append(root_rot)
    #TEST DELETE
    ifinger = 0
    itjoint = 1
    for finger in iterator:
      finger_pose = []
      q = root_rot
      
      #q = qutils.quaternion.from_float_array([1,0,0,0]) #DELETE
      
      ijoint = 0
      #itjoint = itjoint + 1
      for joint in finger:
        phalange_rot     = qutils.euler2quaternion(np.radians(joint[1])[order_axi], order_rot)
        q = q * phalange_rot

        pred = pose_prediction[itjoint*4:itjoint*4+4]
        w=pred[0]
        x=pred[1]
        y=pred[2]
        z=pred[3]
        #XYZ -> XZY -> ZYX
        pred = np.array([w,x,y,z]) #XZY -> ZYX
        print(itjoint)
        print(pred)
        print(qutils.quaternion.from_float_array(pred).inverse())
        #pred = np.array([pred[], pred[], pred[]])
        #bone_space = qutils.quaternion.from_float_array(pred).inverse()
        bone_space = q.inverse() * qutils.quaternion.from_float_array(pred).inverse() * q #World to bone space
        #bone_space = qutils.quaternion.from_float_array(pred)
        print(bone_space)
        finger_pose.append(bone_space.inverse())

        ijoint = ijoint + 1
        itjoint = itjoint + 1

      output_pose.append(finger_pose)
      ifinger = ifinger + 1

    return output_pose

#pose_reference list of lists
#pose_prediction list/array
#World rotation to bone space
def rebuild_pose_from_quaternion2(pose_reference, pose_prediction):
    #hands = ["left_hand", "right_hand"]
    hands = ["left_hand"]

    output_hands = []
    #print(len(pose_reference["right_hand"]))
    for hand in hands:
      pose = new_pose(pose_reference, pose_prediction, hand)
      output_hand = []
      ijoint = 0
      iterator = iter(pose_reference[hand])

      #Root rotation
      order_axi = [0,1,2] #201 UE
      root   = next(iterator) #pos = 0 orien = 1 (p,y,r)
      #order_rot = ['Z','X','Y'] #Really XYZ #YXZ PRY
      #order_rot = ['Y','X','Z'] #Really ZYX #YXZ PRY
      order_rot = ['Z','X','Y'] #Really XYZ -> Order real ZYX -> 
      
      #order_rot = ['X','Y','Z']
      #order_rot = ['Z','Y','X']
      print(root[1])
      root_rot     = qutils.euler2quaternion(np.radians(root[1])[order_axi], order_rot) #FROM UNREAL
      pos = np.array(root[0])[order_axi]
      print("pos")
      print(root[0])
      print(pos)
      root_pos     = qutils.quaternion.from_float_array([0, pos[0], pos[1], pos[2]]) #Test y-zx

      root_rotated = root_pos

      output_hand.append(qutils.quaternion.as_float_array(root_rotated)[1:4])

      #TEST DELETE
      ifinger = 0
      itjoint = 1
      for finger in iterator:
        output_finger = []
        q = root_rot
        qa = root_rot
        #q = qutils.quaternion.from_float_array([1,0,0,0]) #DELETE
        phalange_pos = None
        phalange_rotated = None
        bone_space = qutils.quaternion.from_float_array([1,0,0,0])
        ijoint = 0
        #itjoint = itjoint + 1
        for joint in finger:
          #FIRST APPLY ROTATION
          pos = np.array(joint[0])[order_axi]
          phalange_pos     = qutils.quaternion.from_float_array([0, pos[0], pos[1], pos[2]]) #Test y-zx
          #q = qutils.quaternion.from_float_array([1,0,0,0]) #DELETE
          #print(phalange_pos)
          #q = qutils.quaternion.from_float_array([1,0,0,0])
          phalange_rotated = q * phalange_pos * q.inverse() #Phalange vector
          #phalange_rotated = bone_space.inverse() * phalange_rotated * bone_space
          #print(phalange_rotated)
          point_pos = qutils.quaternion.as_float_array(phalange_rotated)[1:4]

          #Quaternion ROTATION ORIGINAL angles are 'r' 'p' 'y' swaped into 'p' 'y' 'r'
          #UPDATE ROTATION FOR NEXT BONE 
          phalange_rot     = qutils.euler2quaternion(np.radians(joint[1])[order_axi], order_rot)
          q=phalange_rot
          #print(phalange_rot) #AXIS NEURON ROT PRY
          #phalange_rot     = qutils.euler2quaternion(np.radians(joint[1])[[2,0,1]], ['Y', 'X', 'Z'])
          #phalange_rot     = qutils.quaternion.as_float_array(phalange_rot)[[0, 1, 3, 2]]
          #phalange_rot[1]  = -phalange_rot[1]
          #phalange_rot[2]  = -phalange_rot[2]
          #phalange_rot  = qutils.quaternion.from_float_array(phalange_rot)
          #print(phalange_rot)
          #qa = phalange_rot * qa # q-ref
          #Apply prediction quaternion
          point_end = point_pos
          #print("IFINFER {} {}".format(ifinger, ijoint))
          #if ifinger == 2:
          #  print("FINGER BONE SPACE")
            #phalange_pred = q * qutils.quaternion.from_float_array(pose_prediction[ijoint*4:ijoint*4+4]) * q.inverse() #Pose reference - World to bone space
            #q = q.inverse() * qutils.euler2quaternion(np.radians([0,90,0]), ['Y', 'X', 'Z']) * q #World to bone space
          pred = pose[itjoint*4:itjoint*4+4]
          w=pred[0]
          x=pred[1]
          y=pred[2]
          z=pred[3]
          #XYZ -> XZY -> ZYX
          pred = np.array([w,x,y,z]) #XZY -> ZYX
          print(itjoint)
          print(pred)
          print(qutils.quaternion.from_float_array(pred).inverse())
          #pred = np.array([pred[], pred[], pred[]])
          #bone_space = qutils.quaternion.from_float_array(pred).inverse()
          bone_space = qa.inverse() * qutils.quaternion.from_float_array(pred).inverse() * qa #World to bone space
          print(bone_space)
          print(qa)
          #bone_space = qutils.euler2quaternion(np.radians([90,0,0]), ['X','Y','Z'])
          q = bone_space
          #qa = phalange_rot * qa # q-ref
          #qa = bone_space * qa
          qa = qa * bone_space
          #q = q * bone_space
          #phalange_pred = q * phalange_pos * q.inverse()
          #point_end = qutils.quaternion.as_float_array(phalange_pred)[1:4]
          ####
          output_finger.append(point_end)

          ijoint = ijoint + 1
          itjoint = itjoint + 1
          #if ijoint >= 2:
          #  break

        #Final finger bone
        #q = qutils.quaternion.from_float_array([1,0,0,0])
        phalange_rotated = q * phalange_pos * q.inverse() #Phalange vector
        point = qutils.quaternion.as_float_array(phalange_pos)[1:4]
        output_finger.append(point)

        output_hand.append(output_finger)
        ifinger = ifinger + 1
      
      output_hands.append(output_hand)

    return output_hands

def rebuild_pose_from_quaternion3(pose_reference, pose_prediction):
    #hands = ["left_hand", "right_hand"]
    hands = ["left_hand"]

    output_hands = []
    #print(len(pose_reference["right_hand"]))
    for hand in hands:
      pose = new_pose(pose_reference, pose_prediction, hand)
      output_hand = []
      ijoint = 0
      iterator = iter(pose_reference[hand])
      qiterator = iter(pose)

      #Root rotation
      order_axi = [0,1,2] #201 UE
      root   = next(iterator)
      root_rot     = next(qiterator)
      pos = np.array(root[0])[order_axi]

      root_pos     = qutils.quaternion.from_float_array([0, pos[0], pos[1], pos[2]]) #Test y-zx

      root_rotated = root_pos

      output_hand.append(qutils.quaternion.as_float_array(root_rotated)[1:4])

      #TEST DELETE
      ifinger = 0
      itjoint = 1
      for finger in iterator:
        qfinger = iter(next(qiterator))
        output_finger = []
        q = root_rot
        #q = qutils.quaternion.from_float_array([1,0,0,0]) #DELETE
        phalange_pos = None
        phalange_rotated = None
        ijoint = 0
        #itjoint = itjoint + 1
        for joint in finger:
          #FIRST APPLY ROTATION
          pos = np.array(joint[0])[order_axi]
          phalange_pos     = qutils.quaternion.from_float_array([0, pos[0], pos[1], pos[2]]) #Test y-zx
          #q = qutils.quaternion.from_float_array([1,0,0,0]) #DELETE
          #print(phalange_pos)
          #q = qutils.quaternion.from_float_array([1,0,0,0])
          phalange_rotated = q * phalange_pos * q.inverse() #Phalange vector
          #phalange_rotated = bone_space.inverse() * phalange_rotated * bone_space
          #print(phalange_rotated)
          point_pos = qutils.quaternion.as_float_array(phalange_rotated)[1:4]

          #Quaternion ROTATION ORIGINAL angles are 'r' 'p' 'y' swaped into 'p' 'y' 'r'
          #UPDATE ROTATION FOR NEXT BONE 
          q = next(qfinger)

          output_finger.append(point_pos)

          ijoint = ijoint + 1
          itjoint = itjoint + 1
          #if ijoint >= 2:
          #  break

        #Final finger bone
        #q = qutils.quaternion.from_float_array([1,0,0,0])
        phalange_rotated = q * phalange_pos * q.inverse() #Phalange vector
        point = qutils.quaternion.as_float_array(phalange_pos)[1:4]
        output_finger.append(point)

        output_hand.append(output_finger)
        ifinger = ifinger + 1
      
      output_hands.append(output_hand)

    return output_hands

def world2bonespace(pose_reference, pose_prediction):
    hands = ["left_hand", "right_hand"]

    pose_points = []

    for hand in hands:
      ijoint = 0
      iterator = iter(pose_reference[hand])
      root = qutils.from_float_array(next(iterator)[1]) #1 orientation
      for finger in iterator:
        q = root #* qutils.from_float_array([1, 0, 0, 0])
        for joint in finger:
          #qposei = qutils.from_float_array(joint[0])
          qjoint = qutils.from_float_array(joint[1]) #1 orientation
          q = qjoint * q # q-ref
          qf = q.inverse() * qutils.from_float_array(pose_prediction[ijoint*4:ijoint*4+4]) * q
          ijoint = ijoint + 1



if __name__ == "__main__":

    PARSER_ = argparse.ArgumentParser(description="Parameters")
    PARSER_.add_argument("--pose_reference", nargs="?", type=str, default=None, help="Json with pose reference")
    PARSER_.add_argument("--output_error", nargs="?", type=str, default=None, help="Joints angles prediction")
    PARSER_.add_argument("--pointcloud", nargs="?", type=str, default=None, help="Route to pointcloud")

    ARGS_ = PARSER_.parse_args()

    pose_reference = None
    with open(ARGS_.pose_reference) as f:
      pose_reference = json.load(f)

    output_error = None
    with open(ARGS_.output_error) as f:
      output_error = json.load(f)

    cloud = None
    with open(ARGS_.pointcloud, 'rb') as f:
      cloud = PlyData.read(f)
      cloud = [cloud['vertex']['x'], cloud['vertex']['y'], cloud['vertex']['z']]
      #np.vstack((cloud['vertex']['x'],
      #  cloud['vertex']['y'],
      #  cloud['vertex']['z']))


    #pose = rebuild_pose_from_quaternion(pose_reference, output_error['output'])
    pose = rebuild_pose_from_quaternion3(pose_reference, output_error['output_ground_truth'])


    print(len(pose))

    plot_pose(pose, cloud, title="Pose")
    plt.show()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train(ARGS_)

