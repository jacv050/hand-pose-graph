import argparse
#import pytorch3d
import utils.qutils as qutils
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

#%matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

def plot_pose(pose_points, title=""):
    x = []
    y = []
    z = []
    for point in pose_points:
      x.append(point[0])
      y.append(point[1])
      z.append(point[2])
    # Sample points uniformly from the surface of the mesh.
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(np.array(x), np.array(z), -np.array(y))
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

#pose_reference list of lists
#pose_prediction list/array
#World rotation to bone space
def rebuild_pose_from_quaternion(pose_reference, pose_prediction):
    #hands = ["left_hand", "right_hand"]
    hands = ["right_hand"]

    pose_points = []

    for hand in hands:
      ijoint = 0
      iterator = iter(pose_reference[hand])

      #Root rotation
      root   = next(iterator) #pos = 0 orien = 1 (p,y,r)
      root_rot     = qutils.euler2quaternion(root[1], ['Y', 'X', 'Z']) #FROM UNREAL
      #root_rot     = qutils.euler2quaternion(np.array(root[1])[[2,0,1]], ['X', 'Y', 'Z'])
      root_pos     = qutils.quaternion.from_float_array([0, 1, 1, 1]) #Test
      root_rotated = root_rot * root_pos * root_rot.inverse() #Root vector

      pose_points.append(qutils.quaternion.as_float_array(root_pos)[1:4])

      #TEST DELETE
      next(iterator)
      next(iterator)
      next(iterator)
      next(iterator)
      for finger in iterator:
        #q = root_rot * root_pos * root_rot.inverse() #* qutils.from_float_array([1, 0, 0, 0])
        #Quaternion ROTATION
        vec  = root_pos
        #rotated VECTOR
        q = root_rot
        for joint in finger:
          #Quaternion ROTATION ORIGINAL angles are 'r' 'p' 'y' swaped into 'p' 'y' 'r'
          phalange_rot     = qutils.euler2quaternion(joint[1], ['Y', 'X', 'Z'])
          #phalange_rot     = qutils.euler2quaternion(np.array(joint[1])[[2,0,1]], ['X', 'Y', 'Z'])
          q  = phalange_rot * q # q-ref
          qn = q.normalized()
          #Quaternion VECTOR
          phalange_pos     = qutils.quaternion.from_float_array([0, 0.5, 0.5, 0.5]) #Test joint[0]
          #Quaternion VECTOR
          phalange_rotated = qn * phalange_pos * qn.inverse() #Phalange vector
          #Quaternion ROT * ROT
          #World to Bone space ROT
          #bone_space_rot = qn.inverse() * qutils.quaternion.from_float_array(pose_prediction[ijoint*4:ijoint*4+4]) * qn #Pose reference - World to bone space
          #Apply bone space rot to phalange
          bone = phalange_rotated
          #bone = bone_space_rot * phalange_rotated * bone_space_rot.inverse() #Apply axis neuron angle read
          #Final point
          #vec = qutils.quaternion.as_float_array(vec + bone)[1:4]
          vec = vec + bone
          point = qutils.quaternion.as_float_array(vec)[1:4]

          pose_points.append(point)

          ijoint = ijoint + 1

    return pose_points


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

    ARGS_ = PARSER_.parse_args()

    pose_reference = None
    with open(ARGS_.pose_reference) as f:
      pose_reference = json.load(f)

    output_error = None
    with open(ARGS_.output_error) as f:
      output_error = json.load(f)

    #pose = rebuild_pose_from_quaternion(pose_reference, output_error['output'])
    pose = rebuild_pose_from_quaternion(pose_reference, output_error['output_ground_truth'])


    print(len(pose))

    plot_pose(pose, title="Pose")
    plt.show()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train(ARGS_)

