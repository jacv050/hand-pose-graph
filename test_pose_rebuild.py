import argparse
import pytorch3d
import utils.qutils as qutils
import numpy as np
import matplotlib.pyplot as plt
import json

#pose_reference list of lists
#pose_prediction list/array
#World rotation to bone space
def rebuild_pose_from_quaternion(pose_reference, pose_prediction):
    hands = ["left_hand", "right_hand"]

    pose_points = []

    for hand in hands:
      ijoint = 0
      iterator = iter(pose_reference[hand])

      #Root rotation
      root   = next(iterator) #pos = 0 orien = 1 (p,y,r)
      root_rot     = qutils.euler2quaternion(root[1], ['Y', 'X', 'Z'])
      root_pos     = qutils.from_float_array([0, 1, 1, 1]) #Test
      root_rotated = root_rot * root_pos * root_rot.inverse() #Root vector
      for finger in iterator:
        #q = root_rot * root_pos * root_rot.inverse() #* qutils.from_float_array([1, 0, 0, 0])
        #Quaternion ROTATION
        qrot = root_rot
        #rotated VECTOR
        vec  = root_pos
        for joint in finger:
          #Quaternion ROTATION
          phalange_rot     = qutils.euler2quaternion(joint[1], ['Y', 'X', 'Z'])
          #Quaternion VECTOR
          phalange_pos     = qutils.from_float_array([0, 0.5, 0.5, 0.5]) #Test joint[0]
          #Quaternion VECTOR
          phalange_rotated = phalange_rot * phalange_pos * phalange_rot.inverse() #Phalange vector
          #Quaternion ROT * ROT
          q  = phalange_rot * q # q-ref
          qn = q.normalized()
          #World to Bone space ROT
          bone_space_rot = qn.inverse() * qutils.from_float_array(pose_prediction[ijoint*4:ijoint*4+4]) * qn #Pose reference - World to bone space
          #Apply bone space rot to phalange
          bone = bone_space_rot * phalange_rotated * bone_space_rot.inverse() #Apply axis neuron angle read
          #Final point
          vec = qutils.as_float_array(vec + bone)
          pose_points.append(vec)

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

    pose = rebuild_pose_from_quaternion(pose_reference, output_error['output'])

    plt.plot(pose)
    plt.show()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train(ARGS_)

