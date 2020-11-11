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

def get_position(dict_json):
    """ Return an array with UnrealEngine position with coordinate system fixed.

    Args: 
        dict_json: dictionary with UnrealEngine object information.
    """
    return np.array([dict_json["position"]["y"], dict_json["position"]["z"], dict_json["position"]["x"]])/100

def get_rotation(dict_json):
    """ Return an array with UnrealEngine orientation with coordinate system fixed.

    Args:
        dict_json: dictionary with UnrealEngine object information. 
    """
    #UnrealEngine X=Roll, Y=Pitch, Z=Yaw
    #return np.array([dict_json["rotation"]["p"], dict_json["rotation"]["y"], dict_json["rotation"]["r"]])
    return np.array([dict_json["rotation"]["p"], dict_json["rotation"]["y"], dict_json["rotation"]["r"]])
    
def get_kinematics(dict_json):
    return np.array([dict_json["kinematics"]["p"],dict_json["kinematics"]["y"],dict_json["kinematics"]["r"]])

def get_bone(bone_name, list_of_bones_json):
    """ Return bone from a list of bones

    Args:
        bone_name: An string with the bone name.
        list_of_bones_json: List of bones with their properties (position, orientation).
    """
    for bone in list_of_bones_json:
        if (bone["name"] == bone_name):
            return bone
    print(type(bone_name))
    print("ERROR: Bone " + bone_name + " not found...")

def filter_joints(skeleton_json, list_of_bones):
    """ Filter joints from an UnrealEngine skeleton json and a list of bones.
    The list of bones must be a list of lists where the first element is the root.
    Ex. list_of_bones = [root_hand, [joints_finger_1], [joints_finger_2], ...]

    Args:
        skeleton_json = dictionary with UnrealEngine skeleton information. 
    """
    output_list = []

    iterator = iter(list_of_bones)

    for in_finger_list in iterator:
        iterator_finger = iter(in_finger_list)#iterator for finger
        root_finger_name = next(iterator_finger)
        root_finger_bone_json = get_bone(root_finger_name, skeleton_json)
        
        root_finger = np.zeros((3,3))
        root_finger[0][:] = get_position(root_finger_bone_json)
        root_finger[1][:] = get_rotation(root_finger_bone_json)
        root_finger[2][:] = get_kinematics(root_finger_bone_json)
        
        out_finger_list = [] #List of phalanges
        out_finger_list.append(root_finger)
        for bone_name in iterator_finger:
            bone_json = get_bone(bone_name, skeleton_json)

            bone = np.zeros((3,3))
            bone[0][:] = get_position(bone_json)
            bone[1][:] = get_rotation(bone_json)
            bone[2][:] = get_kinematics(bone_json)

            #bone = quaternion.from_euler_angles([bone_json["rotation"]["r"], bone_json["rotation"]["p"], bone_json["rotation"]["y"]])
            out_finger_list.append(bone)
        output_list.append(out_finger_list)

    return output_list

def kinematics_filtered_skeleton(processed_bones):
    """ Generate a kinematic list.

    Args:
        processed_bones: list of filtered bones.
        camera_json: UnrealEngine camera information.
    """
    output_list = []

    iterator = iter(processed_bones)

    #Get root position
    for finger_list in iterator:
        #output_list.append(root_hand[2])
        #Get root position of finger
        out_finger_list = []
        for bone in finger_list:
            out_finger_list.append(bone[2])
            print(bone[2])
        print("")

        output_list.append(out_finger_list)

    return output_list

def skeleton2quaternion(ground_truth):
    iterator = iter(ground_truth)

    #root = next(iterator)
    #[Y,Z,X]
    order_axi = [2,0,1] #XYZ
    order_rot   = ['Y','X','Z']
    neg = np.array([1,1,1])
    output_hand = []

    for finger in iterator:
        print(finger)
        for joint in finger:
            print(np.array(joint))
            q = qutils.euler2quaternion(np.radians(joint)[order_axi]*neg, order_rot) #P R Y
            output_hand = output_hand + qutils.quaternion.as_float_array(q).tolist()


    return output_hand

if __name__ == "__main__":

    PARSER_ = argparse.ArgumentParser(description="Parameters")
    #PARSER_.add_argument("--pose_reference", nargs="?", type=str, default=None, help="Json with pose reference")
    PARSER_.add_argument("--scene_json", nargs="?", type=str, default=None, help="Json scene")
    PARSER_.add_argument("--output_error", nargs="?", type=str, default=None, help="Joints angles prediction")
    PARSER_.add_argument("--bones_names", nargs="?", type=str, default=None, help="Bones names")

    ARGS_ = PARSER_.parse_args()

    #pose_reference = None
    #with open(ARGS_.pose_reference) as f:
    #  pose_reference = json.load(f)

    output_error = None
    with open(ARGS_.output_error) as f:
      output_error = json.load(f)

    skeleton = None
    with open(ARGS_.scene_json) as f:
        skeleton = json.load(f)["frames"][60]["skeletons"][0]["bones"]

    bones_names = None
    with open(ARGS_.bones_names) as f:
        bones_names = json.load(f)
    """
    bones_names["left_hand"] = ["hand_l", 
                ["index_01_l", "index_02_l", "index_03_l"], 
                ["middle_01_l", "middle_02_l", "middle_03_l"],
                ["pinky_01_l", "pinky_02_l", "pinky_03_l"],
                ["ring_01_l", "ring_02_l", "ring_03_l"],
                ["thumb_01_l", "thumb_02_l", "thumb_03_l"]]
    bones_names["right_hand"] = ["hand_r", 
                ["index_01_r", "index_02_r", "index_03_r"], 
                ["middle_01_r", "middle_02_r", "middle_03_r"],
                ["pinky_01_r", "pinky_02_r", "pinky_03_r"],
                ["ring_01_r", "ring_02_r", "ring_03_r"],
                ["thumb_01_r", "thumb_02_r", "thumb_03_r"]]
    """
    
    filtered = filter_joints(skeleton, bones_names["right_hand"])
    gt = kinematics_filtered_skeleton(filtered)
    gt = skeleton2quaternion(gt)

    output_error["output_ground_truth"] = gt
    with open("output_error_final.json", 'w') as f:
        json.dump(output_error, f, indent=2)

    print(gt)

