import logging
import os
import json

import numpy as np

from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

import scipy.spatial
from scipy.cluster.vq import vq, kmeans2, whiten

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import sys
import glob

import copy
import utils.qutils as qutils

import multiprocessing.dummy as mp

LOG = logging.getLogger(__name__)

class UnrealHands(Dataset):

  def _create_graph(self, cloud, k, labels, radius):

    graph_x_ = torch.tensor(np.vstack((cloud['vertex']['red'],
                                      cloud['vertex']['green'],
                                      cloud['vertex']['blue'],
                                      cloud['vertex']['x'],
                                      cloud['vertex']['y'],
                                      cloud['vertex']['z'],)), dtype=torch.float).transpose(0, 1)

    points_ = np.transpose(np.vstack((cloud['vertex']['x'],
                                      cloud['vertex']['y'],
                                      cloud['vertex']['z'])), (1,0)) #N,3

    #clusters = kmeans2(points_, 2)

    tree_ = scipy.spatial.cKDTree(points_)

    idxs_ = None
    if radius is None:
      _, idxs_ = tree_.query(points_, k=k + 1) # Closest point will be the point itself, so k + 1
    else:
      _, idxs_ = tree_.query(points_, k=k + 1, distance_upper_bound=radius) # Closest point will be the point itself, so k + 1
 
    idxs_ = idxs_[:, 1:] # Remove closest point, which is the point itself

    if len(cloud['vertex']['x']) > self.aux_max:
      self.aux_max = len(cloud['vertex']['x'])
      print("\n\n{}\n\n".format(self.aux_max))

    edge_origins_ = np.repeat(np.arange(len(points_)), k)
    edge_ends_ = np.reshape(idxs_, (-1))
    #print(edge_ends_.shape)

    graph_edge_index_ = torch.tensor([edge_origins_, edge_ends_], dtype=torch.long)

    graph_pos_ = torch.tensor(np.vstack((cloud['vertex']['x'],
                                        cloud['vertex']['y'],
                                        cloud['vertex']['z'])), dtype=torch.float).transpose(0, 1)

    #graph_y_ = torch.tensor(cloud['vertex']['label'], dtype=torch.long)
    graph_y_ = None
    indexed = False
    quaternion = True
    if labels is None:
      kin = np.vstack(cloud['gt']['axiskin'])
      kinf = None
      #print(kin)
      l=[4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 38, 39, 41, 42]
      if indexed:
        kinf = kin[l]
      else:
        kinf = kin
      #Convert to quaternion from kinematics
      q = None
      #Quaternion from not indexed specific angles
      if quaternion and not indexed:
        q = np.zeros(int(kin.shape[0]/3*4), dtype=np.float64) #bones angles divided 3 * 4 -> 64
        for i in range(int(kin.size/3)):
          #Correct pyr -> rpy
          qaux = qutils.quaternion.as_float_array(qutils.euler2quaternion(np.radians([kin[i*3+2], kin[i*3], kin[i*3+1]]), ['Y', 'X', 'Z']))
          q[i*4] = qaux[0]
          q[i*4+1] = qaux[1]
          q[i*4+2] = qaux[2]
          q[i*4+3] = qaux[3]

        kinf = q
      #Quaternion from indexed specific angles TODO
      else:
        print('Disable indexed')

      graph_y_ = torch.tensor(kinf, dtype=torch.float)
    else:
      graph_y_ = torch.tensor(labels, dtype=torch.float)


    #FIX SIZE
    diff = 1395 - len(cloud['vertex']['x'])
    graph_x_ = torch.cat((graph_x_, torch.tensor(np.repeat([[0,0,0,0,0,0]], diff, axis=0), dtype=torch.float)), 0)
    graph_pos_ = torch.cat((graph_pos_, torch.zeros((diff,3))))
    #FIXED

    data_ = Data(x = graph_x_, edge_index = graph_edge_index_, pos = graph_pos_, y = graph_y_)
    return data_

  def __init__(self, root, k=3, radius=None, transform=None, pre_transform=None):
    self.cloud_folder = "cloud_sampled"
    self.aux_max = 0
    self.k = k
    self.radius=radius

    super(UnrealHands, self).__init__(root, transform, pre_transform)

  @property
  def num_features(self):
    return self[0].num_features

  @property
  def num_classes(self):
    """The number of classes in the dataset."""
    return 96

  @property
  def raw_file_names(self):
    output = []
    for scene in os.listdir(self.raw_dir):
      rpath = scene + "/" + self.cloud_folder + "/"
      for camera in os.listdir(self.raw_dir + "/" + rpath):
        crpath = rpath + camera + "/"
        output += [crpath + s for s in os.listdir(self.raw_dir + "/" + crpath)]
    return output
    #return ["cloud/" + s for s in os.listdir(self.raw_dir + "/cloud/")]

  @property
  def processed_file_names(self):#CAMBIAR ESTO
    #return os.listdir(self.processed_dir)
    return ["unrealhands_k" + str(self.k) + "_" + str(i) + ".pt" for i in range(len(self.raw_file_names))]

  def __len__(self):
    return len(self.processed_file_names)

  def download(self):
    LOG.info("Data not found...")
    raise RuntimeError("Dataset not found, please download it!")

  def read_joints_json(self, input_file):
    output_dict = {}

    with open(input_file) as f:
        joints = json.load(f)

        for key, value in joints.items():
          output = []
          iterator = iter(value)
          for root in iterator:
            #output.append(root)
            output += root

            for finger_list in iterator:
              for bone in finger_list:
                #output.append(bone)
                output += bone
          output_dict[key] = output

    return output_dict

  def read_joints_json2(self, input_file):
    output_dict = {}

    with open(input_file) as f:
        joints = json.load(f)

        for key, value in joints.items():
          output = []
          iterator = iter(value)
          for root in iterator:
            #output.append(root)
            output += root
            nproot = np.array(root)

            for finger_list in iterator:
              iterator2 = iter(finger_list)
              aux = np.array(next(iterator2)) + nproot
              output += aux.tolist()
              for bone in iterator2:
                #output.append(bone)
                aux = np.array(bone)+aux
                output += aux.tolist()
          output_dict[key] = output

    return output_dict

  def generate_listofpoints(self, labels):
    output = []
    for i in range(0,len(labels),3):
      output.append(labels[i:i+3])

    print(output)

  def process_threaded(self, p):
    path_cloud = self.raw_paths_processed[p]
    #TODO Parameterize
    old_joints = False #OLD_JOINTS
    labels = None
    if old_joints: #DEFAULT FALSE
      path_joints = (path_cloud[:len(path_cloud)-3] + "json").replace(self.cloud_folder, "joints")
      hands_ = self.read_joints_json2(path_joints)
      labels = hands_["left_hand"]+hands_["right_hand"]

    with open(self.raw_paths_processed[p], 'rb') as f:
      print(self.raw_paths_processed[p])
      cloud_ = PlyData.read(f)
      if(len(cloud_['vertex']['x']) > 0):
      	graph_ = self._create_graph(cloud_, self.k, labels, self.radius)
      	torch.save(graph_, os.path.join(self.processed_dir, "unrealhands_k{0}_{1}.pt".format(self.k, p)))

  def process(self):
    LOG.info("Processing dataset...")
    self.raw_paths_processed = self.raw_paths
    p = mp.Pool(12)
    #p.start()
    for i, _ in enumerate(p.imap_unordered(self.process_threaded, range(len(self.raw_paths_processed)), 1)):
      sys.stderr.write('\rdone {0:%}\n'.format(i/len(self.raw_paths_processed)))
    p.close()
    p.join()

  def process_deprecated(self):
    LOG.info("Processing dataset...")

    for p in range(len(self.raw_paths)):
      path_cloud = self.raw_paths[p]
      path_joints = (path_cloud[:len(path_cloud)-3] + "json").replace("cloud", "joints")
      LOG.info("Processing cloud {0} out of {1}".format(p, len(self.raw_paths)))
      LOG.info(path_cloud)
      
      LOG.info(path_joints)
      hands_ = self.read_joints_json(path_joints)
      labels = hands_["left_hand"]+hands_["right_hand"]
      #print(labels)

      with open(self.raw_paths[p], 'rb') as f:
        print(self.raw_paths[p])
        cloud_ = PlyData.read(f)
        graph_ = self._create_graph(cloud_, self.k, labels)
        torch.save(graph_, os.path.join(self.processed_dir, "unrealhands_k{0}_{1}.pt".format(self.k, p)))

  def get(self, idx):
    data_ = torch.load(os.path.join(self.processed_dir, "unrealhands_k{0}_{1}.pt".format(self.k, idx)))
    #data_ = torch.load(os.path.join(self.processed_dir, "unrealhands_k{0}_{1}.pt".format(self.k, idx)))
    return data_
