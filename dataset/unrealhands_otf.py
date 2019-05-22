import logging
import os
import json

import numpy as np

from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

import scipy.spatial

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import sys
import glob

import multiprocessing.dummy as mp

LOG = logging.getLogger(__name__)

class UnrealHands(Dataset):

  def _create_graph(self, cloud, k, labels):

    graph_x_ = torch.tensor(np.vstack((cloud['vertex']['red'],
                                      cloud['vertex']['green'],
                                      cloud['vertex']['blue'])), dtype=torch.float).transpose(0, 1)

    points_ = np.transpose(np.vstack((cloud['vertex']['x'],
                                      cloud['vertex']['y'],
                                      cloud['vertex']['z'])), (1, 0))
    tree_ = scipy.spatial.cKDTree(points_)
    
    _, idxs_ = tree_.query(points_, k=k + 1) # Closest point will be the point itself, so k + 1

    idxs_ = idxs_[:, 1:] # Remove closest point, which is the point itself

    edge_origins_ = np.repeat(np.arange(len(points_)), k)
    edge_ends_ = np.reshape(idxs_, (-1))

    graph_edge_index_ = torch.tensor([edge_origins_, edge_ends_], dtype=torch.long)
    
    graph_pos_ = torch.tensor(np.vstack((cloud['vertex']['x'],
                                        cloud['vertex']['y'],
                                        cloud['vertex']['z'])), dtype=torch.float).transpose(0, 1)

    #graph_y_ = torch.tensor(cloud['vertex']['label'], dtype=torch.long)
    graph_y_ = torch.tensor(labels, dtype=torch.float)
    
    data_ = Data(x = graph_x_, edge_index = graph_edge_index_, pos = graph_pos_, y = graph_y_)
    return data_

  def __init__(self, root, k=3, transform=None, pre_transform=None):
    self.k = k

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
      rpath = scene + "/cloud/"
      for camera in os.listdir(self.raw_dir + "/" + rpath):
        crpath = rpath + camera + "/"
        output += [crpath + s for s in os.listdir(self.raw_dir + "/" + crpath)]

    return output
    #return ["cloud/" + s for s in os.listdir(self.raw_dir + "/cloud/")]

  @property
  def processed_file_names(self):#CAMBIAR ESTO
    return os.listdir(self.processed_dir)
    #return ["unrealhands_k" + str(self.k) + "_" + str(i) + ".pt" for i in range(len(self.raw_file_names))]

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

  def process_threaded(self, p):
    path_cloud = self.raw_paths_processed[p]
    path_joints = (path_cloud[:len(path_cloud)-3] + "json").replace("cloud", "joints")
    #LOG.info("Processing cloud {0} out of {1}".format(p, len(self.raw_paths_processed)))

    #LOG.info(path_cloud)
    #LOG.info(path_joints)
    hands_ = self.read_joints_json(path_joints)
    labels = hands_["left_hand"]+hands_["right_hand"]
    #print(labels)

    with open(self.raw_paths_processed[p], 'rb') as f:
      print(self.raw_paths_processed[p])
      cloud_ = PlyData.read(f)
      if(len(cloud_['vertex']['x']) > 0):
      	graph_ = self._create_graph(cloud_, self.k, labels)
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
