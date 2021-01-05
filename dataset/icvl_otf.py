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
from dataset.datahand import DataHand
import sys
import glob

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import copy
import utils.qutils as qutils
import random

import pcl
import multiprocessing.dummy as mp
from dataset.utils.plot_pointcloud import Plot

LOG = logging.getLogger(__name__)

class ICVL(Dataset):

  def _create_graph(self, cloud, k, labels, radius):
    max_points=1395
    csize=cloud['vertex']['x'].shape[0]

    points_ = np.transpose(np.vstack((cloud['vertex']['x'],
                                      cloud['vertex']['y'],
                                      cloud['vertex']['z'])), (1,0)) #N,3

    #PCL cloud
    p = pcl.PointCloud()
    p.from_array(np.array(points_, dtype=np.float32))
    #Filter outlier
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(1)
    fil.set_negative(False)
    p = fil.filter()
    points_ = p.to_array()[:, :3]
    #Normalize sphere
    points_normalized = self.normalize_points(points_)
    #PCA
    pca = PCA(n_components=3)
    pca.fit(points_normalized)
    components = np.copy(pca.components_)
    if pca.components_[1,0] < 0:
	      components[:,0] = -pca.components_[:,0]
    if pca.components_[2,2]:
      components[:,2] = -pca.components_[:,2]
    components[:,1]=np.cross(components[:,2], components[:,0])

    #Sampling random points Max points 1395
    randomlist = None
    if points_.shape[0] < max_points:
      randomlist = [ i for i in range(0, points_.shape[0])]
      randomlist += random.sample(range(0, points_.shape[0]), max_points-points_.shape[0])
    else:
      randomlist = random.sample(range(0, points_.shape[0]), max_points)

    points_ = points_[randomlist]

    #Rotate points
    points_rotated = np.matmul(points_, components)
    #compute Surface normals
    #normal_k = 30
    #cloud_normals = self.Surface_normals(p, normal_k).to_array()[:,:3]
    #cloud_normals_rotated = np.matmul(cloud_normals, components)

    #Normalize pointcloud
    x_min_max2 = [np.min(points_[:,0]), np.max(points_[:,0])]
    y_min_max2 = [np.min(points_[:,1]), np.max(points_[:,1])]
    z_min_max2 = [np.min(points_[:,2]), np.max(points_[:,2])]

    x_min_max = [np.min(points_rotated[:,0]), np.max(points_rotated[:,0])]
    y_min_max = [np.min(points_rotated[:,1]), np.max(points_rotated[:,1])]
    z_min_max = [np.min(points_rotated[:,2]), np.max(points_rotated[:,2])]

    #Define bounding box
    scale = 1.2
    bb3d_x_len = scale*(x_min_max[1]-x_min_max[0])
    bb3d_y_len = scale*(y_min_max[1]-y_min_max[0])
    bb3d_z_len = scale*(z_min_max[1]-z_min_max[0])
    max_bb3d_len = bb3d_x_len

    offset = np.mean(points_rotated)/max_bb3d_len

    cloud_normalized = points_rotated/max_bb3d_len - offset

    #plot_ = Plot()
    #plot_.cloud(points_, [x_min_max2, y_min_max2, z_min_max2])
    #plot_.cloud(points_rotated, [x_min_max, y_min_max, z_min_max])
    #plot_.cloud(cloud_normalized, [x_min_max, y_min_max, z_min_max])
    #plot_.show()

    #fake rgb TODO delete
    rgb=np.zeros(max_points)
    graph_x_ = torch.tensor(np.vstack((rgb,rgb,rgb,cloud_normalized[:,0],
                                      cloud_normalized[:,1],
                                      cloud_normalized[:,2])), dtype=torch.float).transpose(0, 1)

    #Define graph connections
    tree_ = scipy.spatial.cKDTree(points_)

    #If radius closest point by sphere. With none knn
    idxs_ = None
    if radius is None:
      _, idxs_ = tree_.query(points_, k=k + 1) # Closest point will be the point itself, so k + 1
    else:
      _, idxs_ = tree_.query(points_, k=k + 1, distance_upper_bound=radius) # Closest point will be the point itself, so k + 1

    idxs_ = idxs_[:, 1:] # Remove closest point, which is the point itself

    if len(cloud['vertex']['x']) > self.aux_max:
      self.aux_max = len(cloud['vertex']['x'])

    edge_origins_ = np.repeat(np.arange(len(points_)), k)
    edge_ends_ = np.reshape(idxs_, (-1))

    graph_edge_index_ = torch.tensor([edge_origins_, edge_ends_], dtype=torch.long)

    graph_pos_ = torch.tensor(np.vstack((cloud['vertex']['x'],
                                        cloud['vertex']['y'],
                                        cloud['vertex']['z'])), dtype=torch.float).transpose(0, 1)

    fingers_index= [0, 1, 2, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 39, 40, 41, 42, 43, 44, 45, 46, 47,30, 31, 32, 33, 34,
        35, 36, 37, 38, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    labels = np.matmul(np.array(cloud['gt']['gt'][fingers_index]).reshape((16,3)), components)/max_bb3d_len - offset
    graph_y_ = torch.tensor(labels).view(-1)

    #FIX SIZE TODO delete
    diff = max_points - graph_x_.size(0)
    graph_x_ = torch.cat((graph_x_, torch.tensor(np.repeat([[0,0,0,0,0,0]], diff, axis=0), dtype=torch.float)), 0)
    graph_pos_ = torch.cat((graph_pos_, torch.zeros((diff,3))))
    #FIXED

    data_ = DataHand(x = graph_x_, edge_index = graph_edge_index_, edge_attr=None, y=graph_y_, pos = graph_pos_, normal=None, face=None, rotate=torch.tensor(components), length=max_bb3d_len, offset=offset)
    return data_

  def filter_outliers(self, cloud, stdev_threshold):
    mean = cloud.mean(axis=0)
    stdev = 0
    for point in cloud:
      stdev += np.power( np.linalg.norm(point-mean), 2)
    stdev = np.sqrt(stdev/cloud.shape[0])

    output = []
    for point in cloud:
      distance = np.power( np.linalg.norm(point-mean), 2)
      if distance < stdev*5:
        print(stdev)
        output.append(point)

    return output

  def Surface_normals(self, cloud, normal_k):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(normal_k)

    cloud_normals = ne.compute()
    return cloud_normals.to_array()[:,:3]

  def normalize_points(self, points):
    centroid = points.mean(axis=0)
    meanzero = points-centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(meanzero)**2,axis=-1)))
    return meanzero / furthest_distance

  def __init__(self, root, k=3, radius=None, transform=None, pre_transform=None):
    #self.cloud_folder = "cloud_sampled"
    self.cloud_folder = "cloud"
    self.aux_max = 0
    self.k = k
    self.radius=radius

    super(ICVL, self).__init__(root, transform, pre_transform)

  @property
  def num_features(self):
    return self[0].num_features

  @property
  def num_classes(self):
    """The number of classes in the dataset."""
    return 96

  @property
  def raw_file_names(self):
    """ I
    output = []
    for scene in os.listdir(self.raw_dir):
      rpath = scene + "/" + self.cloud_folder + "/"
      for camera in os.listdir(self.raw_dir + "/" + rpath):
        crpath = rpath + camera + "/"
        output += [crpath + s for s in os.listdir(self.raw_dir + "/training/" + crpath)]
    """
    return [self.cloud_folder + "/" + s for s in os.listdir(self.raw_dir + "/" + self.cloud_folder)]

  @property
  def processed_file_names(self):#CAMBIAR ESTO
    return ["icvl_k" + str(self.k) + "_" + str(i) + ".pt" for i in range(len(self.raw_file_names))]

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
            output += root

            for finger_list in iterator:
              for bone in finger_list:
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
            output += root
            nproot = np.array(root)

            for finger_list in iterator:
              iterator2 = iter(finger_list)
              aux = np.array(next(iterator2)) + nproot
              output += aux.tolist()
              for bone in iterator2:
                aux = np.array(bone)+aux
                output += aux.tolist()
          output_dict[key] = output

    return output_dict

  def generate_listofpoints(self, labels):
    output = []
    for i in range(0,len(labels),3):
      output.append(labels[i:i+3])

  def process_threaded(self, p):
    path_cloud = self.raw_paths_processed[p]
    old_joints = False #OLD_JOINTS
    labels = None
    if old_joints: #DEFAULT FALSE
      path_joints = (path_cloud[:len(path_cloud)-3] + "json").replace(self.cloud_folder, "joints")
      hands_ = self.read_joints_json2(path_joints)
      labels = hands_["left_hand"]+hands_["right_hand"]

    with open(self.raw_paths_processed[p], 'rb') as f:
      cloud_ = PlyData.read(f)
      if(len(cloud_['vertex']['x']) > 0):
      	graph_ = self._create_graph(cloud_, self.k, labels, self.radius)
      	torch.save(graph_, os.path.join(self.processed_dir, "icvl_k{0}_{1}.pt".format(self.k, p)))

  def process(self):
    LOG.info("Processing dataset...")
    self.raw_paths_processed = self.raw_paths
    #TODO DELETE NEXT LINE
    self.process_threaded(0)
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

      with open(self.raw_paths[p], 'rb') as f:
        print(self.raw_paths[p])
        cloud_ = PlyData.read(f)
        graph_ = self._create_graph(cloud_, self.k, labels)
        torch.save(graph_, os.path.join(self.processed_dir, "icvl_k{0}_{1}.pt".format(self.k, p)))

  def get(self, idx):
    data_ = torch.load(os.path.join(self.processed_dir, "icvl_k{0}_{1}.pt".format(self.k, idx)))
    return data_
