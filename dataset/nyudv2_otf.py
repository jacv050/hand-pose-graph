import logging
import os

import numpy as np

from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

import scipy.spatial

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data

LOG = logging.getLogger(__name__)

class NYUDv2(Dataset):

  def _create_graph(self, cloud, k):

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

    graph_y_ = torch.tensor(cloud['vertex']['label'], dtype=torch.long)
    
    data_ = Data(x = graph_x_, edge_index = graph_edge_index_, pos = graph_pos_, y = graph_y_)
    return data_

  def __init__(self, root, k=3, transform=None, pre_transform=None):

    self.k = k

    super(NYUDv2, self).__init__(root, transform, pre_transform)

  @property
  def num_features(self):
    return self[0].num_features

  @property
  def num_classes(self):
    r"""The number of classes in the dataset."""
    return 41

  @property
  def raw_file_names(self):
    return ["cloud/" + s for s in os.listdir(self.raw_dir + "/cloud/")]

  @property
  def processed_file_names(self):
    return ["nyudv2_k" + str(self.k) + "_" + str(i) + ".pt" for i in range(len(self.raw_file_names))]

  def __len__(self):
    return len(self.processed_file_names)

  def download(self):
    LOG.info("Data not found...")
    raise RuntimeError("Dataset not found, please download it!")

  def process(self):

    LOG.info("Processing dataset...")

    for p in range(len(self.raw_paths)):

      LOG.info("Processing cloud {0} out of {1}".format(p, len(self.raw_paths)))
      LOG.info(self.raw_paths[p])

      with open(self.raw_paths[p], 'rb') as f:

        cloud_ = PlyData.read(f)
        graph_ = self._create_graph(cloud_, self.k)
        torch.save(graph_, os.path.join(self.processed_dir, "nyudv2_k{0}_{1}.pt".format(self.k, p)))

  def get(self, idx):

    data_ = torch.load(os.path.join(self.processed_dir, "nyudv2_k{0}_{1}.pt".format(self.k, idx)))
    return data_