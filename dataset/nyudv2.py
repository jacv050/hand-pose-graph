import logging
import os

import numpy as np

from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

import scipy.spatial

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

log = logging.getLogger(__name__)

class NYUDv2(InMemoryDataset):

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

		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return ["cloud/" + s for s in os.listdir(self.raw_dir + "/cloud/")]

	@property
	def processed_file_names(self):
		return ["nyudv2_k"+ str(self.k) + ".pt"]

	def download(self):
		log.info("Data not found...")
		raise RuntimeError("Dataset not found, please download it!")

	def process(self):

		data_list_ = []

		log.info("Processing dataset...")

		for p in range(len(self.raw_paths)):
			log.info("Processing cloud {0} out of {1}".format(p, len(self.raw_paths)))
			log.info(self.raw_paths[p])
			with open(self.raw_paths[p], 'rb') as f:
				cloud_ = PlyData.read(f)
				graph_ = self._create_graph(cloud_, self.k)
				data_list_.append(graph_)

		data_ = self.collate(data_list_)
		torch.save(data_, self.processed_paths[0])
