import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric.nn import GCNConv, TopKPooling, max_pool
import random

class GCN_test(torch.nn.Module):

  def __init__(self, numFeatures, numClasses):
    super(GCN_test, self).__init__()
    #self.conv1 = GCNConv(numFeatures, 16)
    #self.conv2 = GCNConv(16, numClasses)

    self.conv1 = GCNConv(numFeatures, 16)
    self.top_pooling1 = TopKPooling(16, ratio=0.5)
    self.conv2 = GCNConv(16, 16)
    self.top_pooling2 = TopKPooling(16, ratio=0.5)
    self.conv3 = GCNConv(16, 16)
    self.top_pooling3 = TopKPooling(16, ratio=0.5)
    self.conv4 = GCNConv(16, 16)
    self.top_pooling4 = TopKPooling(16, ratio=0.5)

    #13056
    self.fc1   = nn.Linear(1408, 1024)
    self.fc2   = nn.Linear(1024, 512)
    self.fc3   = nn.Linear(512, 96)

  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    """
    size=x.size(0)
    print(size)
    if(size > 500):
      diff = size - 500
      indexes = np.arange(size)
      random.shuffle(indexes)
      print(len(indexes))
      x = x[indexes[diff:]] #deleted 'diff'
      for index in indexes[:diff]:
        v = edge_index == index
        col = v.size(1)
        np.arange(col)
        l1 = torch.arange(col)[v[0,:]]
        l2 = torch.arange(col)[v[1,:]]
        l = np.concatenate((l1,l2))
        #print(edge_index.size())
        edge_index=edge_index[:, np.delete(np.arange(col), l)]
        #print(edge_index.size())
    """

    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x, edge_index, _, batch, _ = self.top_pooling1(x, edge_index, None, batch)
    #x = F.dropout(x, training=self.training)
    x = self.conv2(x, edge_index)
    x = F.relu(x)
    x, edge_index, _, batch, _ = self.top_pooling2(x, edge_index, None, batch)


    x = self.conv3(x, edge_index)
    x = F.relu(x)
    x, edge_index, _, batch, _ = self.top_pooling3(x, edge_index, None, batch)
    x = self.conv4(x, edge_index)
    x = F.relu(x)
    x, edge_index, _, batch, _ = self.top_pooling4(x, edge_index, None, batch)


    #x = F.log_softmax(x, dim=1)
    #x = torch.mean(x,dim=0)
    x = x.view(-1)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    #x = F.dropout(x, training=self.training)
    #print(x.size())
    return x
