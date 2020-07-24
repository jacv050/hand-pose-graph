import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric.nn import GCNConv, TopKPooling, max_pool

class GCN_test(torch.nn.Module):

  def __init__(self, numFeatures, numClasses):
    super(GCN_test, self).__init__()
    #self.conv1 = GCNConv(numFeatures, 16)
    #self.conv2 = GCNConv(16, numClasses)

    self.conv1 = GCNConv(numFeatures, 16)
    self.top_pooling1 = TopKPooling(16, ratio=0.25)
    self.conv2 = GCNConv(16, 16)
    self.top_pooling2 = TopKPooling(16, ratio=0.25)
    self.conv3 = GCNConv(16, 16)
    self.top_pooling3 = TopKPooling(16, ratio=0.10)

    #print(numFeatures)
    #133524
    self.fc1   = nn.Linear(10000, 256)
    self.fc2   = nn.Linear(256, 96)


  def forward(self, data):

    x, edge_index, batch = data.x, data.edge_index, data.batch

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


    #print(x.size())
    #x = torch.transpose(x,0,1)
    #print(x.size())

    #x = F.log_softmax(x, dim=1)
    #x = torch.mean(x,dim=0)
    #print(x.size())
    x = self.fc1(x.view(-1))
    #x = self.fc1(x)
    x = self.fc2(x)
    #x = F.dropout(x, training=self.training)
    #print(x.size())
    return x
