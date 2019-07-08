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
    self.dropout = 0.5

    self.conv1 = GCNConv(numFeatures, 16)
    torch.nn.init.xavier_uniform_(self.conv1.weight)
    self.relu1 = nn.LeakyReLU(0.2, inplace=True)
    #self.top_pooling1 = TopKPooling(16, ratio=0.8)
    self.conv2 = GCNConv(16, 16)
    torch.nn.init.xavier_uniform_(self.conv2.weight)
    self.relu2 = nn.LeakyReLU(0.2, inplace=True)
    self.conv3 = GCNConv(16, 16)
    torch.nn.init.xavier_uniform_(self.conv3.weight)
    self.relu3 = nn.LeakyReLU(0.2, inplace=True)
    """
    self.conv4 = GCNConv(32, 64)
    torch.nn.init.xavier_uniform_(self.conv4.weight)
    self.conv5 = GCNConv(32, 32)
    torch.nn.init.xavier_uniform_(self.conv5.weight)
    self.conv6 = GCNConv(32,32)
    torch.nn.init.xavier_uniform_(self.conv6.weight)
    self.conv7 = GCNConv(32, 32)
    torch.nn.init.xavier_uniform_(self.conv7.weight)
    self.conv8 = GCNConv(32, 32)
    torch.nn.init.xavier_uniform_(self.conv8.weight)
    self.conv9 = GCNConv(32, 32)
    torch.nn.init.xavier_uniform_(self.conv9.weight)
    self.conv10 = GCNConv(32, 32)
    torch.nn.init.xavier_uniform_(self.conv10.weight)
    """
    #self.top_pooling2 = TopKPooling(16, ratio=0.8)
    #self.conv3 = GCNConv(128, 96)
    #torch.nn.init.xavier_uniform_(self.conv2.weight)
    #self.top_pooling2 = TopKPooling(128, ratio=0.3)

    #self.conv1_2 = GCNConv(numFeatures, 16)
    #torch.nn.init.xavier_uniform_(self.conv1_2.weight)
    #self.top_pooling1_2 = TopKPooling(16, ratio=0.5)
    #self.conv2_2 = GCNConv(16, 16)
    #torch.nn.init.xavier_uniform_(self.conv2_2.weight)
    #self.top_pooling2_2 = TopKPooling(16, ratio=0.5)

    #13056 5584 704 14288 22320 44640 89280 11160
    self.fc1_1   = nn.Linear(22320, 1024)
    torch.nn.init.xavier_uniform_(self.fc1_1.weight)
    self.fc2_1   = nn.Linear(1024, 1024)
    torch.nn.init.xavier_uniform_(self.fc2_1.weight)
    self.fc3_1   = nn.Linear(1024, 96)
    torch.nn.init.xavier_uniform_(self.fc3_1.weight)

    #self.fc1_2   = nn.Linear(5584, 48)
    #torch.nn.init.xavier_uniform_(self.fc1_2.weight)
    #torch.nn.init.xavier_uniform_(self.fc1.bias)
    #self.fc2   = nn.Linear(512, 256)
    #torch.nn.init.xavier_uniform_(self.fc2.weight)

  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    """ I
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

    #x[:,:3] = torch.div(x[:,:3], 255)
    #x[:,3:] = torch.div(x[:,3:],0.6)

    x1 = self.conv1(x, edge_index)
    x1 = self.relu1(x1)
    #x1 = torch.tanh(x1)
    edge_index1 = edge_index
    #x1, edge_index1, _, batch1, _ = self.top_pooling1(x1, edge_index, None, batch)
    #x1 = F.dropout(x1, p=0.1, training=self.training)
    x1 = self.conv2(x1, edge_index1)
    x1 = self.relu2(x1)
    #x1 = torch.tanh(x1)
    #x1 = F.dropout(x1, p=0.1, training=self.training)
    x1 = self.conv3(x1, edge_index1)
    x1 = self.relu3(x1)
    #x1 = torch.tanh(x1)
    #x1 = F.dropout(x1, p=0.1, training=self.training)
    """
    x1 = self.conv4(x1, edge_index1)
    x1 = F.relu(x1)
    x1 = F.dropout(x1, p=self.dropout, training=self.training)
    x1 = self.conv5(x1, edge_index1)
    x1 = F.relu(x1)
    x1 = F.dropout(x1, p=self.dropout, training=self.training)
    x1 = self.conv6(x1, edge_index1)
    x1 = F.relu(x1)
    x1 = F.dropout(x1, p=self.dropout, training=self.training)
    x1 = self.conv7(x1, edge_index1)
    x1 = F.relu(x1)
    x1 = F.dropout(x1, p=self.dropout, training=self.training)
    x1 = self.conv8(x1, edge_index1)
    x1 = F.relu(x1)
    x1 = F.dropout(x1, p=self.dropout, training=self.training)
    x1 = self.conv9(x1, edge_index1)
    x1 = F.relu(x1)
    x1 = F.dropout(x1, p=self.dropout, training=self.training)
    x1 = self.conv10(x1, edge_index1)
    x1 = F.relu(x1)
    x1 = F.dropout(x1, p=self.dropout, training=self.training)
    """

    #x1, edge_index1, _, batch1, _ = self.top_pooling2(x1, edge_index1, None, batch1)

    """
    x2 = self.conv1_2(x, edge_index)
    x2 = F.relu(x2)
    x2, edge_index2, _, batch2, _ = self.top_pooling1_2(x2, edge_index, None, batch)
    x2 = self.conv2_2(x2, edge_index2)
    x2 = F.relu(x2)
    x2, edge_index2, _, batch2, _ = self.top_pooling2_2(x2, edge_index2, None, batch2)
    """

    x1 = x1.view(-1)
    x1 = self.fc1_1(x1)
    #x1 = torch.tanh(x1)
    #x1 = F.dropout(x1, p=self.dropout, training=self.training)
    x1 = self.fc2_1(x1)
    #x1 = torch.tanh(x1)
    #x1 = F.dropout(x1, p=self.dropout, training=self.training)
    x1 = self.fc3_1(x1)
    #x1 = torch.tanh(x1)
    #x1 = F.dropout(x1, p=self.dropout, training=self.training)
    #x1 = torch.tanh(x1)

    """
    x2 = x2.view(-1)
    x2 = self.fc1_2(x2)
    x2 = torch.tanh(x2)
    """

    #return torch.cat([x1,x2])
    return x1
