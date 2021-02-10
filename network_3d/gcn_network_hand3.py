import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric.nn import GCNConv, TopKPooling, max_pool_x, graclus, voxel_grid, max_pool, avg_pool
import random
import math

class GCN_test(torch.nn.Module):

  def __init__(self, numFeatures, numClasses):
    super(GCN_test, self).__init__()
    #self.conv1 = GCNConv(numFeatures, 16)
    #self.conv2 = GCNConv(16, numClasses)

    self.conv1 = GCNConv(numFeatures, 1)
    torch.nn.init.xavier_uniform_(self.conv1.weight)
    self.relu1 = nn.LeakyReLU(0.2, inplace=True)
    self.conv2 = GCNConv(1, 1)
    torch.nn.init.xavier_uniform_(self.conv2.weight)
    self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    #13056 5584 704 14288 22320 44640 89280 11160
    #22320
    self.fc1_1   = nn.Linear(8370, 2048) #4185
    torch.nn.init.xavier_uniform_(self.fc1_1.weight)
    #self.dropout1 = nn.Dropout(p=0.3)
    #self.relufc1_1 = nn.LeakyReLU(0.2, inplace=True)
    self.fc2_1   = nn.Linear(2048, 4096)
    torch.nn.init.xavier_uniform_(self.fc2_1.weight)
    #self.dropout2 = nn.Dropout(p=0.3)
    #self.relufc2_1 = nn.LeakyReLU(0.2, inplace=True)

    self.conv2d1 = nn.Conv2d(1,32,(7,7))
    torch.nn.init.xavier_uniform_(self.conv2d1.weight)
    self.relu2d1 = nn.LeakyReLU(0.2, inplace=True)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2d2 = nn.Conv2d(32,32,(5,5))
    torch.nn.init.xavier_uniform_(self.conv2d2.weight)
    self.relu2d2 = nn.LeakyReLU(0.2, inplace=True)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv2d3 = nn.Conv2d(32,64,(3,3))
    torch.nn.init.xavier_uniform_(self.conv2d3.weight)
    self.relu2d3 = nn.LeakyReLU(0.2, inplace=True)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv2d4 = nn.Conv2d(64,64,(3,3))
    torch.nn.init.xavier_uniform_(self.conv2d4.weight)
    self.relu2d4 = nn.LeakyReLU(0.2, inplace=True)
    #self.bn4 = nn.BatchNorm2d(64)

    self.fc3_1   = nn.Linear(160000, 2048)
    torch.nn.init.xavier_uniform_(self.fc3_1.weight)
    #self.dropout3 = nn.Dropout(p=0.3)
    #self.relufc3_1 = nn.LeakyReLU(0.2, inplace=True)
    self.fc4_1   = nn.Linear(2048, 2048)
    torch.nn.init.xavier_uniform_(self.fc4_1.weight)
    #self.dropout4 = nn.Dropout(p=0.3)
    #self.relufc4_1 = nn.LeakyReLU(0.2, inplace=True)
    self.fc5_1   = nn.Linear(2048, 48)
    torch.nn.init.xavier_uniform_(self.fc5_1.weight)
    #self.relufc5_1 = nn.LeakyReLU(0.2, inplace=True)

    #self.nll = NLinearLayer(1395, 96)

    #self.fc1_2   = nn.Linear(5584, 48)
    #torch.nn.init.xavier_uniform_(self.fc1_2.weight)
    #torch.nn.init.xavier_uniform_(self.fc1.bias)
    #self.fc2   = nn.Linear(512, 256)
    #torch.nn.init.xavier_uniform_(self.fc2.weight)

  def forward(self, data):
    x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos

    #x[:,:3] = 0 #torch.div(x[:,:3], 255)
    #x[:,3:] = torch.div(x[:,3:],0.6)

    x1 = self.conv1(x, edge_index)
    x1 = self.relu1(x1)
    #x1 = torch.tanh(x1)
    edge_index1 = edge_index

    x1 = self.conv2(x1, edge_index1)
    x1 = self.relu2(x1)

    x1 = x1*x

    #x1, edge_index1, _, batch1, _ = self.top_pooling2(x1, edge_index1, None, batch1)

    x1 = x1.view(-1)
    #x1 = self.nll(x1.t())
    #x1 = torch.sum(x1, dim=0)
    #print(x1)

    x1 = self.fc1_1(x1)
    #x1 = self.dropout1(x1)
    #x1 = self.relufc1_1(x1)
    x1 = self.fc2_1(x1)
    #x1 = self.dropout2(x1)
    #x1 = self.relufc2_1(x1)

    x1 = x1.view(64,64).unsqueeze(0).unsqueeze(0)

    x1 = self.conv2d1(x1)
    x1 = self.relu2d1(x1)
    x1 = self.bn1(x1)
    x1 = self.conv2d2(x1)
    x1 = self.relu2d2(x1)
    x1 = self.bn2(x1)
    x1 = self.conv2d3(x1)
    x1 = self.relu2d3(x1)
    x1 = self.bn3(x1)
    x1 = self.conv2d4(x1)
    x1 = self.relu2d4(x1)
    #x1 = self.bn4(x1)

    x1 = self.fc3_1(x1.view(-1))
    #x1 = self.dropout3(x1)
    #x1 = self.relufc3_1(x1)
    x1 = self.fc4_1(x1)
    #x1 = self.dropout4(x1)
    #x1 = self.relufc4_1(x1)
    x1 = self.fc5_1(x1)
    x1 = torch.tanh(x1)

    return x1
