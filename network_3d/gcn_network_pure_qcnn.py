import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric.nn import GCNConv, TopKPooling, max_pool_x, graclus, voxel_grid, max_pool, avg_pool
import random
import math
import sys
sys.path.insert(1, 'network_3d/qcnn')
#from application.app.qcnn.Qconv import QConv1d, QConv2d
from Qconv import QConv1d, QConv2d

class GCN_test(torch.nn.Module):

  def __init__(self, numFeatures, numClasses):
    super(GCN_test, self).__init__()
    #self.conv1 = GCNConv(numFeatures, 16)
    #self.conv2 = GCNConv(16, numClasses)
    self.dropout = 0.5

    self.conv1 = GCNConv(numFeatures, 32)
    torch.nn.init.xavier_uniform_(self.conv1.weight)
    #self.conv1.weight.data.uniform_(0.00001, 0.000001)
    #self.conv1.weight.data.fill_(0.0001)
    self.relu1 = nn.LeakyReLU(0.2, inplace=True)
    #self.top_pooling1 = TopKPooling(16, ratio=0.8)
    self.conv2 = GCNConv(32, 30)
    torch.nn.init.xavier_uniform_(self.conv2.weight)
    #self.conv2.weight.data.uniform_(0.00001, 0.01)
    #self.conv2.weight.data.fill_(0.0001)
    self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    #13056 5584 704 14288 22320 44640 89280 11160
    #22320
    #4185
    self.qconv2d1 = QConv2d(30, 64)
    self.qconv2d2 = QConv2d(64*3, 64)
    self.qconv2d3 = QConv2d(64*3, 64)

    #Before
    #self.conv2d1 = nn.Conv2d(1, 3, (3, 2))
    #self.conv2d2 = nn.Conv2d(3, 6, (3, 2))
    #self.conv2d3 = nn.Conv2d(6, 3, (3, 2))
    #self.conv2d4 = nn.Conv2d(3, 1, (3, 2))
    #self.pool3 = nn.MaxPool2d((3,1),(1,1))
    #self.pool2 = nn.MaxPool2d((2,1),(2,1))
    #self.fc1_1   = nn.Linear(2380, 1024)#1680

    self.fc1_1   = nn.Linear(267840, 1024)#1680
    torch.nn.init.xavier_uniform_(self.fc1_1.weight)
    self.fc2_1   = nn.Linear(1024, 1024)
    torch.nn.init.xavier_uniform_(self.fc2_1.weight)
    self.fc3_1   = nn.Linear(1024, 64) #64 16
    torch.nn.init.xavier_uniform_(self.fc3_1.weight)

  def forward(self, data):
    x, edge_index, batch, pos = data.x, data.edge_index, data.batch, data.pos

    x[:,:3] = 0 #torch.div(x[:,:3], 255)
    #x[:,3:] = torch.div(x[:,3:],0.6)

    x1 = self.conv1(x, edge_index)
    x1 = self.relu1(x1)
    #x1 = torch.tanh(x1)
    edge_index1 = edge_index

    x1 = self.conv2(x1, edge_index1)
    x1 = self.relu2(x1)

    x1 = x1.unsqueeze(2).unsqueeze(2)

    x1 = self.qconv2d1(x1)
    x1 = self.qconv2d2(x1)
    x1 = self.qconv2d3(x1)

    #x1 = self.pool2(F.relu(self.conv2d1(x1.unsqueeze(0).unsqueeze(0))))
    #x1 = self.pool2(F.relu(self.conv2d2(x1)))
    #x1 = self.pool2(F.relu(self.conv2d3(x1)))
    #x1 = self.pool2(F.relu(self.conv2d4(x1)))
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

    """ I
    x2 = x2.view(-1)
    x2 = self.fc1_2(x2)
    x2 = torch.tanh(x2)
    """
    #print(x1)
    #return x1
    #return torch.cat([x1,x2])

    magnitude=torch.zeros(x1.shape[0], requires_grad=False, device=torch.cuda.current_device())
    for i in range(int(magnitude.shape[0]/4)):
      aux=torch.pow(x1[i*4:i*4+4],2)
      magnitude[i*4:i*4+4]=torch.sqrt(torch.sum(aux))
    x1=torch.div(x1,magnitude)

    return x1
