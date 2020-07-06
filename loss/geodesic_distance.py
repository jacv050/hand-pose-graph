import torch
import torch.nn.functional as F
import torch.nn as nn

# Recommend
class GeodesicDistanceLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(GeodesicDistanceLoss, self).__init__()
    self.mse = nn.MSELoss(reduction='sum')

  def forward(self, inputs, targets):
    #w = torch.mul(inputs[0], targets[0])
    #x = torch.mul(inputs[1], targets[1])
    #y = torch.mul(inputs[2], targets[2])
    #z = torch.mul(inputs[3], targets[3])
    #return torch.sub(1, torch.sum(torch.tensor([w,x,y,z])))
    #print(inputs*targets)
    #return torch.sub(targets.shape[0]/4, torch.sum(inputs*targets))
    return torch.abs(torch.sub(targets.shape[0]/4, torch.sum(inputs*targets)))
