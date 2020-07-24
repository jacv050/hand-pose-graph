import torch
import torch.nn.functional as F
import torch.nn as nn

# Recommend
class GeodesicDistanceWeightedLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(GeodesicDistanceWeightedLoss, self).__init__()
    self.mse = nn.MSELoss(reduction='sum')
    self.mae = nn.L1Loss()
    self.eps = 1e-6

  def forward(self, inputs, targets):
    prod=inputs*targets
    inner=torch.zeros(int(targets.shape[0]/4), requires_grad=False)
    for i in range(inner.shape[0]):
      aux = torch.sum(prod[i*4:i*4+4])
      inner[i] = aux*aux
      #inner[i]=torch.pow(torch.sum(prod[i*4:i*4+4]),2)
    #inner[0] = inner[0]*1.5 #Weight
    #inner=torch.sum(inner)
    return self.mae(inner, torch.ones(int(targets.shape[0]/4)))
    #return torch.sqrt(self.mse(inner, torch.ones(int(targets.shape[0]/4))) + self.eps)
    #return torch.abs(torch.sub(targets.shape[0]/4, inner))
