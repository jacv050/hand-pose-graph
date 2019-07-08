import torch
import torch.nn.functional as F
import torch.nn as nn

# Recommend
class RootMeanSquareErrorLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(RootMeanSquareErrorLoss, self).__init__()
    self.mse = nn.MSELoss()
    self.eps = 1e-6

  def forward(self, inputs, targets):
    return torch.sqrt(self.mse(inputs, targets) + self.eps)
