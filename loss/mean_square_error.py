import torch
import torch.nn.functional as F
import torch.nn as nn

# Recommend
class MeanSquareErrorLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(MeanSquareErrorLoss, self).__init__()
    self.mse = nn.MSELoss(reduction='sum')

  def forward(self, inputs, targets):
    return self.mse(inputs, targets)