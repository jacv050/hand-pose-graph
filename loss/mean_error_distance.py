import torch
import torch.nn.functional as F
import torch.nn as nn

# Recommend
class MeanErrorDistanceLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(MeanErrorDistanceLoss, self).__init__()
    self.mae = nn.L1Loss()

  def forward(self, inputs, targets):
    N = int(inputs.shape[0]/3)
    return torch.mean(torch.norm(inputs.view(N, 3)-targets.view(N, 3), dim=1))

  def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
