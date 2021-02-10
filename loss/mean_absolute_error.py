import torch
import torch.nn.functional as F
import torch.nn as nn

# Recommend
class MeanAbsoluteErrorLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(MeanAbsoluteErrorLoss, self).__init__()
    self.mae = nn.L1Loss()

  def forward(self, inputs, targets):
    return self.mae(inputs, targets.view(-1))

  def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
