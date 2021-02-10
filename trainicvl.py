# _*_ coding: utf-8 _*_

""" 2D-3D-SeGCN training script.

This is the main training script for the 2D-3D-SeGCN project. For the moment, it just trains the 3D
part of the network.

Todo:
    * Loss function (criterion) factory
"""
__author__ = "Alberto Garcia-Garcia"
__copyright__ = "Copyright 2019, 3D Perception Lab"
__credits__ = ["Alberto Garcia-Garcia"]

__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Alberto Garcia-Garcia"
__email__ = "agarcia@dtic.ua.es"
__status__ = "Development"

import argparse
import logging
import sys

from timeit import default_timer as timer

import png
import numpy as np
import torch
from torch_geometric.data import DataLoader

import loss.factory
import network_3d.utils
#import dataset.icvl_otf
import dataset.icvl_normalized_otf
import json
import tqdm

import utils.gnt_cloud as gnt_cloud

import time

LOG = logging.getLogger(__name__)

NUM_WORKERS = 2 
""" int: Number of thread workers to use when loading the dataset. """

PALETTE = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (191, 0, 0),
    (0, 191, 0),
    (0, 0, 191),
    (191, 191, 0),
    (191, 0, 191),
    (0, 191, 191),
    (191, 162, 155),
    (197, 168, 161),
    (203, 174, 167),
    (209, 180, 173),
    (215, 186, 179),
    (221, 192, 185),
    (227, 198, 191),
    (233, 204, 197),
    (239, 210, 203),
    (245, 216, 209),
    (251, 222, 215),
    (1, 228, 221),
    (7, 234, 227),
    (13, 240, 233),
    (19, 246, 239),
    (25, 252, 245),
    (31, 2, 251),
    (37, 8, 1),
    (43, 14, 7),
    (49, 20, 13),
    (55, 26, 19),
    (61, 32, 25),
    (67, 38, 31),
    (73, 44, 37),
    (79, 50, 43),
    (85, 56, 49),
    (91, 62, 55),
    (97, 68, 61),]
""" ndarray(dtype=(u1, u1, u1)): List of tuples (RGB triplets from 0-255) that conform the palette
    of the dataset to color the ground truth or predictions. """

def train(args):
    """ Training function for the 3D network.

    This function is responsible for orchestrating the whole training procedure for the 3D network:
        (1) creating the dataset and its loaders.
        (2) selecting a compute device.
        (3) building the model.
        (4) creating the criterion for the loss and the optimizer.
        (5) running the training loop.

    Args:
        args : The arguments of the main program.
    """

    #dataset_ = dataset.icvl_otf.ICVL(root="../icvl/training/", k=args.k) #radius = ?
    dataset_ = dataset.icvl_normalized_otf.ICVL(root="../icvl/training/", k=args.k) #radius = ?
    LOG.info("Training dataset...")
    LOG.info(dataset_)

    train_loader_ = DataLoader(dataset_,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=NUM_WORKERS)
    iterator = iter(train_loader_)
    print(next(iterator).y.shape)

    ## Select CUDA device
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    LOG.info(device_)
    LOG.info(torch.cuda.get_device_name(0))

    ## Build model
    model_ = network_3d.utils.get_network(args.net,
                                          dataset_.num_features,
                                          dataset_.num_classes).to(device_)
    LOG.info(model_)

    ## Loss criterion
    criterion_ = loss.factory.get_loss(args.loss)
    LOG.info(criterion_)

    ## Optimizer
    #optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.lr) #5e-8
    #optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.lr, weight_decay=1e-10) #5e-8
    optimizer_ = torch.optim.SGD(model_.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    LOG.info(optimizer_)

    mepoch = 54
    #checkpoint = torch.load('../icvl/training/models2020/model_{}.pt'.format(str(mepoch-1).zfill(3)))
    #model_.load_state_dict(checkpoint)

    time_start_ = timer()
    for epoch in range(args.epochs):
    #for epoch in range(mepoch, args.epochs, 1):
        model_.train()
        LOG.info("Training epoch {0} out of {1}".format(epoch+1, args.epochs))

        loss_all = 0

        i = 1
        counter = 0
        batch_counter = 0
        losses_ = []
        for batch in tqdm.tqdm(train_loader_):

            batch = batch.to(device_)
            optimizer_.zero_grad()
            output_ = model_(batch)
            #print(output_)

            losses_.append(criterion_(output_, batch.y))
            #print(batch.y)

            counter += 1 #batch.y.size(0) #size is not [1,96]

            i = i+1

            batch_size = 32
            if (counter % batch_size) == 0:
              batch_counter += 1
              #loss_ = torch.div(sum(losses_), batch_size)
              loss_ = sum(losses_)
              loss_all += loss_.item()

              #regularization
              #reg_loss = 0
              #for param in model_.parameters():
              #  reg_loss = torch.sum(torch.abs(param)) + reg_loss
              #l1_lambda = 0.00005
              #loss_ = loss_ + l1_lambda * reg_loss

              loss_.backward()
              optimizer_.step()
              losses_ = []

              printable = 480
              if (counter % printable) == 0:
                print(loss_all/counter)

        LOG.info("Training loss {0}".format(loss_all/counter))
        with open('losses_outputs/output_{}.txt'.format(str(epoch).zfill(3)), 'w') as f:
          f.write(str(loss_all/counter))

        # Evaluate on training set
        if (epoch + 1) % 1 == 0:
            torch.save(model_.state_dict(), '../icvl/training/models2020/model_{}.pt'.format(str(epoch).zfill(3)))

    time_end_ = timer()
    LOG.info("Training took {0} seconds".format(time_end_ - time_start_))

#Generate points from relative
def generate_listofpoints(labels):
  output = []
  root_hand = np.array(labels[0:3])
  output.append(root_hand)

  i = 3
  while i < len(labels):
    aux_finger = np.array(labels[i:i+3])+root_hand
    output.append(aux_finger)
    i=i+3
    aux_finger = np.array(labels[i:i+3])+aux_finger
    output.append(aux_finger)
    i = i+3
    aux_finger = np.array(labels[i:i+3])+aux_finger
    output.append(aux_finger)
    i = i+3

  return output

#Generate points from absolute
def generate_listofpoints2(labels):
  output = []
  root_hand = np.array(labels[0:3])
  output.append(root_hand)

  i = 3
  while i < len(labels):
    aux_finger = np.array(labels[i:i+3])
    output.append(aux_finger)
    i=i+3
    aux_finger = np.array(labels[i:i+3])
    output.append(aux_finger)
    i = i+3
    aux_finger = np.array(labels[i:i+3])
    output.append(aux_finger)
    i = i+3

  return output

if __name__ == "__main__":

    PARSER_ = argparse.ArgumentParser(description="Parameters")
    PARSER_.add_argument("--batch_size", nargs="?", type=int, default=1, help="Batch Size")
    PARSER_.add_argument("--epochs", nargs="?", type=int, default=512, help="Training Epochs")
    PARSER_.add_argument("--lr", nargs="?", type=float, default=0.01, help="Learning Rate")
    PARSER_.add_argument("--k", nargs="?", type=int, default=7, help="k Nearest Neighbors")
    PARSER_.add_argument("--net", nargs="?", default="GCN_hand3", help="Network model")
    PARSER_.add_argument("--loss", nargs="?", default="mean_absolute_error", help="Loss criterion")

    ARGS_ = PARSER_.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train(ARGS_)
