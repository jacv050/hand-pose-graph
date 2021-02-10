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
from dataset.datahand import DataHand

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

    dataset_ = dataset.icvl_normalized_otf.ICVL(root="../icvl/testing/", k=args.k)
    #dataset_ = dataset.icvl_otf.ICVL(root="../icvl/testing/", k=args.k)
    LOG.info("Training dataset...")
    LOG.info(dataset_)
    train_loader_ = DataLoader(dataset_,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=NUM_WORKERS)
    iterator = iter(train_loader_)
    print(next(iterator).y.shape)

    ## Select CUDA device
    #device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_ = torch.device('cpu')
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
    #optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.lr, weight_decay=1e-10) #5e-8
    optimizer_ = torch.optim.SGD(model_.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    #optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.lr)
    LOG.info(optimizer_)

    mepoch = 512
    #checkpoint = torch.load('data/models/model_{}.pt'.format(str(mepoch-1).zfill(3)))
    #model_.load_state_dict(checkpoint)

    loss_gen = False

    time_start_ = timer()
    for epoch in range(40, 60,5): #range(237, 700, 1):
        torch.cuda.empty_cache()
        mepoch = epoch+1
        #icvl weights
        #checkpoint = torch.load('../icvl/training/models2020/model_{}.pt'.format(str(mepoch-1).zfill(3)))
        checkpoint = torch.load('../icvl/training/models2020/model_{}.pt'.format(str(mepoch-1).zfill(3)), map_location=device_)
        #checkpoint = torch.load('../icvl/training/models2020_gcn_hand/model_{}.pt'.format(str(61).zfill(3)), map_location=device_)
        #checkpoint = torch.load('../models2020/model_{}.pt'.format(str(mepoch-1).zfill(3)))
        model_.load_state_dict(checkpoint)

        model_.eval()

        LOG.info("Training epoch {0} out of {1}".format(epoch+1, args.epochs))

        loss_all = 0

        j = 1
        counter = 0
        cmean = 0
        eucmean = 0
        meanj = 0
        for batch in tqdm.tqdm(train_loader_):
            counter += 1

            rotate = batch.rotate
            length = batch.length
            offset = batch.offset
            batch = batch.to(device_)
            output_ = model_(batch)
            #print(rotate.size())

            if loss_gen:
              loss_ = criterion_(output_, batch.y)
              loss_all += loss_.item()
            else:
              save_test = True
              #pred_ = model_(batch)

              #aux = pred_
              aux = output_
              aux2 = batch.y.view(-1)
              #l = np.array([aux[i].item() for i in range(48)])
              l = np.matmul(length.detach().numpy()*(np.array([aux[i].item() for i in range(48)]).reshape((16,3)) + offset.detach().numpy()), rotate.transpose(0,1).numpy()).flatten() #* length.detach().numpy()
              #ground_truth = np.array([aux2[i].item() for i in range(48)])
              ground_truth = np.matmul(length.detach().numpy()*(np.array([aux2[i].item() for i in range(48)]).reshape((16,3)) + offset.detach().numpy()), rotate.transpose(0,1).numpy()).flatten() #* length.detach().numpy()

              N = int(aux.size(0)/3)
              meanj += torch.mean(torch.norm(aux.view(N,3)-aux2.view(N,3), dim=1))*length

              if save_test :
                #gnt_cloud.save_ply_cloud(np.transpose(batch.pos.cpu(), (0,1)), np.transpose( batch.x.cpu()[:,:3]*255, (0,1)), 'outputs_clouds_validation/output_cloud_{}.ply'.format(j+1))
                #joints  = np.array(generate_listofpoints2(l))
                #gnt_cloud.save_ply_cloud(joints, np.repeat([[255,255,255]], joints.shape[0], axis=0),'outputs_joints_validation/output_joints_{}.ply'.format(j+1))
                #output_error
                with open('output_error_validation/output_{}.json'.format(str(j+1).zfill(3)), 'w') as f:
                  #error = np.abs(np.array(ground_truth)-np.array(l))
                  error = np.abs(l-ground_truth)
                  output_error = {}
                  output_error["output"] = l.tolist()
                  output_error["output_ground_truth"] = ground_truth.tolist()
                  output_error["error"] = error.tolist()
                  output_error["mean_error"] = error.mean()
                  euclidean_distance = [ np.sqrt(np.power(error[i:i+3],2).sum()) for i in range(0,error.shape[0],3)]
                  output_error["euclidean_distance"] = euclidean_distance
                  output_error["euclidean_distance_mean"] = np.array(euclidean_distance).mean()
                  json.dump(output_error, f)
                  cmean += output_error["mean_error"]
                  eucmean += output_error["euclidean_distance_mean"]
              #"""

            j = j+1
        print("Mean error: {}".format(cmean/counter))
        print("Mean error2: {}".format(meanj/counter))
        print("Mean euclidean error: {}".format(eucmean/counter))
        LOG.info("Training loss {0}".format(loss_all/counter))
        if loss_gen:
          with open('losses_outputs_validation/output_{}.txt'.format(str(mepoch).zfill(3)), 'w') as f:
            f.write(str(loss_all/counter))

    time_end_ = timer()
    LOG.info("Training took {0} seconds".format(time_end_ - time_start_))

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
    PARSER_.add_argument("--lr", nargs="?", type=float, default=0.0001, help="Learning Rate")
    PARSER_.add_argument("--k", nargs="?", type=int, default=7, help="k Nearest Neighbors")
    PARSER_.add_argument("--net", nargs="?", default="GCN_hand3", help="Network model")
    PARSER_.add_argument("--loss", nargs="?", default="mean_absolute_error", help="Loss criterion")

    ARGS_ = PARSER_.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train(ARGS_)
