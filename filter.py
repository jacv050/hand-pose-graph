import logging
import os
import json

import numpy as np

from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)

import scipy.spatial

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import sys
import glob
import tqdm
import multiprocessing.dummy as mp

LOG = logging.getLogger(__name__)

output = []
raw_dir = "/workspace/hand-pose-graph/data/unrealhands/raw"

def threaded(idx):
    #print(output)
    #print(output[idx])
    #print(os.path.join(raw_dir, output[idx]))
    with open(os.path.join(raw_dir, output[idx]), 'rb') as f:
      cloud_ = PlyData.read(f)
      if(len(cloud_['vertex']['x']) == 0):
        os.remove(os.path.join(raw_dir, output[idx]))
        print(os.path.join(raw_dir, output[idx]))


def main():
    pool = mp.Pool(12)

    #raw_dir = "/workspace/hand-pose-graph/data/unrealhands/raw"
    for scene in tqdm.tqdm(os.listdir(raw_dir)):
      rpath = scene + "/cloud/"
      for camera in os.listdir(raw_dir + "/" + rpath):
        crpath = rpath + camera + "/"
        global output
        output += [crpath + s for s in os.listdir(raw_dir + "/" + crpath)]

        #pool.map(threaded, range(len(output)))
        """
        for p in output:
          with open(os.path.join(raw_dir, p), 'rb') as f:
            cloud_ = PlyData.read(f)
            if(len(cloud_['vertex']['x']) == 0):
              print(os.path.join(raw_dir, p))
        """
              #os.remove(os.path.join(raw_dir, p))
    for i, _ in enumerate(pool.imap_unordered(threaded, range(len(output)),1)):
      sys.stdout.write('\rdone {0:%}\n'.format(i/len(output)))
      sys.stdout.flush()
    #pool.map(threaded, range(len(output)))


if __name__ == "__main__":
    main()
