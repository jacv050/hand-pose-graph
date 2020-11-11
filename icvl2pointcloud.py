import numpy as np
import argparse
import pathlib
import multiprocessing.dummy as mp
import sys
from PIL import Image
from plyfile import (PlyData, PlyElement)

def depth2cloud(depth, fx, fy, rx, ry):
  points_ = []
  #fx = 463.889
  #fy = 463.889
  #fx = 240.99
  #fy = 240.96
  #fov = 74.0
  #focal = (rx*0.5)/np.tan(fov*0.5*np.pi/180.0)
  focal = 241.42
  #print(focal)
  fx = focal
  fy = focal
  cx = (rx-1)/2.0
  cy = (ry-1)/2.0

  #print(np.where(depth < 2000))
  #print("Min: {}, Max: {}".format(depth.min(), depth.max()))
  for i in range(0, depth.shape[1]-1):
    for j in range(0, depth.shape[0]-1):
      point_z_ = float(depth[j][i])/1000

      #TODO FIX camera values
      if point_z_ < 2.0:
        point_x_ = (i - cx) * (point_z_ / fx)
        point_y_ = (j - cy) * (point_z_ / fy)

        #TODO no color
        #point_color_ = [color[j][i][0], color[j][i][1], colo

        points_.append([point_x_, point_y_, point_z_])

  return np.array(points_) # np.array(colors_)

def csv_camera_info(line):
  splitted = line.split(',')
  return float(splitted[0]),float(splitted[1]),float(splitted[2]),float(splitted[3])

def csv_annotation(line):
  splitted = line.split(',')
  return int(splitted[0]), np.array([float(v) for v in splitted[1:]])

def save_ply_cloud(points, gt, filename):
  vertex_ = np.zeros(points.shape[0],
    dtype=[('x', 'f4'),('y', 'f4'),('z', 'f4')])

  joints_position = np.zeros(int(gt.shape[0]/3), dtype=[('x','f4'),('y','f4'),('z','f4')])
  gto = np.zeros(gt.shape[0], dtype=[('gt', 'f4')])

  for i in range(points.shape[0]):
    vertex_[i] = (points[i][0], points[i][1], points[i][2])

  for i in range(joints_position.shape[0]):
    joints_position[i] = (gt[i*3], gt[i*3+1], gt[i*3+2])
    gto[i]=gt[i]
    gto[i*3+1]=gt[i*3+1]
    gto[i*3+2]=gt[i*3+2]

  #gt_ = np.zeros(points.shape[0], dtype=[('gt', 'f4')])
  #gt_ = gt

  e1 = PlyElement.describe(vertex_, 'vertex', comments=['vertices'])
  e2 = PlyElement.describe(gto, 'gt', comments=['GroundTruth'])
  e3 = PlyElement.describe(joints_position, 'jointsp', comments=['Joints Position'])
  ply_out = PlyData([e1,e2,e3])
  ply_out.write(filename)

def thread_cloud(line):
  image, gt = csv_annotation(line)
  depth_path = dataset_path + '/depth/depth_1_{}.png'.format(str(image).zfill(7))
  im = Image.open(depth_path)
  depth_file = np.asarray(im)
  #print(depth_file)
  points = depth2cloud(depth_file, fx, fy, rx, ry)
  cloud_file = dataset_path + '/cloud/depth_1_{}.ply'.format(str(image).zfill(7))
  save_ply_cloud(points,gt,cloud_file)

if __name__ == "__main__":

  PARSER_ = argparse.ArgumentParser(description="Parameters")
  PARSER_.add_argument("--annotation", nargs="?", type=str, default=None, help="File annotation path")
  PARSER_.add_argument("--path", nargs="?", type=str, default=None, help="Dataset path")

  ARGS_ = PARSER_.parse_args()
  annotation_file = ARGS_.annotation
  dataset_path = ARGS_.path
  cloud_path = dataset_path+'/cloud'
  pathlib.Path(cloud_path).mkdir(parents=True, exist_ok=True)
  with open(annotation_file, 'r') as f:
    f.readline()
    fx, fy, rx, ry = csv_camera_info(f.readline())
    f.readline()
    f.readline()
    pf = mp.Pool(12)
    for i, _ in enumerate(pf.imap_unordered(thread_cloud, f, 1)):
      #sys.stderr.write('\rdone {0:%}\n'.format(i/scene_.m_total_frames))
      sys.stderr.write('\rdone: {}'.format(i))







