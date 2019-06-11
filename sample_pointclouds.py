import open3d as o3d
import numpy as np
import os
import pathlib

ROOT = "data/unrealhands/raw/"

def main():
  print(ROOT)
  max_points = 0
  for dir in os.listdir(ROOT):
    in_dir  = ROOT + dir + "/cloud"
    out_dir = ROOT + dir + "/cloud_sampled"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(in_dir)
    for camera in os.listdir(in_dir):
      cin_dir  = in_dir + "/" + camera
      cout_dir = out_dir + "/" + camera
      pathlib.Path(cout_dir).mkdir(parents=True, exist_ok=True)
      for cloud in os.listdir(cin_dir):
        pcd     = o3d.io.read_point_cloud(cin_dir + "/" + cloud)
        downpcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=0.01)
        print(np.asarray(pcd.points).shape)
        size = np.asarray(downpcd.points).shape
        print(size)
        if(size[0] > max_points):
          max_points = size[0]
        o3d.io.write_point_cloud(cout_dir + "/" + cloud ,downpcd)

  print("Maxpoints")
  print(max_points)
  #pcd = o3d.geometry.PointCloud()
  #pcd.points

if __name__ == "__main__":
  main()
