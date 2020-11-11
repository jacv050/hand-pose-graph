import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import os

frames = 2828

euclidean_distance_index = range(40)
mean_error_index = range(40)
euclidean_distance = []
mean_error = []

path = "output_error/"

#PRY
def quaternion2rotation(q):
  qw, qx, qy, qz = q
  c, d, e, f = qx, qy, qz, qw
  g, h, k = c+c, d+d, e+e
  a = c*g
  l=c*h
  c=c*k
  m=d*h
  d=d*k
  e=e*k
  g=f*g
  h=f*h
  f=f*k
  mat = np.array([[1-(m+e),1-f,c+h],[l+f,1-(a+e),d-g],[c-h,d+g,1-(a+m)]])
  a, f, g = mat[0][0], mat[0][1], mat[0][2]
  h, k, l = mat[1][0], mat[1][1], mat[1][2]
  m, n, e = mat[2][0], mat[2][1], mat[2][2]
  eulerx = -np.arcsin(np.clip(l,-1,1))
  eulery = 0
  eulerz = 0
  if .99999>np.abs(l):
    eulery=np.arctan2(g,e)
    eulerz=np.arctan2(h,k)
  else:
    eulery=np.atan2(-m,a)
    eulerz=0

  return [eulerx, eulery, eulerz]


euler_gt  = []
euler_out = []
for dir in os.listdir(path):
  #epoch = path + dir + "/"
  epoch = path + "00799" + "/"
  for output_error in os.listdir(epoch):
    batch_gt = []
    batch_out= []
    with open(epoch + output_error, 'ro') as r:
      dict  = json.load(r)
      ogt = dict['output_ground_truth']
      out = dict['output']
      for i in range(0,len(ogt),4):
        batch_gt.append(np.degrees(quaternion2rotation(ogt[i:i+4])))
        batch_out.append(np.degrees(quaternion2rotation(out[i:i+4])))
      euler_gt.append(np.array(batch_gt))
      euler_out.append(np.array(batch_out))
  break;

#[frames,joint,axis]
gt   = np.array(euler_gt)
out  = np.array(euler_out)
diff = np.abs(out-gt)
#Test plot euler x
plotx=diff[:,:,0].mean(axis=1)
plt.plot(plotx)
plt.axhline(plotx.mean(), color='r', linestyle='--', label='Mean'plot/whisker )
plt.axhline(np.median(plotx), color='g', linestyle='-', label='Median')
plt.ylabel('Error x angles')
plt.xlabel('Frames/Batches')
plt.savefig('eulerx.png')
plt.clf()
ploty=diff[:,:,1].mean(axis=1)
plt.plot(ploty)
plt.axhline(ploty.mean(), color='r', linestyle='--', label='Mean')
plt.axhline(np.median(ploty), color='g', linestyle='-', label='Median')
plt.ylabel('Error y angles')
plt.xlabel('Frames/Batches')
plt.savefig('eulery.png')
plt.clf()
plotz=diff[:,:,2].mean(axis=1)
plt.plot(plotz)
plt.ylabel('Error z angles')
plt.axhline(plotz.mean(), color='r', linestyle='--', label='Mean')
plt.axhline(np.median(plotz), color='g', linestyle='-', label='Median')
plt.xlabel('Frames/Batches')
plt.savefig('eulerz.png')
