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

def onelinenexttrycatch(iterator):
  try:
    return next(iterator)
  except StopIteration:
    return None

euler_gt  = []
euler_out = []
for dir in os.listdir(path):
  #epoch = path + dir + "/"
  epoch = path + "00799" + "/" #Process only the outputs for one epoch
  #Process the output_error for each sample
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

#mean error for each joint
diff_mean = diff.mean(axis=1)

#Sort mean error for each axis
diff_mean[:,0].sort()
diff_mean[:,1].sort()
diff_mean[:,2].sort()

#Ceil max error for each axis
index = diff.shape[0]-1
max_x = math.ceil(diff_mean[index,0])
max_y = math.ceil(diff_mean[index,1])
max_z = math.ceil(diff_mean[index,2])

#Set max error
max = int(np.array([max_x, max_y, max_z]).max())
print(max)

#Resolution to plot thresholded
resolution = 1
output_x, output_y, output_z = [], [], []
samples_x, samples_y, samples_z = 0, 0, 0

iterx = iter(diff_mean[:,0])
itery = iter(diff_mean[:,1])
iterz = iter(diff_mean[:,2])

meanx = next(iterx)
meany = next(itery)
meanz = next(iterz)

for threshold in range(0,max,resolution):
  #Axis X
  while meanx and meanx < threshold:
    samples_x += 1
    meanx = onelinenexttrycatch(iterx)
  output_x.append(samples_x)
  #Axis Y
  while meany and meany < threshold:
    samples_y += 1
    meany = onelinenexttrycatch(itery)
  output_y.append(samples_y)
  #Axis Zd
  while meanz and meanz < threshold:
    samples_z += 1
    meanz = onelinenexttrycatch(iterz)
  output_z.append(samples_z)

#Test plot euler x
plt.plot(np.array(output_x)/float(diff.shape[0]), label='X')
plt.plot(np.array(output_y)/float(diff.shape[0]), label='Y')
plt.plot(np.array(output_z)/float(diff.shape[0]), label='Z')
plt.ylabel('Samples percentage')
#plt.ylim(0,1.0)
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel('Mean error degrees')
plt.text(15,0.00, 'Mean error X: {:.2f}'.format(diff_mean[:,0].mean()))
plt.text(15,0.05, 'Mean error Y: {:.2f}'.format(diff_mean[:,1].mean()))
plt.text(15,0.10, 'Mean error Z: {:.2f}'.format(diff_mean[:,2].mean()))
plt.text(15,0.15, 'Median error X: {:.2f}'.format(np.median(diff_mean[:,0])))
plt.text(15,0.20, 'Median error Y: {:.2f}'.format(np.median(diff_mean[:,1])))
plt.text(15,0.25, 'Median error Z: {:.2f}'.format(np.median(diff_mean[:,2])))

plt.legend()
#plt.xticks(range(0, 600, 10))
plt.savefig('euler.png')
