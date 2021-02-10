import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import math
import sys
#frames = 2828
frames = 1597
#frames = 16007

euclidean_distance_index = range(40)
mean_error_index = range(40)
euclidean_distance = []
mean_error = []
euclidean_distance_mean = 0.0
counter = 0.0
for i in range(2,frames+1):
  with open('output_error_validation/output_{}.json'.format(str(i).zfill(3))) as r:
    dict = json.load(r)
    counter += 1.0
    euclidean_distance_mean += dict["euclidean_distance_mean"]
    euclidean_distance.append(np.array(dict['euclidean_distance']))

indices = [0,1,3,4,6,7,9,10,12,13,15]
#JOINTS=16
#x1 = np.array(range(0,JOINTS*2,2))
x1 = np.array(range(0,12*2,2))
#x2 = x1+1
y1 = np.array(euclidean_distance).mean(axis=0)*1000
y1 = y1[indices].tolist() + [euclidean_distance_mean/counter*1000]
#ICVL DeepModel
x2 = x1+1
y2=[9,9.1,13.8,9.8,16.2,6.16,15, 5.8, 15.5, 9.2, 9.4, 11.6]

plt.bar(x1, y1, align="center")
plt.bar(x2, y2, align="center")
#plt.xlabel('Euclidean distance threshold (mm). Mean: {:.2f}'.format(np.array(euclidean_distance).mean()*1000))
plt.ylabel('Mean error distance (mm)')
#plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

plt.margins(0,0)

labels = ["Palm",
          "Index R", "Index 02", "Index T",
          "Middle R", "Middle 02", "Middle T",
          "Pinky R", "Pinky 02", "Pinky T",
          "Ring R", "Ring 02", "Ring T",
          "Thumb R", "Thumb 02", "Thumb T"]



labels = [labels[i] for i in indices] + ["Mean"]
plt.xticks(x1+0.5, labels, rotation=45)

plt.legend(["Ours", "DeepModel"])

""" 
plt.axvline(10, color='gray', linestyle='--')
plt.axvline(20, color='gray', linestyle='--')
plt.axvline(30, color='gray', linestyle='--')
plt.axvline(40, color='gray', linestyle='--')
plt.axvline(50, color='gray', linestyle='--')
plt.axvline(60, color='gray', linestyle='--')
plt.axvline(70, color='gray', linestyle='--')
plt.axhline(0.1, color='gray', linestyle='--')
plt.axhline(0.2, color='gray', linestyle='--')
plt.axhline(0.3, color='gray', linestyle='--')
plt.axhline(0.4, color='gray', linestyle='--')
plt.axhline(0.5, color='gray', linestyle='--')
plt.axhline(0.6, color='gray', linestyle='--')
plt.axhline(0.7, color='gray', linestyle='--')
plt.axhline(0.8, color='gray', linestyle='--')
plt.axhline(0.9, color='gray', linestyle='--')
"""
#plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
#plt.xticks(range(0, max, 10))
#plt.xticks(range(0, 80, 10))
#plt.savefig('euclidean_distance.png')
plt.savefig('joints_error.png', bbox_inches='tight', transparent="True", pad_inches=0.0)
