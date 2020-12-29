import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import math

#frames = 2828
frames = 1597
#frames = 16007

euclidean_distance_index = range(40)
mean_error_index = range(40)
euclidean_distance = []
mean_error = []

for i in range(2,frames+1):
  with open('output_error_validation/output_{}.json'.format(str(i).zfill(3))) as r:
    dict = json.load(r)
    #mean_error.append(dict['mean_error'])
    euclidean_distance.append(dict['euclidean_distance_mean'])

euclidean_distance.sort()
#mean_error.sort()

#print(mean_error[len(mean_error)-1])
max = math.ceil(euclidean_distance[len(euclidean_distance)-1]*1000)
print(max)
gen = iter(range(0, max, 1))

i = next(gen)

output = []
aux = 0
diterator = iter(euclidean_distance)
distance = next(diterator)
for i in gen:
  if distance*1000 > 80:
    continue
  while(distance*1000 < i):
    aux += 1
    distance = next(diterator)
  output.append(aux)


""" I
for distance in euclidean_distance:
  if distance*1000 > i:
    output.append(aux)
    i = next(gen)
  aux += 1

output.append(aux)
"""

plt.plot(np.array(output)/len(euclidean_distance))
plt.ylabel('% of frames')
#plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel('Euclidean distance threshold (mm). Mean: {:.2f}'.format(np.array(euclidean_distance).mean()*1000))
#plt.xticks(range(0, max, 10))
plt.xticks(range(0, 80, 10))
plt.savefig('euclidean_distance.png')
