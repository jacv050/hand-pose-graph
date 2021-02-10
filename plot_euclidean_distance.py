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

plt.xlabel('Euclidean distance threshold (mm). Mean: {:.2f}'.format(np.array(euclidean_distance).mean()*1000))
plt.ylabel('% of frames')
plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

fill_x = 80
plt.xticks(range(0,fill_x,10))
aux_max = max
while(aux_max < fill_x):
  output.append(aux)
  aux_max += 1

plt.margins(0,0)

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

plt.plot(np.array(output)/len(euclidean_distance))
#plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
#plt.xticks(range(0, max, 10))
#plt.xticks(range(0, 80, 10))
#plt.savefig('euclidean_distance.png')
plt.savefig('euclidean_distance.png', bbox_inches='tight', transparent="True", pad_inches=0.0)
