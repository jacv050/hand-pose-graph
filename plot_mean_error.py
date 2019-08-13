import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import math

#frames = 3116
frames = 2844

euclidean_distance_index = range(40)
mean_error_index = range(40)
euclidean_distance = []
mean_error = []

for i in range(2,frames+1):
  with open('output_{}.json'.format(str(i).zfill(3))) as r:
    dict = json.load(r)
    mean_error.append(dict['mean_error'])
    #euclidean_distance(dict['euclidean_distance_mean'])

#euclidean_distance.sort()
mean_error.sort()

#print(mean_error[len(mean_error)-1])
max = math.ceil(mean_error[len(mean_error)-1]*1000)
print(max)
gen = iter(range(0, max+1, 1))

i = next(gen)

output = []
aux = 0
for mean in mean_error:
  if mean*1000 > i: #mean*1000 > i
    output.append(aux)
    i = next(gen)
  aux += 1

output.append(aux)

plt.plot(np.array(output)/len(mean_error))
plt.ylabel('% of frames')
#plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel('Mean error threshold (mm)')
plt.xticks(range(0, max, 10))
plt.savefig('mean_error.png')
