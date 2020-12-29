import matplotlib as mpl
mpl.use('Agg')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import math

#frames = 3116
frames = 1597

euclidean_distance_index = range(40)
mean_error_index = range(40)
euclidean_distance = []
mean_error = []

for i in range(2,frames+1):
  with open('output_error_validation/output_{}.json'.format(str(i).zfill(3))) as r:
    dict = json.load(r)
    mean_error.append(dict['mean_error'])
    #euclidean_distance(dict['euclidean_distance_mean'])

mean_error.sort()

max = math.ceil(mean_error[len(mean_error)-1]*1000)
print(max)
resolution = 1
gen = iter(range(0, max, resolution))

i = next(gen)

output = []
aux = 0
eiterator = iter(mean_error)
error = next(eiterator)
for i in gen:
  while(error*1000 < i):
    aux += 1
    error = next(eiterator)
  output.append(aux)

plt.ylabel('% of frames')
plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.xlabel('Mean error threshold ({:.2f}mm), max error {}'.format(np.array(mean_error).mean()*1000, max))
#plt.xticks(range(0, max, 10))

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

plt.plot(np.array(output)/len(mean_error))
#plt.gca().set_axis_off()
#plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#            hspace = 0, wspace = 0)
plt.savefig('mean_error.png', bbox_inches='tight', transparent="True", pad_inches=0.0)
