import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

losses = []
indexes = []

min = 400.0

for i in range(1,256, 1):
  indexes.append(i)
  with open('output_{}.txt'.format(str(i).zfill(3)), 'r') as r:
    loss = float(r.read())
    if (loss < min):
      print('Loss: {} i: {}'.format(loss, i))
      min = loss

    losses.append(loss)

plt.plot(indexes, losses)
plt.ylabel('Loss value')
plt.xlabel('Epoch')
#plt.xticks(np.array(losses), np.array(indexes))
plt.savefig('losses.png')

print(losses)
