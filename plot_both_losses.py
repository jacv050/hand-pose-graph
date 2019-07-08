import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

losses = []
losses_validation = []
indexes = []

min = 400.0

l = []
l = l + [i for i in range(4,388,4)]
l = l + [i for i in range(389, 510, 4)]
#l = l + [i for i in range(512, 632, 4)]

#for i in range(4,1024, 4):
for i in l:
  indexes.append(i)
  with open('losses_outputs_validation/output_{}.txt'.format(str(i).zfill(3)), 'r') as r:
    loss = float(r.read())*1000
    if (loss < min):
      print('Loss: {} i: {}'.format(loss, i))
      min = loss
    losses_validation.append(loss)
  with open('losses_outputs/output_{}.txt'.format(str(i).zfill(3)), 'r') as r:
    loss = float(r.read())*1000
    losses.append(loss)

plt.plot(indexes, losses, label="Training loss")
plt.plot(indexes, losses_validation, label="Validation loss")
plt.legend()
plt.ylabel('Loss value')
plt.xlabel('Epoch')
#plt.xticks(np.array(losses), np.array(indexes))
plt.savefig('losses.png')

#print(losses)
