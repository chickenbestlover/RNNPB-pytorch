import math
import numpy as np
import torch
from matplotlib import pyplot as plt
T = 30
L = 100
N = 3
np.random.seed(2)
x = np.empty((N, L), 'int64')

x[:] = np.array(range(L))

data = np.empty((N, L), 'float64')

data[0] = np.sin(x[0] / 1.0 / T).astype('float64')

data[1] = np.sin(x[0] / 1.0 / (2*T)).astype('float64')
data[1]*=0.75
#data[1]+= 0.05*np.random.rand(1,L).flatten()

data[2] = np.sin(x[0] / 1.0 / (4*T)).astype('float64')
data[2]*=0.5
#data[2]+= 0.2*np.random.rand(1,L).flatten()

plt.plot(range(L),data[0],'r')
plt.plot(range(L),data[1],'g')
plt.plot(range(L),data[2],'b')

plt.show()

print data.shape
torch.save(data, open('traindata.pt', 'wb'))

