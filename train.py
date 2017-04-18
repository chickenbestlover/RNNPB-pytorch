from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from RNNPB import RNNPB


if __name__ == '__main__':
    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load(open('traindata.pt'))
    input = Variable(torch.from_numpy(data[:, :-1]), requires_grad=False)#Exclude one at the end.
    target = Variable(torch.from_numpy(data[:, 1:]), requires_grad=False)#Exclude one at the start
    # build the model
    seq = RNNPB()
    seq.double()
    #seq.cuda()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(),lr=0.5,)

    print(list(seq.parameters()))
    #begin to train
    for i in range(50):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq.forward(input)
            loss = criterion(out, target)
            print('loss:', loss.data.cpu().numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)

        # begin to predict
        pred = seq.forward(input[:], GENERATE=True)

        # When you want to test with a single Parametric bias value
        #pb = seq.pb.data
        #seq.pb.data = pb[0].view(1, 2)
        #pred = seq.forward(input[0].resize(1, 99), GENERATE=True)
        #seq.pb.data=pb

        y = pred.data.cpu().numpy()
        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are true values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.plot(np.arange(input.size(1)), y[0], 'r', linewidth=2.0)
        plt.plot(np.arange(input.size(1)), y[1], 'g', linewidth=2.0)
        plt.plot(np.arange(input.size(1)), y[2], 'b', linewidth=2.0)

        plt.plot(np.arange(input.size(1)),target.data[0].numpy(),'r:',linewidth=2.0)
        plt.plot(np.arange(input.size(1)), target.data[1].numpy(), 'g:',linewidth=2.0)
        plt.plot(np.arange(input.size(1)), target.data[2].numpy(), 'b:',linewidth=2.0)

        plt.show()
        plt.savefig('results/predict%d.pdf' % i)
        plt.close()