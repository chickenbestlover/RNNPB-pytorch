from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class RNNPB(nn.Module):
    def __init__(self):
        super(RNNPB, self).__init__()

        self.data_dim=1
        self.data_num=3
        self.ctx_dim = 20
        self.pb_dim=2
        self.hidden_dim=40
        self.input_dim = self.data_dim + self.pb_dim + self.ctx_dim
        self.output_dim = self.data_dim + self.ctx_dim

        self.fc1_data = nn.Linear(self.data_dim,self.hidden_dim)
        self.fc1_pb = nn.Linear(self.pb_dim, self.hidden_dim)
        self.fc1_ctx = nn.Linear(self.ctx_dim, self.hidden_dim)

        self.fc2_data = nn.Linear(self.hidden_dim,self.data_dim)
        self.fc2_ctx = nn.Linear(self.hidden_dim, self.ctx_dim)

        self.pb = nn.Parameter(data=torch.zeros(self.data_num,self.pb_dim))
    def RNNPBCell(self, data, ctx):
        hiddenOut = self.fc1_data(data)
        hiddenOut += self.fc1_pb(self.pb)
        hiddenOut += self.fc1_ctx(ctx)
        hiddenOut = F.relu(hiddenOut)
        out = self.fc2_data(hiddenOut)
        ctx = self.fc2_ctx(hiddenOut)

        return out, ctx

    def forward(self,data,GENERATE=False):
        outputs = []

        ctx = Variable(torch.zeros(data.size(0), self.ctx_dim).double(),requires_grad=True)
        out = Variable(torch.zeros(data.size(0), self.data_dim).double(),requires_grad=True)
        for i, input_t in enumerate(data.chunk(data.size(1), dim=1)):
            if GENERATE:
                out, ctx = self.RNNPBCell(out,ctx)
            else:
                out, ctx = self.RNNPBCell(input_t,ctx)
            outputs += [out]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs