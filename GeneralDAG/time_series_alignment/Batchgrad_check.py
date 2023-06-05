import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import pdb
from torch.distributions import Categorical
import torch
from scipy.special import logsumexp,softmax
import math
import random 
import bisect
import collections
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from BayesianDTW import BayesianDTW
from MLE import MLE
from torch.autograd import Variable
# Get two GT

model1 = MLE(5,6)
model2 = MLE(4,5)


sample  = torch.zeros((2,6,7))
x_len = torch.tensor((5,4))
y_len = torch.tensor((6,5))

# Get 0,1 mask
mask = torch.zeros((2,6,7))
mask[0,:5,:6] = 1
mask[1,:4,:5] = 1

sample[0,:5,:6] = torch.tensor(model1.W)
sample[1,:4,:5] = torch.tensor(model2.W)

# Inite parameter theta

theta = torch.tensor(torch.rand((2,6,7)),requires_grad=True).double()
dtw = BayesianDTW(15)
# pdb.set_trace()
mu,pi =dtw(theta,mask.unsqueeze(-1).repeat((1,1,1,3)))

# Get sample
batch_path, batch_log_probs,log_probs = dtw.sample(theta,pi,x_len,y_len,mu)

loss1 = -torch.sum(batch_log_probs)
loss1 = Variable(loss1, requires_grad = True)
pdb.set_trace()
loss2 = -torch.sum(log_probs)
pdb.set_trace()
loss1.backward()
pdb.set_trace()
