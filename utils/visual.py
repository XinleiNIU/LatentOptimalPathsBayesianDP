import re
import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pdb

matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(pi,x_length,y_length):
	'''
		Sampling a batch of optimal absolute path
		Input args:
		W: [batch,Na,Nb] distance matrix
		mu: [batch,Na+1,Nb+1] location parameter
		x_length: batch of Na dim
		y_length: batch of Nb dim
		# mask: [batch,Na,Nb] (0,1) mask
		prob_pmf: 
			True: compute log probability by Gibbs PMF, 
			False: compute by transition matrix pi. 
	'''

	Y = torch.zeros((x_length,y_length))
	logprobY = 0
	i = x_length-1; j = y_length-1
	Y[i,j] = 1
	while i>= 0 and j >= 0:
		if i == 0 and j == 0:
			break

		dist = Categorical(pi[i,j,:])
		idx = dist.sample().item()

		logprobY +=torch.log(pi[i,j,idx])
		if idx ==0:
			Y[i-1,j] = 1
			i-=1
		elif idx ==1:
			Y[i-1,j-1] = 1
			i-=1;j-=1
		else:
			raise Exception("Incorrect sample direction!")

	return Y,logprobY

def sample_from_pi(pi, x_length, y_length, n):
	transition = pi[0].item()
	sample_dict ={}
	for i in range(n):
		sample_i = sample(pi,x_length,y_length)
		if sample_i in sample_dict:
			sample_dict[sample_i] +=1
		else:
			sample_dict[sample_i] = 1
	return sample_dict


