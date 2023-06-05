import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import pdb
from torch.distributions import Categorical
import torch
from scipy.special import logsumexp,softmax
import math
from dtw import dtw
from FindPaths import FindPaths
import random 
import bisect
import collections
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
# from Gradient_check import DTW_func

class MLE(nn.Module):
	def __init__(self,Na,Nb):
		super(MLE, self).__init__()

		# self.alpha = np.random.rand() * 2 + 0.05
		self.alpha = 15
		self.Na = Na
		self.Nb = Nb
		self.FindPaths = FindPaths(self.Na,self.Nb)
		self.all_path = self.allpath().astype(int)
		self.dtw = dtw(self.alpha)
		self.W =  self.distance(True)
		self.GT =  torch.nn.Parameter(torch.tensor(self.W),requires_grad=False)
		# self.W = np.array([[1.5,2.5],[1.3,3.4]])
		self.mu_GT = self.compute_location(self.W)
		self.omega =  self.compute_omega(self.GT)
		print('Inite Distance matrix...')
		print('W:', self.W)

		self.path_weight = self.get_path_exp_weight()
		self.gibbs = self.gibbs_distribution()
		# self.theta = torch.nn.Parameter(torch.ones(self.Na,self.Nb)*0.01,requires_grad = True)
		self.theta = torch.nn.Parameter(torch.rand(self.Na,self.Nb),requires_grad = True)
		# self.theta = torch.nn.Parameter(torch.tensor(self.W),requires_grad = True)
	

	def distance(self,scale):
		rng = np.random.RandomState(0)
		X = rng.randn(self.Na, 3)
		Y = rng.randn(self.Nb, 3)
		W =  pairwise_distances(X, Y) 
		# W =  - W
		# W = W.max()+0.8 - W
		if scale:
			return -W/10
		else:
			return W
		# W = np.array([[0.1,-1,10],[-0.4,0.9,8],[4,1,-1.4]])
		# W = np.array([[0.1,4,2],[8, 0.9,1],[0.8,6,10]])
		# #  W = np.array([[-16.9405, -21.7124],[-18.2243, -17.2990]])
		# return W


	def allpath(self):
		'''
			Return (N,Na,Nb) matrix for all possible alignment matrixs
		'''
		return self.FindPaths.all_path

	def get_path_exp_weight(self):
		weight_dict = self.placehoder()
		for path in weight_dict.keys():
			Y = np.asarray(path)
			score = self.dtw.compute_weight(Y,self.W,True)
			weight_dict[path] = math.exp(score)
		return weight_dict

	def compute_location(self,W):
		mu = torch.zeros((self.Na+1,self.Nb+1))
		mu[0,:] = -1e20
		mu[:,0] = -1e20
		mu[0,0] = 0
		for i in range(1,self.Na+1):
			for j in range(1,self.Nb+1):
				mu[i,j]  = torch.logsumexp(torch.stack((mu[i-1,j]+self.alpha*W[i-1,j-1],mu[i,j-1]+self.alpha*W[i-1,j-1],mu[i-1,j-1]+self.alpha*W[i-1,j-1])),0)
		
		return mu

	def compute_location_new(self,W):

		mu = torch.zeros((self.Na+1,self.Nb+1))
		mu[1,1]= self.alpha*W[0,0]
		# Calculate first row
		for i in range(1,self.Na+1):
			mu[i,1] = torch.log(torch.exp(mu[i-1,1]+self.alpha*W[i-1,0]))
		for j in range(1,self.Nb+1):
			mu[1,j] = torch.log(torch.exp(mu[1,j-1]+self.alpha*W[0,j-1]))
		for i in range(2,self.Na+1):
			for j in range(2,self.Nb+1):

		# mu = torch.zeros((self.Na,self.Nb))
		# # Calculate first row
		# for i in range(1,self.Na):
		# 	mu[i,0] = torch.log(torch.exp(mu[i-1,0]+self.alpha*W[i-1,0]))
		# for j in range(1,self.Nb):
		# 	mu[0,j] = torch.log(torch.exp(mu[0,j-1]+self.alpha*W[0,j-1]))
		# for i in range(1,self.Na):
		# 	for j in range(1,self.Nb):

				mu[i,j]  = torch.logsumexp(torch.stack((mu[i-1,j]+self.alpha*W[i-1,j-1],mu[i,j-1]+self.alpha*W[i-1,j-1],mu[i-1,j-1]+self.alpha*W[i-1,j-1])),0)
		return mu

	def gibbs_distribution(self):
		dict_gibbs = self.placehoder()
		overall_weight = sum(self.path_weight.values())
		for path in dict_gibbs.keys():
			dict_gibbs[path] = self.path_weight[path]/overall_weight
		return dict_gibbs

	def cdf(self, path_prob):
		'''
			Compute CDF given a distribution
		'''
		total = sum(path_prob.values())
		result = []
		cumsum = 0
		for path,w in path_prob.items():
			cumsum += w
			result.append(cumsum/total)

		return result

	def placehoder(self):
		'''
			Generate a dict placeholder of all paths with 0 values.
		'''
		dic = {}
		for i in range(len(self.all_path)):
			Y = self.all_path[i,:,:]
			ty = tuple(map(tuple,Y))
			dic[ty] = 0 
		return dic

	def inverse_sampling(self,path_prob):
		'''
			Inverse sampling given a dict with samples and distribution
		'''

		cdf_vals = self.cdf(path_prob)
		x = random.random()
		idx = bisect.bisect(cdf_vals,x)
		return list(self.gibbs)[idx]

	def get_sample(self,n):
		sample = torch.zeros((n,self.Na,self.Nb))
		for i in range(n):
			y = self.inverse_sampling(self.gibbs)
			sample[i] = torch.tensor(y)
		return sample

	def compute_weight(self,Y,W,scale):
		
		if scale == True:
			return torch.sum(Y*W,(0,1))*self.alpha
		else:
			return torch.sum(Y*W,(0,1))

	def total_weight(self,theta):
		'''
			Compute denominator of gibbs given parameter W
		'''
		total = 0
		for path in self.gibbs.keys():
			Y = torch.tensor(path)
			score = self.compute_weight(Y,theta,True)
			total+= torch.exp(score)
		return total

	def log_prob(self,theta,sample):
		'''
			Compute log prob for each sample
		'''
		total = 0
		for path in self.gibbs.keys():
			Y = torch.tensor(path)
			score = self.compute_weight(Y,theta,False)
			total+= torch.exp(score)

		logp = torch.zeros(len(sample))
		for i in range(len(sample)):
			y = sample[i,:,:]
			score = self.compute_weight(y,theta,False)
			logp[i] = torch.log(torch.exp(score)/total)
		return logp
	
	def compute_log_transition(self,W,reduced):
		Na,Nb = W.shape
		mu = self.compute_location(W)
		pi = torch.zeros((Na+2,Nb+2,3))
		pi[0,0] = torch.tensor(([0,0,1]))
		pi[-1,-1]  = torch.tensor(([0,0,1]))
		for i in reversed(range(1,Na+1)):
			for j in reversed(range(1,Nb+1)):			
				log_pi = torch.stack((mu[i-1,j]+self.alpha*W[i-1,j-1],mu[i,j-1]+self.alpha*W[i-1,j-1],mu[i-1,j-1]+self.alpha*W[i-1,j-1]))-mu[i,j]
				pi[i,j] = log_pi
		if reduced == True:
			return pi[1:Na+1,1:Nb+1]
		else:
			return pi

	def compute_transition(self,W,reduced):
		Na,Nb = W.shape
		mu = self.compute_location(W)
		pi = torch.zeros((Na+2,Nb+2,3))
		pi[0,0] = torch.tensor(([0,0,1]))
		pi[-1,-1]  = torch.tensor(([0,0,1]))
		for i in reversed(range(1,Na+1)):
			for j in reversed(range(1,Nb+1)):			
				log_pi = torch.stack((mu[i-1,j]+self.alpha*W[i-1,j-1],mu[i,j-1]+self.alpha*W[i-1,j-1],mu[i-1,j-1]+self.alpha*W[i-1,j-1]))-mu[i,j]
				pi[i,j] = torch.exp(log_pi)
		if reduced == True:
			return pi[1:Na+1,1:Nb+1]
		else:
			return pi

	def compute_log_likelihood(self,W,Y,pi):

		Na, Nb = W.shape
		if pi is not None:
			pass
		else:
			pi = self.compute_log_transition(W,True)

		likelihood = 0
		i = Na-1; j=Nb-1
		while i!= 0 and j!= 0 :

			# pdb.set_trace()
			if Y[i-1,j] ==1 :
				likelihood = likelihood+pi[i,j,0]
				i -= 1
			elif Y[i,j-1] ==1:
				likelihood = likelihood+pi[i,j,1]
				j-=1
			elif Y[i-1,j-1] ==1:
				likelihood = likelihood+pi[i,j,2]
				i-=1
				j-=1
		return likelihood

	def compute_omega(self,W,pi=None):
		if pi is None:
			pi = self.compute_transition(W,False)
	
		Na,Nb = W.shape
		omega = torch.zeros_like(pi)
		
		lam = torch.zeros((Na+1,Nb+1))
		rho = torch.zeros((Na+1,Nb+1))

		lam[0,:] = 0; lam[:,0] = 0; lam[0,0] = 1
		rho[-1,:] = 0; rho[:,-1] = 0; rho[-1,-1]=1

		# Topological iter for lam
		for i in range(1,Na+1):
			for j in range(1,Nb+1):
				lam[i,j] = torch.matmul(torch.stack(([lam[i-1,j],lam[i,j-1],lam[i-1,j-1]]),0),pi[i,j])
			
		# Reversed iter for rho
		for i in reversed(range(Na)):
			for j in reversed(range(Nb)):
				rho[i,j] = torch.matmul(torch.stack(([rho[i+1,j],rho[i,j+1],rho[i+1,j+1]]),0),torch.stack(([pi[i+2,j+1,0],pi[i+1,j+2,1],pi[i+2,j+2,2]]),0))
				# print(i,j)
				# print(np.array(([pi[i+2,j+1,0],pi[i+1,j+2,1],pi[i+2,j+2,2]])),[rho[i+1,j],rho[i,j+1],rho[i+1,j+1]])
				# pdb.set_trace()
		# Compute omega
		for i in range(1,Na+1):
			for j in range(1,Nb+1):
				omega[i,j] = torch.stack(([lam[i-1,j],lam[i,j-1],lam[i-1,j-1]]),0)*pi[i,j]*rho[i-1,j-1]
		
		return omega[1:Na+1,1:Nb+1]

	# def get_expectation_path_omega(self,W =None):

	# 	if W is not None:
	# 		return np.sum(W*np.sum(self.omega,-1),(0,1))
	# 	else:
	# 		return np.sum(self.W*np.sum(self.omega,-1),(0,1))

	def KLD(self,mu):
		
		# mu = self.compute_location(theta)
		# logprob_q = self.mu_GT[-1,-1] -self.alpha*torch.sum(self.GT.unsqueeze(-1).repeat(1,1,3)*self.omega)

		# logprob_p = mu[-1,-1] - self.alpha*torch.sum(self.theta.unsqueeze(-1).repeat(1,1,3)*self.omega)
		# self.alpha*torch.sum((self.GT-self.theta).unsqueeze(-1).repeat(1,1,3)*self.omega)
		# kl = -mu[-1,-1] +  self.alpha*torch.sum((self.theta).unsqueeze(-1).repeat(1,1,3)*self.omega)
		path_expectation_Wr= torch.sum(self.theta*torch.sum(self.omega,-1),(0,1))
		path_expectation_W = torch.sum(self.GT*torch.sum(self.omega,-1),(0,1))
		# kl = mu[-1,-1] - self.mu_GT[-1,-1] + self.alpha*(path_expectation_W - path_expectation_Wr)
		logp =  self.alpha* path_expectation_Wr - mu[-1,-1]
		logq =  self.alpha*path_expectation_W - self.mu_GT[-1,-1] 
		kl = logq - logp
		return kl


	def forward(self,sample,KLD=True):
		
		# Compute log prob and prob at a given x
		# return self.log_prob(self.theta,sample)
		# pi = self.compute_transition(self.theta,True)		

		# # mu = self.compute_location(self.theta)
		# mu = self.compute_location_new(self.theta)
		# pdb.set_trace()

		mu = self.compute_location(self.theta)

		if KLD :
			self.kl_divergence = self.KLD(mu)

			return self.kl_divergence
		else:		
			logp = torch.zeros(len(sample))
			for i in range(len(sample)):
				# y = torch.tensor(sample[i,:,:].clone()).double()
				y = sample[i,:,:]
				# logprob= self.compute_log_likelihood(self.theta,y,pi)
				score = self.compute_weight(y,self.theta,True)
				logprob = torch.log(torch.exp(score)) - mu[-1,-1]
				# logprob = torch.exp(score)/torch.exp(mu[-1,-1])
				logp[i] = logprob
			self.kl_divergence = self.KLD(mu)
			return logp

	def check_logprob_transition(self,y):
		'''
			y -> a tensor
		'''
		y = np.array(y).astype(int)
		y = tuple(map(tuple,y))
		pi = self.compute_log_transition(self.W,True)
		logprob= self.compute_log_likelihood(self.W,y,pi)
		prob = torch.exp(logprob)
		return prob
	
	def check_mu(self,y):
		mu = self.compute_location(self.W)
		score = self.compute_weight(y,self.W,True)
		logprob = torch.log(torch.exp(score)) - mu[-1,-1]
		return torch.exp(logprob)
	
	def check_mu_new(self,y):
		mu = self.compute_location_new(self.W)
		score = self.compute_weight(y,self.W,True)
		logprob = torch.log(torch.exp(score)) - mu[-1,-1]
		return torch.exp(logprob)
	
	def tensor2tup(self,y):
		y = np.array(y).astype(int)
		y = tuple(map(tuple,y))
		return y

	def NLL_loss(self,sample):
		'''
			Return logprob given samples
		'''
		logprob,_ = self.forward(sample)
		
		return  -torch.sum(logprob),self.theta





	# def forward(self,sample):
		
	# 	# Compute log prob and prob at a given x
	# 	# return self.log_prob(self.theta,sample)
	# 	pi = self.compute_transition(self.theta,True)
	# 	logp = torch.zeros(len(sample))
	# 	for i in range(len(sample)):
	# 		y = sample[i,:,:]
	# 		prob = self.compute_log_likelihood(self.theta,y,pi)
	# 		logp[i] = prob
	# 	return logp
		



def train():
	from Gradient_check import DTW_func

	batch_size =  1
	sample_size = 1

	model =MLE(3,3).double()

	# Sample from the preset gibbs distribution
	sample = model.get_sample(sample_size)
	# Shuffle index
	idx = torch.randperm(sample.shape[0])
	# func = DTW_func.apply 
	# init_theta = model.theta.data.clone()
	# dx= torch.autograd.functional.jacobian(func,(init_theta,sample))
	# print('float type Jacobian by autograd:')
	# print(dx[0])
	# init_theta = model.theta.data.clone().double()
	# dx= torch.autograd.functional.jacobian(func,(init_theta,sample))
	# print('double type Jacobian by autograd:')
	# print(dx[0])


	opt = torch.optim.Adam(model.parameters(),lr= 0.00005)
	for i in range(100000):
		for batch in range(0,sample.shape[0],batch_size):
			# opt.zero_grad()
			model.zero_grad()

			indices = idx[batch:batch+batch_size]
			batch_sample = sample[indices]
			
			logprob = model(batch_sample)
			loss = logprob
			# loss = -torch.sum(logprob)
			# loss =  torch.prod(logprob)
			loss.backward()
			# print('Backward gradient(theta.grad):',)
			# print( model.theta.grad)
			# pdb.set_trace()
			opt.step()

		if i % 500 ==0:

			print('Iteration',i,' Loss:',loss.item(),)
			print('Ground Truth W:', model.W)
			print('Theta:', model.theta.data)
			print('KLD:',model.kl_divergence)
			print('  ')
			# pdb.set_trace()

			# print('GT mu:', model.mu_GT)
			# print('Retrieve mu:', mu)

		
		

	print('Complete training...')
	print('GT is,')
	print(model.W)
	print('Estimate is',  model.theta)

	print('GT gibbs:', model.gibbs)
	restore = model.gibbs.copy()
	total = model.total_weight(model.theta)
	for path in model.gibbs.keys():
		y = torch.tensor(path)
		score = model.compute_weight(y,model.theta,True)
		restore[path] = torch.exp(score)/total
	print('Restore gibbs:', restore)

	pdb.set_trace()

train()