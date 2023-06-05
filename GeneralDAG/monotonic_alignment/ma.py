import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import pdb
from torch.distributions import Categorical
import torch
from scipy.special import logsumexp,softmax
import math



class ma(object):
	def __init__(self,alpha):
		self.alpha = alpha

	def make_data(self,Na,Nb):
		rng = np.random.RandomState(0)
		X = rng.randn(Na, 3)
		Y = rng.randn(Nb, 3)
		return - pairwise_distances(X, Y) / 10

	def preset_data(self):
		# W = np.array([[20,20,100,10,10,10],
		# 			[10,10,100,10,10,10],
		# 			[10,10,10,100,100,10],
		# 			[10,10,10,10,10,100],])
		W = -np.array([[20,30,50],[40,50,50]])
		return W

	def compute_location(self,W):
		Na,Nb = W.shape
		mu = np.ones((Na+1,Nb+1))*-np.inf

		mu[0,0] = 0
		# Compute first row
		# for j in range(1,Nb-Na+1):
		# 	mu[1,j] = np.log(np.exp(mu[1,j-1]+self.alpha*W[1,j-1]))
		# pdb.set_trace()
		for j in range(1,Nb+1):
			for i in range(1,min(j,Na)+1):
				# pdb.set_trace()
				mu[i,j]  = logsumexp([mu[i,j-1]+self.alpha*W[i-1,j-1],mu[i-1,j-1]+self.alpha*W[i-1,j-1]])
	
		return mu

	def compute_transition(self,W,reduced=True):
		Na,Nb = W.shape
		mu = self.compute_location(W)
		pi = np.zeros((Na+2,Nb+2,2))
		pi[0,0] = np.array(([0,1]))
		pi[-1,-1]  = np.array(([0,1]))
	
		for j in reversed(range(1,Nb+1)):
			for i in reversed(range(max(j-(Nb-Na+1)+1,1),min(j,Na)+1)):		
				log_pi = np.array((mu[i,j-1]+self.alpha*W[i-1,j-1],mu[i-1,j-1]+self.alpha*W[i-1,j-1]))-mu[i,j]
				pi[i,j] = np.exp(log_pi)

		if reduced == True:
			return pi[1:Na+1,1:Nb+1]
		else:
			return pi

	# def compute_transition_softmax(self,W):
	# 	Na,Nb = W.shape
	# 	mu = self.compute_location(W)
	# 	pi = np.zeros((Na+1,Nb+1,3))
	# 	for i in reversed(range(1,Na+1)):
	# 		for j in reversed(range(1,Nb+1)):				
	# 			log_pi = np.array((mu[i-1,j]+self.alpha*W[i-1,j-1],mu[i,j-1]+self.alpha*W[i-1,j-1],mu[i-1,j-1]+self.alpha*W[i-1,j-1]))
	# 			pi[i,j] = softmax(log_pi)
	# 	return pi[1:,1:]


	def sampling(self,W):
		pi = self.compute_transition(W,True)
		Na,Nb = W.shape
		Y = np.zeros_like(W)
		Y[-1,-1] = 1
		i = Na-1; j = Nb-1
		likelihood = 1
		while i >= 0 and j >= 0:
			if i ==0 and j==0:
				break
			prob = pi[i,j,:]
			dist = Categorical(torch.tensor(prob))
			idx = dist.sample().item()
			likelihood = likelihood*prob[idx]
			if idx == 0:
				Y[i,j-1] = 1
				j -= 1
			elif idx == 1:
				Y[i-1,j-1] = 1
				i -= 1
				j -= 1
			else:
				raise Exception("Incorrect sample!")
		return Y,likelihood

	def compute_likelihood(self,W,Y,pi):

		Na, Nb = W.shape
		if pi is not None:
			pass
		else:
			pi = self.compute_transition(W,True)

		likelihood = 1
		i = Na-1; j=Nb-1
		while i> 0 and j>0:
			if Y[i,j-1] ==1:
				likelihood = likelihood*pi[i,j,0]
				j-=1
			elif Y[i-1,j-1] ==1:
				likelihood = likelihood*pi[i,j,1]
				i-=1
				j-=1
		return likelihood

	def compute_weight(self,Y,W,scale):
		if scale is True:
			return np.sum(Y*W,(0,1))*self.alpha
		else:
			return np.sum(Y*W,(0,1))

	def compute_gibbs(self,all_path,W):
		overall_weight = 0
		dict_gibbs = {}
		for i in range(len(all_path)):
			Y = all_path[i,:,:]
			gibbs_weight = np.exp(self.compute_weight(Y,W,True))
			overall_weight += gibbs_weight

		for i in range(len(all_path)):
			Y = all_path[i,:,:]
			gibbs_weight = np.exp(self.compute_weight(Y,W,True))
			# Y = Y.tolist()
			dict_gibbs[i] = gibbs_weight/overall_weight
		return dict_gibbs

	def compute_omega(self,W):
		pi = self.compute_transition(W,False)
		
		Na,Nb = W.shape
		omega = np.zeros_like(pi)
		
		lam = np.zeros((Na+1,Nb+1))
		rho = np.zeros((Na+1,Nb+1))

		lam[0,:] = 0; lam[:,0] = 0; lam[0,0] = 1
		rho[-1,:] = 0; rho[:,-1] = 0; rho[-1,-1]=1

		# Topological iter for lam
		for i in range(1,Na+1):
			for j in range(1,Nb+1):
				lam[i,j] = np.array(([lam[i,j-1],lam[i-1,j-1]]))@pi[i,j]
		
		# Reversed iter for rho
		for i in reversed(range(Na)):
			for j in reversed(range(Nb)):
				rho[i,j] = np.array(([rho[i,j+1],rho[i+1,j+1]]))@np.array(([pi[i+1,j+2,0],pi[i+2,j+2,1]]))
				# print(i,j)
				# print(np.array(([pi[i+2,j+1,0],pi[i+1,j+2,1],pi[i+2,j+2,2]])),[rho[i+1,j],rho[i,j+1],rho[i+1,j+1]])
				# pdb.set_trace()
		# Compute omega
		for i in range(1,Na+1):
			for j in range(1,Nb+1):
				omega[i,j] = np.array(([lam[i,j-1],lam[i-1,j-1]]))*pi[i,j]*rho[i-1,j-1]
		
		return omega[1:Na+1,1:Nb+1]

	def compute_omega_bruce(self,all_path,W):
		dict_gibbs = self.compute_gibbs(all_path,W)

		Na,Nb = W.shape
		omega = np.zeros((Na+1,Nb+1,2))

		# Pad the path with (0,0) = 1
		all_path = np.pad(all_path,(1,0))[1:,:,:]
		all_path[:,0,0] = 1; all_path[:,-1,-1]= 1
		
		for i in range(1,Na+1):
			for j in range(1,Nb+1):
				for path in range(len(dict_gibbs.keys())):
	
					# For two directions
					if all_path[path,i,j] == 1 and all_path[path,i,j-1] == 1:
						omega[i,j,0]+= dict_gibbs[path]
					elif all_path[path,i,j] == 1 and all_path[path,i-1,j-1] == 1:
						omega[i,j,1]+= dict_gibbs[path]
					else:
						pass
		
		return omega[1:,1:,:]





def run():
	from FindPaths import FindPaths

	alpha = 1
	model = ma(alpha)
	W = model.make_data(3,5)

	omega = model.compute_omega(W)
	all_path = FindPaths(3,5).list_to_mat()




# run()