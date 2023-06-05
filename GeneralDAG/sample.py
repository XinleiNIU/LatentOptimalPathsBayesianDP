import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import pdb
from torch.distributions import Categorical
import torch
from reinforce_shortest_path import WeightedDAG
from scipy.special import logsumexp
import math


class sample(object):
	def __init__(self,alpha, W):
		# self.alpha = alpha
		self.WeightedDAG = WeightedDAG
		self.alpha = alpha
		self.W = W

	def get_matrix(self,W):
		'''
			The W is a (n,n) distance matrix,
			W[i,:] -> edge weight of i to its childs
			W[:,i] -> edge weight of i to its parents 
		'''
		return W

	def make_mask(self,W):
		'''
		   make a binary mask for the distance matrix
		'''

		return (W > -np.inf).astype(int)

	def compute_mu(self,W):
		'''
			Computr location parameter mu
			INPUT: W (n,n) distance matrix
			RETURN: U (n) location parameter 

		'''
		m,n = W.shape
		U = np.zeros((n))
		mask = self.make_mask(W)

		# Toplogical iteration calculate mu
		for i in range(1,n):
			U[i] = logsumexp(U*mask[:,i]+self.alpha*W[:,i])
		return U

	def compute_mu_bf(self,W):
		'''
			Compute location parameter mu by brute force
			INPUT: W (n,n) distance matrix
			RETURN: U (n) location parameter 
		'''
		m,n = W.shape
		U = np.zeros((n)) # mu_0 = 0

		# Iterate each node
		for node in range(1,n):
			# Init mu
			cum_mu = 0
			# Iterate parents of the node
			for state in range(n):
				# If the state j is i's parent
				if W[state,node] > -np.inf:
					cum_mu = np.exp(U[state]+self.alpha*W[state,node]) +cum_mu
			U[node] = np.log(cum_mu+1e-20)
		return U


	def check_location(self,W):
		'''
			Assert compute mu and compute mu bf get same result
		'''
		U_bf = self.compute_mu_bf(W)
		U = self.compute_mu(W)
		assert np.allclose(U_bf,U), 'Incorret implementation!'


	# TODO: double check bugs on sampling part

	def sampling(self,W):
		'''
			Sample the shorest path reversely by the transition matrix
		'''
		# Generate transition matrix 
		P = self.compute_transition(W)
		m,n = P.shape
		# Init v = N
		v= n-1
		Y = [v]
		# Inite weight_score 
		weight_score = 0
		# Start sampling
		path_prob = 1
		while v>0:
			pi_uv = P[:,v]
			# Sample u by categorical distribution
			u_sampler =Categorical(torch.tensor(pi_uv))
			u = u_sampler.sample().item()
			weight_score += W[u,v]
			path_prob = pi_uv[u] * path_prob
			v = u
			Y.append(v)
		return Y[::-1],weight_score,path_prob

	def backward_sample(self,W):
		'''
			Find the shorest path by the argmax the transition matrix
		'''
		P = self.compute_transition(W)
		weight_score = 0
		m,n = W.shape
		v = n-1
		Y=[v]
		while v >0:
			pi_uv = P[:,v]
			u = np.argmax(pi_uv)
			weight_score += W[u,v]
			v = u 
			Y.append(v)
		return Y[::-1],weight_score


	def compute_weight(self,Y,W,alpha):
		'''
			Set alpha to None to get unscaled path score
		'''
		weight = 0
		for edge in zip(Y,Y[1:]):
			u = edge[0]
			v = edge[1]
			if alpha is not None:
				weight += self.alpha*W[u,v]
			else:
				weight += W[u,v]
		return weight

	def compute_log_transition_bf(self,W):
		'''
			Compute the log transition matrix by O(N2)
		'''
		# Compute mu
		U = self.compute_mu(W)
		m,n = W.shape
		log_transition_matrix = np.zeros_like(W)
		# Iterate all possible nodes v 
		for v in range(1,n):
			# Iterate v's parent
			for u in range(n):
				if W[u,v] > -np.inf:
					log_transition_matrix[u,v] = (U[u]+self.alpha*W[u,v]) - U[v]

		return log_transition_matrix


	def compute_log_transition(self,W):
		'''
			Computing the log transition matrix by O(N)
		'''
		# Compute mu
		U = self.compute_mu(W)
		m,n = W.shape
		log_transition_matrix = np.zeros_like(W)
		# Get the edge mask
		mask = self.make_mask(W)
		# Iterate all possible nodes v 
		for v in range(1,n):
			log_transition_matrix[:,v] = (U*mask[:,v]+self.alpha*W[:,v]) - U[v]
			# pdb.set_trace()
		return log_transition_matrix

	def compute_transition(self,W):
		'''
			Computing the transition matrix by O(N)
		'''
		# Compute mu
		U = self.compute_mu(W)
		m,n = W.shape
		transition_matrix = np.zeros_like(W)
		# Get the edge mask
		mask = self.make_mask(W)
		# Iterate all possible nodes v 
		for v in range(1,n):
			transition_matrix[:,v] = np.exp(U*mask[:,v]+self.alpha*W[:,v])/np.exp(U[v])
			# pdb.set_trace()
		return transition_matrix


	def check_transition(self,W,Y):
		'''
			Get the path probability by the transition matrix.
		'''
		log_transition_matrix = self.compute_log_transition(W)
		log_transition_matrix_bf = self.compute_log_transition_bf(W)
		
		pi_uv = 0
		for edge in zip(Y,Y[1:]):
			u = edge[0]
			v = edge[1]
			pi_uv = pi_uv+log_transition_matrix[u,v]
		return np.exp(pi_uv)




	def compute_likelihood(self,Y,W,P):
		'''
			The the prob. given a path.
			INPUT: 
				Y: a list of path
				W: distance matrix of the DAG
			OUTPUT:
				prob: probability of the given path Y.
		'''
		
		prob = 1
		if P is None:
			U = self.compute_mu(W)
			for edge in zip(Y,Y[1:]):
				u = edge[0]
				v = edge[1]
				pi_uv = np.exp(U[u]+self.alpha*W[u,v])/np.exp(U[v])
				prob = prob*pi_uv
		else:
			P = self.compute_transition(W)
			for edge in zip(Y,Y[1:]):
				u = edge[0]
				v = edge[1]
				prob = prob*P[u,v]

		return prob

	def compute_omega_bf(self,W,P):
		'''
			Compute marginal probability by bf
			INPUT:
				P: log probability transition matrix
			RETURN:
				omega: omega[u,v] = omega for edge(u,v)
		'''

		# Topological iteration for lam

		m,n = W.shape
		if P is None:
			P = self.compute_transition(W)
		lam = np.zeros(n)
		lam[0] = 1

		for v in range(1,n):
			for u in range(n):
				# Find v's parent
				if W[u,v] > -np.inf :
					lam[v] += lam[u]*P[u,v]


		# Revered iteration for rho
		rho = np.zeros(n)
		rho[-1] = 1

		for u in reversed(range(n-1)):
			# pdb.set_trace()
			for v in range(n):
				# Find u's child
				if W[u,v] > -np.inf :
					# pdb.set_trace()
					rho[u] += rho[v]*P[u,v]
			
		# Compute omega
		omega = np.zeros_like(W)
		for v in range(n):
			for u in range(n):
				omega[u,v] = P[u,v]*lam[u]*rho[v]
		return omega,lam,rho


	def compute_omega(self,W,P):
		'''
			Compute total probability of path (u,v)
			Input arg:
				P: [m,n] transition matrix
			Output arg:
				omega: [m,n] probability of path include [i,j: i -> j] edge
				# omega[i,j] -> lam[i]P[i,j]lo[j]
		'''

		m,n = W.shape
		mask = self.make_mask(W)
		if P is None:
			P = self.compute_transition(W)
	
		lam = np.zeros(n)
		lam[0] = 1

		for v in range(1,n):
			lam[v] = np.sum(lam*mask[:,v]*P[:,v])
			# pdb.set_trace()


		# Revered iteration for rho
		rho = np.zeros(n)
		rho[-1] = 1
		for u in reversed(range(n-1)):
			rho[u]= np.sum(rho*mask[u,:]*P[u,:])

		# Compute omega
		# pdb.set_trace()
		omega = np.tile(lam,(n,1))*P* np.tile(rho,(n,1))
		#np.tile(rho.reshape(-1,1),(1,n))
		return omega,lam,rho
		# omega = np.exp(P)
		# for v in range(1,n):
		# 	for u in range(n):
		# 		if P[u,v] != 1:
		# 			omega[u,v] = np.exp(P[u,v])*lam[u]*rho[v]
		# return omega,lam,rho

	def check_omega(self,W):
		omega_bf,lam_bf,rho_bf = self.compute_omega_bf(W,None)
		omega,lam,rho = self.compute_omega(W,None)
		assert np.allclose(lam_bf,lam), 'Incorrect lambda!'
		assert np.allclose(rho_bf,rho), 'Incorrect rho!'
		assert np.allclose(omega_bf,omega), 'Incorrect omega!'


	def KL_Divergence(self,U,Ur,W,Wr,alpha):
		'''
		Compute KL divergence between two distribution
			Input args:
				U: [1,n] location parameter of posterior
				Ur: [1,n] location parameter of prior
				W: [m,n] distance matrix of posterior
				Wr: [m,n] distrance matrix of piror
				alpha: temperature
			Output args:
				KLD
q
		'''
		omega = self.compute_omega(W,None)
		pdb.set_trace()
		KLD = Ur[-1]-U[-1] + self.alpha*np.sum(omega*(W-Wr))
		return KLD

	def compute_path_socre(self,Y,W):
		m,n = W.shape
		U = 0
		for edge in zip(Y,Y[1:]):
			score = U +self.alpha*W[edge[0],edge[1]]
			U = score
		return score


	def Gibbs_prob(self,Y,W):
		'''
			Compute path probability by gibbs distribution
			INPUT: 
				Y: a list of sampled path
				W: DAG distance matrix
			OUTPUT:
				prob: probability of given Y and W.
		'''
		U = self.compute_mu(W)
		score = self.compute_path_socre(Y,W)
		prob = np.exp(score)/np.exp(U[-1])
		return prob

	def check_likelihood(self,Y,W):
		prob_gibbs = self.Gibbs_prob(Y,W)
		
		prob_sample = self.compute_likelihood(Y,W)
		
		prob_tran = self.check_transition(W,Y)
		P = self.compute_log_transition_bf(W)

		assert math.isclose(prob_gibbs,prob_sample), 'Incorrect path prob for gibbs probability v.s. sampled probability!'
		assert math.isclose(prob_tran,prob_gibbs), 'Incorrect path prob for transition matrix v.s. gibbs probability!'
		assert math.isclose(prob_tran,prob_sample), 'Incorrect path prob for transition matrix v.s. sampled probability!'




if __name__ == '__main__':

	alpha = np.random.rand() * 2 + 0.05
	W = np.array([[-np.inf,   10,  5, -np.inf],
				  [-np.inf,-np.inf,   10,  5],
				  [-np.inf,-np.inf,-np.inf,   10],
				  [-np.inf,-np.inf,-np.inf,-np.inf]])
	allpath = [[0,2,3],[0,1,2,3],[0,1,3]]

	sample = sample(alpha,W)
	path = [0,1,3]
	mu = sample.compute_mu(W)
	prob = sample.compute_likelihood(path,W,None)
	# cum = 0
	# for path in allpath:
	# 	prob = sample.compute_likelihood(path,W,None)
	# 	# print(prob)
	# 	cum = cum+prob
	# assert math.isclose(cum,1), 'Cum is not 1!'

	pdb.set_trace()

