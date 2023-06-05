import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import pdb
from torch.distributions import Categorical
import torch
from scipy.special import logsumexp,softmax
import math
from ma import ma
from FindPaths import FindPaths
import random 
import bisect
import collections

class test_ma(object):
	def __init__(self):
		# self.alpha = np.random.rand() * 2 + 0.05
		self.alpha =1
		print(self.alpha)
		self.Na = 5
		self.Nb = 10
		self.W = self.distance(False)
		self.FindPaths = FindPaths(self.Na,self.Nb)
		self.all_path = self.allpath().astype(int)
		self.dtw = ma(self.alpha)
		self.path_weight = self.get_path_exp_weight()
		self.gibbs = self.gibbs_distribution()
		self.pi = self.dtw.compute_transition(self.W,True)
		self.omega = self.dtw.compute_omega(self.W)
		# self.mu = self.dtw.compute_location(self.W)
		self.self_test()


	def distance(self,scale):
		rng = np.random.RandomState(0)
		X = rng.randn(self.Na, 3)
		Y = rng.randn(self.Nb, 3)
		W = - pairwise_distances(X, Y) 
		if scale:
			return W/10
		else:
			return W

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

	def gibbs_sample(self):
		sample_dic = self.placehoder()

		for i in range(100000):
			y = self.inverse_sampling(self.gibbs)
			sample_dic[y] += 1
		return max(sample_dic,key=sample_dic.get)

	def bayesian_sample(self):
		sample_dic = self.placehoder()
		for i in range(100000):
			Y,_ = self.dtw.sampling(self.W)
			ty = tuple(map(tuple,Y.astype(int)))
			sample_dic[ty]+= 1
		return max(sample_dic,key=sample_dic.get)
	
	def argmin_path(self):
		'''
			Get the argmin path (argmax for negative W).
		'''
		return max(self.path_weight, key=self.path_weight.get)

	def check_likelihood(self):
		# Get transition matri P
		prob_dict = {}
		
		for path in self.gibbs.keys():
			Y = np.asarray(path)
			prob = self.dtw.compute_likelihood(self.W,Y,self.pi)
			prob_dict[path] = prob
			print('Checking the probability of path', path, '...', end = ' ')
			assert math.isclose(self.gibbs[path], prob_dict[path]), 'Path prob between likelihood and gibbs are not equal!'
			print('Done')
		return prob_dict

	def get_expectation_path_bruce(self):

		expect = 0 
		
		for path in self.gibbs.keys():
			Y = np.asarray(path)
			expect+= self.gibbs[path]*self.dtw.compute_weight(self.W,Y,False)
		return expect


	def get_expectation_path_omega(self):

		return np.sum(self.W*np.sum(self.omega,-1),(0,1))

	def get_omega_bruce(self):
		omega = np.zeros((self.Na+1,self.Nb+1,2))

		# Pad the path with (0,0) = 1
		all_path = np.pad(self.all_path,(1,0))[1:,:,:]
		all_path[:,0,0] = 1; all_path[:,-1,-1]= 1
		
		for i in range(1,self.Na+1):
			for j in range(1,self.Nb+1):
				for path in range(len(self.gibbs.keys())):
					# pdb.set_trace()
					# For three directions
					path_ty = tuple(map(tuple,self.all_path[path,:,:]))
					# if all_path[path,i,j] == 1 and all_path[path,i-1,j] == 1:
					# 	omega[i,j,0]+= self.gibbs[path_ty]
					if all_path[path,i,j] == 1 and all_path[path,i,j-1] == 1:
						omega[i,j,0]+= self.gibbs[path_ty]
					elif all_path[path,i,j] == 1 and all_path[path,i-1,j-1] == 1:
						omega[i,j,1] += self.gibbs[path_ty]
					else:
						pass
		
		return omega[1:,1:,:]

	def variance(self,expecation):
		n = self.all_path.shape[0]
		var = 0
		for sample in range(n):
			Y =  self.all_path[sample]
			score_i = self.dtw.compute_weight(self.W,Y,True)
			var += (score_i-expecation)**2
			
		return var/(n-1)


	def self_test(self):

		

		# Test whether find the all possible paths 
		print('Checking cumulative probability of all paths ...', end = ' ')
		assert math.isclose(sum(self.gibbs.values()),1), 'The cumulative probability of all paths is not 1!'
		print('Done')

		# Check probability of paths
		self.check_likelihood()

		# Check omega
		print('Checking correctness of omega...', end = ' ')
		omega_bruce = self.get_omega_bruce()
		assert np.allclose(omega_bruce,self.omega), 'Incorrect omega values!'
		print('Done')

		# Check expectation
		print('Checking expectation of path score...', end = ' ')
		expect_bf = self.get_expectation_path_bruce()
		expect_omega = self.get_expectation_path_omega()
		assert math.isclose(expect_bf,expect_omega), 'Unequal expectation values!'
		print('Done')

		import operator
		print(max(self.gibbs.items(), key=operator.itemgetter(1))[1])
		var = self.variance(expect_omega)

		pdb.set_trace()
		# Sample a path from gibbs
		print('Start sample optimal path for 100000 rounds...', end = ' ')
		gibbs_path = self.gibbs_sample()
		bayesian_path = self.bayesian_sample()

		argmin_path = self.argmin_path()
		print('Optimal path by argmin is:', argmin_path,'Exp weight:', self.path_weight[argmin_path])
		print('Optimal path sampled from Gibbs is:', gibbs_path,'Exp weight:', self.path_weight[gibbs_path])
		print('Optimal path sampled from Bayesian DP is:', bayesian_path,'Exp weight:', self.path_weight[bayesian_path])

		assert argmin_path == bayesian_path, 'Argmin path and Bayesian sample are not equal!'
		assert gibbs_path == argmin_path, 'Gibbs sample and optimal argmin path are not equal!'

		print('End of self checking ')


test = test_ma()
pdb.set_trace()