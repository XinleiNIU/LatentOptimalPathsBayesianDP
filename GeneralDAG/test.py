from reinforce_shortest_path import WeightedDAG
from sample import sample
import numpy as np
import pdb
import math
import numpy as np
import random 
import bisect
import collections
import matplotlib.pyplot as plt

class test:
	def __init__(self):
		# self.alpha = np.random.rand() * 2 + 0.05
		self.alpha = 0.5
		self.wdag = WeightedDAG.random_dag(n=6, threshold=0.5, maxiter=9999, alpha=self.alpha)
		self.W = -self.wdag.to_matrix() # Convert argmin to argmax problem
		self.sampler = sample(self.alpha,self.W)

		self.path_prob()
		self.sample_dict = self.batch_bayesian_sample(100,True)
		self.sample_dict_50 = self.batch_bayesian_sample(50,True)
		self.sample_dict_250 = self.batch_bayesian_sample(250,True)
		self.plot_compare_figure(save_name="alpha_"+str(self.alpha))
		pdb.set_trace()
		self.path_prob_backward()
		self.check_cumulative()
		self.check_sampling()
		self.check_omega()
	
	def plot_helper(self,dic):
		keys = list(dic.keys())
		values = list(dic.values())
		return keys, values

	def plot_compare_figure(self, save_name = None):
		# GT key& value
		keys, values_GT = self.plot_helper(self.path_prob)

		# 50 samples key & value
		_, values_50 = self.plot_helper(self.sample_dict_50)

		# 100 samples
		_, values_100 = self.plot_helper(self.sample_dict)

		# 250 samples
		_, values_250 = self.plot_helper(self.sample_dict_250)

		# keys_text = list(self.sample_dict.keys())
		# values_BDP = list(self.sample_dict.values())
		# values_GT = list(self.path_prob.values())
		keys_num = np.arange(len(keys))
		fig, ax = plt.subplots()
		plt.xticks(rotation=45,fontsize=8)
		ax.plot(keys_num, np.array(values_50), marker='o', linestyle='-',label='BDP: 50 samples')
		ax.plot(keys_num, np.array(values_100), marker='o', linestyle='-',label='BDP: 100 samples')
		ax.plot(keys_num, np.array(values_250), marker='o', linestyle='-',label='BDP: 250 samples')
		ax.plot(keys_num, np.array(values_GT), marker='o', linestyle='-',label='GT')
		plt.xticks(keys_num, keys)
		plt.legend()
		plt.title(f'Distribution of the latent path with alpha = {self.alpha}')

		# Add labels for the x and y axes
		# plt.xlabel('Latent paths')
		plt.ylabel('Probability')
		if save_name is not None:
			plt.savefig('./'+str(save_name)+'.png',bbox_inches='tight')
		else:
			plt.show()

	def check_sampling(self):

		# Check forward sample 
		Y_argmax, weight_score_argmax = self.sampler.backward_sample(self.W)
		# Check forward sample and backward sample
		print('Backward sample path:', Y_argmax,'; weight score:', weight_score_argmax)
		print('Christian\'s shortest path:', self.wdag.shortest_path_dp()[-1], 'weight score:', -self.wdag.shortest_path_dp()[0])
		assert math.isclose(-weight_score_argmax,self.wdag.shortest_path_dp()[0]), 'Incorrect sampled weight score!'
		assert tuple(Y_argmax) == self.wdag.shortest_path_dp()[-1], 'Incorrect sampled path!'
		gibbs_sample = self.gibbs_sample()
		bayesian_sample = self.bayesian_sample()
		assert gibbs_sample == bayesian_sample, 'Gibbs sampled path is differ to bayesian sampled path!'
		assert bayesian_sample == tuple(Y_argmax),'Mode sample is differ to the backward argmax sample!'

		
	def check_cumulative(self):
		'''
			Check the cumulative prob of all paths
		'''
		cum = 0
		for Y in self.wdag.complete_paths():
			prob = self.sampler.compute_likelihood(list(Y),self.W,None)
			cum += prob
		print('Complete calculating cum of all path...')
		assert math.isclose(cum,1),'Cumulative is not 1!'
		print('All done! No problem for the likelihood!')

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

	def inverse_sampling(self,path_prob):
		'''
			Inverse sampling given a dict with samples and distribution
		'''

		cdf_vals = self.cdf(path_prob)
		x = random.random()
		idx = bisect.bisect(cdf_vals,x)
		return list(path_prob.keys())[idx]

	def gibbs_sample(self):
		# Compute weight path
		path_dict = {}
		for Y in self.wdag.complete_paths():
			score = self.sampler.compute_weight(list(Y),self.W,True)
			path_dict[Y] = math.exp(score)
		total_score = sum(path_dict.values())

		sample_dic = dict((path,0) for path in path_dict.keys())

		# Sample a path N times with given gibbs distribution
		for r in range(10000):
			# Sample a path by inverse sampling with gibbs distribution
			sampled_path = self.inverse_sampling(path_dict)
			sample_dic[sampled_path] += 1
		print('*************************************************************************')
		print('Complete sampling paths by gibbs distribution...')
		print('Sample outcomes:', sample_dic)
		print('Mode path:',max(sample_dic,key=sample_dic.get))
		print('*************************************************************************')
		# Return its mode 
		return max(sample_dic,key=sample_dic.get)

	def bayesian_sample(self):
		sample_dic = dict((path,0) for path in self.wdag.complete_paths())
		for r in range(10000):
		# print('*************************************************************************')
		# Return its mode 
			sampled_path,_,_ = self.sampler.sampling(self.W)

			sample_dic[tuple(sampled_path)] += 1
		# print('*************************************************************************')
		# print('Complete sampling paths by the transition matrix...')
		# print('Sample outcomes:', sample_dic)
		# print('Mode path:',max(sample_dic,key=sample_dic.get))
		return max(sample_dic,key=sample_dic.get)

	def batch_bayesian_sample(self,n,return_prob=False):
		"""
			Return sampled path dict
		"""
		sample_dic = dict((path,0) for path in self.wdag.complete_paths())
		for r in range(n):
		# print('*************************************************************************')
		# Return its mode 
			sampled_path,_,_ = self.sampler.sampling(self.W)

			sample_dic[tuple(sampled_path)] += 1
		# obtain sample approximate probabilty
		if return_prob:
			for key in sample_dic:
				sample_dic[key] = sample_dic[key]/n
		# print('*************************************************************************')
		# print('Complete sampling paths by the transition matrix...')
		# print('Sample outcomes:', sample_dic)
		# print('Mode path:',max(sample_dic,key=sample_dic.get))
		return sample_dic

	def path_prob(self):
		path_dict = {}
		for Y in self.wdag.complete_paths():
			score = self.sampler.compute_weight(list(Y),self.W,True)
			path_dict[Y] = math.exp(score)
		total_score = sum(path_dict.values())
		self.path_prob = dict((path,value/total_score) for path,value in path_dict.items())
	
	def path_prob_backward(self):
		path_dict = {}
		for Y in self.wdag.complete_paths():
			prob = self.sampler.compute_likelihood(list(Y),self.W,None)
			path_dict[Y] = prob
		self.path_prob_back = path_dict

	def omega_forward(self):
		'''
			Compute marginal distribution of any edges by bf
		'''
		paths, probs = zip(*self.path_prob_back.items())
		omega = collections.defaultdict(lambda : 0)
		for path, prob in zip(paths, probs):
			for u, v in zip(path[:-1], path[1:]):
				omega[(u,v)] += prob
		omega_matrix = np.zeros_like(self.W)
		for edge,value in omega.items():
			omega_matrix[edge[0],edge[1]] = value
		return omega_matrix

	def check_omega(self):
		transitionc = np.zeros_like(self.W)
		for u,children in enumerate(self.wdag.transition_log_probabilities()):
			for v,prob in children.items():
				transitionc[u,v] = math.exp(prob)
	

		# Calculate the expectation by gibbs distribution
		mean_gibbs = 0
		# Expectation of the distribution
		for Y,prob in self.path_prob.items():
			score =  self.sampler.compute_weight(list(Y),self.W,None)
			mean_gibbs += score*prob
		omega_forward = self.omega_forward()
		# Get the expectation by bayesian 
		omega,lam,rho = self.sampler.compute_omega_bf(self.W,None)
		mean_omega = np.nansum(omega*self.W,(0,1))
		mean_omega_forward = np.nansum(omega_forward*self.W,(0,1))
		
		print( 'Expectation of W by gibbs:', mean_gibbs)
		print( 'Expectation of W by omega', mean_omega)
		print( 'Expectation of W by omega bruce', mean_omega_forward)
		assert math.isclose(mean_gibbs, mean_omega), 'Unequal value for the expectation computed by gibbs distribution and omega'
		assert math.isclose(mean_omega_forward, mean_omega), 'Unequal value for the expectation computed by omega forward and omega'
		

		

test = test()


pdb.set_trace()