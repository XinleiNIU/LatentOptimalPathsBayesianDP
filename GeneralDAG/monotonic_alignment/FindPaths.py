import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import pdb
from torch.distributions import Categorical
import torch
import math
import collections



class FindPaths(object):
	'''
		Given lengths of two sequence, return all possible alignment matrices on monotonic alignment DAG
	'''
	def __init__(self,Na,Nb):
		self.Na = Na
		self.Nb = Nb
		self.W = self.make_data()
		self.grid = self.make_grid()
		self.printAllPaths()
		self.all_path = self.list_to_mat()

	def make_data(self):
		rng = np.random.RandomState(0)
		X = rng.randn(self.Na, 3)
		Y = rng.randn(self.Nb, 3)
		return - pairwise_distances(X, Y) / 10
	def make_grid(self):
		return np.arange(1,self.Na*self.Nb+1).reshape(self.Na,self.Nb).tolist()
	
	def printAllPathsUtil(self, mat, i, j, m, n, path, pi):
		if (i == m -1):
			for k in range(j,n):
				path[pi+k-j] = mat[i][k]

			for l in range(pi+n-j):

				self.all_path_list.append(list(dict.fromkeys(path)))
				# print(path, end=" ")
				# print( )
				return

		if (j == n-1):
			for k in range(i,m):
				path[pi+k-i] = mat[k][j]

			for l in range(pi+m-i):
				self.all_path_list.append(list(dict.fromkeys(path)))
				# print(path, end = " ")
				# print( )
				return
 
		# Add the current cell
		# to the path being generated
		path[pi] = mat[i][j]

		# Print all the paths
		# that are possible after moving down
		# self.printAllPathsUtil(mat, i + 1, j, m, n, path, pi + 1)

		# Print all the paths
		# that are possible after moving right
		self.printAllPathsUtil(mat, i, j + 1, m, n, path, pi + 1)

		# Print all the paths
		# that are possible after moving down right
		self.printAllPathsUtil(mat, i + 1, j + 1, m, n, path, pi + 1)

	def printAllPaths(self):
		mat = self.grid
		m = self.Na 
		n = self.Nb
		path = [0 for i in range(m+n-1)]
		self.all_path_list = []
		self.printAllPathsUtil(mat,0,0,m,n,path,0)

	def list_to_mat(self):
		
		for (idx,path) in enumerate(self.all_path_list):
			path_mat = np.zeros((self.Na*self.Nb))
			path_mat[np.array(path)-1]=1
			path_mat = path_mat.reshape(self.Na,self.Nb)
			# ignore paths with down move
			# pdb.set_trace()
			if idx ==0:
				all_path_mat = np.zeros_like(path_mat)
			if np.array_equal(np.ones(self.Nb),np.sum(path_mat,0)):
				all_path_mat = np.concatenate((all_path_mat,path_mat))
			output = all_path_mat.reshape(-1,self.Na,self.Nb)
		return output[1:,:,:]



	