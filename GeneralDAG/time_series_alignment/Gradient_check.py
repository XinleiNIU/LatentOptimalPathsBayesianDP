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
from BayesianDTW import BayesianDTW


def compute_location(theta,alpha):
	Na,Nb = theta.shape
	mu = torch.zeros((Na+1,Nb+1))
	mu[0,:] = -1e20
	mu[:,0] = -1e20
	mu[0,0] = 0
	for i in range(1,Na+1):
		for j in range(1,Nb+1):

			mu[i,j]  = torch.logsumexp(torch.stack((mu[i-1,j]+alpha*theta[i-1,j-1],mu[i,j-1]+alpha*theta[i-1,j-1],mu[i-1,j-1]+alpha*theta[i-1,j-1])),0)
	
	return mu

def compute_location2(theta,alpha):
	Na,Nb = theta.shape
	mu = torch.zeros((Na,Nb))

	for i in range(Na):
		for j in range(Nb):
			if i ==0 and j ==0 :
				mu[i,j] = torch.log(torch.exp(alpha*theta[i,j]))
			elif i ==0 and j!=0:
				mu[i,j] = torch.log(torch.exp(mu[i,j-1]+alpha*theta[i,j]))
			elif j==0 and i!=0:
				mu[i,j] = torch.log(torch.exp(mu[i-1,j]+alpha*theta[i,j]))
			else:
				mu[i,j]  = torch.logsumexp(torch.stack((mu[i-1,j]+alpha*theta[i,j],mu[i,j-1]+alpha*theta[i,j],mu[i-1,j-1]+alpha*theta[i,j])),0)
	
	return mu[-1,-1]

def compute_location_batch(W,alpha,scale = False):

		batch,Na,Nb = W.shape
		mu = torch.zeros((batch,Na+1,Nb+1))
		mu[:,0,:] = -1e20
		mu[:,:,0] = -1e20
		mu[:,0,0] = 0

		for i in range(1,Na+1):
			for j in range(1,Nb+1):

				mu[:,i,j]  = torch.logsumexp(torch.stack((mu[:,i-1,j]+alpha*W[:,i-1,j-1],mu[:,i,j-1]+
					alpha*W[:,i-1,j-1],mu[:,i-1,j-1]+alpha*W[:,i-1,j-1]),1),1)
		if scale:
			return mu[:,1:,1:]
		else:
			return mu

def compute_pi_batch(W,mu,mask,alpha):

		batch,Na,Nb = W.shape
		pi = torch.zeros((batch,Na+2,Nb+2,3))
		pi[:,0,0] = torch.tensor(([0,0,1]))

		for i in range(1,Na+1):
			for j in range(1,Nb+1):
				pi[:,i,j] = torch.stack((mu[:,i-1,j]+alpha*W[:,i-1,j-1],mu[:,i,j-1]+alpha*W[:,i-1,j-1],mu[:,i-1,j-1]+alpha*W[:,i-1,j-1]),1)
		pi = F.softmax(pi,-1)
	
		if mask == None:
			return pi[:,1:Na+1,1:Nb+1]
		else:
			return pi[:,1:Na+1,1:Nb+1]*mask

def compute_pi(W,mu,alpha):

		Na,Nb = W.shape
		pi = torch.zeros((Na+2,Nb+2,3))
		pi[0,0] = torch.tensor(([0,0,1]))

		for i in range(1,Na+1):
			for j in range(1,Nb+1):
				# pdb.set_trace()
				pi[i,j] = torch.stack((mu[i-1,j]+alpha*W[i-1,j-1],mu[i,j-1]+alpha*W[i-1,j-1],mu[i-1,j-1]+alpha*W[i-1,j-1]),0)
		pi = F.softmax(pi,-1)
	
		
		return pi[1:Na+1,1:Nb+1]


def sample(W,pi,x_length,y_length,mu,alpha):

		batch,Na,Nb = W.shape
		# pi = self.compute_transition(W,mu,x_length,y_length,True,True)
		

		Y = torch.zeros_like(W)
		logprobY = torch.zeros(batch)
		logprobY_bf = torch.zeros(batch)
		# Sample optimal path one by one
		for sample in range(batch):
			i = x_length[sample]-1; j = y_length[sample]-1
			Y[sample,i,j] = 1
			# pdb.set_trace()
			# Sample one path
			while i >= 0 and j >= 0:
				if i ==0 and j==0:
					break
				dist = Categorical(pi[sample,i,j,:])

				idx = dist.sample().item()
			 
				logprobY[sample] +=torch.log(pi[sample,i,j,idx])
				# pdb.set_trace()
				if idx == 0:
					Y[sample,i-1,j] = 1
					i -= 1
				elif idx == 1:
					Y[sample,i,j-1] = 1
					j -= 1
				elif idx == 2:
					Y[sample,i-1,j-1] = 1
					i-=1
					j-=1
				else:
					raise Exception("Incorrect sample!")
			logprobY_bf[sample] = torch.sum(Y[sample]*W[sample])*alpha - mu[sample,x_length[sample],y_length[sample]]
		
		return Y,logprobY,logprobY_bf

def compute_likelihood(W,Y,pi):

	Na, Nb = W.shape
	likelihood = 1
	i = Na-1; j=Nb-1
	while i> 0 and j>0:
		# pdb.set_trace()
		if Y[i-1,j] ==1 :
			likelihood = likelihood*pi[i,j,0]
			i -= 1
		elif Y[i,j-1] ==1:
			likelihood = likelihood*pi[i,j,1]
			j-=1
		elif Y[i-1,j-1] ==1:
			likelihood = likelihood*pi[i,j,2]
			i-=1
			j-=1
	return torch.log(likelihood)

def compute_logprob(theta,sample,bf=False):
	
	alpha = 15
	if bf == True:
		all_paths = model.all_path
		mu = 0 
		for path in range(len(all_paths)):
			yi = torch.tensor(all_paths[path,:,:],dtype=torch.float64)
			mu = mu+ torch.exp(torch.sum(yi*theta*alpha))
		# # sample = torch.tensor([[[0., 1.],[1., 0.]]],dtype=torch.float64)
		# y3 = torch.tensor([[[1., 0.],[0., 1.]]],dtype=torch.float64)
		# y1 = torch.tensor([[[1., 1.],[0., 1.]]],dtype=torch.float64)
		# y2 = torch.tensor([[[1., 0.],[1., 1.]]],dtype=torch.float64)
		# mu = torch.exp(torch.sum(y3*theta*alpha)) +torch.exp(torch.sum(y1*theta*alpha)) +torch.exp(torch.sum(y2*theta*alpha))
		score = torch.sum(sample*theta*alpha) 
		# logprob = torch.log(torch.exp(score)) - torch.log(mu)
		logprob = torch.exp(score)/mu
	else:
		mu = compute_location(theta,alpha) 
		score = torch.sum(sample*theta*alpha) 
		logprob = score - mu[-1,-1]
		# pdb.set_trace()
		# logprob = torch.exp(score)/torch.exp(mu)

	# mu = compute_location(theta,alpha)	  
	# mu = torch.exp(torch.sum(y3*theta*alpha)) +torch.exp(torch.sum(y1*theta*alpha)) +torch.exp(torch.sum(y2*theta*alpha))


	# logprob = torch.exp(score)/torch.exp(mu)
	return logprob

def compute_logprob_iter(theta,sample):
	alpha = 15
	mu = compute_location(theta,alpha) 
	score = torch.sum(sample*theta*alpha) 
	logprob = torch.log(torch.exp(score)) - mu
	# logprob = torch.exp(score)/torch.exp(mu)
	return -logprob



def compute_logprob_batch(theta,sample):
	alpha = 15
	mu = compute_location(theta,alpha) 
	
	pi = compute_pi(theta,mu,alpha)
	# pdb.set_trace()
	logprob = compute_likelihood(theta,sample,pi)


	return logprob

class DTW_func(torch.autograd.Function):

	
	@staticmethod
	def forward(ctx,theta,sample):
		logprob = compute_logprob(theta,sample)
		ctx.save_for_backward(theta,sample)
		# model = MLE()
		# logprob,_ = model(sample)
		# opt = torch.optim.Adam(model.parameters(),lr= 0.00005)
		# opt.zero_grad()
		# loss = logprob.clone()
		# loss.backward()
		# ctx.save_for_backward(model.theta.grad)

		return logprob
	
	@staticmethod
	def backward(ctx,grad_output):
		(theta,sample) = ctx.saved_tensors
	
		dx = torch.autograd.functional.jacobian(compute_logprob,(theta,sample))

		return dx[0]*grad_output,dx[1]*grad_output


def compute_batch_prob1(theta,sample):
	alpha = 15
	x_len = torch.tensor((5,4),requires_grad=False)
	y_len = torch.tensor((6,5),requires_grad=False)
	# Get 0,1 mask
	mu_batch = compute_location_batch(theta,alpha)

	pi_batch = compute_pi_batch(theta,mu_batch,None,alpha)
	logprob = compute_likelihood(theta[0,:5,:6],sample,pi_batch[0,:5,:6])
	return logprob

def compute_batch_prob(theta,):
	alpha = 15
	x_len = torch.tensor((5,4),requires_grad=False)
	y_len = torch.tensor((6,5),requires_grad=False)
	# Get 0,1 mask
	mask = torch.zeros((2,6,7),requires_grad=False)
	mask[0,:5,:6] = 1
	mask[1,:4,:5] = 1
	mask = mask.unsqueeze(-1).repeat((1,1,1,3))
	mu_batch = compute_location_batch(theta,alpha)
	pi_batch = compute_pi_batch(theta,mu_batch,mask,alpha)
	path_batch,logprob_batch,logprob_bf = sample(theta,pi_batch,x_len,y_len,mu_batch,alpha)
	return path_batch,logprob_batch

class batch_DTW_func(torch.autograd.Function):

	
	@staticmethod
	def forward(ctx,theta):
		alpha = 15
		x_len = torch.tensor((5,4),requires_grad=False)
		y_len = torch.tensor((6,5),requires_grad=False)
		# Get 0,1 mask
		mask = torch.zeros((2,6,7),requires_grad=False)
		mask[0,:5,:6] = 1
		mask[1,:4,:5] = 1
		mask = mask.unsqueeze(-1).repeat((1,1,1,3))
		mu_batch = compute_location_batch(theta,alpha)
		pi_batch = compute_pi_batch(theta,mu_batch,mask,alpha)
		path_batch,logprob_batch,logprob_bf = sample(theta,pi_batch,x_len,y_len,mu_batch,alpha)
		ctx.save_for_backward(theta)
		# ctx.others(mask,x_len,y_len)

		return logprob_batch,path_batch
	
	@staticmethod
	def backward(ctx,grad_output):
		(theta,) = ctx.saved_tensors

		dx1 = torch.autograd.functional.jacobian(compute_batch_prob,(theta,))
		# dx2 = torch.autograd.functional.jacobian(compute_batch_prob1,(theta,))
		pdb.set_trace()
		return None


def test_jacobian_1D():
	x =  torch.nn.Parameter(torch.tensor(-math.pi))
	tfunc = testfunc.apply

	# loss.backward()
	dx = torch.autograd.functional.jacobian(tfunc,x)
	print(dx)
	pdb.set_trace()
	test = gradcheck(tfunc, x, eps=1e-6, atol=1e-10)
	print(test)
	pdb.set_trace()

def Jacobian_analytical(theta, logspace=True):
	y1 = torch.exp(theta[0,0]+theta[0,1]+theta[1,1]) 
	y2 = torch.exp(theta[0,0]+theta[1,0]+theta[1,1])
	y3 = torch.exp(theta[0,0] +theta[1,1])

	J = torch.zeros_like(theta)
	if logspace == False:
		J[0,0] = y3*(y1+y2+y3)/(y1+y2+y3)**2 -  y3*(y1+y2+y3)/(y1+y2+y3)**2
		# J[1,0] = (y2)/(y1+y2+y3)
		# J[0,1] = (y1)/(y1+y2+y3)
		J[1,0] = -(y2*y3)/(y1+y2+y3)**2
		J[0,1] = -(y1*y3)/(y1+y2+y3)**2
		J[1,1] = y3*(y1+y2+y3)/(y1+y2+y3)**2 -  y3*(y1+y2+y3)/(y1+y2+y3)**2
	else:
		J[0,0] = - 1+ (y1+y2+y3)/(y1+y2+y3)
		J[1,1] = - 1+ (y1+y2+y3)/(y1+y2+y3)
		J[0,1] = -(y1)/(y1+y2+y3)
		J[1,0] =- (y2)/(y1+y2+y3)

	return J


def test():
	from MLE import MLE
	alpha = 15
	model1 = MLE(5,6)
	model2 = MLE(4,5)
	W  = torch.zeros((2,6,7))
	x_len = torch.tensor((5,4),requires_grad=False)
	y_len = torch.tensor((6,5),requires_grad=False)
	# Get 0,1 mask
	mask = torch.zeros((2,6,7),requires_grad=False)
	mask[0,:5,:6] = 1
	mask[1,:4,:5] = 1
	W[0,:5,:6] = torch.tensor(model1.W)
	W[1,:4,:5] = torch.tensor(model2.W)
	theta = torch.tensor(torch.rand((2,6,7)),requires_grad=True).float()
	theta = theta*mask
	mu_batch = compute_location_batch(theta,alpha)
	pi_batch = compute_pi_batch(theta,mu_batch,mask.unsqueeze(-1).repeat((1,1,1,3)),alpha)

	# Get two samples 
	batch_sample,logprob_batch,logprob_bf = sample(theta,pi_batch,x_len,y_len,mu_batch,alpha)

	sample1 = batch_sample[0,:5,:6]
	sample2 = batch_sample[1,:4,:5]
		
	# Get Jacobian of W1 by iter compute
	dx1 = torch.autograd.functional.jacobian(compute_logprob,(theta[0,:5,:6],sample1))
	dx2 = torch.autograd.functional.jacobian(compute_batch_prob1,(theta,sample1))
	dx3 = torch.autograd.functional.jacobian(compute_logprob_batch,(theta[0,:5,:6],sample1))
	print('(dx1) Jacobian with respect to the theta by log prob from gibbs PMF:')
	print(dx1[0])
	print('(dx2) Jacobian with respect to the theta by log prob from pi (batch compute):')
	print(dx2[0][0][:5,:6])
	print('(dx3) Jacobian with respect to the theta by log prob from pi:')
	print(dx3[0])
	print('check dx1 == dx2')
	print(torch.isclose(dx1[0],dx2[0][0][:5,:6]))
	print('check dx2 == dx3')
	print(torch.isclose(dx3[0],dx2[0][0][:5,:6]))
	d1 = dx1[0]
	d2 = dx2[0][0][:5,:6]
	d3 = dx3[0]
	pdb.set_trace()


test()

def test1():
	from MLE import MLE
	# test_jacobian_1D()
	model = MLE()
	sample = model.get_sample(1)

	# sample = torch.tensor([[[1, 1.,0],[0, 1., 0.],[0,0,1]]])

	# sample = torch.tensor([[[1.,0],[0, 1.]]]).double()

	# W = torch.tensor(model.W,requires_grad=True).double()
	# theta = torch.nn.Parameter(torch.rand(W.shape),requires_grad = True)
	# theta = W
	theta = torch.tensor(model.theta.data,requires_grad=True).double()
	# theta = torch.tensor([[1.2,1.3],[2.3,0.4]],requires_grad=True,dtype=torch.float64)
	# theta = torch.tensor([[1.5,2.5,7],[1.3,3.4,0.5],[1,4,1.5]],requires_grad=True,dtype=torch.float64)
	alpha = model.alpha

	# Define optimizer
	opt = torch.optim.Adam(model.parameters(),lr= 0.00005)

	# # Get auto grad
	model.zero_grad()
	logprob,_= model(sample)
	loss = - logprob
	loss.backward()
	# opt.step()

	func = DTW_func.apply

	# print('Manually compute Jacobian:', Jacobian_analytical(theta))

	dx1 = torch.autograd.functional.jacobian(compute_logprob_iter,(theta,sample))
	print('iter grad:', dx1[0])
	dx2 = torch.autograd.functional.jacobian(compute_logprob,(theta,sample))
	print('bf grad', dx2[0])
	print('Check equality between iter grad and bf grad',torch.isclose(dx1[0],dx2[0]))
	# nll = func(theta,sample)
	# (V_grad,) = torch.autograd.grad(nll,(theta,),create_graph=True)
	print(model.theta.grad.data)


	# print('The theta is', theta)
	# print('The sample is', sample)
	pdb.set_trace()
	input = (theta,torch.tensor(sample,requires_grad=False).double())
	# pdb.set_trace()
	test = gradcheck(func, input,eps=1e-6, atol=1e-4,rtol=1e-2)
	print(test)
	pdb.set_trace()
# test()

def ff1(input):
	return 0.5 * (5 * input ** 3 - 3 * input)

class testfunc(torch.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	@staticmethod
	def forward(ctx, input):
		"""
		In the forward pass we receive a Tensor containing the input and return
		a Tensor containing the output. ctx is a context object that can be used
		to stash information for backward computation. You can cache arbitrary
		objects for use in the backward pass using the ctx.save_for_backward method.
		"""
		ctx.save_for_backward(input)
		return 0.5 * (5 * input ** 3 - 3 * input)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		input, = ctx.saved_tensors
		return grad_output * 1.5 * (5 * input ** 2 - 1)
		# dx = torch.autograd.functional.jacobian(ff1,input)
		# return dx*grad_output
