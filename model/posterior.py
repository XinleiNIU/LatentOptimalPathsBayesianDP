import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from .utils import LinearNorm, PreNet, PositionalEncoding,PostNet
from .attention import CrossAttentionBlock,MultiHeadScaledProductAttention
from utils.tools import get_mask_from_lengths,get_2d_mask_from_lengths
import pdb
from .BayesianDTW import BayesianDTW
from .BayesianPDA import BayesianPDA
import monotonic_align

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	
class BasePosterior(nn.Module):
	"""Encode the target sequence into latent distributions"""

	def __init__(self):
		super(BasePosterior, self).__init__()

	def forward(self, inputs, src_enc, src_lengths=None, target_lengths=None
				):
		raise NotImplementedError

	@staticmethod
	def reparameterize(mu, logvar, nsamples=1, random=True):
		"""
		:param mu: [batch, max_time, dim]
		:param logvar: [batch, max_time, dim]
		:param nsamples: int
		:param random: whether sample from N(0, 1) or just use zeros
		:return: samples, noises, [batch, nsamples, max_time, dim]
		"""
		# print('tracing back at posterior reparameterize')
		batch, max_time, dim = mu.shape
		std = torch.exp(0.5 * logvar)
		if random:
			eps = torch.normal(0.0, 1.0, [batch, nsamples, max_time, dim]).to(mu.device)
		else:
			eps = torch.zeros([batch, nsamples, max_time, dim], device=mu.device)
		samples = eps * std.unsqueeze(1) + mu.unsqueeze(1)
		return samples, eps


	def sample(self, inputs, src_enc, input_lengths, src_lengths,
			   nsamples=1, random=True):
		"""
		:param inputs: [batch, tgt_max_time, in_dim]
		:param src_enc: [batch, src_max_time, emb_dim]
		:param input_lengths: [batch, ]
		:param src_lengths: [batch, ]
		:param nsamples:
		:param random:
		:return:
		tensor1: samples from the posterior, [batch, nsamples, tgt_max_time, dim]
		tensor2: log-probabilities, [batch, nsamples]
		"""
		raise NotImplementedError




class TransformerPosterior(BasePosterior):
	def __init__(self, num_mels, n_conv, conv_kernel, post_hidden,
				 pos_drop_rate,alpha):
		super(TransformerPosterior, self).__init__()
		self.pos_weight = nn.Parameter(torch.tensor(1.0, device=device))

		self.postnet = PostNet(n_conv = n_conv,
								hidden=num_mels,
							conv_filters =post_hidden,
							conv_kernel = conv_kernel,
	        	        	drop_rate= pos_drop_rate)

		self.alpha = alpha 
		self.dtw = BayesianPDA(self.alpha)


	def pairwise_distance(self,a,b,p=2):
		# return torch.cdist(a.double(),b.double(),p)
		return - torch.cdist(a,b,p)

	def log_probability(self, mu, W, pi,input_lengths,src_lengths):
		'''
			log prob for KLD
		'''
		mu_mask = torch.zeros_like(mu)
		mu_mask[torch.arange(len(mu)),input_lengths,src_lengths] = 1
		mu_n = torch.sum(mu*mu_mask,(1,2))
		omega = self.dtw.compute_omega(pi,input_lengths,src_lengths)
		path_exp = torch.sum(omega*W.unsqueeze(3).repeat(1,1,1,2),(1,2,3))
		logprob_q = path_exp*self.alpha  - mu_n 
		return logprob_q,omega

	def forward(self, inputs, src_enc, src_lengths=None, target_lengths=None, mask = None, f0_embd = None):
		# print('tracing back at posterior call')

		postnet_outs = self.postnet(inputs)
		if f0_embd is not None:
			Wspec = postnet_outs + f0_embd
		else:
			Wspec = postnet_outs
		if mask is None:
			mask = self.dtw.W_mask(inputs.shape[0],src_lengths,target_lengths)

		W = torch.matmul(Wspec,src_enc.permute(0,2,1))
		alignments = F.softmax(W,1)
		mu,pi = self.dtw(alignments,mask)
			
		return mu,pi,alignments,postnet_outs
		

	def sample(self, inputs, src_enc, input_lengths, src_lengths, mask = None, f0_embd = None,
			   nsamples=1, random=True):
		"""
			input_length: text_length
			src_length: mel_length
		"""
		if mask is None:
			mask = self.dtw.W_mask(inputs.shape[0],input_lengths,src_lengths)
		mu, pi, W, spec_embs= self.forward(inputs, src_enc, input_lengths, src_lengths, mask, f0_embd)
		# omega = self.log_exp_path(W, pi,input_lengths,src_lengths)
		samples = monotonic_align.maximum_path(pi,src_lengths,input_lengths)
		log_probs = self.dtw.get_logprob(samples,W,mu,src_lengths,input_lengths)
		return samples, log_probs, mu, pi, W, spec_embs

	def sample_random(self,W, input_lengths,src_lengths,mask):
		mu,pi = self.dtw(W,mask)
		samples = monotonic_align.maximum_path(pi,src_lengths,input_lengths)
		return samples

