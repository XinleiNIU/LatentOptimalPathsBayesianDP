import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from numpy import pi as PI
from .glow import Glow
from utils.tools import get_mask_from_lengths
# from .BayesianDTW import BayesianDTW
from .BayesianPDA import BayesianPDA
import monotonic_align
import pdb

class Prior(nn.Module):
    """ P(z|x): prior that generate the latent variables conditioned on x
    """

    def __init__(self, channels):
        super(Prior, self).__init__()
        self.channels = channels

    def _initial_sample(self, targets_lengths, temperature=1.0):
        """
        :param targets_lengths: [batch,]
        :param temperature: standard deviation
        :return: initial samples with shape [batch_size, length, channels],
                 log-probabilities: [batch, ]
        """
        batch_size = targets_lengths.shape[0]
        length = torch.max(targets_lengths).long()
        epsilon = torch.normal(0.0, temperature, [batch_size, length, self.channels]).to(targets_lengths.device)

        logprobs = -0.5 * (epsilon ** 2 + math.log(2 * PI))
        seq_mask = get_mask_from_lengths(targets_lengths).unsqueeze(-1)  # [batch, max_time, 1]
        logprobs = torch.sum(seq_mask * logprobs, dim=[1, 2])  # [batch, ]
        return epsilon, logprobs

    def forward(self, inputs, targets_lengths, condition_lengths):
        """
        :param targets_lengths: [batch, ]
        :param inputs: condition_inputs
        :param condition_lengths:
        :return: tensor1: outputs, tensor2: log_probabilities
        """
        raise NotImplementedError

    def log_probability(self, z, condition_inputs, z_lengths=None, condition_lengths=None
                        ):
        """
        compute the log-probability of given latent variables, first run through the flow
        inversely to get the initial sample, then compute the
        :param z: latent variables
        :param condition_inputs: condition inputs
        :param z_lengths:
        :param condition_lengths:
        :return: the log-probability
        """
        raise NotImplementedError

    def init(self, *inputs, **kwargs):
        """
        Initiate the weights according to the initial input data
        :param inputs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class TransformerPrior(Prior):
    def __init__(self, n_blk, channels, n_transformer_blk, embd_dim, attention_dim,
                 attention_heads, temperature, ffn_hidden,alpha):
        super(TransformerPrior, self).__init__(channels)
        self.glow = Glow(n_blk, channels, n_transformer_blk, embd_dim, attention_dim,
                         attention_heads, temperature, ffn_hidden)

        self.alpha = alpha 
        self.dtw = BayesianPDA(self.alpha)
    def pairwise_distance(self,a,b,p=2):
        return -torch.cdist(a,b,p)


    def forward(self, targets_lengths, conditional_inputs: torch.Tensor, condition_lengths, f0_embd = None, temperature=1.0, mask = None):
        # get initial noise
        epsilon, logprobs = self._initial_sample(targets_lengths, temperature=temperature)

        z = epsilon
        # Get estimated spectrogram feature
        z, logdet = self.glow(z, conditional_inputs, targets_lengths, condition_lengths)
        logprobs += logdet

        if f0_embd is not None:
            z = z+f0_embd
        
        if mask is None:
            mask = self.dtw.W_mask(z.shape[0],condition_lengths,targets_lengths)

        Wr = torch.matmul(z,conditional_inputs.permute(0,2,1))
        Wr = F.softmax(Wr,1)
        mu,pi = self.dtw(Wr,mask)
        samples =  monotonic_align.maximum_path(pi,targets_lengths,condition_lengths)
        sample_probs = self.dtw.get_logprob(samples,Wr,mu,targets_lengths,condition_lengths)
        logprobs += sample_probs
    
        return samples, logprobs

    def log_probability(self, z, condition_inputs, z_lengths=None, condition_lengths=None, mask = None, omega=None):
        """
        :param z: [batch, max_time, dim]
        :param condition_inputs:
        :param z_lengths:
        :param condition_lengths:
        :return: log-probabilities of z, [batch]
        """
        if mask is None:
            mask = self.dtw.W_mask(z.shape[0],condition_lengths,z_lengths)

        epsilon, logdet = self.glow.inverse(z.detach(), condition_inputs, z_lengths, condition_lengths)

        # Standard gaussian logprob.
        logprobs_x = -0.5 * (epsilon ** 2 + math.log(2 * PI))
        max_time = z.shape[1]
        seq_mask = get_mask_from_lengths(z_lengths, max_time).unsqueeze(-1)  # [batch, max_time]

        logprobs_x = torch.sum(seq_mask * logprobs_x, dim=[1, 2])  # [batch, ]
        std_norm = logprobs_x
        logprobs_x = logprobs_x + logdet

        if omega is not None:
            # Wr = self.pairwise_distance(z,condition_inputs)
            Wr = torch.matmul(z,conditional_inputs.permute(0,2,1))
            Wr = F.softmax(Wr,1)
            mu_r = self.dtw.compute_location_new(Wr)
            mu_r_n = self.dtw.get_mu_N(mu,z_lengths,condition_lengths)    
            path_expectation = torch.sum(omega*Wr.unsqueeze(3).repeat(1,1,1,2),(1,2,3))
            logprob_p = path_expectation*self.alpha - mu_r_n 
            logprobs_x += logprob_p 
        return logprobs_x


    def init(self, inputs: torch.Tensor, targets_lengths, condition_lengths):
        """ init flow by gaussian noise """

        # get initial noise
        epsilon, logprobs = self._initial_sample(targets_lengths)

        z = epsilon
        z, logdet = self.glow.init(z, inputs, targets_lengths, condition_lengths)
        mask = self.dtw.W_mask(z.shape[0],condition_lengths,targets_lengths)
        Wr = torch.matmul(z,inputs.permute(0,2,1))
        Wr = F.softmax(Wr,1)
        
        sample = self.dtw.init(Wr,mask,targets_lengths,condition_lengths)
        return sample, logprobs
