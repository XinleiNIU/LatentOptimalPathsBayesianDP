import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import pdb
from torch.distributions import Categorical
import torch
import math
import random 
import bisect
import collections
import torch.nn as nn
import torch.nn.functional as F
import monotonic_align
from utils.tools import get_2d_mask_from_lengths

class BayesianDTW(nn.Module):
    def __init__(self,alpha):
        super(BayesianDTW, self).__init__()
        self.alpha = alpha


    def W_mask(self,batch,x_length,y_length):
        """
            Generate parallelogram mask given xlen ylen
        """
        mask = get_2d_mask_from_lengths(y_length,x_length)
        return mask

    def compute_location(self,W,scale = False):
        """ Compute mu by N(Na*Nb) complexity """

        batch,Na,Nb = W.shape
        mu = torch.zeros((batch,Na+1,Nb+1),device=W.device)
        mu[:,0,:] = -1e20
        mu[:,:,0] = -1e20
        mu[:,0,0] = 0
        for i in range(1,Na+1):
            for j in range(1,Nb+1):

                mu[:,i,j]  = torch.logsumexp(torch.stack((mu[:,i-1,j]+self.alpha*W[:,i-1,j-1],mu[:,i,j-1]+
                    self.alpha*W[:,i-1,j-1],mu[:,i-1,j-1]+self.alpha*W[:,i-1,j-1]),1),1)
        if scale:
            return mu[:,1:,1:]
        else:
            return mu

    def compute_location_new(self,W,scale = False):
        """ Compute mu by N(Na+Nb-1) complexity """

        batch,Na,Nb = W.shape
        # Init mu
        mu = torch.zeros((batch,Na+1,Nb+1),device=W.device)
        mu[:,0,:] = -1e20
        mu[:,:,0] = -1e20
        mu[:,0,0] = 0
        for i in range(Na+Nb-1):
            output = self.sub_compute_mu(mu,W)
            mu = F.pad(output,(1,0,1,0),"constant",-1e20)
            mu[:,0,0] = 0
        if scale:
            return mu[:,1:,1:]
        else:
            return mu

    def compute_location_2(self,W,scale = False):
        """ Compute mu by N(Na+Nb-1) complexity """

        batch,Na,Nb = W.shape
        # Init mu
        mu = torch.zeros((batch,Na+1,Nb+1),device=W.device)
        mu[:0,:] = -1e20
        mu[:,:,0] = -1e20
        mu[:,0,0] = 0


        # Init a mask to indicate if the node have been computed.
        computed_mask = torch.zeros((Na+1,Nb+1),device=W.device).bool() # Size: [Na+1,Nb+1]
        computed_mask[:,0] = True
        computed_mask[0,:] = True

        # Compute
        for i in range(Na+Nb-1):
            output,output_mask,computed_mask= self.sub_compute_mu_2(mu,W,batch,computed_mask)
            mu[:,output_mask] = output.view(batch,-1)

        if scale:
            return mu[:,1:,1:]
        else:
            return mu

    def sub_compute_mu(self,mu,W):

        prev = torch.stack((mu[:,:-1,1:],mu[:,1:,:-1], mu[:,:-1,:-1]),3)
        output = prev + self.alpha*W.unsqueeze(3).repeat(1,1,1,3)
        output = torch.logsumexp(output,-1)
        return output

    def sub_compute_mu_2(self,mu,W,batch,computed_mask):
        """ Compute part of mu on the ith step """
 
        # Compute index of selected tensors from i
        nodes_with_valid_parents = computed_mask[:-1,:-1] & computed_mask[1:,:-1] & computed_mask[:-1,1:]
        nodes_not_computed_yet = ~computed_mask[1:,1:]
       
        notes_to_be_compute_in_this_step_i = nodes_with_valid_parents & nodes_not_computed_yet


        parent_top_left = mu[:,:-1,:-1].masked_select(notes_to_be_compute_in_this_step_i)
        parent_left =  mu[:,1:,:-1].masked_select(notes_to_be_compute_in_this_step_i) 
        parent_top = mu[:,:-1,1:].masked_select(notes_to_be_compute_in_this_step_i)
        prev = torch.stack((parent_left,parent_top,parent_top_left),dim=1)

        # Fetch W
        fetched_w = W.masked_select(notes_to_be_compute_in_this_step_i)

        # Compute mu for selected tensors.
        output = torch.logsumexp(prev + self.alpha*fetched_w.unsqueeze(1).repeat(1,3),-1)

        # Pad mask to (Na+1,Nb+1)
        output_mask = F.pad(notes_to_be_compute_in_this_step_i,(1,0,1,0),'constant',False)

        computed_mask = F.pad(nodes_with_valid_parents,(1,0,1,0),'constant',True)
 
        return output,output_mask,computed_mask



    def compute_pi_new(self,W,mu,mask,reduced = True):
        ''' 
            Batch compute transition matrix pi
        '''

        mu_stack= torch.stack((mu[:,:-1,1:],mu[:,1:,:-1],mu[:,:-1,:-1]),3)
        # pi[top, left, top_left]
        pi = mu_stack+self.alpha*W.unsqueeze(3).repeat(1,1,1,3)
        pi = F.softmax(pi,-1)
    
        if reduced == True:
            return pi*mask.unsqueeze(-1).repeat(1,1,1,3)
        else:
            # pad mask with 1 on left, right, top, bottom dim
            mask_pad = F.pad(mask,(0,0,1,1,1,1),value = 1)
            return pi*mask_pad

    def sample(self,W,pi,x_length,y_length,prob_pmf=False):
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

        batch,Na,Nb = W.shape
        # pi = self.compute_transition(W,mu,x_length,y_length,True,True)
        

        Y = torch.zeros_like(W,device = W.device)
        logprobY = torch.zeros(batch,device = W.device)
        # Sample optimal path one by one
        for sample in range(batch):
            i = x_length[sample]-1; j = y_length[sample]-1
            Y[sample,i,j] = 1
            # pdb.set_trace()
            # Sample one path
            while i >= 0 and j >= 0:
                if i ==0 and j==0:
                    break
                # prob = pi[i,j,:]
                dist = Categorical(pi[sample,i,j,:])

                idx = dist.sample().item()
                if prob_pmf == False:
                    logprobY[sample] +=torch.log(pi[sample,i,j,idx])
                
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
            if prob_pmf == True:
                # Compute log prob of Y by PMF of Gibbs
                logprobY[sample] = torch.sum(Y[sample]*W[sample])*self.alpha - mu[sample,x_length[sample],y_length[sample]]
        
        return Y,logprobY

    def sub_compute_lambda(self,lam, pi):

        prev = torch.stack((lam[:,:-1,1:],lam[:,1:,:-1],lam[:,:-1,:-1]),3)
        output = torch.matmul(prev.unsqueeze(3),pi.unsqueeze(-1)).view(lam[:,:-1,:-1].shape)
        return output

    def sub_compute_lambda_2(self,lam,pi,batch,computed_mask):
        """ Compute part of mu on the ith step """
 
        # Compute index of selected tensors from i
        nodes_with_valid_parents = computed_mask[:-1,:-1] & computed_mask[1:,:-1] & computed_mask[:-1,1:]
        nodes_not_computed_yet = ~computed_mask[1:,1:]
       
        notes_to_be_compute_in_this_step_i = nodes_with_valid_parents & nodes_not_computed_yet


        parent_top_left = lam[:,:-1,:-1].masked_select(notes_to_be_compute_in_this_step_i)
        parent_left =  lam[:,1:,:-1].masked_select(notes_to_be_compute_in_this_step_i) 
        parent_top = lam[:,:-1,1:].masked_select(notes_to_be_compute_in_this_step_i)
        prev = torch.stack((parent_top,parent_left,parent_top_left),dim=1)


        # # Fetch W
        fetch_pi = pi.masked_select(notes_to_be_compute_in_this_step_i.unsqueeze(-1).repeat(1,1,1,3)).view(-1,3)

        # # Compute mu for selected tensors.
        update_lam = torch.matmul(prev.unsqueeze(1),fetch_pi.unsqueeze(-1)).view(batch,-1)

        # # Pad mask to (Na+1,Nb+1)
        output_mask = F.pad(notes_to_be_compute_in_this_step_i,(1,0,1,0),'constant',False)
        # output[notes_to_be_compute_in_this_step_i] = computed_mu
        
        lam[:,output_mask] = update_lam

        # del output_mask
        computed_mask = F.pad(nodes_with_valid_parents,(1,0,1,0),'constant',True)
 
        return lam,computed_mask


    def sub_compute_rho(self,rho, pi):
        prev_rho = torch.stack((rho[:,1:,:-1],rho[:,:-1,1:],rho[:,1:,1:]),3)
        pi_stack = torch.stack((pi[:,1:,:-1,0],pi[:,:-1,1:,1],pi[:,1:,1:,2]),3)
        output = torch.matmul(prev_rho.unsqueeze(3),pi_stack.unsqueeze(-1)).view(rho[:,:-1,:-1].shape)
        return output

    def sub_compute_rho_2(self,rho,pi,batch,computed_mask):
        """ Compute part of mu on the ith step """
 
        # Compute index of selected tensors from i
        nodes_with_valid_parents = computed_mask[1:,:-1] & computed_mask[:-1,1:] & computed_mask[1:,1:]
        nodes_not_computed_yet = ~computed_mask[:-1,:-1]
       
        notes_to_be_compute_in_this_step_i = nodes_with_valid_parents & nodes_not_computed_yet


        parent_top_left = rho[:,1:,1:].masked_select(notes_to_be_compute_in_this_step_i)
        parent_left =  rho[:,:-1,1:].masked_select(notes_to_be_compute_in_this_step_i) 
        parent_top = rho[:,1:,:-1].masked_select(notes_to_be_compute_in_this_step_i)
        prev = torch.stack((parent_top,parent_left,parent_top_left),dim=1)

        
        # # Fetch pi
        pi_top = pi[:,1:,:-1].masked_select(notes_to_be_compute_in_this_step_i.unsqueeze(-1).repeat(1,1,1,3)).view(-1,3)[:,0]
        pi_left = pi[:,:-1,1:].masked_select(notes_to_be_compute_in_this_step_i.unsqueeze(-1).repeat(1,1,1,3)).view(-1,3)[:,1]
        pi_top_left = pi[:,1:,1:].masked_select(notes_to_be_compute_in_this_step_i.unsqueeze(-1).repeat(1,1,1,3)).view(-1,3)[:,2]

        fetch_pi = torch.stack((pi_top,pi_left,pi_top_left),dim=1) 
       
        # # Compute mu for selected tensors.
        update_rho = torch.matmul(prev.unsqueeze(1),fetch_pi.unsqueeze(-1)).view(batch,-1)
      
        # # Pad mask to (Na+1,Nb+1)
        output_mask = F.pad(notes_to_be_compute_in_this_step_i,(0,1,0,1),'constant',False)
        # output[notes_to_be_compute_in_this_step_i] = computed_mu
        
        rho[:,output_mask] = update_rho

        # del output_mask
        computed_mask = F.pad(nodes_with_valid_parents,(0,1,0,1),'constant',True)
 
        return rho,computed_mask
    
    def change_pad_top_left(self,input,batch,Na,Nb,x_length,y_length):
        """
            Change padding from bottom right to top left
        """    
        placehoder = torch.zeros_like(input)
        for i in range(batch):
            xlen = x_length[i]
            ylen = y_length[i]
            placehoder[i,Na-xlen:,Nb-ylen:] = input[i,:xlen,:ylen]
        return placehoder

    def change_pad_bottom_right(self,input,batch,Na,Nb,x_length,y_length):
        """
            Change padding from bottom right to top left
        """    
        placehoder = torch.zeros_like(input)
        for i in range(batch):
            xlen = x_length[i]
            ylen = y_length[i]
          
            placehoder[i,:xlen+1,:ylen+1] = input[i,Na-xlen:,Nb-ylen:]
           
        return placehoder

    def compute_omega(self,W,pi,x_length,y_length):

        batch,Na,Nb = W.shape
        
        lam = torch.zeros((batch,Na+1,Nb+1),device = W.device)
        rho = torch.zeros((batch,Na+1,Nb+1),device = W.device)

        lam[:,0,0] = 1
        rho[:,-1,-1] = 1
    

      
        for i in range(Na+Nb-1):
            # lam,computed_mask = self.sub_compute_lambda_2(lam,pi,batch,computed_mask)
            output = self.sub_compute_lambda(lam, pi)
            lam = F.pad(output,(1,0,1,0),"constant",0)
            lam[:,0,0] = 1

        # Form a new pi with top-left padding
        pi_new = self.change_pad_top_left(pi,batch,Na,Nb,x_length,y_length)
        pi_new = F.pad(pi_new,(0,0,0,1,0,1),value = 0)
        pi_new[:,-1,-1] = torch.tensor(([0,0,1]),device = W.device).repeat(batch,1).float()

        
        for i in range(Na+Nb-1):
            # rho,computed_mask = self.sub_compute_rho_2(rho,pi_new,batch,computed_mask)
            output = self.sub_compute_rho(rho,pi_new)
            rho = F.pad(output,(0,1,0,1),"constant",0)
            rho[:,-1,-1] = 1

        rho_new = self.change_pad_bottom_right(rho,batch,Na,Nb,x_length,y_length)

        omega =  torch.stack((lam[:,:-1,1:],lam[:,1:,:-1],lam[:,:-1,:-1]),3) * pi * rho_new[:,:-1,:-1].unsqueeze(-1).repeat(1,1,1,3)
        return omega
    def get_mu_N(self,mu,x_length,y_length):
        """
            Get mu_N given mu matrix
        """
        mu_mask = torch.zeros_like(mu)
        mu_mask[torch.arange(len(mu)),x_length,y_length] = 1
        return torch.sum(mu*mu_mask,(1,2))
        
    def get_logprob(self,sample,W,mu,x_length,y_length):
        """
            Get log-probablity given batch of paths
        """
        mu_N = self.get_mu_N(mu,x_length,y_length)

        logprob = torch.sum(sample*W,(1,2))*self.alpha- mu_N
        return logprob

    def compute_omega_new(self,W,pi,x_length,y_length):

        batch,Na,Nb = W.shape
        
        lam = torch.zeros((batch,Na+1,Nb+1),device = W.device)
        rho = torch.zeros((batch,Na+1,Nb+1),device = W.device)

        lam[:,0,0] = 1
        rho[:,-1,-1] = 1
        
        computed_mask = torch.zeros((Na+1,Nb+1),device=W.device).bool() # Size: [Na+1,Nb+1]
        computed_mask[:,0] = True
        computed_mask[0,:] = True

      
        for i in range(Na+Nb-1):
            lam,computed_mask = self.sub_compute_lambda_2(lam,pi,batch,computed_mask)


        # Form a new pi with top-left padding
        pi_new = self.change_pad_top_left(pi,batch,Na,Nb,x_length,y_length)
        pi_new = F.pad(pi_new,(0,0,0,1,0,1),value = 0)
        pi_new[:,-1,-1] = torch.tensor(([0,0,1]),device = W.device).repeat(batch,1).float()

        computed_mask = torch.zeros((Na+1,Nb+1),device=W.device).bool() # Size: [Na+1,Nb+1]
        computed_mask[:,-1] = True
        computed_mask[-1,:] = True
        
        for i in range(Na+Nb-1):
            rho,computed_mask = self.sub_compute_rho_2(rho,pi_new,batch,computed_mask)

        rho_new = self.change_pad_bottom_right(rho,batch,Na,Nb,x_length,y_length)

        omega =  torch.stack((lam[:,:-1,1:],lam[:,1:,:-1],lam[:,:-1,:-1]),3) * pi * rho_new[:,:-1,:-1].unsqueeze(-1).repeat(1,1,1,3)
        return omega

    
    def init(self,W,mask,x_length,y_length):
        W = W*mask
        mu = self.compute_location_new(W)
        pi  = self.compute_pi_new(W,mu,mask,True)
        samples =  monotonic_align.maximum_path(pi,x_length,y_length)
        return samples

    def forward(self,W,mask):
        mu= self.compute_location_new(W) 
        pi = self.compute_pi_new(W,mu,mask,True)
        return mu,pi

