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




class BayesianDTW(nn.Module):
    def __init__(self,alpha):
        super(BayesianDTW, self).__init__()
        self.alpha = alpha

    def compute_location(self,W,scale = False):

        batch,Na,Nb = W.shape
        mu = torch.zeros((batch,Na+1,Nb+1))
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

    def compute_transition(self,W,mu,x_length,y_length,reduced,sapar):
        batch,Na,Nb = W.shape
        pi = torch.zeros((batch,Na+2,Nb+2,3))
        pi[:,0,0] = torch.tensor(([0,0,1]))
        pi[:,-1,-1]  = torch.tensor(([0,0,1]))
        if sapar:
            # Compute transition matrix sample by sample
            for sample in range(batch):
                for i in reversed(range(1,x_length[sample]+1)):
                    for j in reversed(range(1,y_length[sample]+1)):
                        
                        pi[sample,i,j] = torch.stack((mu[sample,i-1,j]+self.alpha*W[sample,i-1,j-1],mu[sample,i,j-1]+self.alpha*W[sample,i-1,j-1],mu[sample,i-1,j-1]+self.alpha*W[sample,i-1,j-1]),0)-mu[sample,i,j]

        else:
            for i in reversed(range(1,Na+1)):
                for j in reversed(range(1,Nb+1)):            
                    pi[:,i,j] = torch.stack((mu[:,i-1,j]+self.alpha*W[:,i-1,j-1],mu[:,i,j-1]+self.alpha*W[:,i-1,j-1],mu[:,i-1,j-1]+self.alpha*W[:,i-1,j-1]),1)-mu[:,i,j].unsqueeze(1)
                    # pi[:,i,j] = torch.exp(log_pi)
        if reduced == True:
            return pi[:,1:Na+1,1:Nb+1]
        else:
            return pi
    def compute_pi(self,W,mu,mask,reduced = True):
        ''' 
            Batch compute transition matrix pi
        '''

        batch,Na,Nb = W.shape
        pi = torch.zeros((batch,Na+2,Nb+2,3))
        pi[:,0,0] = torch.tensor(([0,0,1]))
        # pi[torch.arange(batch)]
        # pi[:,-1,-1]  = torch.tensor(([0,0,1]))
        # Stack mat as [Na,Nb,3] matrix
        for i in range(1,Na+1):
            for j in range(1,Nb+1):
                pi[:,i,j] = torch.stack((mu[:,i-1,j]+self.alpha*W[:,i-1,j-1],mu[:,i,j-1]+self.alpha*W[:,i-1,j-1],mu[:,i-1,j-1]+self.alpha*W[:,i-1,j-1]),1)
        # get softmax value of pi
        pi = F.softmax(pi,-1)

        if reduced == True:
            return pi[:,1:Na+1,1:Nb+1]*mask
        else:
            # pad mask with 1 on left, right, top, bottom dim
            mask_pad = F.pad(mask,(0,0,1,1,1,1),value = 1)
            return pi*mask_pad


    def sample(self,W,pi,x_length,y_length,mu):
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
           
            # if prob_pmf == True:
            #     # Compute log prob of Y by PMF of Gibbs
            logprobY_bf[sample] = torch.sum(Y[sample]*W[sample])*self.alpha - mu[sample,x_length[sample],y_length[sample]]
        
        return Y,logprobY,logprobY_bf

    
    def compute_omega(self,W,pi,x_length,y_length):
        
        batch,Na,Nb = W.shape
        
        
        lam = torch.zeros((batch,Na+1,Nb+1))
        rho = torch.zeros((batch,Na+1,Nb+1))

        lam[:,0,0] = 1
        # rho[-1,:] = 0; rho[:,-1] = 0; rho[-1,-1]=1
        rho[torch.arange(batch),x_length,y_length] = 1

        

        # # Pad pi 
        pi = F.pad(pi,(0,0,1,1,1,1),value = 0)
        pi[:,0,0] = torch.tensor(([0,0,1]))
        pi[torch.arange(batch),x_length+1,y_length+1] = torch.tensor(([0,0,1])).float().cuda()

        omega = torch.zeros_like(pi).cuda()
        # pdb.set_trace()
        # Topological iter for lam
        for i in range(1,Na+1):
            for j in range(1,Nb+1):
                
                lam[:,i,j] = torch.matmul(torch.stack(([lam[:,i-1,j],lam[:,i,j-1],lam[:,i-1,j-1]]),1).unsqueeze(1),pi[:,i,j].unsqueeze(-1)).view(-1)
          
        
        # Reversed iter for rho one by one
        # for idx in range(batch):
        #     xlen = x_length[idx]
        #     ylen = y_length[idx]
        #     for i in reversed(range(xlen)):
        #         for j in reversed(range(ylen)):
                  
        #             rho[idx,i,j] = torch.matmul(torch.stack((rho[idx,i+1,j],rho[idx,i,j+1],rho[idx,i+1,j+1]),0), torch.stack((pi[idx,i+2,j+1,0],pi[idx,i+2,j+1,1],pi[idx,i+2,j+2,2]),0))

        for i in reversed(range(Na)):
            for j in reversed(range(Nb)):

                rho[:,i,j] = torch.matmul(torch.stack((rho[:,i+1,j],rho[:,i,j+1],rho[:,i+1,j+1]),1).unsqueeze(1),torch.stack((pi[:,i+2,j+1,0],pi[:,i+1,j+2,1],pi[:,i+2,j+2,2]),1).unsqueeze(-1)).view(-1)
                # pdb.set_trace()
                rho[torch.arange(batch),x_length,y_length] = 1
                # pi_prob = torch.sum(torch.stack((pi[:,i+2,j+1,0],pi[:,i+1,j+2,1],pi[:,i+2,j+2,2]),1),-1)
                # if pi_prob == 0:
                #     rho[:,i,j] = pi_prob
                # else:
                # rho[:,i,j] = torch.sum(torch.stack((pi[:,i+2,j+1,0],pi[:,i+1,j+2,1],pi[:,i+2,j+2,2]),1))

                # rho[i,j] = np.array(([rho[i+1,j],rho[i,j+1],rho[i+1,j+1]]))@np.array(([pi[i+1,j,0],pi[i,j+1,1],pi[i+1,j+1,2]]))
  
        for i in range(1,Na+1):
            for j in range(1,Nb+1):
               
                omega[:,i,j] = torch.stack((lam[:,i-1,j],lam[:,i,j-1],lam[:,i-1,j-1]),1)*pi[:,i,j]*rho[:,i-1,j-1].unsqueeze(1).repeat(1,3)
     
        return omega[:,1:Na+1,1:Nb+1]

    def forward(self,W,mask):
        mu = self.compute_location(W)
        pi  = self.compute_pi(W,mu,mask,True)
        # pdb.set_trace()
        return mu,pi

