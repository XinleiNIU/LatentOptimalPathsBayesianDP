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

from pdb import set_trace as st


class BayesianPDA(nn.Module):
    def __init__(self,alpha):
        super(BayesianPDA, self).__init__()
        self.alpha = alpha

    def W_mask(self,batch,x_length,y_length):
        """
            Generate parallelogram mask given xlen ylen
        """
        Nb =x_length.max(); Na = y_length.max()
        mask = torch.zeros((batch,Na,Nb),device=x_length.device).bool()
        
        for b in range(batch):
            xlen = x_length[b]
            ylen = y_length[b]
            assert ylen>xlen
            mask_i = torch.ones(ylen,xlen,device=x_length.device)
            
            # get upper triangular mask
            tri_up = torch.tril(mask_i).bool()
            tri_lo = torch.triu(mask_i,-ylen+xlen).bool()

            # tri_up = torch.triu(mask_i).bool()
            # tri_lo = torch.tril(mask_i,ylen-xlen).bool()
            mask[b,:ylen,:xlen] = (tri_up&tri_lo) 
        return mask


    def compute_location(self,W,scale = False):
        """ Compute mu by N(Na*Nb) complexity """

        batch,Na,Nb = W.shape
        mu = torch.ones((batch,Na+1,Nb+1),device=W.device)*-1e20
        mu[:,0,0] = 0

        for i in range(1,Na+1):
            for j in range(1,min(i,Nb)+1):

                mu[:,i,j]  = torch.logsumexp(torch.stack((mu[:,i-1,j]+self.alpha*W[:,i-1,j-1],mu[:,i-1,j-1]+self.alpha*W[:,i-1,j-1]),1),1)
        if scale:
            return mu[:,1:,1:]
        else:
            return mu

    def compute_location_new(self,W,scale = False):
        """ Compute mu by N(Na+Nb-1) complexity """

        batch,Na,Nb = W.shape
        # Init mu
        mu = torch.ones((batch,Na+1,Nb+1),device=W.device)*-1e20
        mu[:,0,0] = 0
        for i in range(Na+Nb-1):
            output = self.sub_compute_mu(mu,W)
            mu = F.pad(output,(1,0,1,0),"constant",-1e20)
            mu[:,0,0] = 0
        if scale:
            return mu[:,1:,1:]
        else:
            return mu

    def sub_compute_mu(self,mu,W):

        prev = torch.stack((mu[:,:-1,1:], mu[:,:-1,:-1]),3)
        output = prev + self.alpha*W.unsqueeze(3).repeat(1,1,1,2)
        output = torch.logsumexp(output,-1)
        return output

    def compute_location_2(self,W,scale = False):
        """ Compute mu by N(Na+Nb-1) complexity """

        batch,Na,Nb = W.shape

        mu = []
        # Init mu
        for i in range(Na+1):
            row = []
            for j in range(Nb+1):
                row.append(torch.ones((batch),device=W.device)*-1e20)
            mu.append(row)

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

    def sub_compute_mu_2(self,mu,W,batch,computed_mask):
        """ Compute part of mu on the ith step """
 
        # Compute index of selected tensors from i
        nodes_with_valid_parents = computed_mask[:-1,:-1]& computed_mask[:-1,1:]
        nodes_not_computed_yet = ~computed_mask[1:,1:]
       
        notes_to_be_compute_in_this_step_i = nodes_with_valid_parents & nodes_not_computed_yet

        parent_top_left = mu[:,:-1,:-1].masked_select(notes_to_be_compute_in_this_step_i)
        parent_left =  mu[:,1:,:-1].masked_select(notes_to_be_compute_in_this_step_i) 
        parent_top = mu[:,:-1,1:].masked_select(notes_to_be_compute_in_this_step_i)
        prev = torch.stack((parent_left,parent_top,parent_top_left),dim=1)

        # Fetch W
        fetched_w = W.masked_select(notes_to_be_compute_in_this_step_i)

        # Compute mu for selected tensors.
        # prev = torch.stack((mu[:,:-1,1:],mu[:,1:,:-1], mu[:,:-1,:-1]),3)
        # output = prev + self.alpha*W.unsqueeze(3).repeat(1,1,1,3)
        output = torch.logsumexp(prev + self.alpha*fetched_w.unsqueeze(1).repeat(1,3),-1)

        # Pad mask to (Na+1,Nb+1)
        output_mask = F.pad(notes_to_be_compute_in_this_step_i,(1,0,1,0),'constant',False)
        # output[notes_to_be_compute_in_this_step_i] = computed_mu

        computed_mask = F.pad(nodes_with_valid_parents,(1,0,1,0),'constant',True)
 
        return output,output_mask,computed_mask

    def compute_location_3(self,W,scale = False):
        """
            Compute location parameter by a O(N) iteration
        """
        batch,Na,Nb = W.shape
        
        # Init mu
        # old_mu = torch.ones((batch,Na+1,Nb+1),device=W.device)*-1e20
        mu = [torch.ones((batch,Nb+1),device=W.device)*-1e20 for row in range(Na+1)]

        # old_mu[:,0,0] = 0
        mu[0][:,0] = 0

        # # Old method
        # for i in range(1,Na+1):
        #     top_left = old_mu[:,i-1,:-1]
        #     top = old_mu[:,i-1,1:]
        #     # Update mu
        #     old_mu[:,i,1:] = torch.logsumexp(torch.stack([top_left + self.alpha*W[:,i-1,:], top+self.alpha*W[:,i-1,:]],1),1)

        # New method
        for i in range(1,Na+1):
            top_left = mu[i-1][:,:-1]
            top = mu[i-1][:,1:]
            # Update mu
            mu[i][:,1:] = torch.logsumexp(torch.stack([top_left + self.alpha*W[:,i-1,:], top+self.alpha*W[:,i-1,:]],1),1)

        mu = torch.stack(mu,dim=1)
        if scale:
            return mu[:,1:,1:]
        else:
            return mu

    def get_mu_N(self,mu,x_length,y_length):
        """
            Get mu_N given mu matrix
        """
        mu_mask = torch.zeros_like(mu)
        mu_mask[torch.arange(len(mu)),x_length,y_length] = 1
        return torch.sum(mu*mu_mask,(1,2))

    def compute_pi_bf(self,W,mu,x_length,y_length,reduced=True):
        W = W[-1]
        mu = mu[-1]

        Na = x_length[-1]
        Nb = y_length[-1]

        pi = torch.zeros((Nb+2,Na+2,2),device = W.device)
        pi[0,0] = torch.tensor(([0,1]))
        pi[-1,-1]  = torch.tensor(([0,1]))
      
        for i in reversed(range(1,Nb+1)):
            for j in reversed(range(max(i-(Nb-Na+1)+1,1),min(i,Na)+1)):     
                
                log_pi = torch.stack((mu[i-1,j]+self.alpha*W[i-1,j-1],mu[i-1,j-1]+self.alpha*W[i-1,j-1]),0)-mu[i,j]
                pi[i,j] = torch.exp(log_pi)

        if reduced == True:
            return pi[1:Nb+1,1:Na+1]
        else:
            return pi

    def compute_pi_new(self,W,mu,mask,reduced = True):
        ''' 
            Batch compute transition matrix pi
        '''

        mu_stack= torch.stack((mu[:,:-1,1:],mu[:,:-1,:-1]),3)
        # pi[top, left, top_left]
        pi = mu_stack+self.alpha*W.unsqueeze(3).repeat(1,1,1,2)
        pi = F.softmax(pi,-1)
     
        if reduced == True:
            return pi*mask.unsqueeze(-1).repeat(1,1,1,2)
        else:
            # pad mask with 1 on left, right, top, bottom dim
            mask_pad = F.pad(mask,(0,0,1,1,1,1),value = 1)
            return pi*mask_pad


    def sample(self,W,pi,mu,x_length,y_length,prob_pmf=False):
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
        # logprobY = torch.zeros(batch,device = W.device)
        # logprobY = []
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

                # if torch.rand(1,device = W.device)< pi[sample,i,j,0]:
                #     Y[sample,i-1,j] = 1
                #     i -= 1
                # else: 
                #     Y[sample,i-1,j-1] = 1
                #     i-=1
                #     j-=1
                # if prob_pmf == False:
                #     logprobY[sample] +=torch.log(pi[sample,i,j,idx])
            
                if idx == 0:
                    Y[sample,i-1,j] = 1
                    i -= 1
                elif idx == 1:
                    Y[sample,i-1,j-1] = 1
                    i-=1
                    j-=1
                else:
                    raise Exception("Incorrect sample!")

        #     if prob_pmf == True:
        #         # Compute log prob of Y by PMF of Gibbs
        #         logprobY.append(torch.sum(Y[sample]*W[sample])*self.alpha - mu[sample,x_length[sample],y_length[sample]])

        logprobY = self.get_logprob(Y,W,mu,x_length,y_length)

        return Y,logprobY

    def get_logprob(self,sample,W,mu,x_length,y_length):
        """
            Get log-probablity given batch of paths
        """
        mu_N = self.get_mu_N(mu,x_length,y_length)

        logprob = torch.sum(sample*W,(1,2))*self.alpha- mu_N
        return logprob

    def sample_from_one(self,W,pi,x_length,y_length):
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

        Y = torch.zeros((x_length,y_length),device = W.device)
        # Sample optimal path one by one
        i = x_length-1; j = y_length-1
        Y[i,j] = 1

        # pdb.set_trace()
        # Sample one path
        while i >= 0 and j >= 0:
            if i ==0 and j==0:
                break
            # prob = pi[i,j,:]
            dist = Categorical(pi[0,i,j,:])

            idx = dist.sample().item()
   
            if idx == 0:
                Y[i-1,j] = 1
                i -= 1
            elif idx == 1:
                Y[i-1,j-1] = 1
                i-=1
                j-=1
            else:
                raise Exception("Incorrect sample!")
        
        return Y

    def sub_compute_lambda(self,lam, pi):

        prev = torch.stack((lam[:,:-1,1:],lam[:,:-1,:-1]),3)
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
        prev_rho = torch.stack((rho[:,1:,:-1],rho[:,1:,1:]),3)
        pi_stack = torch.stack((pi[:,1:,:-1,0],pi[:,1:,1:,1]),3)
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

    def compute_omega_bf(self,W,pi,x_length,y_length):
        """
            Compute omega by O(n^2)
        """
        
        Na = x_length[0]; Nb = y_length[0]
        W = W[0,:Na,:Nb].detach().cpu().numpy()
        pi_old = pi[0,:Na,:Nb].detach().cpu().numpy()

        pi = np.zeros((Na+2,Nb+2,2))
        pi[1:Na+1,1:Nb+1]= pi_old
        pi[0,0] = np.array(([0,1]))
        pi[-1,-1] = np.array(([0,1]))

        omega = np.zeros_like(pi)
        
        lam = np.zeros((Na+1,Nb+1))
        rho = np.zeros((Na+1,Nb+1))

        lam[0,:] = 0; lam[:,0] = 0; lam[0,0] = 1
        rho[-1,:] = 0; rho[:,-1] = 0; rho[-1,-1]=1

        # Topological iter for lam
        for i in range(1,Na+1):
            for j in range(1,Nb+1):
                lam[i,j] = np.array(([lam[i-1,j],lam[i-1,j-1]]))@pi[i,j]
        
        # Reversed iter for rho
        for i in reversed(range(Na)):
            for j in reversed(range(Nb)):
                rho[i,j] = np.array(([rho[i+1,j],rho[i+1,j+1]]))@np.array(([pi[i+2,j+1,0],pi[i+2,j+2,1]]))

        # Compute omega
        for i in range(1,Na+1):
            for j in range(1,Nb+1):
                omega[i,j] = np.array(([lam[i-1,j],lam[i-1,j-1]]))*pi[i,j]*rho[i-1,j-1]
        
        return omega[1:Na+1,1:Nb+1]

    def compute_omega2(self,pi,x_length,y_length):
        """
            Compute omega by O(N+1) 
        """

        batch,Na,Nb,_ = pi.shape
        
        lam = torch.zeros((batch,Na+1,Nb+1),device = pi.device)
        rho = torch.zeros((batch,Na+1,Nb+1),device = pi.device)

        lam[:,0,0] = 1
        rho[:,-1,-1] = 1
    
   
        for i in range(Na+1):
            # lam,computed_mask = self.sub_compute_lambda_2(lam,pi,batch,computed_mask)
            output = self.sub_compute_lambda(lam, pi)
            lam = F.pad(output,(1,0,1,0),"constant",0)
            lam[:,0,0] = 1
 
        # Form a new pi with top-left padding
        pi_new = self.change_pad_top_left(pi,batch,Na,Nb,x_length,y_length)
        pi_new = F.pad(pi_new,(0,0,0,1,0,1),value = 0)
        pi_new[:,-1,-1] = torch.tensor(([0, 1]),device = pi.device).repeat(batch,1).float()

        
        for i in range(Na+1):
            # rho,computed_mask = self.sub_compute_rho_2(rho,pi_new,batch,computed_mask)
            output = self.sub_compute_rho(rho,pi_new)
            rho = F.pad(output,(0,1,0,1),"constant",0)
            rho[:,-1,-1] = 1
        
        rho_new = self.change_pad_bottom_right(rho,batch,Na,Nb,x_length,y_length)

        omega =  torch.stack((lam[:,:-1,1:],lam[:,:-1,:-1]),3) * pi * rho_new[:,:-1,:-1].unsqueeze(-1).repeat(1,1,1,2)
        return omega


    def compute_omega(self,pi,x_length,y_length):
        """
            Compute omega by O(N) method
        """
        batch,Na,Nb,_ = pi.shape
        
        # old_lam = torch.zeros((batch,Na+1,Nb+1),device = pi.device)
        lam = [torch.zeros((batch, Nb+1),device = pi.device) for rows in range(Na+1)]
        # old_rho = torch.zeros((batch,Na+1,Nb+1),device = pi.device)
        rho = [torch.zeros((batch,Nb+1),device = pi.device) for rows in range(Na+1)]

        # old_lam[:,0,0] = 1
        lam[0][:,0] = 1
        # old_rho[:,-1,-1] = 1
        rho[-1][:,-1] = 1

        # # Original method
        # # Topological iter for lam
        # for i in range(1,Na+1):
        #     top_left = old_lam[:,i-1,:-1]
        #     top = old_lam[:,i-1,1:]

        #     old_lam[:,i,1:] = torch.sum(torch.stack([top,top_left],2)* pi[:,i-1,:],-1)
        #     # pdb.set_trace()
        #     # lam[:,i,1:] = torch.matmul(torch.stack([top,top_left],2).unsqueeze(2),pi[:,i-1,:].unsqueeze(-1)).view(batch,-1)

        # New method
        # Topological iter for lam
        for i in range(1,Na+1):
            top_left = lam[i-1][:,:-1]
            top = lam[i-1][:,1:]

            lam[i][:,1:] = torch.sum(torch.stack([top,top_left],2)* pi[:,i-1,:],-1)
            # pdb.set_trace()
            # lam[:,i,1:] = torch.matmul(torch.stack([top,top_left],2).unsqueeze(2),pi[:,i-1,:].unsqueeze(-1)).view(batch,-1)

        lam = torch.stack(lam,dim=1)
        # st()

        # Form a new pi with top-left padding
        pi_new = self.change_pad_top_left(pi,batch,Na,Nb,x_length,y_length)
        pi_new = F.pad(pi_new,(0,0,0,1,0,1),value = 0)
        pi_new[:,-1,-1] = torch.tensor(([0, 1]),device = pi.device).repeat(batch,1).float()

        # # Original method
        # # Reversed iter for rho
        # for i in reversed(range(Na)):
        #     prev_rho = torch.stack([rho[:,i+1,:-1], rho[:,i+1,1:]],2)
        #     pi_stack = torch.stack((pi_new[:,i+1,:-1,0],pi_new[:,i+1,1:,1]),2)
        #     rho[:,i,:-1] = torch.sum(prev_rho*pi_stack,-1)

        # New method
        # Reversed iter for rho
        for i in reversed(range(Na)):
            prev_rho = torch.stack([rho[i+1][:,:-1], rho[i+1][:,1:]],2)
            pi_stack = torch.stack((pi_new[:,i+1,:-1,0],pi_new[:,i+1,1:,1]),2)
            rho[i][:,:-1] = torch.sum(prev_rho*pi_stack,-1)

        rho = torch.stack(rho,dim=1)

        # for i in range(Na+1):
        #     # rho,computed_mask = self.sub_compute_rho_2(rho,pi_new,batch,computed_mask)
        #     output = self.sub_compute_rho(old_rho,pi_new)
        #     old_rho = F.pad(output,(0,1,0,1),"constant",0)
        #     old_rho[:,-1,-1] = 1

        rho_new = self.change_pad_bottom_right(rho,batch,Na,Nb,x_length,y_length)

        omega =  torch.stack((lam[:,:-1,1:],lam[:,:-1,:-1]),3) * pi * rho_new[:,:-1,:-1].unsqueeze(-1).repeat(1,1,1,2)
        return omega

    def _compute_omega_(self,W,pi,x_length,y_length):

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
            # output = self.sub_compute_lambda(lam, pi)
            # lam = F.pad(output,(1,0,1,0),"constant",0)
            # lam[:,0,0] = 1

        # Form a new pi with top-left padding
        pi_new = self.change_pad_top_left(pi,batch,Na,Nb,x_length,y_length)
        pi_new = F.pad(pi_new,(0,0,0,1,0,1),value = 0)
        pi_new[:,-1,-1] = torch.tensor(([0,0,1]),device = W.device).repeat(batch,1).float()

        computed_mask = torch.zeros((Na+1,Nb+1),device=W.device).bool() # Size: [Na+1,Nb+1]
        computed_mask[:,-1] = True
        computed_mask[-1,:] = True
        
        for i in range(Na+Nb-1):
            rho,computed_mask = self.sub_compute_rho_2(rho,pi_new,batch,computed_mask)
            # output = self.sub_compute_rho(rho,pi_new)
            # rho = F.pad(output,(0,1,0,1),"constant",0)
            # rho[:,-1,-1] = 1

        rho_new = self.change_pad_bottom_right(rho,batch,Na,Nb,x_length,y_length)

        omega =  torch.stack((lam[:,:-1,1:],lam[:,1:,:-1],lam[:,:-1,:-1]),3) * pi * rho_new[:,:-1,:-1].unsqueeze(-1).repeat(1,1,1,3)
        return omega

    def check_correctness(self,W,mu,x_length,y_length,mask):
        p2 = self.compute_pi_bf(W,mu,y_length,x_length)
        pi  = self.compute_pi_new(W,mu,mask,True)
        p3 = pi[-1]*mask[-1].unsqueeze(-1).repeat(1,1,2)
        p3 = p3[:x_length[-1],:y_length[-1]]
        isclose = torch.isclose(p2,p3)
        return (p2-p3).max()

    def plot_sample(self,samples):
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 1, squeeze=False)
        axes[0][0].imshow(samples[0,:,:].cpu().detach().numpy())
        axes[0][0].set_ylim(0, samples[0,:,:].shape[0])
        plt.savefig('./test.png')

    def init(self,W,mask,x_length,y_length):
        W = W*mask
        # mu1 = self.compute_location(W)
        # mu = self.compute_location_new(W)
        mu = self.compute_location_3(W)
        pi  = self.compute_pi_new(W,mu,mask,True)
        # samples,_ = self.sample(W,pi,mu,x_length,y_length,True)
        samples =  monotonic_align.maximum_path(pi,x_length,y_length)
        return samples

    def forward(self,W,mask):
        W = W*mask

        # mu = self.compute_location_new(W)
        mu = self.compute_location_3(W)
        pi = self.compute_pi_new(W,mu,mask,True)
        return mu,pi

    def sample_to_dict(self,W,mask,x_length,y_length,n):
        W = W*mask
        mu = self.compute_location_3(W)
        pi = self.compute_pi_new(W,mu,mask,True)
        
        for i in range(n):
            sample_i = self.sample_from_one(W,pi,x_length,y_length)
            if i ==0 :
                sample = sample_i
            else:
                sample += sample_i
        return sample

    # def compute_transition(self,W,mu,x_length,y_length,reduced,sapar):
    #     batch,Na,Nb = W.shape
    #     pi = torch.zeros((batch,Na+2,Nb+2,3)).to(device)
    #     pi[:,0,0] = torch.tensor(([0,0,1]))
    #     pi[:,-1,-1]  = torch.tensor(([0,0,1]))
    #     if sapar:
    #         # Compute transition matrix sample by sample
    #         for sample in range(batch):
    #             for i in reversed(range(1,x_length[sample]+1)):
    #                 for j in reversed(range(1,y_length[sample]+1)):
                        
    #                     pi[sample,i,j] = torch.stack((mu[sample,i-1,j]+self.alpha*W[sample,i-1,j-1],mu[sample,i,j-1]+self.alpha*W[sample,i-1,j-1],mu[sample,i-1,j-1]+self.alpha*W[sample,i-1,j-1]),0)-mu[sample,i,j]

    #     else:
    #         for i in reversed(range(1,Na+1)):
    #             for j in reversed(range(1,Nb+1)):            
    #                 pi[:,i,j] = torch.stack((mu[:,i-1,j]+self.alpha*W[:,i-1,j-1],mu[:,i,j-1]+self.alpha*W[:,i-1,j-1],mu[:,i-1,j-1]+self.alpha*W[:,i-1,j-1]),1)-mu[:,i,j].unsqueeze(1)
    #                 # pi[:,i,j] = torch.exp(log_pi)
    #     if reduced == True:
    #         return pi[:,1:Na+1,1:Nb+1]
    #     else:
    #         return pi
            
    # def compute_pi(self,W,mu,mask,reduced = True):
    #     ''' 
    #         Batch compute transition matrix pi
    #     '''

    #     batch,Na,Nb = W.shape
    #     pi = torch.zeros((batch,Na+2,Nb+2,3)).to(device)
    #     pi[:,0,0] = torch.tensor(([0,0,1]))
    #     # pi[torch.arange(batch)]
    #     # pi[:,-1,-1]  = torch.tensor(([0,0,1]))
    #     # Stack mat as [Na,Nb,3] matrix
    #     for i in range(1,Na+1):
    #         for j in range(1,Nb+1):
    #             pi[:,i,j] = torch.stack((mu[:,i-1,j]+self.alpha*W[:,i-1,j-1],mu[:,i,j-1]+self.alpha*W[:,i-1,j-1],mu[:,i-1,j-1]+self.alpha*W[:,i-1,j-1]),1)
    #     # get softmax value of pi
    #     pi = F.softmax(pi,-1)

    #     if reduced == True:
    #         return pi[:,1:Na+1,1:Nb+1]*mask
    #     else:
    #         # pad mask with 1 on left, right, top, bottom dim
    #         mask_pad = F.pad(mask,(0,0,1,1,1,1),value = 1)
    #         return pi*mask_pad


    # def compute_omega_old(self,W,pi,x_length,y_length):
        
    #     batch,Na,Nb = W.shape
        
        
    #     lam = torch.zeros((batch,Na+1,Nb+1)).cuda()
    #     rho = torch.zeros((batch,Na+1,Nb+1)).cuda()

    #     lam[:,0,0] = 1
    #     # rho[-1,:] = 0; rho[:,-1] = 0; rho[-1,-1]=1
    #     rho[torch.arange(batch),x_length,y_length] = 1
        

    #     # # Pad pi 
    #     pi = F.pad(pi,(0,0,1,1,1,1),value = 0)
        
    #     pi[torch.arange(batch),x_length+1,y_length+1] = torch.tensor(([0,0,1])).float().cuda()

    #     omega = torch.zeros_like(pi).cuda()
    #     # pdb.set_trace()
    #     # Topological iter for lam
    #     # for i in range(1,Na+1):
    #     #     for j in range(1,Nb+1):
                
    #     #         lam[:,i,j] = torch.matmul(torch.stack(([lam[:,i-1,j],lam[:,i,j-1],lam[:,i-1,j-1]]),1).unsqueeze(1),pi[:,i,j].unsqueeze(-1)).view(-1)
    #     for idx in range(batch):
    #         xlen = x_length[idx]
    #         ylen = y_length[idx]
    #         for i in range(1,xlen+1):
    #             for j in range(1,ylen+1):

    #                 lam[idx,i,j] = torch.matmul(torch.stack(([lam[idx,i-1,j],lam[idx,i,j-1],lam[idx,i-1,j-1]]),0),pi[idx,i,j])

        
    #     # Reversed iter for rho one by one
    #     for idx in range(batch):
    #         xlen = x_length[idx]
    #         ylen = y_length[idx]
    #         for i in reversed(range(xlen)):
    #             for j in reversed(range(ylen)):
                 
    #                 rho[idx,i,j] = torch.matmul(torch.stack((rho[idx,i+1,j],rho[idx,i,j+1],rho[idx,i+1,j+1]),0), torch.stack((pi[idx,i+2,j+1,0],pi[idx,i+1,j+2,1],pi[idx,i+2,j+2,2]),0))

    #     # for i in reversed(range(Na)):
    #     #     for j in reversed(range(Nb)):

    #     #         rho[:,i,j] = torch.matmul(torch.stack((rho[:,i+1,j],rho[:,i,j+1],rho[:,i+1,j+1]),1).unsqueeze(1),torch.stack((pi[:,i+2,j+1,0],pi[:,i+1,j+2,1],pi[:,i+2,j+2,2]),1).unsqueeze(-1)).view(-1)
    #     #         # pdb.set_trace()
    #     #         rho[torch.arange(batch),x_length,y_length] = 1
    #     #         # pi_prob = torch.sum(torch.stack((pi[:,i+2,j+1,0],pi[:,i+1,j+2,1],pi[:,i+2,j+2,2]),1),-1)
    #     #         # if pi_prob == 0:
    #     #         #     rho[:,i,j] = pi_prob
    #     #         # else:
    #     #         # rho[:,i,j] = torch.sum(torch.stack((pi[:,i+2,j+1,0],pi[:,i+1,j+2,1],pi[:,i+2,j+2,2]),1))

    #     #         # rho[i,j] = np.array(([rho[i+1,j],rho[i,j+1],rho[i+1,j+1]]))@np.array(([pi[i+1,j,0],pi[i,j+1,1],pi[i+1,j+1,2]]))
    #     for idx in range(batch):
    #         xlen = x_length[idx]
    #         ylen = y_length[idx]
    #         for i in reversed(range(1,xlen+1)):
    #             for j in reversed(range(1,ylen+1)):
    #                 omega[idx,i,j] = torch.stack((lam[idx,i-1,j],lam[idx,i,j-1],lam[idx,i-1,j-1]),0)*pi[idx,i,j]*rho[idx,i-1,j-1].unsqueeze(0).repeat(3)
    #     # for i in range(1,Na+1):
    #     #     for j in range(1,Nb+1):
               
    #     #         omega[:,i,j] = torch.stack((lam[:,i-1,j],lam[:,i,j-1],lam[:,i-1,j-1]),1)*pi[:,i,j]*rho[:,i-1,j-1].unsqueeze(1).repeat(1,3)
     
    #     return omega[:,1:Na+1,1:Nb+1],lam,rho,pi