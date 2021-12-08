from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Multiview(nn.Module):
    """Multi-view attention distiller"""
    def __init__(self, p=1, n_theta=10, 
                 gamma=1.0, 
                 eta=0.5, 
                 normalize=True,
                 reduction='sum'):
        super(Multiview, self).__init__()
        self.p = p
        self.n_theta = n_theta
        self.gamma = gamma
        self.eta = eta
        self.normalize = normalize
        self.margin = 0.012
        self.reduction = getattr(torch, reduction)
        
    def forward(self, g_s, g_t, labels, weights):
        return [self.mtat_loss(f_s, f_t, labels)*w for f_s, f_t, w in zip(g_s, g_t, weights) if w > 0]

    def mtat_loss(self, f_s, f_t, labels):

        s_C, t_C = f_s.shape[1], f_t.shape[1]
        num_group = np.gcd(s_C, t_C) // 2 # GCD of Channels
        c, w, h = np.indices(f_s.shape[1:])
        indeces = np.column_stack((c.flatten(), w.flatten(), h.flatten() )).transpose()
        indeces = torch.tensor(indeces, requires_grad=False, device=f_s.device,  dtype=torch.float32).T

        theta = torch.randn((self.n_theta, 3), requires_grad=False, device=f_s.device, dtype=torch.float32)
        theta = F.normalize(theta, dim=-1, p=2).T
        
        pro_A = (indeces@theta).T
        sort_pro_A, argsort_pro_A = torch.sort(pro_A, dim=1)

        f_s = f_s.view(f_s.size(0),-1) # B x (C x W x H)
        f_t = f_t.view(f_t.size(0),-1) # B x (C x W x H)
        losses = [self.swd_constrastive(f_s, f_t, argsort_pro_A[i], sort_pro_A[i], num_group, labels) for i in range(self.n_theta)]
        return sum(losses)/len(losses)

    
    def swd_constrastive(self, f_s, f_t, idx_sort, value_sort, num_group, labels):
        """Sliced Wasseitern Distance with contrastive learning"""
        if self.normalize :
            f_s = F.normalize(f_s, dim=-1, p=2) ** 2 ## Sum up to 1 along dim 1
            f_t = F.normalize(f_t, dim=-1, p=2) ** 2 ## Sum up to 1 along dim 1

        ## Quantile the project vector
        quantile = torch.arange(0, 1+1.0/num_group, 1.0/num_group).to(f_s.device)

        value_sort_quantile =  (value_sort[-1] - value_sort[0]) * quantile + value_sort[0] # (max-min )* range + min
        index_quantile = torch.searchsorted(value_sort, value_sort_quantile)[1:-1] ## Exlude the zero and last
        index_quantile = index_quantile.unsqueeze(0).repeat(f_s.size(0),1)

        ## Cumulated sumation 
        c_f_s = torch.cumsum(f_s, dim=-1)
        c_f_t = torch.cumsum(f_t, dim=-1)
        
        ## Using quantile index 
        c_f_s = c_f_s.gather(1, index_quantile)
        c_f_t = c_f_t.gather(1, index_quantile)
        
        ## Add zero to batch dim
        c_f_s = torch.cat((torch.zeros(c_f_s.size(0)).unsqueeze(1).to(f_s.device), c_f_s), dim=1)
        c_f_t = torch.cat((torch.zeros(c_f_t.size(0)).unsqueeze(1).to(f_s.device), c_f_t), dim=1)
        
        ## Get sum on each interval
        i_f_s = c_f_s[:, 1:] - c_f_s[:, :-1]
        i_f_t = c_f_t[:, 1:] - c_f_t[:, :-1]
        
        loss =  self.gamma * self.reduction(torch.abs(i_f_s - i_f_t).pow(self.p), dim=1).mean() + self.eta * (self.contrastive_loss(i_f_s, i_f_t, labels) + self.contrastive_loss(i_f_t, i_f_s, labels))
        return loss
    
    def contrastive_loss(self, f_s, f_t, labels):
        """
        f_s: B x f
        f_t: B x f
        labels: B x 1
        """
        target = labels.cpu().numpy()
        target_pos = np.where(target>0)[0]
        target_neg = np.where(target<1)[0]
        
        # Random choice positive and negative samples
        target_true_pos = np.random.permutation(target_pos)
        target_false_pos = np.random.choice(target_neg, size=len(target_pos))
        target_true_neg = np.random.permutation(target_neg)
        target_false_neg = np.random.choice(target_pos, size=len(target_neg))
        
        ## Similarity with positive distribution
        pos_sim = torch.cat((self.reduction(torch.abs(f_s[target_pos]-f_t[target_true_pos]).pow(self.p), dim=1),
                    self.reduction(torch.abs(f_s[target_neg]-f_t[target_true_neg]).pow(self.p), dim=1)), dim=0)
        
        ## Similarity with negative distribution
        neg_sim = torch.cat((self.reduction(torch.abs(f_s[target_pos]-f_t[target_false_pos]).pow(self.p), dim=1),
                            self.reduction(torch.abs(f_s[target_neg]-f_t[target_false_neg]).pow(self.p), dim=1)), dim=0)
        
        loss = pos_sim.mean() +  torch.clamp(self.margin - neg_sim, min=0.0).mean() ## Contrastive loss
        return loss
    
    
