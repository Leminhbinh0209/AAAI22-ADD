from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class Frequency(nn.Module):
    def __init__(self,  kernel='l2', gamma=1, reduction='sum'):
        super(Frequency, self).__init__()

        self.kernel = kernel
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, g_s, g_t, weights):
        return [self.frequecy_loss(f_s, f_t) * w for f_s, f_t, w in zip(g_s, g_t, weights) if w > 0]

    def frequecy_loss(self, s_out, t_out, mask=None):
        """
        Frequency different
        input:
            t_out:  B x C x W x H
            s_out:  B x C x W x H
        output: loss
        """
        assert self.kernel in ['l2', 'cosine', 'gaussian']
        reduction_ = getattr(torch, self.reduction)

        if mask is not None:
            t_out = torch.einsum("bcwh, b -> bcwh", t_out, mask)
            s_out = torch.einsum("bcwh, b -> bcwh", s_out, mask)
        t_out_fft = torch.rfft(t_out, 2, normalized=True, onesided=False) / np.sqrt(t_out.size(-2) * t_out.size(-1)) # B x C x W x H x 2
        s_out_fft = torch.rfft(s_out, 2, normalized=True, onesided=False) / np.sqrt(s_out.size(-2) * s_out.size(-1)) # B x C x W x H x 2

        if self.kernel == 'l2':
            entriloss = torch.norm(t_out_fft-s_out_fft, dim=-1)**2  # B x C x W x H 
        if self.kernel == 'cosine':
            t_out_fft = F.normalize(t_out_fft, dim=-1, p=2) # B x C x W x H x 2
            s_out_fft = F.normalize(s_out_fft, dim=-1, p=2) # B x C x W x H x 2
            entriloss = torch.einsum("bcwhk, bcwhk -> bcwh", t_out_fft, s_out_fft)

        weights = torch.empty_like(entriloss).copy_(entriloss).detach()
        weights = torch.mean(weights, dim=1) # B x  W x H
        weights = torch.exp(self.gamma*weights)

        loss = reduction_(torch.einsum("bcwh, bwh -> bcwh", entriloss, weights) , dim=(-3,-2,-1)) # B
        return torch.mean(loss)

