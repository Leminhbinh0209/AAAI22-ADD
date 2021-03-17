from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

"""Reference from this work: https://github.com/"""
class NonLocal(nn.Module):
    def __init__(self, reduction="sum"):
        super(NonLocal, self).__init__()
        self.reduction  = getattr(torch, reduction)
        
        self.att = nn.ModuleList([
            SelfAttention(in_channel=256),
            SelfAttention(in_channel=512),
            SelfAttention(in_channel=1024),
            SelfAttention(in_channel=2048)
        ])
        
    def forward(self, g_s, g_t, weights):
        return [self.at_loss(f_s, f_t, i)*w for i, (f_s, f_t, w) in enumerate(zip(g_s, g_t, weights)) if w > 0]

    def at_loss(self, f_s, f_t, k_th):
        return torch.mean(self.reduction((self.att[k_th](f_s) - self.att[k_th](f_t)).pow(2), dim=(-3,-2,-1)))
    
class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)
        
def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module
def spectral_init(module, gain=1):
    torch.nn.init.kaiming_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)


def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, gain=1):
        super().__init__()

        self.query = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                   gain=gain)
        self.key = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                 gain=gain)
        self.value = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out