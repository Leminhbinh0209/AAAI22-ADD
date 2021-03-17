from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
#         self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
#         loss = self.crit(f_s, f_t)
#         return loss
        tea_ft_norm =  F.normalize(f_t, dim=-1, p=2)
        stu_ft_norm =  F.normalize(f_s, dim=-1, p=2)
        l2_loss = 2 - 2 * (tea_ft_norm * stu_ft_norm).sum(dim=-1)
        return  l2_loss.mean()