import os
import json
import sys
import numpy as np
import pandas as pd
from time import time
from scipy.special import softmax

from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import importlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

def ECE(y_true, y_pred, n_bins=20):
    """
    Expected Calibration Error and Max-Expected Calibration Error 
    lower is better
    input:
        y_true: labels (1: fake , 0: real) B
        y_pred: propbability of y_true   B x 2
        n_bins: the number of binning groups
    output:
        ECE
    """
    n_samples = y_pred.shape[0]

    y_pred = softmax(y_pred, axis=1)
    y_top1 = np.argmax(y_pred, axis=1)
    y_conf = np.max(y_pred, axis=1)
    indeces_sort = np.argsort(y_conf)

    y_top1 = y_top1[indeces_sort]
    y_conf = y_conf[indeces_sort]
    y_true = y_true[indeces_sort]

    ece = 0.0
    max_ece = 0.0
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        idx = np.where((y_conf > a) & (y_conf <=b))[0]
        if not len(idx): continue
        acc = np.sum(y_true[idx] == y_top1[idx]) / len(idx)
        conf = np.mean(y_conf[idx])
        diff = np.abs(acc - conf) * len(idx) / n_samples
        max_ece = np.abs(acc - conf) if max_ece < np.abs(acc - conf) else max_ece
        ece += diff
    return (ece, max_ece)

def pairwise_dist(ref, query, fill_diag=True):
    """
    Pairwise distance 
    input:
        ref: reference matrix : (N, D)
        query: Query matrix (M, D)
        fill_diag: Fill the diagonal with inf value
    """
    n = ref.size(0)
    m = query.size(0)
    xx, yy, zz = torch.mm(ref,ref.t()), torch.mm(query,query.t()), torch.mm(ref, query.t())
    rx = (xx.diag().unsqueeze(0).expand(m, n))
    ry = (yy.diag().unsqueeze(0).expand(n, m))
    dist = (rx.t() + ry - 2*zz).t()
    if fill_diag : 
        dist.fill_diagonal_(np.inf)
    return dist
def recall_at_k(ref, query, y_ref, y_query, fill_diag=True):
    """
    Recall @ K
    input: 
        ref, query: Embeding space: M x D, N x D, tensor
        y_ref, y_query: label of emb: M, N, numpy array
        fill_diag: Fill the diagonal with inf value
    """
    dist = pairwise_dist(ref, query, fill_diag) # N x M
    dist = dist.numpy()
    neighbor_idx = np.argmin(dist, axis=1) # N values in [0, M-1]
    neighbor_label = y_ref[neighbor_idx]
    return np.sum(neighbor_label==y_query) / len(y_query)


def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            if isinstance(num, str): self.file.write(num)
            else: self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k / batch_size * 100.0)
#     return res


def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    # add datetime postfix
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    exp_name = config.exp_name + '_{}_bs{}_lr{}_wd{}'.format(config.dataset, config.batch_size, config.lr, config.wd)
    exp_name += ('_' + timestamp)

    # create some important directories to be used for that experiments
    config.summary_dir = os.path.join('experiments', 'tb', exp_name)
    config.checkpoint_dir = os.path.join('experiments', 'save', exp_name, 'checkpoints/')
    config.result_dir = os.path.join('experiments', 'save', exp_name, 'results/')
    for dir in [config.summary_dir, config.checkpoint_dir, config.result_dir]:
        ensure_dir(dir)

    # save config
    write_json(vars(config), os.path.join('experiments', 'save', exp_name, 'config.json'))

    return config

def get_model_parameters(model):
    "Counting number of params"
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters

def count_parameters(model):
    "Couting number of trainable params"
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(output, target, topk=[1]):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def frozen_param(model):
    for param in model.parameters():
        param.requires_grad = False

def set_trainable(model, boolean:bool=True, except_layers: list=[], device_ids: list=[]):
    if boolean:
        for i, param in model.named_parameters():
                param.requires_grad = True
        if len(except_layers) > 0: # Except some layers
            for layer in except_layers:
                assert layer is not None
                if len(device_ids) <=1:
                    for param in getattr(model, layer).parameters():
                        param.requires_grad = False
                else:
                    for param in getattr(model.module, layer).parameters():
                        param.requires_grad = False
        return model
    else:
#         assert len(except_layers) > 0, "Require free layer"
        for i, param in model.named_parameters():
                param.requires_grad = False
        for layer in except_layers: # Except some layers
            assert layer is not None
            if len(device_ids) <=1:
                for param in getattr(model, layer).parameters():
                    param.requires_grad = True
            else:
                for param in getattr(model.module, layer).parameters():
                    param.requires_grad = True
        return model

def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str('_'.join([save_dir, 'current.pth']))
    torch.save(state, filename)

    if best:
        filename = str('_'.join([save_dir, 'best.pth']))
        torch.save(state, filename)
        
def kd_schedule(epoch, end_epoch, trigger=6):
    if epoch < trigger:
        return 1.0
    elif epoch > end_epoch:
        return 0.0
    k = epoch - trigger
    K = end_epoch - trigger
    return (np.cos(np.pi*k / K) + 1.0) / 2.0

class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

def sqd_kd(t_out, s_out, label, norm_type='mse'):
    """
    Sliced Wasseitern distance for logit feature by positive and negative feature
    input:
        t_tensor: B x C 
        s_tensor: B x C
    output:
        swd distance
    """
    t_out = t_out.detach().clone()
    t_out = t_out.unsqueeze(-1).unsqueeze(-1) # B x C x 1 x 1
    s_out = s_out.unsqueeze(-1).unsqueeze(-1) # B x C x 1 x 1
    pos_idx = [np.where(label.cpu().numpy()==1)[0]]
    neg_idx = [np.where(label.cpu().numpy()==0)[0]]
    t_out_pos = t_out[pos_idx]
    s_out_pos  = s_out[pos_idx]
    
    t_out_neg = t_out[neg_idx]
    s_out_neg  = s_out[neg_idx]
    
    pos_all = torch.cat((t_out_pos, s_out_pos), dim=0)
    neg_all = torch.cat((t_out_neg, s_out_neg), dim=0)
    
    min_samples = min(pos_all.size(0), neg_all.size(0))
    pos_all = pos_all[:min_samples]
    neg_all = neg_all[:min_samples]
    
    dist =  SWD(t_out_pos, s_out_pos, norm_type=norm_type) + \
            SWD(t_out_neg, s_out_neg, norm_type=norm_type) + \
            10*1/(SWD(pos_all, neg_all, norm_type=norm_type)+torch.finfo(torch.float32).eps)
    
    return dist

def wasserstein1d(x, y, norm_type='mse'):   
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    z = (x1-y1).view(x.size(0),-1)

    if norm_type in ['MSE', 'mse', 'Frob', 'F']:
        return torch.mean(torch.square(torch.norm(z, p=2, dim=1)))
    elif norm_type in ['L1', 'l1']:
        return torch.mean(torch.norm(z, p=1, dim=1))
    elif norm_type in ['LAD', 'lad', 'L21', 'l21', 'L2', 'l2']:
        return torch.mean(torch.norm(z, p=2, dim=1))
    else:
        raise Exception("Norm type error!")   

def SWD(t_out, s_out, norm_type='mse', mask=None):
    """
    Sliced Wasseitern distance for tensors by taking the channel-wise attention
    input:
        t_tensor: B x C x W x H
        s_tensor: B x C x W x H
    output:
        swd distance
    """
    
    t_out = t_out.detach().clone()
    if mask is not None:
        t_out = torch.einsum("bcwh, b -> bcwh", t_out, mask)
        s_out = torch.einsum("bcwh, b -> bcwh", s_out, mask)
        
    # Average pooling
    t_out_avg = torch.mean(t_out, dim=(-2,-1)) # B x C 
    s_out_avg = torch.mean(s_out, dim=(-2,-1)) # B x C 
    
    # random theta
    theta = torch.randn((s_out_avg.size(1), 10000), requires_grad=False, device=t_out.device)
    theta = theta/torch.norm(theta, dim=0)[None, :]
    t_1d = t_out_avg@theta
    s_1d = s_out_avg@theta
    return wasserstein1d(t_1d, s_1d, norm_type)

class Principal(nn.Module):
    """"""
    def __init__(self,  
                 in_features, 
                 intrinsic_size , 
                 normalize=False,
                 norm_type='l1',
                 lambda1 = 2.0,
                 lambda2 = 0.1):
        super(Principal, self).__init__()
        
        self.normalize = normalize
        self.norm_type = norm_type
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        self.intrinsic_size = intrinsic_size
        self.A = nn.Parameter(torch.randn((in_features, self.intrinsic_size), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_uniform_(self.A)
        
    def forward(self, g_s, g_t, compute_loss):
        # g_s: B x in_features
        if compute_loss:
            g_s = g_s.detach().clone()
            g_t = g_t.detach().clone()
            
        z_s = torch.matmul(g_s, self.A)  # : B x intrinsic_size
        z_t = torch.matmul(g_t, self.A)  # : B x intrinsic_size
        
        if self.normalize:
            z_s = F.normalize(z_s, dim=-1, p=2)
            z_t = F.normalize(z_t, dim=-1, p=2)
            
        loss =  self.lambda1 *self.pca_error(g_t, z_t) + \
                self.lambda1 * self.pca_error(g_s, z_s) +\
                self.lambda2 * self.proj_error() 
        
        return z_s, z_t, loss
    
    def pca_error(self, y, z):
#         y = y.detach().clone()
#         z = z.detach().clone()
        # : B x intrinsic_size
        norm_type = self.norm_type
        z = torch.matmul(z, self.A.T) #  B x in_features
        if norm_type in ['MSE', 'mse', 'Frob', 'F']:
            return torch.mean(torch.square(torch.norm(y-z, p=2, dim=1)))
        elif norm_type in ['L1', 'l1']:
            return torch.mean(torch.norm(y-z, p=1, dim=1))
        elif norm_type in ['LAD', 'lad', 'L21', 'l21', 'L2', 'l2']:
            return torch.mean(torch.norm(y-z, p=2, dim=1))
        else:
            raise Exception("Norm type error!")   

    def proj_error(self):
        return torch.mean(torch.square(torch.matmul(self.A.T, self.A) \
                                        - torch.eye(self.intrinsic_size, device=self.A.device)) )


# Get from https://github.com/kimiandj/gsw
class GSW():
    def __init__(self,ftype='linear',nofprojections=10,degree=2,radius=1000,use_cuda=True):
        self.ftype=ftype
        self.nofprojections=nofprojections
        self.degree=degree
        self.radius=radius
        if torch.cuda.is_available() and use_cuda:
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.theta=None # This is for max-GSW

    def gsw(self,X,Y,theta=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)

        Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        Yslices_sorted=torch.sort(Yslices,dim=0)[0]
        return torch.sqrt(torch.sum((Xslices_sorted-Yslices_sorted)**2))
    
    def max_gsw(self,X,Y,iterations=50,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        assert dn==dm and M==N
#         if self.theta is None:
        if self.ftype=='linear':
            theta=torch.randn((1,dn),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        elif self.ftype=='poly':
            dpoly=self.homopoly(dn,self.degree)
            theta=torch.randn((1,dpoly),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        elif self.ftype=='circular':
            theta=torch.randn((1,dn),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        self.theta=theta

        optimizer=optim.Adam([self.theta],lr=lr)
        total_loss=np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.gsw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            total_loss[i]=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data/=torch.sqrt(torch.sum(self.theta.data**2))

        return self.gsw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))

    def gsl2(self,X,Y,theta=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        '''
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)

        Yslices_sorted=torch.sort(Yslices,dim=0)

        return torch.sqrt(torch.sum((Xslices-Yslices)**2))

    def get_slice(self,X,theta):
        ''' Slices samples from distribution X~P_X
            Inputs:
                X:  Nxd matrix of N data samples
                theta: parameters of g (e.g., a d vector in the linear case)
        '''
        if self.ftype=='linear':
            return self.linear(X,theta)
        elif self.ftype=='poly':
            return self.poly(X,theta)
        elif self.ftype=='circular':
            return self.circular(X,theta)
        else:
            raise Exception('Defining function not implemented')

    def random_slice(self,dim):
        if self.ftype=='linear':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='poly':
            dpoly=self.homopoly(dim,self.degree)
            theta=torch.randn((self.nofprojections,dpoly))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='circular':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        return theta.to(self.device)

    def linear(self,X,theta):
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.t())

    def poly(self,X,theta):
        ''' The polynomial defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
            degree: degree of the polynomial
        '''
        N,d=X.shape
        assert theta.shape[1]==self.homopoly(d,self.degree)
        powers=list(self.get_powers(d,self.degree))
        HX=torch.ones((N,len(powers))).to(self.device)
        for k,power in enumerate(powers):
            for i,p in enumerate(power):
                HX[:,k]*=X[:,i]**p
        if len(theta.shape)==1:
            return torch.matmul(HX,theta)
        else:
            return torch.matmul(HX,theta.t())

    def circular(self,X,theta):
        ''' The circular defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
        '''
        N,d=X.shape
        if len(theta.shape)==1:
            return torch.sqrt(torch.sum((X-self.radius*theta)**2,dim=1))
        else:
            return torch.stack([torch.sqrt(torch.sum((X-self.radius*th)**2,dim=1)) for th in theta],1)

    def get_powers(self,dim,degree):
        '''
        This function calculates the powers of a homogeneous polynomial
        e.g.
        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]
        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
        '''
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in self.get_powers(dim - 1,degree - value):
                    yield (value,) + permutation

    def homopoly(self,dim,degree):
        '''
        calculates the number of elements in a homogeneous polynomial
        '''
        return len(list(self.get_powers(dim,degree)))