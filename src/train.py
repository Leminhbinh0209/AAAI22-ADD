import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys

import torch
import torchvision
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import SubsetRandomSampler

import operator
from functools import reduce
from functools import partial

from matplotlib import pyplot as plt
import dlib
import math
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
from efficientnet_pytorch import EfficientNet
import numpy as np
import cv2
import random
from random import random
import skimage
from skimage import measure, io
from io import BytesIO
from PIL import Image

from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.metrics import  roc_auc_score, f1_score, precision_score, recall_score
import easydict
import yaml
import json
import pretrainedmodels
from glob import glob
import pandas as pd
import albumentations
import albumentations.pytorch
import importlib
from distiller_zoo import Attention, HintLoss, Correlation, VIDLoss, RKDLoss,  PKT, DistillKL, Frequency, AttentionPx, Multiview, NonLocal
import torch.multiprocessing as mp
import torch.utils.data.distributed
from utils import *
from resnet_baseline import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set random seed
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


def JPEGcompression(image, qf):
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

class FaceDataset(Dataset):
    def __init__(self, img_paths, labels,
                     compression_qf=100,
                     quality='c40',
                     aug_transform=None,
                     tensor_transform=None,
                     mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.compression_qf = compression_qf
        self.quality = quality
        self.aug_transform = aug_transform
        self.tensor_transform = tensor_transform
       
    def __getitem__(self, index):
        img_path =  self.img_paths[index]
        label = self.labels[index]
        img_raw = Image.open(img_path)
        if self.compression_qf < 100: 
            img_com = JPEGcompression(img_raw, self.compression_qf) 
        else:
            img_com = Image.open(img_path.replace('raw', self.quality)) 
            
        if random() < 0.5: 
            img_raw = img_raw.transpose(Image.FLIP_LEFT_RIGHT)
            img_com = img_com.transpose(Image.FLIP_LEFT_RIGHT)
            
        if self.aug_transform is not None:
            img_raw = self.aug_transform(img_raw)
            img_com = self.aug_transform(img_com)
            
        if self.tensor_transform is not None:
            img_raw = self.tensor_transform(img_raw)
            img_com = self.tensor_transform(img_com)
            
        return img_raw, img_com, label
    
    def __len__(self):
        return len(self.labels)
    
def main(config):
    # ---------------------- LOAD DATA ----------------------
    current_time = str(time())[:10]
    real_type = 'real'
    fake_type = config.dataset

    with open(f"../data/face_data_ontraining_{fake_type}_raw_v1.json") as json_file:
        data_info = json.load(json_file)
            
    tensor_transform = transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    train_dataset = FaceDataset(img_paths=data_info['train']['dir'],
                                labels=data_info['train']['lb'],
                                quality = config.data_quality,
                                compression_qf=config.compression_qf,
                                aug_transform=None,
                                tensor_transform=tensor_transform,
                                mode='train')
    val_dataset = FaceDataset(img_paths=data_info['val']['dir'],
                               labels=data_info['val']['lb'],
                               quality = config.data_quality,
                               compression_qf=config.compression_qf,
                               tensor_transform=tensor_transform,
                               mode='val')

    test_dataset = FaceDataset(img_paths=data_info['test']['dir'],
                               labels=data_info['test']['lb'],
                               quality = config.data_quality,
                               compression_qf=config.compression_qf,
                               tensor_transform=tensor_transform,
                               mode='test')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    
    # ---------------------- DEFINE MODEL ----------------------
    # Create logger and checkpoints directory
    logger_dir = os.path.join(config.result_dir, config.dataset, config.data_quality)
    checkpoint_dir = os.path.join(config.checkpoint_dir, config.dataset, config.data_quality)
    os.makedirs(logger_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device, device_ids = setup_device(config.n_gpu)

    # Define teacher and student model
    if 'resnet50' in config.model_name:
        teacher_model = resnet50(pretrained=False,  num_classes=2)
        student_model = resnet50(pretrained=True,  num_classes=2)
    elif 'resnet34' in config.model_name:
        teacher_model = resnet34(pretrained=False,  num_classes=2)
        student_model = resnet34(pretrained=True,  num_classes=2)
    elif 'resnet18' in config.model_name:
        teacher_model = resnet18(pretrained=False,  num_classes=2)
        student_model = resnet18(pretrained=True,  num_classes=2)
        
    elif 'efficientnetb0' in config.model_name:
        teacher_model =  EfficientNet.from_pretrained('efficientnet-b0', dropout_rate=0, num_classes=2)
        student_model = EfficientNet.from_pretrained('efficientnet-b0', dropout_rate=0, num_classes=2)

        
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # Load pretrained-weights of teacher model
    pretrained_weights = torch.load(os.path.join(config.checkpoint_dir, config.dataset, 'raw', config.pretrained_path))['state_dict']
    teacher_model.load_state_dict(pretrained_weights)

    for param in teacher_model.parameters():
        param.requires_grad = False

    teacher_model.eval()
    teacher_model_weights = {}
    for name, param in teacher_model.named_parameters():
        teacher_model_weights[name] = param.detach()

    
    if len(device_ids) > 1:
            student_model = nn.DataParallel(student_model, device_ids=device_ids)
    print("Number of student model parameters: ", get_model_parameters(student_model))
    
    # training parameters
    trainable_list = nn.ModuleList([])
    trainable_list.append(student_model)

    # ---------------------- Loss and Optimizer ----------------------
    criterion_cls = nn.CrossEntropyLoss().cuda()   

    if config.is_proj:
        proj_module = Principal(2048, config.intrinsic_size, norm_type=config.norm_proj)
        proj_module = proj_module.to(device)
        if len(device_ids) > 1:
            proj_module = nn.DataParallel(proj_module, device_ids=device_ids)
        trainable_list.append(proj_module)
        
    
    if config.is_freq:
        config.model_name += '_freq'
        frequecy_loss = Frequency(kernel=config.kernel_fr, gamma=config.gamma_fr)
        
    # Divergence loss - Hinton
    criterion_div = DistillKL(config.kd_T)
    
    # Distillation loss 
    if config.distill == 'hint':
        config.model_name += '_hint'
        criterion_kd = HintLoss()
        
    elif config.distill == 'non-local':
        config.model_name += '_non-local'
        criterion_kd = NonLocal()        
        criterion_kd = criterion_kd.to(device)
        if len(device_ids) > 1:
                criterion_kd = nn.DataParallel(criterion_kd, device_ids=device_ids)
        trainable_list.append(criterion_kd)
        
    elif config.distill == 'attention':
        config.model_name += '_attention'
        criterion_kd = Attention()
        
    elif config.distill == 'swd':
        config.model_name += '_swd'
        criterion_kd = Multiview(p=config.p_swd, n_theta=config.n_theta_swd,
                                 gamma=config.gamma_swd,
                                 eta=config.eta_swd,
                                 normalize=True,
                                 reduction=config.reduction_swd)
    
    optimizer = torch.optim.Adam(
        params=trainable_list.parameters(),
        lr=config.lr, 
        betas=(0.9, 0.999), 
        eps=1e-08, weight_decay=0, 
        amsgrad=False)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.05,
        total_steps=None)
    
        
    # ---------------------- Backbone features ----------------------
    backbone_layers = []
    if "efficientnetb0" in config.model_name: 
        backbone_layers = config.backbone_layers_b0
    
    # ---------------------- Training phase  ----------------------
    print(f"***Start training with {fake_type} {config.data_quality}***")
    best_acc = 0.0
    watch_interval = 0
    patience = 0
    early_stop =  False
    # Frozen pre-trained params
    if 'resnet' in config.model_name:
        set_trainable(student_model, False, ['fc'], device_ids)
    if 'efficientnet' in config.model_name:
        set_trainable(student_model, False, ['_fc'], device_ids)
    config.is_freq = False
    print("Training params: ", count_parameters(student_model))
    
    for epoch in range(1, config.epochs + 1):
        
        # ===================Start: warming up=====================
        if epoch == config.warm_up:
            config.is_freq = True
            # Unfrozen pre-trained params
            watch_interval = 0
            set_trainable(student_model, True, [], device_ids)
            print("Training params: ", count_parameters(student_model))
            
        # ----------------------End: warming up----------------------
        
        student_model.train()
        train_fr = AverageMeter()
        train_kd = AverageMeter()
        train_ce = AverageMeter()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        train_auroc = AverageMeter()
        
        # ===================Start: training loop=====================
        for batch_idx, (batch_data_raw, batch_data_com, batch_target) in enumerate(train_loader):
            watch_interval += 1
            
            batch_data_raw = batch_data_raw.to(device=device, dtype=torch.float32)
            batch_data_com = batch_data_com.to(device=device, dtype=torch.float32)
            batch_target = batch_target.to(device=device, dtype=torch.long)
            
            middle_raw_tensors, batch_raw_feat, batch_raw_pred = teacher_model(batch_data_raw, True, backbone_layers)
            middle_com_tensors, batch_com_feat, batch_com_pred = student_model(batch_data_com, True, backbone_layers)
            
            # ===================Start: losses===================
            
            loss_fr = 0
            loss_kd = 0
            loss_div = 0
            loss_cls = criterion_cls(batch_com_pred, batch_target)
            
            # ===================Start: frequency loss=====================
            # Frequency loss
            if config.is_freq:
                loss_fr = sum(frequecy_loss(middle_com_tensors, middle_raw_tensors, config.layer_fr))
                
            # ----------------------End: frequency loss----------------------
            
            # Distillation loss at penulimate layer
            if config.distill == 'hint':
                loss_kd = criterion_kd(batch_com_feat, batch_raw_feat)
                
            if config.distill == 'atcn':
                loss_kd = sum(criterion_kd(middle_com_tensors, middle_raw_tensors, config.layer_atcn))
                
            if config.distill == 'non-local':
                loss_kd = sum(criterion_kd(middle_com_tensors, middle_raw_tensors, config.layer_atcn))
                
            if config.distill == 'attention':
                loss_kd = sum(criterion_kd(middle_com_tensors, middle_raw_tensors, config.layer_atcn))
                
            if config.distill == 'swd':
                loss_kd = sum(criterion_kd(middle_com_tensors, middle_raw_tensors, batch_target, config.layer_swd))
                
            loss =  config.lambda_fr*loss_fr +\
                    config.lambda_kd * loss_kd + \
                    config.lambda_ce * loss_cls +\
                    config.lambda_div * loss_div
           
            # ----------------------End: losses----------------------
            
            # ===================Start: backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # ----------------------End: backward----------------------
            
            acc1 = accuracy(batch_com_pred, batch_target)
            auc = roc_auc_score(batch_target.cpu().detach().numpy(), batch_com_pred.cpu().detach().numpy()[:,1])
            
            train_fr.update(loss_fr.item() if config.is_freq else 0, batch_data_com.size(0))
            train_kd.update(loss_kd.item(), batch_data_com.size(0))
            train_ce.update(loss_cls.item(), batch_data_com.size(0))
            train_loss.update(loss.item(), batch_data_com.size(0))
            train_acc.update(acc1[0], batch_data_com.size(0))
            train_auroc.update(auc, batch_data_com.size(0))
            
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write("Train Epoch: {e:02d} Batch: {batch:04d}/{size:04d} | L_F: {l_fr:.4f} | Loss_KD: {l_kd:.4f} |  Loss_CE: {l_ce:.4f} | Loss:{loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}"\
                             .format(e=epoch, batch=batch_idx+1, size=len(train_loader), \
                                     l_fr=train_fr.avg, l_kd=train_kd.avg, l_ce=train_ce.avg, \
                                     loss=train_loss.avg, acc=train_acc.avg, auc=train_auroc.avg))            
            
            # ===================Start: validation=====================
            if (watch_interval == len(train_loader)//10) & (epoch >= config.warm_up):
                watch_interval = 0
                
                student_model.eval()
                val_ce = AverageMeter()
                val_loss = AverageMeter()
                val_acc = AverageMeter()
                val_auroc = AverageMeter()

                with torch.no_grad():
                    for batch_idx, (batch_data_raw, batch_data_com, batch_target) in enumerate(val_loader):

                        batch_data_raw = batch_data_raw.to(device=device, dtype=torch.float32)
                        batch_data_com = batch_data_com.to(device=device, dtype=torch.float32)
                        batch_target = batch_target.to(device=device, dtype=torch.long)

                        _, batch_raw_feat, batch_raw_pred = teacher_model(batch_data_raw, True, backbone_layers)
                        _, batch_com_feat, batch_com_pred = student_model(batch_data_com, True, backbone_layers)
                        loss_cls = criterion_cls(batch_com_pred, batch_target)
                        
                        acc1= accuracy(batch_com_pred, batch_target)
                        auc = roc_auc_score(y_true=batch_target.cpu().detach().numpy(), y_score=batch_com_pred.cpu().detach().numpy()[:,1])

                        
                        val_ce.update(loss_cls.item(), batch_data_com.size(0))                        
                        val_acc.update(acc1[0], batch_data_com.size(0))
                        val_auroc.update(auc, batch_data_com.size(0))


                sys.stdout.write("\n\t\tValidation:  Loss_CE: {l_ce:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}\n"
                                .format( l_ce=val_ce.avg,  acc=val_acc.avg, auc=val_auroc.avg))

                best = False
                if val_acc.avg > best_acc:
                    print("Val Acc \033[0;32m improved \033[0;0m from {acc_past:.4f} to {acc_new:.4f} ".format(acc_past=best_acc, acc_new=val_acc.avg))
                    best_acc = val_acc.avg
                    best = True
                    patience = 0

                else:            
                    print("Val Acc does \033[1;31m NOT \033[0;0m improve from {acc:.4f}".format(acc=best_acc))
                    patience += 1

                save_model(os.path.join(checkpoint_dir, config.model_name), epoch, student_model, optimizer, lr_scheduler, device_ids, best)
            
            # ----------------------End: validation----------------------
                student_model.train()
                if (patience>=config.early_stop):
                    # early stopping
                    early_stop =  True
                    break
                
        if early_stop:
            print("Early Stopping...")
            break
                
    # ===================Start: Testing=====================

    pretrained_weights = torch.load(str('_'.join([os.path.join(checkpoint_dir, config.model_name ),'best.pth'])))['state_dict']
    student_model.load_state_dict(pretrained_weights)
    student_model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    test_auroc = AverageMeter()
    y_true = []
    y_pred = []
    query = []
    with torch.no_grad():
        for batch_idx, (_, batch_data_com, batch_target) in enumerate(test_loader):

                batch_data_com = batch_data_com.to(device=device, dtype=torch.float32)
                batch_target = batch_target.to(device=device, dtype=torch.long)

                _, batch_com_feat, batch_com_pred = student_model(batch_data_com, True)
                loss_cls = criterion_cls(batch_com_pred, batch_target)
                acc1= accuracy(batch_com_pred, batch_target)
                auc = roc_auc_score(y_true=batch_target.cpu().detach().numpy(), y_score=batch_com_pred.cpu().detach().numpy()[:,1])
                
                test_loss.update(loss_cls.item(), batch_data_com.size(0))
                test_acc.update(acc1[0], batch_data_com.size(0))
                test_auroc.update(auc, batch_data_com.size(0))
                
                y_true = np.concatenate((y_true, batch_target.cpu().detach().numpy()), axis=0) if len(y_true) else batch_target.cpu().detach().numpy()
                y_pred = np.concatenate((y_pred, batch_com_pred.cpu().detach().numpy()), axis=0) if len(y_pred) else batch_com_pred.cpu().detach().numpy()
                query = torch.cat((query, batch_com_feat.cpu().detach()), axis=0) if len(query) else batch_com_feat.cpu().detach()
                
    ece_score = ECE(y_true=y_true, y_pred=y_pred, n_bins=20)
    test_auroc = roc_auc_score(y_true=y_true, y_score=y_pred[:,1])
    test_precision = precision_score(y_true=y_true, y_pred=np.argmax(y_pred, axis=1))
    test_recall = recall_score(y_true=y_true, y_pred=np.argmax(y_pred, axis=1))
    test_f1 = f1_score(y_true=y_true, y_pred=np.argmax(y_pred, axis=1))
    
    sys.stdout.write("\033[0;32m Test loss: {loss:.4f} \n Test ACC: {acc:.4f} \n Test AUC: {auc:.4f} \n Test Precision: {pre:.4f} \n Test Recall: {rec:.4f} \n Test F1: {f1:.4f} \033[0;0m\n"
            .format(loss=test_loss.avg, acc=test_acc.avg, auc=test_auroc, pre=test_precision, rec=test_recall, f1=test_f1))

        
    # ---------------------- Start: Calculate Recal @ K ----------------------
    ref = [] 
    y_ref = [] 
    with torch.no_grad():
                
        for batch_idx, (_, batch_data_com, batch_target) in enumerate(train_loader):
                batch_data_com = batch_data_com.to(device=device, dtype=torch.float32)
                batch_target = batch_target.to(device=device, dtype=torch.long)

                _, batch_com_feat, batch_com_pred = student_model(batch_data_com, True)
                
                y_ref = np.concatenate((y_ref, batch_target.cpu().detach().numpy()), axis=0) if len(y_ref) else batch_target.cpu().detach().numpy()
                ref = torch.cat((ref, batch_com_feat.cpu().detach()), axis=0) if len(ref) else batch_com_feat.cpu().detach()
                
    recall = recall_at_k(ref=ref, query=query, y_ref=y_ref, y_query=y_true, fill_diag=False)
    sys.stdout.write("\033[0;32m Test Recall @ 1: {rc:.4f} \033[0;0m\n".format(rc=recall))
    # ----------------------End:  Calculate Recal @ K ----------------------
    
    # ---------------------- Save embedding space ----------------------
    with open(os.path.join(checkpoint_dir, config.model_name + '_emb.npy'), 'wb') as fileout:
        np.save(fileout, query.numpy())
    with open(os.path.join(checkpoint_dir, config.model_name + '_emb_train.npy'), 'wb') as fileout:
        np.save(fileout, ref.numpy())
    # ---------------------- End: Save embedding space ----------------------
    
    return test_acc.avg, best_acc
    # ----------------------End: testing--------------------

if __name__ == "__main__":
    with open("../configs/resnet_kd_fr_mv.yaml", 'r') as stream: # Change Dataset here
        config = yaml.safe_load(stream)
    config = easydict.EasyDict(config)
    model_name  = config.model_name
    main(config)
