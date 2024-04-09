import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import requests
from requests.adapters import HTTPAdapter
import os
import math
import sys   
import numpy as np

# ***

def define_SlackedNorm(y, y_logit_all, safe_norm):
    
    # https://discuss.pytorch.org/t/how-could-i-delete-one-element-from-each-row-of-a-tensor/84362
    # y_logit_target : positive
    # y_logit_nontarget : hard negative
    mask_target = torch.zeros_like(y_logit_all).scatter_(1, y.unsqueeze(1), 1.0)
    y_logit_target = y_logit_all[ mask_target.bool() ].view(y_logit_all.size(0), 1)

    mask_nontarget = torch.ones_like(y_logit_all).scatter_(1, y.unsqueeze(1), 0.0)
    y_logit_nontarget = y_logit_all[ mask_nontarget.bool() ].view(y_logit_all.size(0), y_logit_all.size(1)-1)
    y_logit_nontarget, _ = torch.sort( y_logit_nontarget, dim = -1, descending = True )
    y_logit_nontarget = y_logit_nontarget[:,0].unsqueeze(-1)

    # ***** *****
    
    # Proposed Proxy for Face Recognizability Estimation in SlackedFace
    # SlackedFace: Learning a Slacked Margin for Low-Resolution Face Recognition
    # https://papers.bmvc2023.org/0282.pdf
    slacked_norm = y_logit_target * ( y_logit_target - y_logit_nontarget )
    # The slope of Sigmoidal function can be fine-tuned.
    slacked_norm = 1.0 / ( 1.0 + torch.exp( -8.0 * slacked_norm ) )
    slacked_norm = safe_norm ** ( 1.0 - slacked_norm )
    
    return slacked_norm

class SlackedAdaFace(nn.Module):
    
    def __init__(self,
                 in_features = 512, # embedding_size
                 out_features = 70722, # classnum
                 s = 64.,
                 m = 0.4,
                 h = 0.333,
                 t_alpha = 0.99,
                 ini_type = 'default'):
        
        super(SlackedAdaFace, self).__init__()
        
        self.out_features = out_features
        
        ##### ##### 
        
        self.kernel = Parameter(torch.FloatTensor(in_features,out_features))

        # initial kernel
        if ini_type.lower() == 'uniform':
            self.kernel.data.uniform_(-1, 1).renorm_(2,0,1.0)
        elif ini_type.lower() == 'xavier':
            nn.init.xavier_uniform_(self.kernel)
            self.kernel.data.renorm_(2,0,1.0) 
        elif ini_type.lower() == 'default':
            self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)

        ##### #####
        
        self.m = m 
        self.eps = 1e-4
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

    def forward(self, embbedings, norms, label, first_batch = True):

        kernel_norm = l2_norm(self.kernel,axis=0)
        embbedings = l2_norm(embbedings,axis=1)
        
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        # Restrict soft_norms to be in the range of [0, 1]
        safe_norms = torch.clip(norms, min=0.001, max=100) / 100 # for stability
        safe_norms = norms.clone().detach()

        ##### #####
        
        # Define slacked margin, and update batch_mean, batch_std
        with torch.no_grad():
            
            slacked_norms = define_SlackedNorm(label, cosine, safe_norms)

            # Reset batch_mean and batch_std for every new epoch
            if first_batch is True:
                self.batch_mean = slacked_norms.mean().detach()
                self.batch_std = slacked_norms.std().detach()
            else:
                self.batch_mean = self.t_alpha * slacked_norms.mean().detach() + ( 1.0 - self.t_alpha ) * self.batch_mean
                self.batch_std =  self.t_alpha * slacked_norms.std().detach() + ( 1.0 - self.t_alpha ) * self.batch_std
            
            # **** 

        margin_scaler = ( slacked_norms - self.batch_mean ) / ( self.batch_std + self.eps ) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        ##### #####
        
        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine_ = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine_.size()[1], device=cosine_.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine_ = cosine_ - m_cos

        # scale
        scaled_cosine_m = cosine_ * self.s
        
        return scaled_cosine_m, cosine

# ***

class AdaFace(nn.Module):
    
    def __init__(self,
                 in_features = 512, # embedding_size
                 out_features = 70722, # classnum
                 s = 64.,
                 m = 0.4,
                 h = 0.333,
                 t_alpha = 0.99,
                 ini_type = 'default'):
        
        super(AdaFace, self).__init__()
        
        self.out_features = out_features
        self.kernel = Parameter(torch.FloatTensor(in_features,out_features))
        # self.kernel = Parameter(torch.FloatTensor(out_features,in_features))

        # initial kernel
        if ini_type.lower() == 'uniform':
            self.kernel.data.uniform_(-1, 1).renorm_(2,0,1.0)
        elif ini_type.lower() == 'xavier':
            nn.init.xavier_uniform_(self.kernel)
            self.kernel.data.renorm_(2,0,1.0) 
        elif ini_type.lower() == 'default':
            self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        # self.reset_parameters() 
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        '''
        print('\AdaFace with the following properties:')
        print('self.s', self.s)
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.t_alpha', self.t_alpha)
        print()
        '''
        
    '''
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel)
    '''
    
    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        embbedings = l2_norm(embbedings,axis=1)
        
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine_ = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine_.size()[1], device=cosine_.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine_ = cosine_ - m_cos

        # scale
        scaled_cosine_m = cosine_ * self.s
        
        return scaled_cosine_m, cosine

