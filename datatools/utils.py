# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:12:41 2023

@author: LP263296
"""

import torch


def Ed(u,E=1,S=1000,I=0.1/12,dl=1/9):
    dx=u.diff(1,-2)
    Et=E*S/2/dl * (dx.norm(2,-1) - dl).pow(2).sum(-1)
    
    dx_=torch.cat((dx,torch.zeros_like(dx)[...,:,[0]]),-1)
    
    gamma=2*torch.cross(dx_[...,1:,:],dx_[...,:-1,:],-1).norm(2,-1)/(dx[...,1:,:].norm(2,-1)*dx[...,:-1,:].norm(2,-1)*(u[...,2:,:]-u[...,:-2,:]).norm(2,-1))
    Ef=E*I*dl/2*gamma.pow(2).sum(-1)
    return (Et+Ef)