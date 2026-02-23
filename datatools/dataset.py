# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:59:16 2023

@author: louen
"""

import numpy as np
import torch

from torch.utils.data import Dataset


class BeamDataset(Dataset):
    """
    1D static beam dataset
    """

    def __init__(self, data_dir, dataset_size=3778, normalize=True):
        """
        Arguments:
            data_dir (string): directory containing all the u_{}.csv and f_{}.csv
            dataset_sizr (int): number of (u,f) samples
            normalize (bool): if True  u and f are divided by a scalar so that they become of unit magnitude 
        """
        self.dataset_size=dataset_size
        
        u=torch.tensor([])
        f=torch.tensor([])
        
        for i in range(dataset_size):
            ui = torch.from_numpy(np.genfromtxt(f"{data_dir}/u{i}.csv", delimiter=";")).to(torch.float32)
            fi = torch.from_numpy(np.genfromtxt(f"{data_dir}/f{i}.csv", delimiter=";")).to(torch.float32)
            u=torch.cat((u,ui.unsqueeze(0)),0)
            f=torch.cat((f,fi.unsqueeze(0)),0)

        if normalize:
            f=f/5000
            u=u/1
            
        self.u = u
        self.f = f

        self.num_node=u.size(1)
        self.num_dims=u.size(2)
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.u[idx], self.f[idx]


class LinearDataset(Dataset):
    """
    ndof linear dataset : T = 1/2 u^t K u,  K.u=f
    """

    def __init__(self,dof=2,K=None, dataset_size=1000):
        """
        Arguments:
            dof (int) : number of degree of freedom
            K (tensor of shape (dof,dof) or None) : symetric stifness matrix, randomly initialised if None
            dataset_sizr (int): number of (u,f) samples
        """
        self.dof=dof
        self.dataset_size=dataset_size
        
        if K is None:
            M=torch.rand(dof,dof)
            K=1/2*(M+M.t())
        self.K=K
        
        u=torch.rand(dataset_size,dof)*2-1
        f=(K@u.t()).t()
        
            
        self.u = u
        self.f = f

        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.u[idx], self.f[idx]
    
    


    
class TwoSpringsDataset(Dataset):
    """
    2dof dataset, two springs  
    
    $----+
         |
         |
         $

    """

    def __init__(self,k1=None,k2=None, dataset_size=1000, convex=False, forward=True):
        """
        Arguments:
            k1 and k2 : floats (stifness of each springs), randomly initialized if None
            dataset_sizr (int): number of (u,f) samples
        """
        self.dataset_size=dataset_size
        
        if k1 is None:
            k1=torch.rand(1)
        if k2 is None:
            k2=torch.rand(1)
        self.K=(k1,k2)
        

        if forward:
            if convex:
                u=(torch.rand(dataset_size,2)*2-1)+0.8 #the energy is convex on this subset of displacement space
                f=self.forward(u) 
            else:
                u=(torch.rand(dataset_size,2)*2-1)*2
                f=self.forward(u) 
        else:
            u0=(torch.rand(dataset_size,2)*2-1)*2
            f=(torch.rand(dataset_size,2)*2-1)*2
            u=self.backward(f,u0)

        self.u = u.detach()
        self.f = f.detach()
    def Ed(self,u):
        x,y=u.t()
        
        l1=torch.sqrt(y**2+(1+x)**2)
        l2=torch.sqrt(x**2+(1+y)**2)
        
        k1,k2=self.K
        return 1/2*k1*(l1-1)**2+1/2*k2*(l2-1)**2
    def forward(self,u):
        x,y=u.t()
        
        l1=torch.sqrt(y**2+(1+x)**2)
        l2=torch.sqrt(x**2+(1+y)**2)

        k1,k2=self.K

        fx=k1*(x+1)*(l1-1)/l1+k2*x*(l2-1)/l2
        fy=k2*(y+1)*(l2-1)/l2+k1*y*(l1-1)/l1

        f=torch.cat((fx.unsqueeze(-1),fy.unsqueeze(-1)),-1)
        return f

    def backward(self,f,u0=None):
        if u0 is None:
            u0=torch.zeros(1,2)
        u=torch.nn.Parameter(u0)
        optim=torch.optim.Adam([u],lr=0.05)
        for i in range(1000):
            optim.zero_grad()
            err=torch.sum(self.Ed(u)-torch.einsum('bi,bi->b',u,f))
            err.backward()
            optim.step()
        return u.detach()
            
            
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.u[idx], self.f[idx]
    




class LinearizedTwoSpringsDataset(Dataset):
    """
    2dof dataset, two springs  
    
    $----+
         |
         |
         $

    """

    def __init__(self,k1=None,k2=None, dataset_size=1000, forward=True):
        """
        Arguments:
            k1 and k2 : floats (stifness of each springs), randomly initialized if None
            dataset_sizr (int): number of (u,f) samples
        """
        self.dataset_size=dataset_size
        
        if k1 is None:
            k1=torch.rand(1)
        if k2 is None:
            k2=torch.rand(1)
        self.K=(k1,k2)
        
        if forward:
            u=(torch.rand(dataset_size,2)*2-1)*2
            f=self.forward(u) 
        else:
            u0=(torch.rand(dataset_size,2)*2-1)*2
            f=(torch.rand(dataset_size,2)*2-1)*2
            u=self.backward(f,u0)

            
        self.u = u.detach()
        self.f = f.detach() 

    def Ed(self,u):
        x,y=u.t()
        k1,k2=self.K
        return 1/2*k1*x**2+1/2*k2*y**2
    def forward(self,u):
        x,y=u.t()

        k1,k2=self.K

        fx=k1*x
        fy=k2*y

        f=torch.cat((fx.unsqueeze(-1),fy.unsqueeze(-1)),-1)
        return f

    def backward(self,f,u0=None):
        fx,fy=f.t()
        k1,k2=self.K
        x=fx/k1
        y=fy/k2
        u=torch.cat((x.unsqueeze(-1),y.unsqueeze(-1)),-1)
        return u
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.u[idx], self.f[idx]
    













