import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm



from datatools.dataset import LinearDataset, TwoSpringsDataset, LinearizedTwoSpringsDataset
from datatools.visualize import plot_energy
from torch.utils.data.dataloader import DataLoader
from models import Model, ModelFtD, FCNN, LEBNN


from torch_lr_finder import LRFinder




bdd='Spr11DtF'

dataset = TwoSpringsDataset(k1=1.,k2=1)

train_set, val_set = torch.utils.data.random_split(dataset, [math.floor(len(dataset)*0.8), math.ceil(len(dataset)*0.2)])

batch_size=128
train_loader, val_loader = DataLoader(train_set, batch_size), DataLoader(val_set, batch_size)



    

    

dof=2
load_size=2

hidden_size=64
latent_size=2


#id autoenc f (only possible if latent_size=2)
lfcu= Model(FCNN(latent_size,hidden_size,latent_size),FCNN(dof,hidden_size,latent_size),FCNN(latent_size,hidden_size,dof))
lebrb = Model(LEBNN(latent_size,hidden_size),FCNN(dof,hidden_size,latent_size),FCNN(latent_size,hidden_size,dof))
lebfc = Model(LEBNN(latent_size,hidden_size,False),FCNN(dof,hidden_size,latent_size),FCNN(latent_size,hidden_size,dof))
lfcf = ModelFtD(FCNN(latent_size,hidden_size,latent_size),FCNN(dof,hidden_size,latent_size),FCNN(latent_size,hidden_size,dof))

"""
#lin autoenc f
lfcu= Model(FCNN(latent_size,hidden_size,latent_size),FCNN(dof,hidden_size,latent_size),FCNN(latent_size,hidden_size,dof),nn.Linear(dof,latent_size,False),nn.Linear(latent_size,dof,False))
lebrb = Model(LEBNN(latent_size,hidden_size),FCNN(dof,hidden_size,latent_size),FCNN(latent_size,hidden_size,dof),nn.Linear(dof,latent_size,False),nn.Linear(latent_size,dof,False))
lebfc = Model(LEBNN(latent_size,hidden_size,False),FCNN(dof,hidden_size,latent_size),FCNN(latent_size,hidden_size,dof),nn.Linear(dof,latent_size,False),nn.Linear(latent_size,dof,False))
lfcf = ModelFtD(FCNN(latent_size,hidden_size,latent_size),FCNN(dof,hidden_size,latent_size),FCNN(latent_size,hidden_size,dof),nn.Linear(dof,latent_size,False),nn.Linear(latent_size,dof,False))
"""


class Model_stacked(nn.Module):
    
    def __init__(self, model):
        super(Model_stacked, self).__init__()
        
        self.model=model
    
    def forward(self, uf): 
        """
        input : uf the concatenation of u and f
        output : uuf=[u_pred, u_autoenc, f_autoenc]
        """
        u,f=uf.transpose(0,1)
        u_pred=self.model(f)
        u_autoenc=self.model.autoenc_u(u)
        f_autoenc=self.model.autoenc_f(f)
        uuf=torch.cat((u_pred.unsqueeze(1),u_autoenc.unsqueeze(1),f_autoenc.unsqueeze(1)),1)
        return uuf


class Training_metric(nn.Module):
    
    def __init__(self, metric=torch.nn.MSELoss(),weight=[1/0.2,1/0.03,10/0.6]):
        super(Training_metric, self).__init__()
        
        self.metric=metric
        self.weight=weight
    def forward(self, uuf_pred, uuf): 
        """
        input : uf the concatenation of u and f
        output : uuf=[u_pred, u_autoenc, f_autoenc]
        """
        u,uae,fae = uuf_pred.transpose(0,1)
        u_obj, _, f_obj = uuf.transpose(0,1)
        
        return self.weight[0]*self.metric(u,u_obj)+self.weight[1]*self.metric(uae,u_obj)+self.weight[2]*self.metric(fae,f_obj)


def Tune_lr(model):

    model_stacked=Model_stacked(model)
    optimizer = torch.optim.Adam(model_stacked.parameters(), lr=0.000001)
    
    trainset = torch.utils.data.TensorDataset(torch.cat((dataset.u.unsqueeze(1),dataset.f.unsqueeze(1)),1), torch.cat((dataset.u.unsqueeze(1),dataset.u.unsqueeze(1),dataset.f.unsqueeze(1)),1))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    lr_finder = LRFinder(model_stacked, optimizer, Training_metric(), device="cpu")
    lr_finder.range_test(trainloader, end_lr=0.4, num_iter=1000)
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset()
  



def train(model, criterion = nn.L1Loss(), epochs=100, lr=0.1, name=''):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    err_train=[]
    err_val=[]
    for epoch in tqdm(range(epochs)):
        err_train.append(0)
        for data in train_loader:
            optimizer.zero_grad()
            u, f = data
            
            fpred=model(u)
            
            err = criterion(fpred,f)
            err_train[-1]+=float(err)/len(train_loader)

            u_reconstruct = model.autoenc_u(u)
            f_reconstruct = model.autoenc_f(f)
            
            err = (err/0.2 + criterion(u_reconstruct,u)/0.03 + criterion(f_reconstruct,f)/0.6) 
            
            err.backward()
            optimizer.step()
            #model.enc_f.weight.data=torch.linalg.inv(model.dec_f.weight.data)

        if min(err_train)==err_train[-1]:
            torch.save(model.state_dict(),name)
        if 1.5*min(err_train)<err_train[-1]:
            model.load_state_dict(torch.load(name))
        err_val.append(0)
        with torch.no_grad():
            for data in val_loader:
                u, f = data
                
                fpred=model(u)
                
                err = criterion(fpred,f)     

                err_val[-1]+=float(err)/len(val_loader)
                
    plt.plot(list(range(epochs)), err_train, list(range(epochs)), err_val)
    return err_train, err_val






def trainFtD(model, criterion = nn.L1Loss(), epochs=100, lr=0.1, name=''):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    err_train=[]
    err_val=[]
    for epoch in tqdm(range(epochs)):
        err_train.append(0)
        for data in train_loader:
            optimizer.zero_grad()
            u, f = data
            
            upred=model.backward(f)
            
            err = criterion(upred,u)
            err_train[-1]+=float(err)/len(train_loader)

            u_reconstruct = model.autoenc_u(u)            
            f_reconstruct = model.autoenc_f(f)

            err = (err/0.2 + criterion(u_reconstruct,u)/0.03 + criterion(f_reconstruct,f)/0.6) 

            err.backward()
            optimizer.step()
            
        if min(err_train)==err_train[-1]:
            torch.save(model.state_dict(),name)
        if 1.5*min(err_train)<err_train[-1]:
            model.load_state_dict(torch.load(name))
        err_val.append(0)
        with torch.no_grad():
            for data in val_loader:
                u, f = data
                
                upred=model.backward(f)
                
                err = criterion(upred,u)     

                err_val[-1]+=float(err)/len(val_loader)
                
    plt.plot(list(range(epochs)), err_train, list(range(epochs)), err_val)
    return err_train, err_val




def err_dtf(loader, model, criterion = nn.L1Loss(reduction='sum')):
    err=0
    for i in loader:
        u,f=i
        f_pred=model.forward(u)
        err+=float(criterion(f_pred,f))
    return err/len(loader.dataset.indices)
        
        

def err_ftd(loader, model ,lr=0.0005, nmax=1000, criterion = nn.L1Loss(reduction='sum'),CI=False):
    err=0
    
    listerr=[]
    
    num_correct=0

    for i in loader:
        u,f=i
        
        if CI:
            u_pred=model.backward(f,u,lr=lr,nmax=nmax)
            err+=float(criterion(u_pred,u))
            num_correct+=torch.sum(torch.mean(torch.abs(u_pred-u),-1)<0.5)
        
            listerr.append((torch.mean(torch.abs(u_pred-u),-1)))

        else:
            u=u[torch.randperm(len(u))] #shuffle along batch dimension
            u_obj = dataset.backward(f,u.clone())
            u_pred=model.backward(f,u ,lr ,nmax)
            err+=float(criterion(u_pred,u_obj))
            
            listerr.append((torch.mean(torch.abs(u_pred-u_obj),-1)))
            
            num_correct+=torch.sum(torch.mean(torch.abs(u_pred-u_obj),-1)<0.5)
        
        
    plt.hist(listerr)

    return err/len(loader.dataset.indices)
        

def count_success(loader, model ,lr=0.0005, nmax=1000, criterion = nn.L1Loss(reduction='sum'),CI=False):
    err=0
    
    listerr=[]
    
    num_correct=0

    for i in loader:
        u,f=i
        
        if CI:
            u_pred=model.backward(f,u,lr=lr,nmax=nmax)
            err+=criterion(u_pred,u) 
            num_correct+=torch.sum(torch.mean(torch.abs(u_pred-u),-1)<0.5)
        
        else:
            u=u[torch.randperm(len(u))] #shuffle along batch dimension
            u_obj = dataset.backward(f,u.clone())
            u_pred=model.backward(f,u ,lr ,nmax)
            err+=criterion(u_pred,u_obj) 
            num_correct+=torch.sum(torch.mean(torch.abs(u_pred-u_obj),-1)<0.5)
        
        
    return float(num_correct/len(loader.dataset.indices))
        





    
epochs=10000
    
trainFtD(lfcf, epochs=epochs, lr=1e-4, name='lfc-f', criterion = nn.MSELoss())
train(lfcu, epochs=epochs, lr=1e-4, name='lfc-u', criterion = nn.MSELoss())
train(lebrb, epochs=epochs, lr=1e-4,name='lebrb', criterion = nn.MSELoss())
train(lebfc, epochs=epochs, lr=1e-4,name='lebfc', criterion = nn.MSELoss())







'''
###Plot energy :
from datatools.visualize import plot_energy

plot_energy(lebrb.latent_model.V,s=2)

'''



