import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm



from datatools.dataset import BeamDataset
from datatools.visualize import plot_uff
from torch.utils.data.dataloader import DataLoader
from models import Model, FCNN, LEBNN, EncoderU, DecoderU, EncoderF, DecoderF, ModelFtD, V


from torch_lr_finder import LRFinder






dataset = BeamDataset('data')

train_set, val_set = torch.utils.data.random_split(dataset, [math.floor(len(dataset)*0.8), math.ceil(len(dataset)*0.2)])

batch_size=128                
train_loader, val_loader = DataLoader(train_set, batch_size), DataLoader(val_set, batch_size)



u0=torch.cat((torch.linspace(0,1,10).unsqueeze(1),torch.zeros(10,1)),-1)


num_nodes=10
dof=2*(num_nodes-1)

hidden_size=64
latent_size=2
encdec=True #learnable autoencoder for f


lfcu = Model(FCNN(latent_size,hidden_size,latent_size),EncoderU(num_nodes, latent_size, hidden_size), DecoderU(num_nodes, latent_size, hidden_size), EncoderF(latent_size,hidden_size,learnable=encdec),  DecoderF(latent_size,hidden_size,learnable=encdec))
lebrb = Model(LEBNN(latent_size,hidden_size),EncoderU(num_nodes, latent_size, hidden_size), DecoderU(num_nodes, latent_size, hidden_size), EncoderF(latent_size,hidden_size,learnable=encdec),  DecoderF(latent_size,hidden_size,learnable=encdec))
lebfc = Model(LEBNN(latent_size,hidden_size,False),EncoderU(num_nodes, latent_size, hidden_size), DecoderU(num_nodes, latent_size, hidden_size), EncoderF(latent_size,hidden_size,learnable=encdec),  DecoderF(latent_size,hidden_size,learnable=encdec))
lfcf= ModelFtD(FCNN(latent_size,hidden_size,latent_size),EncoderU(num_nodes, latent_size, hidden_size), DecoderU(num_nodes, latent_size, hidden_size), EncoderF(latent_size,hidden_size,learnable=encdec),  DecoderF(latent_size,hidden_size,learnable=encdec))





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
    
    def __init__(self, metric=torch.nn.L1Loss(),weight=[1/0.1,1/0.3,1/0.15]):
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
    


'''
Tune_lr(lfcu)
Tune_lr(lebrb)
Tune_lr(lebfc)
'''













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
            
            err = (err/0.1 + criterion(u_reconstruct,u)/0.3 + criterion(f_reconstruct,f)/0.15)+torch.sum(model(u0.unsqueeze(0))**2)/0.03 + torch.sum(model(u0.unsqueeze(0))**2)
            
            err.backward()
            optimizer.step()

        if min(err_train)==err_train[-1]:
            torch.save(model.state_dict(),name)
        if 1.5*min(err_train)<err_train[-1]:
            model.load_state_dict(torch.load(name))
        print(f'|train error {err_train[-1]}|')
        err_val.append(0)
        with torch.no_grad():
            for data in val_loader:
                u, f = data
                
                fpred=model(u)
                
                err = criterion(fpred,f)     

                err_val[-1]+=float(err)/len(val_loader)
        print(f'|val error {err_val[-1]}|')
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
            err = (err/0.1 + criterion(u_reconstruct,u)/0.3 + criterion(f_reconstruct,f)/0.15)

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
        
        


def err_ftd(loader, model ,lr=0.05, nmax=1000, criterion = nn.L1Loss(reduction='sum'),CI=False):
    err=0
    
    listerr=[]
    
    num_correct=0
    u0=torch.cat((torch.linspace(0,1,10).unsqueeze(1),torch.zeros(10,1)),-1)

    for i in loader:
        u,f=i
        
        if CI:
            u_pred=model.backward(f,u,lr=lr,nmax=nmax)
        else:
            u_pred=model.backward(f,u/2+u0/2,lr=lr,nmax=nmax)
        err+=float(criterion(u_pred,u))
        listerr+=torch.mean(torch.abs(u-u_pred),[-1,-2]).tolist()
        
    plt.hist(listerr)

    return err/len(loader.dataset.indices)
        

def count_success(loader, model ,lr=0.05, nmax=1000, criterion = nn.L1Loss(reduction='sum'),CI=False):
    err=0
    
    listerr=[]
    
    num_correct=0
    u0=torch.cat((torch.linspace(0,1,10).unsqueeze(1),torch.zeros(10,1)),-1)

    for i in loader:
        u,f=i
        
        if CI:
            u_pred=model.backward(f,u,lr=lr,nmax=nmax)
        else:
            u_pred=model.backward(f,u/2+u0/2,lr=lr,nmax=nmax)
        num_correct+=float(torch.sum(torch.mean(torch.abs(u_pred-u),[-1,-2])<0.1))
        

    return num_correct/len(loader.dataset.indices)
        





epochs=50000

#trainFtD(lfcf, epochs=epochs, lr=1e-4, name='lfc-f', criterion = nn.L1Loss())
#train(lfcu, epochs=epochs, lr=1.5e-4, name='lfc-u', criterion = nn.L1Loss())
#train(lebrb, epochs=epochs, lr=1.4e-4,name='lebrb', criterion = nn.L1Loss())
train(lebfc, epochs=epochs, lr=2e-4,name='lebfc', criterion = nn.L1Loss())







'''
###Plot energy :
from datatools.visualize import plot_energy

plot_energy(lebrb.latent_model.V,s=2)

'''



