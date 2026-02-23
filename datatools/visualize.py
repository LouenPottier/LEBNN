# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 19:15:36 2022

@author: louen
"""


import matplotlib.pyplot as plt
import torch


import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def plot_uf(u,f):
    if u.dim()==2:
        u=u.unsqueeze(0)
        f=f.unsqueeze(0)
    
    fig, ax = plt.subplots()
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

    for i in range(len(u)):
        ui=u[i].detach()
        fi=f[i].detach()
        line, = ax.plot(ui[:,0], ui[:,1])
        vect=ax.quiver(ui[-1,0], ui[-1,1], fi[0], fi[1])    
    plt.show()
    return


def plot_uff(u,f_pred,f_obj):
    if u.dim()==2:
        u=u.unsqueeze(0)
        f_pred=f_pred.unsqueeze(0)
        f_obj=f_obj.unsqueeze(0)
    
    u=u.detach()
    f_pred=f_pred.detach()
    f_obj=f_obj.detach()

    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.set_xlim(-1.1,1.1)
    ax1.set_ylim(-1.1,1.1)
    ax1.set_title('Input displacement field and output loading vector')    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    
    mean_error=torch.mean((f_pred-f_obj)**2)

    for i in range(min(len(u),500)):
        ui=u[i]
        fi_pred=f_pred[i]
        fi_obj=f_obj[i]
        line, = ax1.plot(ui[:,0], ui[:,1])
        
        if torch.mean((fi_pred-fi_obj)**2)>mean_error:
            color_pred='red'
        else:
            color_pred='green'

        vect_obj=ax1.quiver(ui[-1,0], ui[-1,1], fi_obj[0], fi_obj[1], color='black')    
        vect_pred=ax1.quiver(ui[-1,0], ui[-1,1], fi_pred[0], fi_pred[1], color=color_pred)    
        
        
    ax2.set_title('Error histogram')    
    ax2.set_ylabel('number of sample')
    ax2.set_xlabel('MSE(f_pred,f_obj)')
    ax2.hist(torch.mean((f_pred-f_obj)**2,-1),10)
    
    ax3.set_title('Error in function of the x displacement of the last node')    
    ax3.set_ylabel('u[-1]_x')
    ax3.set_xlabel('MSE(f_pred,f_obj)')
    ax3.scatter(u[:,-1,0]-1,torch.mean((f_pred-f_obj)**2,-1))
    
    plt.show()
    return

def plot_energy(E,H=None,s=1):
    
    n=200
    
    
    if s is not None:
        minx=-2.5*s
        maxx=2.5*s
        miny=-2.5*s
        maxy=2.5*s
    elif H is not None:
        minx = float(torch.min(H[:,0]))
        maxx = float(torch.max(H[:,0]))
        miny = float(torch.min(H[:,1]))
        maxy = float(torch.max(H[:,1]))
    else:
        minx=-2.5
        maxx=2.5
        miny=-2.5
        maxy=2.5
    
    x=torch.linspace(minx,maxx,n)
    y=torch.linspace(miny,maxy,n)
    
    X, Y = torch.meshgrid(x, y)
    
    h=torch.cat((X.flatten().unsqueeze(1),Y.flatten().unsqueeze(1)),1)
    
    
    Z=E(h).view(n,n).detach()
    
    Z=((Z-torch.min(Z))**(1/2)).detach()
    
    mini,maxi,saddle=find_extrema(Z)

    plt.contour(X,Y,Z,levels=50)
    
    

    plt.scatter(H[:,0].detach(),H[:,1].detach(),color='blue')
    L=mini+maxi+saddle
    
    Lx=[i[0] for i in L]
    Ly=[i[1] for i in L]
    print(L)
    plt.scatter(x[Lx],y[Ly],color='red')
    
    return Z#, mini, maxi, saddle


def plot3d_energy(E):
    
    n=200


    # Make data.
    x = torch.linspace(-2.5, 2.5, n)
    y = torch.linspace(-2.5, 2.5, n)
    X, Y = torch.meshgrid(x, y)
    
    h=torch.cat((X.flatten().unsqueeze(1),Y.flatten().unsqueeze(1)),1)
    
    
    Z=E(h).view(n,n).detach()    
    Z=((Z-torch.min(Z))**(1/2)).detach()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.contour(X,Y,Z,200)
    
    



def find_extrema(E):
    n=len(E)
    m=len(E[0])
    
    saddle=[]
    mini=[]
    maxi=[]
    for i in range(1,n-1):
        for j in range(1,m-1):
            if E[i][j]==min(E[i-1:i+2,j]) and E[i][j]==min(E[i,j-1:j+2]):
                mini.append([i,j])
    
            if E[i][j]==max(E[i-1:i+2,j]) and E[i][j]==max(E[i,j-1:j+2]):
                maxi.append([i,j])
    
            if E[i][j]==min(E[i-1:i+2,j]) and E[i][j]==max(E[i,j-1:j+2]):
                saddle.append([i,j])
    
            if E[i][j]==max(E[i-1:i+2,j]) and E[i][j]==min(E[i,j-1:j+2]):
                saddle.append([i,j])
            

    
    return  mini, maxi, saddle


    
    
    
    
    

