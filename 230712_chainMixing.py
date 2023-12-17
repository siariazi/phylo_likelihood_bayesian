#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:39:07 2023

@author: siavashriazi
a script to mix the chain from _mhChain codes and plot the results
"""
import pandas as pd, os, math, numpy as np, matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
#Truncated normal: mean mu, variance sigma, truncated between a and b
def randTNorm(a,b,mu,sigma):
    [aStd,bStd]=[(a - mu) / sigma, (b - mu) / sigma]
    return (truncnorm.rvs(aStd, bStd)*sigma+mu)
def pdfTNorm(a,b,mu,sigma,x):
    return truncnorm.pdf((x-mu)/sigma, (a-mu)/sigma, (b-mu)/sigma)/sigma
def cdfTNorm(a,b,mu,sigma,x):
    return truncnorm.cdf((x-mu)/sigma, (a-mu)/sigma, (b-mu)/sigma)
def round_up_to_even(x):
    return math.ceil(x / 2.) * 2

testPars=[0.25,0.05,0.01,0.001,1000,3,100]#beta,gamma,psi,sigma,kappa,i0,T change to previous to make big tree in short time
beta, gamma, psi, sigma, kappa, i0, T = testPars
# change working directory to the folder containing chains
os.chdir('/Users/siavashriazi/Desktop/SFU/Codes/Python/chains/')

# check working directory
os.getcwd()

# Maintain a list of filenames
filenames = ["chain1.csv", "chain2.csv", "chain3.csv"]

df = pd.read_csv(filenames[0])

# Transpose the dataframe and convert it to a list of lists
data = df.T.values.tolist()
nRep = len(data[0])

biRatio = 0.3

biRep=round_up_to_even(nRep*biRatio) # burn-in replicates

nChain = len(filenames)

# This will be your larger list containing all smaller lists
chainList = []

# this list contain all the psiLists
psiList = []
biList = []

for filename in filenames:
    # Read each csv file into a dataframe
    df = pd.read_csv(filename)
    
    # Transpose the dataframe and convert it to a list of lists
    data = df.values.tolist()
    
    # if the number of n is odd, I subtract 1 to make it even 
    #data = data[:-1]
    
    # Add this list of lists to your larger list
    chainList.append(data)
    # Now chainList contains list of lists from all CSV files
    
    # psiList has the posterior values of all parameters 
    psiList.append(data[biRep:])
    
    # biList has burnin values of parameters
    biList.append(data[:biRep])

# this list contain all the psiLists


nPar=len(chainList[0][0])
n=len(psiList[0]) #number of obs. per chain
halfLen=math.floor(n/2)

nBin = 1000
a = 50
psiListAll=psiList[0]
KDEListAll=np.empty((nPar,2,nBin),dtype=np.double)

def calcKDEAll(nBin,a): # calculate the kernal density estimate across chains
    #Combine psiLists
    global psiListAll, KDEListAll
    #print('size: {}'.format(self.psiListAll.shape))
    for c in range(1,nChain):
        psiListAll=np.concatenate((psiListAll, psiList[c]), axis=0)
        #print('size: {}'.format(self.psiListAll.shape))
   
    for var in range(0,nPar):
        KDEListAll[var][0]= np.linspace(min(psiListAll[:,var]), max(psiListAll[:,var]), nBin)
        dx=KDEListAll[var][0][1]-KDEListAll[var][0][0]
        KDEListAll[var][1] = sum(norm(xi,dx*50).pdf(KDEListAll[var][0]) for xi in psiListAll[:,var])#smooth out over a of the nBin interval
        tot=sum(np.multiply(KDEListAll[var][1],dx))
        KDEListAll[var][1]/=tot 

calcKDEAll(nBin,a) #calculate KDE for the combined chains    

def calcESS(lenTest):
    lenTest=min(lenTest,math.floor(halfLen/10))
    V=np.empty((lenTest,2*nChain,nPar),dtype=np.double)
    for c in range(nChain):
        half1=psiList[c][:halfLen]
        half2=psiList[c][halfLen:]
        for t in range(lenTest):
            temp1=half1[t:]
            temp2=half1[:halfLen-t]
            V[t][2*c]=np.mean(np.multiply(temp1-np.mean(temp1,axis=0),temp2-np.mean(temp2,axis=0)),axis=0)
            temp1=half2[t:]
            temp2=half2[:halfLen-t]
            V[t][2*c+1]=np.mean(np.multiply(temp1-np.mean(temp1,axis=0),temp2-np.mean(temp2,axis=0)),axis=0)
        ESS=np.sum(halfLen/(1+2*np.sum(np.abs(V),axis=0)),axis=0)
    return ESS    

lenTest=200
calcESS(lenTest)


def calcR():
    # calc Variance within V
    WList=np.empty((2*nChain,nPar),dtype=np.double) # list of the variance within each half-chain
    ChainAvg=np.empty((2*nChain,nPar),dtype=np.double) # array of the average outcome for each parameter in each half-chain
    for c in range(nChain): # for each full chain
    
        half1=np.array(psiList[c][:halfLen])
        half2=np.array(psiList[c][halfLen:])
        #run Mean
        ChainAvg[2*c]=sum(half1)/halfLen
        ChainAvg[2*c+1]=sum(half2)/halfLen
        WList[2*c]=np.divide(sum(np.power(np.add(half1,-1.0*ChainAvg[2*c]),2)),halfLen-1)
        WList[2*c+1]=np.divide(sum(np.power(np.add(half2,-1.0*ChainAvg[2*c+1]),2)),halfLen-1)
    WAvg=sum(WList)/len(WList)
    #calc Variance between B
    ChainHat=np.mean(ChainAvg, axis=0)
    B=np.multiply(np.divide(sum(np.power(np.add(ChainAvg,-1.0*ChainHat),2)),nChain-1),halfLen)
    VarPlus=np.add(np.multiply(WAvg,(halfLen-1)/halfLen),np.multiply(B,1/halfLen))
    R=np.power(np.divide(VarPlus,WAvg),0.5)
    return R

calcR()


def findCIAll(var): # find the credible interval of concatonated chains
    global CI, maxPost
    [x_d,kde]=KDEListAll[var]
    dx=x_d[1]-x_d[0]
    #index = np.lexsort((x_d,kde))
    index = np.lexsort((x_d,kde))[::-1] #sort (from largest to smallest) by kde then by x_d
    maxPost=[x_d[index[0]],kde[index[0]]] 
    temp=np.add.accumulate(kde[index]*dx)
    temp2=index[[i for i,v in enumerate(temp) if v < 0.95]] #find pts in credible interval
    indexRev = np.lexsort((kde[temp2],x_d[temp2]))[::-1] #sort (from largest to smallest) by x_d then by kde 
    CI=[list(x_d[temp2][indexRev]),list(kde[temp2][indexRev])]

findCIAll(0)    

def kdeHistAll(var,ax): # plot hisogram of parameter estimate from an individual chain
    [x_d,kde]=KDEListAll[var]
    findCIAll(var)
    #plt.fill_between(x_d, density, alpha=0.5) #plot filled smoothed kernel density
    ax.hist(psiListAll[:,var].tolist(), bins=30,density=True, color = "skyblue",alpha=0.3) #plot histogram
    ax.plot(x_d,kde,'k') #plot smoothed kernel density line
    ax.scatter(CI[0],CI[1],color='pink',alpha=1,marker=".") #plot credible interval
    ax.plot([maxPost[0],maxPost[0]],[0, max(kde)*1.2],'r')# Show maximum postier estiamte
    ax.plot([[beta,gamma][var],[beta,gamma][var]],[0, max(kde)*1.2],'g')# Show true value
    #ax.plot([[beta,gamma,psi][var],[beta,gamma,psi][var]],[0, max(kde)*1.2],'g')# Show true value
    ax.plot(psiListAll[:,var], np.full_like(psiListAll[:,var],-max(kde)*0.05), '|k', markeredgewidth=1) #plot hashes at bottom
    ax.axis([min(x_d), max(x_d), -2*max(kde)*0.05, max(kde)*1.2]); 
    

def plotChain(var,ax,biRatio,c): # plot mixing within var 'var' in a chain
        #fig, ax = plt.subplots()
        biListc = biList[c]
        biListp = [inner_list[var] for inner_list in biListc]
        psiListc = psiList[c]
        psiListp = [inner_list[var] for inner_list in psiListc]
        ax.plot(range(0,biRep),biListp,color='skyblue',alpha=0.5);
        ax.plot(range(biRep,nRep),psiListp,color='blue');    

plt.rcParams.update({'font.size': 15})


fig, axs = plt.subplots(nPar,nChain)
#fig.suptitle('Stochastic chains')
for v in range(nPar):
    for c in range(nChain):
        plotChain(v,axs[v,c],biRatio,c)  
        axs[0,c].set(title='chain {}'.format(c))
    axs[v,0].set(ylabel=[r'$\beta$',r'$\gamma$'][v])
    #axs[v,0].set(ylabel=[r'$\beta$',r'$\gamma$',r'$\psi$'][v])
for ax in fig.get_axes():
    ax.label_outer()
fig.set_figheight(7)
fig.set_figwidth(15)


#fig.savefig('Basic.png', dpi=300)
fig, axs = plt.subplots(nChain)
#fig.suptitle('Stochastic chains')
for c in range(nChain):
    plotChain(0,axs[c],biRatio,c)  
    axs[c].set(title='chain {}'.format(c))
#axs[v,0].set(ylabel=[r'$\beta$',r'$\gamma$'][v])    
    #axs[v,0].set(ylabel=[r'$\beta$',r'$\gamma$',r'$\psi$'][v])
for ax in fig.get_axes():
    ax.label_outer()
fig.set_figheight(7)
fig.set_figwidth(15)

# plotting the posterior
fig, ax2 = plt.subplots(1,nPar)
for v in range(nPar):
    kdeHistAll(v,ax2[v])
ax2[0].set(ylabel=r'$\beta$')
ax2[1].set(ylabel=r'$\gamma$')
#ax2[2].set(ylabel=r'$\psi$')
#ax2[1].set(ylabel='kappa')
#ax2[4].set(ylabel='sigma')

# ax2[0].set(xlim=(0, 10))
# ax2[1].set(xlim=(0, 10))
fig.set_figheight(5)
fig.set_figwidth(29)
maxPost

# plotting the posterior
fig, ax2 = plt.subplots()
kdeHistAll(0,ax2[0])
ax2[0].set(ylabel=r'$\beta$')
#ax2[2].set(ylabel=r'$\psi$')
#ax2[1].set(ylabel='kappa')
#ax2[4].set(ylabel='sigma')

# ax2[0].set(xlim=(0, 10))
# ax2[1].set(xlim=(0, 10))
fig.set_figheight(5)
fig.set_figwidth(29)
maxPost
