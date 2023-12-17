#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:28:55 2023

@author: siavashriazi
"""

import pandas as pd
import random as rnd
import numpy as np
import math
from scipy.stats import truncnorm, norm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import copy
import os


# the following function returns the ODE's
def sir_odes(pars,state):
    beta=pars[0]
    gamma=pars[1]
    psi=pars[2]
    sigma=pars[3]
    kappa=pars[4]
    
    
    t=state[0]
    S=state[1]
    I=state[2]
    R=state[3]
    
    
    dSdt = -beta/kappa * S * I+sigma*R
    dIdt = beta/kappa  * S * I -  (gamma + psi)* I
    dRdt = (gamma + psi) * I -sigma*R
    
    return np.array([1,dSdt,dIdt,dRdt])

def sir_rk4(pars,inits,nStep):
    h=pars[len(pars)-1]/nStep
    out=np.array([0.]+inits)
    temp=out
    for s in range(nStep):
        k1=sir_odes(pars,temp)
        fk1=temp+k1*h
        k2=sir_odes(pars,fk1)
        fk2=temp+k2*h/2
        k3=sir_odes(pars,fk2)
        fk3=temp+k3*h/2
        k4=sir_odes(pars,fk3)
        fk4=temp+k4*h
        temp=temp+(k1+2*k2+2*k3+k4)/6*h
        out=np.vstack((out,temp))
    return out

# plotting a sample output
#testPars = [0.1,0.01,0.01,0.001,100,5,100] 
testPars=[0.5,0.05,0.01,0.001,100,3,20]#beta,gamma,psi,sigma,kappa,i0,T
testPars=[0.25,0.05,0.01,0.001,300,3,100]#beta,gamma,psi,sigma,kappa,i0,T according to Mathematica
testPars=[0.25,0.05,0.01,0.001,300,3,30]#beta,gamma,psi,sigma,kappa,i0,T change to previous to make big tree in short time

beta, gamma, psi, sigma, kappa, i0, T = testPars
testInits=[kappa-i0,i0,0]
nInt=sir_rk4(testPars,testInits,30)
#plt.plot(nInt[:,0],nInt[:,1],nInt[:,0],nInt[:,2],nInt[:,0],nInt[:,3])
# plotting ODE solution
sPlot, = plt.plot(nInt[:,0],nInt[:,1], label="Suscepible")
iPlot, = plt.plot(nInt[:,0],nInt[:,2], label="Infected")
rPlot, = plt.plot(nInt[:,0],nInt[:,3], label="Recovered")
plt.legend(handles=[sPlot,iPlot, rPlot])
plt.xlabel('forward time (t)')
plt.ylabel('Conuts')
plt.show()
######################
numPar = 2 

# Here we simulate a tree from a stochastic SIR model to do so we follow a number of variables.
# treeMtrx is a matrix whoes elemnt i,j gives the shared ancestry between linegaes i and j
# state is a vector that indicates the state of each lineage. See states below.  Note that the alive vector is like the state vector but just denotes who is alive and who has either recovered or been sampled.  There should be a better way to make this alive vector but I don't know how.
# t is the time variable (forward in time)
# epiState is a matrix whoes rows give [t, S, I, R] for each event in the stocahstic process
# States: 1: infected, 0:recovered, -1:sampled
# Events: 1: transmission, 2: recovery, 3:sampling, 4:immunity loss, 0: non-event to itterate to present day
# The class sim tree has four functions:
# 1. init sets the inital parameters
# 2. event updates the tree and epidemic given that event e has occured after Deltat time
# 3. gillespie Simulates a sequence of events arising from an stochastic SIR model
# 4. sampledTree Returns the sampled treeMtrx and a vector of sampling times and birth times measured FORWARD in time

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

class simTree:
    def __init__(self,pars):
        #setting parameters
        self.beta=pars[0]
        self.gamma=pars[1]
        self.psi=pars[2]
        self.sigma = pars[3]
        self.kappa = pars[4]
        self.i0 = pars[5]
        self.T=pars[6]
        
        #setting state
        self.treeMtrx = np.zeros((self.i0, self.i0))

        self.state = [1] * self.i0
        self.alive = [1] * self.i0
        self.epiState=np.array([[0,self.kappa-self.i0,self.i0,0,0]])
        #self.gillespie()#simulate tree
    def event(self,e,Deltat):
        #Add delta t to infected lineages
        self.treeMtrx=np.identity(len(self.treeMtrx))*Deltat*self.alive+self.treeMtrx
        #print(self.treeMtrx)
        if e==1: #infection
            ind=rnd.choice(find_indices(self.state, lambda x: x==1)) #pick parent
            #update tree matrix, state vector, alive vector
            self.treeMtrx=np.vstack((self.treeMtrx,self.treeMtrx[ind])) #add row to tree mtrx
            col=np.transpose(np.hstack((self.treeMtrx[ind],self.treeMtrx[ind,ind])))
            self.treeMtrx=np.vstack((np.transpose(self.treeMtrx),col))#adding column
            #print(self.treeMtrx)
            self.state=self.state+[1]
            self.alive=self.alive+[1]
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,-1,1,0,0]))
        elif e==2:#recovery
            ind=rnd.choice(find_indices(self.state, lambda x: x==1))# pick lineage to die
            self.state[ind]=0
            self.alive[ind]=0
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,0,-1,1,0]))
        elif e==3:#samplint
            ind=rnd.choice(find_indices(self.state, lambda x: x==1))# pick lineage for sampling
            self.state[ind]=-1
            self.alive[ind]=0
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,0,-1,1,1]))
        elif e==4:#waning
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,1,0,-1,0]))
        elif e==0: #Update to present day *empty event*
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,0,0,0,0]))
        else:
            print("ERROR in event id")
    def gillespie(self):
        #initialize
        t=0
        S=self.epiState[-1,1]
        I=self.epiState[-1,2]
        R=self.epiState[-1,3]
        N=self.epiState[-1,4]
        rates=[self.beta/self.kappa*S*I,self.gamma*I,self.psi*I,self.sigma*R]
        totalRate = sum(rates)
        Deltat=round(np.random.exponential(scale=1/totalRate),3)
        e=rnd.choices(np.linspace(1,len(rates),len(rates)), weights=rates)[0]
        while t+Deltat<self.T:
            #perform event
            self.event(e,Deltat)
            t+=Deltat
            #pick new deltat
            S=self.epiState[-1,1]
            I=self.epiState[-1,2]
            R=self.epiState[-1,3]
            N=self.epiState[-1,4]
            rates=[self.beta/self.kappa*S*I,self.gamma*I,self.psi*I,self.sigma*R]
            totalRate = sum(rates)
            if totalRate==0:
                Deltat=self.T-t
                e=0
            else:
                Deltat=round(np.random.exponential(scale=1/totalRate),3)
                e=rnd.choices(np.linspace(1,len(rates),len(rates)), weights=rates)[0]
        #Last step
        self.event(0,self.T-t)
        self.sampledTree()
    def sampledTree(self):
        # Extracts the sampled tree
        # Extracts the observed sampling times recoded FORWARD in time(yVec)
        # Extracts the observed birth times recoded FORWARD in time (xVec)
        inds=find_indices(self.state, lambda x: x==-1)
        self.sampTree=self.treeMtrx[inds][:,inds]
        self.yVec=np.diagonal(self.sampTree)# sampling times are the diagonal
        # birth times are the (non-duplicated) off diagonals greater than 0
        temp2=np.reshape(np.triu(self.sampTree, k=1),len(self.sampTree)*len(self.sampTree))
        temp2=[x for x in temp2 if x > 0]
        self.xVec=np.array(list(dict.fromkeys(temp2)))
    # a recursive matrix to break the matrix 
    def convert_newick(self,mat):
        if np.shape(mat)[0] == 1:
            #return(":"+str(mat[0][0]))
            return "xAz:" + str(mat[0][0])
        elif np.shape(mat)[0] == 2:
            new_mat = mat - np.amin(mat)
            # dv collects non zero elements of the new mat 
            dv = new_mat[np.nonzero(new_mat)]
            #return("(:"+str(dv[0])+",:"+str(dv[1])+"):"+str(np.amin(mat)))
            return "(xAz:" + str(dv[0]) + ",xAz:" + str(dv[1]) + "):" + str(np.amin(mat))
        elif np.shape(mat)[0] > 2:
            branch_length =  np.amin(mat)
            # substracting min value of all elements
            newm = mat - branch_length
            out = self.break_matrix(newm)
            return "(" + self.convert_newick(out[0])  + "," + self.convert_newick(out[1]) + "):" + str(branch_length)

    # break matrix breaks the matrix to two matrices
    def break_matrix(self,mat):
        mat2 = copy.deepcopy(mat)
        k = []
        for i in range(np.shape(mat2)[0]):
            if mat2[0][i] == 0:
                k.append(i)
            #print(i)
        m1 = np.delete(mat2,k,1)
        m1 = np.delete(m1,k,0)
        m2 = mat[np.ix_(k,k)]
        output = [m1,m2]
        return output

    # toNweick outputs the final result
    def toNewick(self):
        out = self.convert_newick(self.treeMtrx)
        self.treeTxt = "("+out+")xA0z;"
        #self.treeTxt = "("+out+");"
    
    def add_label(self):
        j = 1
        textl = list(self.treeTxt)
        label_list = []
        for i in range(0,len(textl)):
            #print(i)
            if textl[i] == 'A':
                textl.insert(i+1,j)
                label_list.append("A"+str(j))
                j += 1
                
        label_list.append("A0")
        self.treeTxtL = ''.join(map(str, textl))


tree1=simTree(testPars)
tree1.gillespie()
np.shape(tree1.sampTree)


my_blue = '#5B9BD5' 
my_green = '#70AD47'
my_purple = '#7030A0'
# plotting stochastic simulation
sPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,1], color='orange', label="Suscepible")
iPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,2], color='red', label="Infected")
rPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,3], color='purple', label="Recovered")
nPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,4], color=my_blue, label="Sampled")

plt.legend(handles=[sPlot,iPlot, rPlot,nPlot])
#plt.plot(tree1.epiState[:,0],tree1.epiState[:,1],tree1.epiState[:,0],tree1.epiState[:,2],tree1.epiState[:,0],tree1.epiState[:,3])
plt.xlabel('forward time (t)')
plt.ylabel('Conuts')
plt.show()
################################
# plotting, blue vertical line is birth event, green is sampling
#plt.plot(tree1.epiState[:,0],tree1.epiState[:,1],tree1.epiState[:,0],tree1.epiState[:,2],tree1.epiState[:,0],tree1.epiState[:,3])

# plotting stochastic simulation
sPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,1], color='orange')
iPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,2], color='red')
rPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,3], color='purple')
#plt.legend(handles=[sPlot,iPlot, rPlot])
for X in tree1.xVec:
    plt.axvline(x=X, color = 'g')
for Y in tree1.yVec:
    plt.axvline(x=Y, color = 'b')
plt.xlabel('forward time (t)')
plt.ylabel('Conuts')
plt.show()
#####################################
# Your existing code to plot the epidemic data
sPlot, = plt.plot(tree1.epiState[:,0], tree1.epiState[:,1], color='black', label="Susceptible")
iPlot, = plt.plot(tree1.epiState[:,0], tree1.epiState[:,2], color='red', label="Infected")
rPlot, = plt.plot(tree1.epiState[:,0], tree1.epiState[:,3], color='grey', label="Recovered")
plt.legend(handles=[sPlot, iPlot, rPlot])
# Adding green points using tree1.xVec as time points
plt.scatter(tree1.xVec, [tree1.epiState[np.where(tree1.epiState[:,0] == x)[0][0], 2] for x in tree1.xVec], color=my_green, label='Birth', zorder=2)
# Adding blue points using tree1.yVec as time points
plt.scatter(tree1.yVec, [tree1.epiState[np.where(tree1.epiState[:,0] == y)[0][0], 2] for y in tree1.yVec], color=my_purple, label='Sampling', zorder=2)
plt.xlabel('forward time (t)')
plt.ylabel('Counts')
plt.legend()
plt.savefig('sampling_birth.png', dpi=300)

plt.show()


os.chdir("/Users/siavashriazi/Desktop")


# plotting ODE
plt.plot(nInt[:,0],nInt[:,1], linestyle = 'dotted',color='black')
plt.plot(nInt[:,0],nInt[:,2], linestyle ='dotted',color='red')
plt.plot(nInt[:,0],nInt[:,3], linestyle ='dotted', color='grey')
sPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,1], color = 'black',label="Suscepible") # S
iPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,2], color = 'red',label="Infected") # I
rPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,3], color = 'grey',label="Recovered") # R
plt.legend(handles=[sPlot,iPlot, rPlot])
plt.xlabel('forward time (t)')
plt.ylabel('Conuts')
plt.savefig('fit_ode.png', dpi=300)

plt.show()