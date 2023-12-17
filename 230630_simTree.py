#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:27:58 2023

@author: siavashriazi
This is a script to only simulate a tree
"""

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import copy

#testPars=[0.5,0.05,0.01,0.001,100,3,20]#beta,gamma,psi,sigma,kappa,i0,T
testPars=[0.25,0.05,0.01,0.001,300,3,100]#beta,gamma,psi,sigma,kappa,i0,T according to Mathematica
#testPars=[0.3,0.1,0.1,0.01,200,5,10] #PtA beta,gamma,psi,sigma,kappa,i0,T according to Ailene's Mathematica BirthDeathSampling_6_29_23
# mean: 8.31, cv (sd/mean)=0.53
#testPars=[1.05,0.25,0.1,0.01,200,5,10] #PtB beta,gamma,psi,sigma,kappa,i0,T according to Ailene's Mathematica BirthDeathSampling_6_29_23
#  mean: 38.72, cv: 0.225
#testPars=[2.475,0.45,0.1,0.01,200,5,10] #PtC beta,gamma,psi,sigma,kappa,i0,T according to Ailene's Mathematica BirthDeathSampling_6_29_23
# mean: 36.37, cv: 0.18

beta, gamma, psi, sigma, kappa, i0, T = testPars
testInits=[kappa-i0,i0,0]

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


#setting state
treeMtrx = np.zeros((i0, i0))

state = [1] * i0
alive = [1] * i0
epiState=np.array([[0,kappa-i0,i0,0]])

def event(e,Deltat):
    #Add delta t to infected lineages
    global treeMtrx, alive, state, epiState
    treeMtrx=np.identity(len(treeMtrx))*Deltat*alive+treeMtrx
    #print(treeMtrx)
    if e==1: #infection
        ind=rnd.choice(find_indices(state, lambda x: x==1)) #pick parent
        #update tree matrix, state vector, alive vector
        treeMtrx=np.vstack((treeMtrx,treeMtrx[ind])) #add row to tree mtrx
        col=np.transpose(np.hstack((treeMtrx[ind],treeMtrx[ind,ind])))
        treeMtrx=np.vstack((np.transpose(treeMtrx),col))#adding column
        #print(treeMtrx)
        state=state+[1]
        alive=alive+[1]
        #update epiState
        epiState=np.vstack((epiState,epiState[-1]+[Deltat,-1,1,0]))
    elif e==2:#recovery
        ind=rnd.choice(find_indices(state, lambda x: x==1))# pick lineage to die
        state[ind]=0
        alive[ind]=0
        #update epiState
        epiState=np.vstack((epiState,epiState[-1]+[Deltat,0,-1,1]))
    elif e==3:#samplint
        ind=rnd.choice(find_indices(state, lambda x: x==1))# pick lineage for sampling
        state[ind]=-1
        alive[ind]=0
        #update epiState
        epiState=np.vstack((epiState,epiState[-1]+[Deltat,0,-1,1]))
    elif e==4:#waning
        #update epiState
        epiState=np.vstack((epiState,epiState[-1]+[Deltat,1,0,-1]))
    elif e==0: #Update to present day *empty event*
        epiState=np.vstack((epiState,epiState[-1]+[Deltat,0,0,0]))
    else:
        print("ERROR in event id")
        
def gillespie():
    global epiState, beta, gamma, psi, sigma, kappa, i0, T 
    #initialize
    t=0
    S=epiState[-1,1]
    I=epiState[-1,2]
    R=epiState[-1,3]
    rates=[beta/kappa*S*I,gamma*I,psi*I,sigma*R]
    totalRate = sum(rates)
    Deltat=round(np.random.exponential(scale=1/totalRate),3)
    e=rnd.choices(np.linspace(1,len(rates),len(rates)), weights=rates)[0]
    while t+Deltat<T:
        #perform event
        event(e,Deltat)
        t+=Deltat
        #pick new deltat
        S=epiState[-1,1]
        I=epiState[-1,2]
        R=epiState[-1,3]
        rates=[beta/kappa*S*I,gamma*I,psi*I,sigma*R]
        totalRate = sum(rates)
        if totalRate==0:
            Deltat=T-t
            e=0
        else:
            Deltat=round(np.random.exponential(scale=1/totalRate),3)
            e=rnd.choices(np.linspace(1,len(rates),len(rates)), weights=rates)[0]
    #Last step
    event(0,T-t)
    sampledTree()
    
def sampledTree():
    global sampTree, xVec, yVec
    # Extracts the sampled tree
    # Extracts the observed sampling times recoded FORWARD in time(yVec)
    # Extracts the observed birth times recoded FORWARD in time (xVec)
    inds=find_indices(state, lambda x: x==-1)
    sampTree=treeMtrx[inds][:,inds]
    yVec=np.diagonal(sampTree)# sampling times are the diagonal
    # birth times are the (non-duplicated) off diagonals greater than 0
    temp2=np.reshape(np.triu(sampTree, k=1),len(sampTree)*len(sampTree))
    temp2=[x for x in temp2 if x > 0]
    xVec=np.array(list(dict.fromkeys(temp2)))
    
gillespie()

# plotting stochastic simulation
#sPlot, = plt.plot(epiState[:,0],epiState[:,1], label="Suscepible")
#iPlot, = plt.plot(epiState[:,0],epiState[:,2], label="Infected")
#rPlot, = plt.plot(epiState[:,0],epiState[:,3], label="Recovered")
#plt.legend(handles=[sPlot,iPlot, rPlot])
#plt.plot(tree1.epiState[:,0],tree1.epiState[:,1],tree1.epiState[:,0],tree1.epiState[:,2],tree1.epiState[:,0],tree1.epiState[:,3])
#plt.xlabel('forward time (t)')
#plt.ylabel('conuts')
#plt.show()
    
# a recursive matrix to break the matrix 
def convert_newick(mat):
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
        out = break_matrix(newm)
        return "(" + convert_newick(out[0])  + "," + convert_newick(out[1]) + "):" + str(branch_length)

# break matrix breaks the matrix to two matrices
def break_matrix(mat):
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
def toNewick():
    global treeMtrx, treeTxt
    out = convert_newick(treeMtrx)
    treeTxt = "("+out+")xA0z;"
    #treeTxt = "("+out+");"

# this function add number to the labled tips
def add_label():
    global treeTxt, treeTxtL
    j = 1
    textl = list(treeTxt)
    label_list = []
    for i in range(0,len(textl)):
        #print(i)
        if textl[i] == 'A':
            textl.insert(i+1,j)
            label_list.append("A"+str(j))
            j += 1
            
    label_list.append("A0")
    treeTxtL = ''.join(map(str, textl))

#toNewick()
#add_label()

np.shape(sampTree)[1]
xVec
yVec
print(size)







