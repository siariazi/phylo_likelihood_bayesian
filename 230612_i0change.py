#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:26:54 2023

@author: siavashriazi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:46:27 2022

@author: siari
# This is a script based on 220924_3parms_AM_SIR_init5.py that incorporates change in initial number of infections in starting distance matrix, so if i0=3 then the initial treeMtrx has three rows and three columns all filled with 0. 

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
        self.epiState=np.array([[0,self.kappa-self.i0,self.i0,0]])
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
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,-1,1,0]))
        elif e==2:#recovery
            ind=rnd.choice(find_indices(self.state, lambda x: x==1))# pick lineage to die
            self.state[ind]=0
            self.alive[ind]=0
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,0,-1,1]))
        elif e==3:#samplint
            ind=rnd.choice(find_indices(self.state, lambda x: x==1))# pick lineage for sampling
            self.state[ind]=-1
            self.alive[ind]=0
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,0,-1,1]))
        elif e==4:#waning
            #update epiState
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,1,0,-1]))
        elif e==0: #Update to present day *empty event*
            self.epiState=np.vstack((self.epiState,self.epiState[-1]+[Deltat,0,0,0]))
        else:
            print("ERROR in event id")
    def gillespie(self):
        #initialize
        t=0
        S=self.epiState[-1,1]
        I=self.epiState[-1,2]
        R=self.epiState[-1,3]
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

# plotting stochastic simulation
sPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,1], color='orange', label="Suscepible")
iPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,2], color='red', label="Infected")
rPlot, = plt.plot(tree1.epiState[:,0],tree1.epiState[:,3], color='purple', label="Recovered")
plt.legend(handles=[sPlot,iPlot, rPlot])
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
sPlot, = plt.plot(tree1.epiState[:,0], tree1.epiState[:,1], color='orange', label="Susceptible")
iPlot, = plt.plot(tree1.epiState[:,0], tree1.epiState[:,2], color='red', label="Infected")
rPlot, = plt.plot(tree1.epiState[:,0], tree1.epiState[:,3], color='purple', label="Recovered")
plt.legend(handles=[sPlot, iPlot, rPlot])
my_blue = '#5B9BD5' 
my_green = '#70AD47'
# Adding green points using tree1.xVec as time points
plt.scatter(tree1.xVec, [tree1.epiState[np.where(tree1.epiState[:,0] == x)[0][0], 2] for x in tree1.xVec], color=my_green, label='Birth')

# Adding blue points using tree1.yVec as time points
plt.scatter(tree1.yVec, [tree1.epiState[np.where(tree1.epiState[:,0] == y)[0][0], 2] for y in tree1.yVec], color=my_blue, label='Sampling')


plt.xlabel('forward time (t)')
plt.ylabel('Counts')
plt.legend()
plt.show()
################################
# plotting alonside
sPlot, = plt.plot(nInt[:,0],nInt[:,1], label="Suscepible")
iPlot, = plt.plot(nInt[:,0],nInt[:,2], label="Infected")
rPlot, = plt.plot(nInt[:,0],nInt[:,3], label="Recovered")
plt.legend(handles=[sPlot,iPlot, rPlot])
plt.plot(tree1.epiState[:,0],tree1.epiState[:,1], color = 'blue') # S
plt.plot(tree1.epiState[:,0],tree1.epiState[:,2], color = 'orange') # I
plt.plot(tree1.epiState[:,0],tree1.epiState[:,3], color = 'green') # R
plt.xlabel('forward time (t)')
plt.ylabel('Conuts')
plt.show()
################################
print(tree1.xVec)
print(tree1.yVec)


tree1.toNewick()
tree1.add_label()
tree1.treeTxtL
# package toytree
import toytree
# printing given trees in Newick format using toytree
# removing name of the root, pakcages don't like it
treeTxt = tree1.treeTxtL[:-5]
treeTxt += ";"
toyTree = toytree.tree(treeTxt, tree_format=0)
toyTree.draw()
print(toyTree)
canvas, axes, mark = toyTree.draw(width=1000, height=700);
import toyplot.pdf
toyplot.pdf.render(canvas, "C:/Siavash/Codes/tree_plot.pdf")

# package ete3
from ete3 import Tree
# printing given trees in Newick format using ete3
eteTree1 = Tree(treeTxt)
print(eteTree1)

# writing the tree to a text file
save_path='/Users/siavashriazi/Desktop/SFU/iqtree-2.2.0-MacOSX'
file_name = "SIRtree.nwk"
completeName = os.path.join(save_path, file_name)
file1 = open(completeName, "w")
file1.write(tree1.treeTxtL)
file1.close()

# after converting the tree to alignment by iqtree, the file needs to be trimmed
file2 = open("/Users/siavashriazi/Desktop/SFU/iqtree-2.2.0-MacOSX/alignment.phy","r")
content = file2.read()
file2.close()
# deleting all the content before A
content2 = content[content.index("x"):]
# removing whitepsaces
content2 = content2.strip()
content2 = content2.replace('x','>')
content2 = content2.replace('z','\n')
file3 = open("/Users/siavashriazi/Desktop/SFU/BEAST files/alignment.fasta", "w")
file3.write(content2)
file3.close()

# after converting the alignment to xml file using beauti2 I write a clean version of newick tree in BEAST folder
text2 = text.replace('x','')
text2 = text2.replace('z','')
file4 = open(r"C:\\Siavash\Codes\Phylogenetic_softwares\BEAST_2.6.7\alignment_tree.txt", "w")
file4.write(text2)
file4.close()


class like():
    def __init__(self,parsL,treeIn):
        #Parameters
        ##Epi parameters
        self.beta = parsL[0]
        self.gamma = parsL[1]
        self.psi = parsL[2]
        self.sigma = parsL[3]
        self.kappa = parsL[4]
        self.i0 = parsL[5]
        self.T = parsL[6]
        #self.T=20
        #self.kappa=100
        #self.sigma=0.001
        
        ##Diversification parameters
        self.SIR=sir_rk4(parsL,[self.kappa-self.i0,self.i0,0],self.T*3)
        self.tauVec=self.T-self.SIR[:,0] # backward time
        self.newTau = self.SIR[:,0] # a vector of time to be attached to E vector for plotting 
        self.lambdaVec=self.SIR[:,1]*self.beta/self.kappa
        self.muVec=np.full((len(self.tauVec)), self.gamma)
        self.psiVec=np.full((len(self.tauVec)), self.psi)
        #data
        self.xVec=(treeIn.T-treeIn.xVec)#x BACKWARD in time 
        self.yVec=(treeIn.T-treeIn.yVec)#y BACKWARD in time
        self.nStep = 30
        self.inits = [1] # initial condition of E
    # calcE() calculates dEdt
    def calcE(self,state,t):
        
        E=state[0]

        S = float(self.SIR[t,1]) # t is the index of time not the value of time

        
        self.dEdt =   -(S*self.beta/self.kappa + self.gamma + self.psi)*E + S*(self.beta/self.kappa)*E**2 + self.gamma  

    # E_euler integrate dEdt
    def E_euler(self,time):
        h = time[0]/self.nStep
        newT = len(time) - 1
        self.out=np.array(self.inits)
        temp=self.out
        for i in range(newT):
            # calculating E bakcward
            self.calcE(temp,newT-i)           
            temp=temp+self.dEdt*h
            self.out=np.vstack((self.out,temp))
       
    def lamda(self,t):
        lamda = self.sValue(self.T-t)*self.beta/self.kappa
        return lamda
    # phi() A returns phi integrad
    # this function should have a return value because we are integrating it over integration interval in quad()   
    def phiA(self,t):
        phiA =  2*self.lamda(t)*self.eValue(t) - (self.lamda(t) + self.gamma + self.psi)
        return phiA
    # likelihood() calls other functions in the class and calculates the (log)likelihood
    def likelihood(self):
        # caclulating dEdt
        self.E_euler(self.tauVec) 
        
        # attaching time to eVec
        self.eVec = np.vstack((self.newTau,np.squeeze(self.out)))
        
        # interpolation of S(t) and E(t)
        self.sValue = interp1d(self.SIR[:,0], self.SIR[:,1], kind = 'cubic')
        self.eValue = interp1d(self.eVec[0,:], self.eVec[1,:], kind = 'cubic')
        
        # the first part of the equation: integrating from 0 to T
        #self.g = math.exp(quad(self.phiA, 0, self.T)[0])
        self.g2 = np.log(math.exp(quad(self.phiA, 0, self.T)[0]))

        # integating over xi to T        
        for i in self.xVec:
            #self.g = self.g*self.lamda(i)*math.exp(quad(self.phiA, 0, i)[0])
            self.g2 = self.g2 + np.log(self.lamda(i)) + np.log(math.exp(quad(self.phiA, 0, i)[0]))
        # integrating over yj to T
        for j in self.yVec:
            #self.g = self.g*self.psi/math.exp(quad(self.phiA, 0, j)[0])
            self.g2 = self.g2 + np.log(self.psi) - np.log(math.exp(quad(self.phiA, 0, j)[0]))
        
        #self.g = np.log(self.g)
 

like1=like(testPars,tree1)
plt.plot(like1.SIR[:,0],like1.SIR[:,3]) # plotting to check
like1.likelihood()
like1.g # loglikelihood
plt.plot(like1.eVec[0,:],like1.eVec[1,:]) # plotting E(tau)

#plt.plot(like1.tauVec,like1.lambdaVec,like1.tauVec,like1.muVec,like1.tauVec,like1.psiVec)
#plt.xlabel('backward time (tau)')
#plt.ylabel('Rates')
#plt.show()

# calculating a likelihood surface 
# calculating a likelihood surface 
def LLsurface(pars):
    gendat = [[],[],[]]
    for i in np.arange(0.05,1,0.05): # i is beta 
        for j in np.arange(0.01,0.1,0.01): # j is gamma
            gendat[0].append(i)
            gendat[1].append(j)
            testPars=[i,j,pars[2],pars[3],pars[4],pars[5],pars[6]] 
            like1 = like(testPars,tree1)
            like1.likelihood()
            gendat[2].append(like1.g)

    gendata = np.array(gendat)
    ax = plt.axes(projection='3d')
    ax.scatter(gendata[0,:], gendata[1,:], gendata[2,:])
    ax.set_xlabel('beta')
    ax.set_ylabel('gamma')
    ax.set_zlabel('LLL')
    
LLsurface(testPars)

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

class Theta:
    def __init__(self,jmpRatio=200):
        self.next=None # pointer for linked list
        self.cnt=0
        # bound of parameters
        self.betaDict={"min":0.01,"max":1.0,"jmpVar":10.0/jmpRatio} # real beta is 0.25
        self.gammaDict={"min":0.01,"max":2,"jmpVar":10.0/jmpRatio} # real gamma is 0.05
        #self.psiDict={"min":0.0,"max":0.5,"jmpVar":10.0/jmpRatio}  # real psi is 0.01
        #self.sd = 0.05
        self.psi = psi
        self.sigma = sigma
        self.kappa = kappa
        self.i0 = i0
        self.T = T
        #self.kappaDict={"min":20.0,"max":200.0,"jmpVar":10.0/jmpRatio} # real kappa is 100
        #self.sigmaDict={"min":0.0,"max":0.5,"jmpVar":10.0/jmpRatio} # real sigma is 0.001

    def randTheta(self):
        self.beta=rnd.uniform(self.betaDict["min"],self.betaDict["max"])  
        self.gamma=rnd.uniform(self.gammaDict["min"],self.gammaDict["max"])  
        #self.psi=rnd.uniform(self.psiDict["min"],self.psiDict["max"])  
        #self.T = 20 # I'm not sure should I pass T as a parm or constant
        #self.kappa=rnd.uniform(self.kappaDict["min"],self.kappaDict["max"])  
        #self.sigma=rnd.uniform(self.sigmaDict["min"],self.sigmaDict["max"])  
        
        parsL1 = [self.beta,self.gamma,self.psi,self.sigma,self.kappa,self.i0,self.T]
        #print(parsL1)
        like1 = like(parsL1, tree1)
        like1.likelihood()
        self.lnLik = like1.g2
        #self.calcEqu()
        #self.calcLnLik()
        self.printTheta('Theta Initial')
    def jump(self,parent):
        self.cnt=parent.cnt+1 # counter 
        self.beta = randTNorm(self.betaDict["min"], self.betaDict["max"], parent.beta, self.betaDict["jmpVar"])
        self.gamma = randTNorm(self.gammaDict["min"], self.gammaDict["max"], parent.gamma, self.gammaDict["jmpVar"])
        #self.psi = randTNorm(self.psiDict["min"], self.psiDict["max"], parent.psi, self.psiDict["jmpVar"])
        #self.kappa = randTNorm(self.kappaDict["min"], self.kappaDict["max"], parent.kappa, self.kappaDict["jmpVar"])
        #self.sigma = randTNorm(self.sigmaDict["min"], self.sigmaDict["max"], parent.sigma, self.sigmaDict["jmpVar"])
        parsL1 = [self.beta,self.gamma,self.psi,self.sigma,self.kappa,self.i0,self.T]
        like1 = like(parsL1, tree1)
        like1.likelihood()
        self.lnLik = like1.g2
        self.calcJumpProb(parent)
    def calcJumpProb(self,parent):
        #Calculate the forward jump probability
        self.jmpProbF=1
        self.jmpProbF*=pdfTNorm(self.betaDict["min"], self.betaDict["max"], parent.beta, self.betaDict["jmpVar"], self.beta)
        self.jmpProbF*=pdfTNorm(self.gammaDict["min"], self.gammaDict["max"], parent.gamma, self.gammaDict["jmpVar"], self.gamma)
        #self.jmpProbF*=pdfTNorm(self.psiDict["min"], self.psiDict["max"], parent.psi, self.psiDict["jmpVar"], self.psi)
        #self.jmpProbF*=pdfTNorm(self.kappaDict["min"], self.kappaDict["max"], parent.kappa, self.kappaDict["jmpVar"], self.kappa)
        #self.jmpProbF*=pdfTNorm(self.sigmaDict["min"], self.sigmaDict["max"], parent.sigma, self.sigmaDict["jmpVar"], self.sigma)        
        #Calculate the backward jump probability
        self.jmpProbB=1
        self.jmpProbB*=pdfTNorm(self.betaDict["min"], self.betaDict["max"], parent.beta, self.betaDict["jmpVar"], self.beta)
        self.jmpProbB*=pdfTNorm(self.gammaDict["min"], self.gammaDict["max"], parent.gamma, self.gammaDict["jmpVar"], self.gamma)
        #self.jmpProbB*=pdfTNorm(self.psiDict["min"], self.psiDict["max"], parent.psi, self.psiDict["jmpVar"], self.psi)
        #self.jmpProbB*=pdfTNorm(self.kappaDict["min"], self.kappaDict["max"], parent.kappa, self.kappaDict["jmpVar"], self.kappa)
        #self.jmpProbB*=pdfTNorm(self.sigmaDict["min"], self.sigmaDict["max"], parent.sigma, self.sigmaDict["jmpVar"], self.sigma)
        #calculate acceptance ratio
        #print([self.jmpProbF,self.jmpProbB,self.Lik,parent.Lik])
        #self.r=(self.Lik/parent.Lik)*(self.jmpProbB/self.jmpProbF)
        self.lnr = self.lnLik - parent.lnLik + math.log(self.jmpProbB) - math.log(self.jmpProbF)
        #print([(self.Lik/parent.Lik),(self.jmpProbB/self.jmpProbF),self.r])

    def printTheta(self,name): 
        nDig=4
        print('{} \n'.format(name))
        print('bete: {}'.format(round(self.beta,nDig)))
        print('gamma: {}'.format(round(self.gamma,nDig)))
        print('psi: {}'.format(round(self.psi,nDig)))
        print('kappa: {}'.format(round(self.kappa,nDig)))
        print('sigma: {}'.format(round(self.sigma,nDig)))
        print('log-likelihood: {}'.format(self.lnLik))
        print('\n')

tree1=simTree(testPars)
tree1.gillespie()
np.shape(tree1.sampTree)
like1=like(testPars,tree1)
like1.likelihood()
like1.g2      
Theta1=Theta()
Theta1.randTheta()
Theta1.printTheta('Theta 1')

class chain:
    def __init__(self,nPar=numPar):
        self.headval = Theta() # setting up the linked list
        self.headval.randTheta() # drawing a random starting point (future models may choice to overdisperse these)
        self.nPar=nPar
    def MHChain(self,nRep): #simulate chain using the Metropolis-Hastings Algorithm
        self.nRep=nRep
        wrkTheta=self.headval
        t=1 # counter of the reps
        acc=0 # conter of accepted jumps
        fail=0 # conter of rejected jumps
        jmpRatio=20 # jump Ratio, scales the variance in jump distn. 
        while t<nRep: 
            propTheta=Theta(jmpRatio) #propose a theta
            propTheta.jump(wrkTheta)
            temp=rnd.random()
            #if temp<propTheta.r: #accept
            if temp<math.exp(propTheta.lnr): #accept
                if (t+1)%(min(nRep/2,50))==0: #priting out progress
                    if((acc/(acc+fail))>0.3): # accepting too many,  
                        jmpRatio/=2  #searches more globallly
                        print('t: {}, accept ratio {}, new jmp Ratio: {}'.format(t+1,acc/(acc+fail),jmpRatio))
                    elif((acc/(acc+fail))<0.1):
                        jmpRatio*=2 #searches more locally
                        print('t: {}, accept ratio {}, new jmp Ratio: {}'.format(t+1,acc/(acc+fail),jmpRatio))
                    else:
                        print('t: {}, accept ratio {}'.format(t+1,acc/(acc+fail)))
                    # reset acc and fail 
                wrkTheta.next=propTheta #link list
                wrkTheta=wrkTheta.next #step along
                t+=1
                acc+=1
#                 print('accept r={} random={}'.format(propTheta.r,temp))
            else:
                fail+=1
                # print('reject r={} random={}'.format(propTheta.r,temp))
    def printChain(self,biRatio): #convert linked list into a np array of parameter values
        out=np.empty((self.nRep,self.nPar),dtype=np.double)
        wrkTheta=self.headval
        row=0
        for row in range(self.nRep):
            out[row][0]=wrkTheta.beta
            out[row][1]=wrkTheta.gamma
            #out[row][2]=wrkTheta.psi
            #out[row][1]=wrkTheta.kappa
            #out[row][4]=wrkTheta.sigma
            wrkTheta=wrkTheta.next
        #Handelling burnin (bi)
        self.biRep=round_up_to_even(self.nRep*biRatio)
        #print('chain length: {}, biRep: {}, n: {}\n'.format(len(out),self.biRep,self.n))
        self.psiList=out[self.biRep:] #posterior chain
        self.biList=out[:self.biRep] #burnin chain
    def plotChain(self,var,ax,biRatio): # plot mixing within var 'var' in a chain
        self.printChain(biRatio)
        #fig, ax = plt.subplots()
        ax.plot(range(0,self.biRep),self.biList[:,var],color='skyblue',alpha=0.5);
        ax.plot(range(self.biRep,self.nRep),self.psiList[:,var],color='blue');
    def calcKDE(self,biRatio,nBin=1000,a=50): # calculate the kernal density estimate from the chain
        self.printChain(biRatio)
        self.KDEList=np.empty((self.nPar+2,2,nBin),dtype=np.double)
        for var in range(0,self.nPar+2):
            self.KDEList[var][0]= np.linspace(min(self.psiList[:,var]), max(self.psiList[:,var]), nBin)
            dx=self.KDEList[var][0][1]-self.KDEList[var][0][0]
            self.KDEList[var][1] = sum(norm(xi,dx*a).pdf(self.KDEList[var][0]) for xi in self.psiList[:,var])#smooth out over 50 of the 1000 intervals
            tot=sum(np.multiply(self.KDEList[var][1],dx))
            self.KDEList[var][1]/=tot
    def findCI(self,var): # find the credible interval
        [x_d,kde]=self.KDEList[var]
        dx=self.KDEList[var][0][1]-self.KDEList[var][0][0]
        #index = np.lexsort((x_d,kde))
        index = np.lexsort((x_d,kde))[::-1] #sort (from largest to smallest) by kde then by x_d 
        self.maxPost=[x_d[index[0]],kde[index[0]]] #maximum posterior estimate
        temp=np.add.accumulate(kde[index]*dx)
        temp2=index[[i for i,v in enumerate(temp) if v < 0.95]] #find pts in credible interval
        indexRev = np.lexsort((kde[temp2],x_d[temp2]))[::-1] #sort (from largest to smallest) by x_d then by kde 
        return [list(x_d[temp2][indexRev]),list(kde[temp2][indexRev])]
    def kdeHist(self,var,ax,biRatio): # plot hisogram of parameter estimate from an individual chain
        self.calcKDE(biRatio)
        [x_d,kde]=self.KDEList[var]
        #plt.fill_between(x_d, density, alpha=0.5) #plot filled smoothed kernel density
        ax.hist(self.psiList[:,var], bins=30,density=True, color = "skyblue",alpha=0.3) #plot histogram
        ax.plot(x_d,kde,'k') #plot smoothed kernel density line
        temp=self.findCI(var)
        ax.scatter(temp[0],temp[1],color='pink',alpha=1,marker=".") #plot credible interval
        ax.plot([self.maxPost[0],self.maxPost[0]],[0, max(kde)*1.2],'r')# Show maximum postier estiamte
        ax.plot([[beta,gamma,psi][var],[beta,gamma,psi][var]],[0, max(kde)*1.2],'g')# Show true value
        ax.plot(self.psiList[:,var], np.full_like(self.psiList[:,var],-max(kde)*0.05), '|k', markeredgewidth=1) #plot hashes at bottom
        ax.axis([min(x_d), max(x_d), -2*max(kde)*0.05, max(kde)*1.2]);

nRep=3000
chain1=chain()
chain1.MHChain(nRep)

#Histograms ##################
biRatio=0.2
chain1.printChain(biRatio)
fig, axs = plt.subplots(1,numPar)
axs[0].hist(chain1.psiList[:,0],bins=20)
axs[1].hist(chain1.psiList[:,1],bins=20)
#axs[2].hist(chain1.psiList[:,2],bins=20)
#axs[3].hist(chain1.psiList[:,3],bins=20)
#axs[4].hist(chain1.psiList[:,4],bins=20)
axs[0].plot([beta,beta],[0, 250],'g')# Show true value
axs[1].plot([gamma,gamma],[0, 300],'g')# Show true value
#axs[2].plot([psi,psi],[0, 550],'g')# Show true value
#axs[1].plot([kappa,kappa],[0, 100],'g')# Show true value
#axs[4].plot([sigma,sigma],[0, 100],'g')# Show true value
fig.set_figheight(3)
fig.set_figwidth(12)
axs[0].set(title='$beta$')
#axs[1].set(title='$kappa$')
axs[1].set(title='$gamma$')
#axs[2].set(title='$psi$')
#axs[4].set(title='$sigma$')
axs[0].set(ylabel='probability density')

# plotting chains ##################
biRatio=0.2
fig, axs = plt.subplots(1,numPar)
fig.suptitle('Stochastic chains')
chain1.plotChain(0,axs[0],biRatio)
chain1.plotChain(1,axs[1],biRatio)
#chain1.plotChain(2,axs[2],biRatio)
#chain1.plotChain(3,axs[3],biRatio)
#chain1.plotChain(4,axs[4],biRatio)
for ax in fig.get_axes():
    ax.label_outer()
axs[0].set(ylabel='beta')
axs[1].set(ylabel='gamma')
#axs[2].set(ylabel='psi')
#axs[1].set(ylabel='kappa')
#axs[4].set(ylabel='sigma')
fig.set_figheight(3)
fig.set_figwidth(8)
#######################################
# simulating multiple chains
def runChains(nChain,nRep):
    chainList=[]
    for c in range(nChain):
        print('Chain #: {}'.format(c))
        chainList.append(chain()) #empty chain
        chainList[c].MHChain(nRep)
    return chainList
chainList=runChains(3,500)

# calculate convergence statistics and compound posterior
class baysianEst:
    def __init__(self,chainsIn,biRatio):
        self.chainList=chainsIn
        self.nChain=len(chainsIn) # of chains
        for c in range(self.nChain): #print chains
            self.chainList[c].printChain(biRatio)
        self.nPar=self.chainList[0].nPar # of parameters in a chain
        self.n=len(self.chainList[0].psiList) #number of obs. per chain
        self.halfLen=math.floor(self.n/2)
        self.calcKDEAll() #calculate KDE for the combined chains
    def calcR(self):
        # calc Variance within V
        WList=np.empty((2*self.nChain,self.nPar),dtype=np.double) # list of the variance within each half-chain
        self.ChainAvg=np.empty((2*self.nChain,self.nPar),dtype=np.double) # array of the average outcome for each parameter in each half-chain
        for c in range(self.nChain): # for each full chain
            half1=self.chainList[c].psiList[:self.halfLen]
            half2=self.chainList[c].psiList[self.halfLen:]
            #run Mean
            self.ChainAvg[2*c]=sum(half1)/self.halfLen
            self.ChainAvg[2*c+1]=sum(half2)/self.halfLen
            WList[2*c]=np.divide(sum(np.power(np.add(half1,-1.0*self.ChainAvg[2*c]),2)),self.halfLen-1)
            WList[2*c+1]=np.divide(sum(np.power(np.add(half2,-1.0*self.ChainAvg[2*c+1]),2)),self.halfLen-1)
        self.WAvg=sum(WList)/len(WList)
        #calc Variance between B
        ChainHat=np.mean(self.ChainAvg, axis=0)
        self.B=np.multiply(np.divide(sum(np.power(np.add(self.ChainAvg,-1.0*ChainHat),2)),self.nChain-1),self.halfLen)
        self.VarPlus=np.add(np.multiply(self.WAvg,(self.halfLen-1)/self.halfLen),np.multiply(self.B,1/self.halfLen))
        self.R=np.power(np.divide(self.VarPlus,self.WAvg),0.5)
        return self.R
    def calcESS(self,lenTest=200):
        lenTest=min(lenTest,math.floor(self.halfLen/10))
        self.V=np.empty((lenTest,2*self.nChain,self.nPar),dtype=np.double)
        for c in range(self.nChain):
            half1=self.chainList[c].psiList[:self.halfLen]
            half2=self.chainList[c].psiList[self.halfLen:]
            for t in range(lenTest):
                temp1=half1[t:]
                temp2=half1[:self.halfLen-t]
                self.V[t][2*c]=np.mean(np.multiply(temp1-np.mean(temp1,axis=0),temp2-np.mean(temp2,axis=0)),axis=0)
                temp1=half2[t:]
                temp2=half2[:self.halfLen-t]
                self.V[t][2*c+1]=np.mean(np.multiply(temp1-np.mean(temp1,axis=0),temp2-np.mean(temp2,axis=0)),axis=0)
            self.ESS=np.sum(self.halfLen/(1+2*np.sum(np.abs(self.V),axis=0)),axis=0)
        return self.ESS
    def calcKDEAll(self,nBin=1000,a=50): # calculate the kernal density estimate across chains
        #Combine psiLists
        self.psiListAll=self.chainList[0].psiList
        #print('size: {}'.format(self.psiListAll.shape))
        for c in range(1,self.nChain):
            self.psiListAll=np.concatenate((self.psiListAll, self.chainList[c].psiList), axis=0)
            #print('size: {}'.format(self.psiListAll.shape))
        self.KDEListAll=np.empty((self.nPar,2,nBin),dtype=np.double)
        for var in range(0,self.nPar):
            self.KDEListAll[var][0]= np.linspace(min(self.psiListAll[:,var]), max(self.psiListAll[:,var]), nBin)
            dx=self.KDEListAll[var][0][1]-self.KDEListAll[var][0][0]
            self.KDEListAll[var][1] = sum(norm(xi,dx*50).pdf(self.KDEListAll[var][0]) for xi in self.psiListAll[:,var])#smooth out over a of the nBin interval
            tot=sum(np.multiply(self.KDEListAll[var][1],dx))
            self.KDEListAll[var][1]/=tot
    def calcSummaryStats(self,nBin=1000,a=50):
        self.summaryStatAll=[self.chainList[0].Fst,self.chainList[0].DeltaHA,self.chainList[0].DeltaLF]
        for c in range(1,self.nChain):
            self.summaryStatAll=np.concatenate((self.summaryStatAll, [self.chainList[1].Fst,self.chainList[1].DeltaHA,self.chainList[1].DeltaLF]), axis=0)
    def findCIAll(self,var): # find the credible interval of concatonated chains
        [x_d,kde]=self.KDEListAll[var]
        dx=x_d[1]-x_d[0]
        #index = np.lexsort((x_d,kde))
        index = np.lexsort((x_d,kde))[::-1] #sort (from largest to smallest) by kde then by x_d
        self.maxPost=[x_d[index[0]],kde[index[0]]] 
        temp=np.add.accumulate(kde[index]*dx)
        temp2=index[[i for i,v in enumerate(temp) if v < 0.95]] #find pts in credible interval
        indexRev = np.lexsort((kde[temp2],x_d[temp2]))[::-1] #sort (from largest to smallest) by x_d then by kde 
        self.CI=[list(x_d[temp2][indexRev]),list(kde[temp2][indexRev])]
    def kdeHistAll(self,var,ax): # plot hisogram of parameter estimate from an individual chain
        [x_d,kde]=self.KDEListAll[var]
        self.findCIAll(var)
        #plt.fill_between(x_d, density, alpha=0.5) #plot filled smoothed kernel density
        ax.hist(self.psiListAll[:,var].tolist(), bins=30,density=True, color = "skyblue",alpha=0.3) #plot histogram
        ax.plot(x_d,kde,'k') #plot smoothed kernel density line
        ax.scatter(self.CI[0],self.CI[1],color='pink',alpha=1,marker=".") #plot credible interval
        ax.plot([self.maxPost[0],self.maxPost[0]],[0, max(kde)*1.2],'r')# Show maximum postier estiamte
        ax.plot([[beta,gamma,psi][var],[beta,gamma,psi][var]],[0, max(kde)*1.2],'g')# Show true value
        ax.plot(self.psiListAll[:,var], np.full_like(self.psiListAll[:,var],-max(kde)*0.05), '|k', markeredgewidth=1) #plot hashes at bottom
        ax.axis([min(x_d), max(x_d), -2*max(kde)*0.05, max(kde)*1.2]);
        
# chain mixing
biRatio=0.3

#tree1=simTree(testPars)
#tree1.gillespie()
chainList=runChains(3,500)
bay1=baysianEst(chainList,biRatio)

plt.rcParams.update({'font.size': 15})


fig, axs = plt.subplots(bay1.nPar,bay1.nChain)
#fig.suptitle('Stochastic chains')
for v in range(bay1.nPar):
    for c in range(bay1.nChain):
        bay1.chainList[c].plotChain(v,axs[v,c],biRatio)
        axs[0,c].set(title='chain {}'.format(c))
    axs[v,0].set(ylabel=[r'$\beta$',r'$\gamma$',r'$\psi$'][v])
for ax in fig.get_axes():
    ax.label_outer()
fig.set_figheight(7)
fig.set_figwidth(15)
#fig.savefig('Basic.png', dpi=300)

#ESS
bay1=baysianEst(chainList,biRatio)
bay1.calcESS()

# convergence ratio
bay1.calcR()


# plotting the posterior
fig, ax2 = plt.subplots(1,bay1.nPar)
for v in range(bay1.nPar):
    bay1.kdeHistAll(v,ax2[v])
ax2[0].set(ylabel=r'$\beta$')
ax2[1].set(ylabel=r'$\gamma$')
#ax2[2].set(ylabel=r'$\psi$')
#ax2[1].set(ylabel='kappa')
#ax2[4].set(ylabel='sigma')

# ax2[0].set(xlim=(0, 10))
# ax2[1].set(xlim=(0, 10))
fig.set_figheight(5)
fig.set_figwidth(29)
bay1.maxPost
