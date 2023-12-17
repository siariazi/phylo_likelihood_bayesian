#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 22:35:56 2023

@author: siavashriazi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 23:03:03 2023

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
import csv

# as an example from a run of 230630_simTree
xVec = np.array([ 1.246,  3.242, 15.677,  6.034,  1.859,  5.76 ,  2.695,  6.015,
        6.353, 13.739, 16.384, 20.625, 37.075,  4.075, 10.918,  9.451,
       10.074,  6.405, 11.417, 20.91 , 10.201, 25.976, 18.438, 10.715,
        8.59 , 12.179, 12.028, 12.735, 12.122, 13.079, 13.914, 25.907,
       15.503, 22.413,  6.985, 17.201,  7.517, 14.59 , 13.113, 16.709,
       32.938, 16.488, 15.981, 17.606, 24.293, 33.73 , 15.506, 23.474,
       29.277, 27.647])

yVec = np.array([17.067,  8.051, 12.112, 29.95 , 44.603, 62.555, 13.201, 10.615,
       67.183, 25.912, 13.985, 14.398, 18.341, 14.323, 16.221, 42.444,
       31.44 , 32.525, 33.094, 34.147, 36.617, 37.94 , 20.519, 49.861,
       36.25 , 20.1  , 31.424, 54.26 , 22.273, 47.795, 48.17 , 31.141,
       40.109, 23.4  , 61.549, 41.354, 76.738, 26.612, 49.929, 42.3  ,
       27.939, 40.519, 47.731, 37.925, 37.85 , 34.442, 35.529, 43.164,
       41.4  , 43.697, 85.095, 46.88 , 86.301])

numPar = 2
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
#testPars=[0.5,0.05,0.01,0.001,100,3,20]#beta,gamma,psi,sigma,kappa,i0,T
testPars=[0.25,0.05,0.01,0.001,300,3,100]#beta,gamma,psi,sigma,kappa,i0,T according to Mathematica
#testPars=[0.25,0.05,0.01,0.001,1000,3,100]#beta,gamma,psi,sigma,kappa,i0,T change to previous to make big tree in short time

beta, gamma, psi, sigma, kappa, i0, T = testPars
testInits=[kappa-i0,i0,0]

class like():
    def __init__(self,parsL,xVec,yVec):
        #Parameters
        ##Epi parameters
        self.beta = parsL[0]
        self.gamma = parsL[1]
        self.psi = parsL[2]
        self.sigma = parsL[3]
        self.kappa = parsL[4]
        self.i0 = parsL[5]
        self.T = parsL[6]
  
        ##Diversification parameters
        self.SIR=sir_rk4(parsL,[self.kappa-self.i0,self.i0,0],self.T*3)
        self.tauVec=self.T-self.SIR[:,0] # backward time
        self.newTau = self.SIR[:,0] # a vector of time to be attached to E vector for plotting 
        #data
        self.xVec=(self.T-xVec)#x BACKWARD in time 
        self.yVec=(self.T-yVec)#y BACKWARD in time
        self.nStep = 30
        self.inits = [1] # initial condition of E
        
        
        # attaching time to eVec
        
        # interpolation of S(t) and E(t)
        self.sValue = interp1d(self.SIR[:,0], self.SIR[:,1], kind = 'cubic')
        #print(self.sValue(10))
        
    # calcE() calculates dEdt
    def calcE(self,state,t):
        
        E=state[0]

        #S = float(self.SIR[t,1]) # t is the index of time not the value of time

        
        self.dEdt = -(self.lamda(t) + self.gamma + self.psi)*E + self.lamda(t)*E**2 + self.gamma  
        #print("E is:",E)
        #print("S is:",S)
        #print("dEdt is:",self.dEdt)
    # E_euler integrate dEdt
    def E_euler(self,time):
        h = self.tauVec[0] - self.tauVec[1]
        #print("h is:",h)
        self.out=np.array(self.inits)
        temp=self.out
        for i in self.tauVec[1:]:
            # calculating E bakcward
            #print("temp is:",temp)
            #print("time of integ:",i)
            #print("value of S at t",self.sValue(i))
            self.calcE(temp,i)   
            #print("dE is:",self.dEdt)
            temp=temp+self.dEdt*h
            self.out=np.vstack((self.out,temp))
       
    def lamda(self,t):
        lamda = self.sValue(t)*self.beta/self.kappa
        #print("lamda is:",lamda)
        return lamda
    # phi() A returns phi integrad
    # this function should have a return value because we are integrating it over integration interval in quad()   
    def phiA(self,t):
        phiA =  2*self.lamda(t)*self.eValue(t) - (self.lamda(t) + self.gamma + self.psi)
        #print("phiA is:",phiA)
        return phiA
    # likelihood() calls other functions in the class and calculates the (log)likelihood
    def likelihood(self):
        # caclulating dEdt
        self.E_euler(self.tauVec) 
        self.eVec = np.vstack((self.newTau,np.squeeze(self.out)))

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
 

like1=like(testPars,xVec,yVec)
plt.plot(like1.SIR[:,0],like1.SIR[:,3]) # plotting to check
like1.likelihood()
like1.g2 # loglikelihood
plt.plot(like1.eVec[0,:],like1.eVec[1,:]) # plotting E(tau)


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

jmpRatio=20
betaDict={"min":0.01,"max":1.0,"jmpVar":10.0/jmpRatio} # real beta is 0.25
gammaDict={"min":0.01,"max":2,"jmpVar":10.0/jmpRatio} 


class Theta:
    def __init__(self):
       global jmpRatio, betaDict, gammaDict, psi, sigma, kappa, i0, T
       
       self.cnt=0
       
       # use global variables
       self.betaDict = betaDict
       self.gammaDict = gammaDict

       self.psi = psi
       self.sigma = sigma
       self.kappa = kappa
       self.i0 = i0
       self.T = T

    def randTheta(self):
        self.beta=rnd.uniform(self.betaDict["min"],self.betaDict["max"])  
        self.gamma=rnd.uniform(self.gammaDict["min"],self.gammaDict["max"])  
        #self.psi=rnd.uniform(self.psiDict["min"],self.psiDict["max"])  
        #self.T = 20 # I'm not sure should I pass T as a parm or constant
        #self.kappa=rnd.uniform(self.kappaDict["min"],self.kappaDict["max"])  
        #self.sigma=rnd.uniform(self.sigmaDict["min"],self.sigmaDict["max"])  
        
        parsL1 = [self.beta,self.gamma,self.psi,self.sigma,self.kappa,self.i0,self.T]
        #print(parsL1)
        like1 = like(parsL1,xVec,yVec)
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
        like1 = like(parsL1,xVec,yVec)
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

#tree1=simTree(testPars)
#tree1.gillespie()
#np.shape(tree1.sampTree)
like1=like(testPars,xVec,yVec)
like1.likelihood()
like1.g2      
Theta1=Theta()
Theta1.randTheta()
Theta1.printTheta('Theta 1')


# Changing the current working directory
os.chdir('/Users/siavashriazi/Desktop/SFU/Codes/Python/chains')
# Open the CSV file in write mode
with open('chain.csv', 'w', newline='') as file1:
    # Create a CSV writer object
    writer1 = csv.writer(file1)
    
    # Write the column names
    writer1.writerow(['beta', 'gamma'])
    
def MHChain(nRep): #simulate chain using the Metropolis-Hastings Algorithm
        global jmpRatio
        # this is the initial theta 
        initTheta=Theta()
        initTheta.randTheta()
        
        while initTheta.lnLik is None: # if calculated likelihood is nan 
            initTheta = Theta()  # Propose a new Theta
            initTheta.randTheta()
            print("here")
        t=1 # counter of the reps
        acc=0 # conter of accepted jumps
        fail=0 # conter of rejected jumps
        #jmpRatio=20 # jump Ratio, scales the variance in jump distn. 
        while t<nRep: 
            propTheta=Theta() #propose a theta
            propTheta.jump(initTheta)
            temp=rnd.random()
            #if temp<propTheta.r: #accept
            if temp<math.exp(propTheta.lnr): #accept
                if (t+1)%(min(nRep/2,50))==0: #priting out progress
                    if((acc/(acc+fail))>0.3): # accepting too many,  
                        jmpRatio/=2  #searches more globallly
                        print('t: {}, accept ratio {}, new jmp Ratio: {}'.format(t+1,acc/(acc+fail),jmpRatio))
                        #print('beta: ',propTheta.beta,' gamma: ',propTheta.gamma)
                    elif((acc/(acc+fail))<0.1):
                        jmpRatio*=2 #searches more locally
                        print('t: {}, accept ratio {}, new jmp Ratio: {}'.format(t+1,acc/(acc+fail),jmpRatio))
                    else:
                        print('t: {}, accept ratio {}'.format(t+1,acc/(acc+fail)))
                    acc = 0
                    fail = 0
                with open('chain.csv', 'a', newline='') as file1:
                    # Create a CSV writer object
                    writer1 = csv.writer(file1)
                    # Write the column names
                    writer1.writerow([propTheta.beta, propTheta.gamma])
                t+=1
                acc+=1
#                 print('accept r={} random={}'.format(propTheta.r,temp))
            else:
                fail+=1
                # print('reject r={} random={}'.format(propTheta.r,temp))    


nRep=3000
MHChain(nRep)


# start with one parameter 
# can we see the likelihood back
# change to Markov Chain with a uniform 
# run single job:
    # sbatch file.sl
# array job
# sbatch --array = 1-10 file.sl 
        