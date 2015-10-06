# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:39:19 2015

@author: Edward
"""

# Project 2 FMNN25

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
class optimizationProblem(object):
    
    def __init__(self,function, dimensions,tolerance, gradient = None):
        self.tol = tolerance
        self.dx = 0.0000001
        self.f = function
        self.dimensions = dimensions
        if gradient:
            self.g = gradient
        else:
            def grad(x):
                return np.array([(self.f(x+self.dx*delta[i])-self.f(x))/self.dx for i in range(dimensions)])
            delta = sp.identity(self.dimensions)
            self.g = grad      
        self.hessian = self.computeHessian()
        
    def computeHessian(self):
        def hess(x):
            G = np.zeros((self.dimensions,self.dimensions))
            for row in range(self.dimensions):
                for col in range(row,self.dimensions):
                    G[row,col] = self.secondDerivative(row,col,x)
                    if col != row:
                        G[col,row] = G[row,col]
            return G
        return hess
        
    def secondDerivative(self,i,j,x):
        deltax = np.zeros(self.dimensions)
        deltay = np.zeros(self.dimensions)
        deltax[i] = self.dx
        deltay[j] = self.dx
        return (self.f(x+deltax + deltay)-self.f(x + deltax) -self.f(x + deltay) + self.f(x))/self.dx**2


class Newton(optimizationProblem):
    
    def step(self,x):
        self.Hinv = np.eye(self.dimensions)
       # U = np.linalg.cholesky(H)
        #Uinv = np.linalg.inv(U)
        #self.Hinv = np.dot(Uinv,Uinv.T)
        while True:
            g=self.g(x)
            sK= -1*np.dot(self.Hinv,g)
            alpha= self.inexactLineSearch(x,sK)
            self.deltaX = alpha*sK
            x= x+self.deltaX
            if abs(np.linalg.norm(self.deltaX)) < self.tol:
                return x
            self.update(x)
            self.posDefCheck()
            #if k > 200:
             #   return x
            #k+=1
    def update(self, x):
        H=self.hessian(x)
        U = np.linalg.cholesky(H)
        Uinv = np.linalg.inv(U)
        self.Hinv = np.dot(Uinv,Uinv.T)
    
    def posDefCheck(self):
        try:
            np.linalg.cholesky(self.Hinv)
        except np.linalg.LinAlgError:
            self.Hinv = np.eye(self.dimensions)
        
                
    def exactLineSearch(self,xk,sK):
        alpha = np.linspace(0.,10**4,10**5)
        
        f_alpha = np.array([self.f(xk+alpha[i]*sK) for i in range(len(alpha))])
        alpha_k = alpha[np.argmin(f_alpha)]
        
        return alpha_k

    
    def inexactLineSearch(self,xk,sK):

        self.rho = 0.1
        self.sigma = 0.7
        self.tau  =0.1
        self.chi = 9.
        self.alpha0 = sK.dot(sK)/sK.dot(self.Hinv.dot(sK))
        self.alphaL = 0.
        self.alphaU = 10**99
        self.computeValues(xk, sK)
        while not (self.LC() and self.RC()):
            if not self.LC():
                self.block1() 
            else:
                self.block2()
            self.computeValues(xk, sK)
            
        return self.alpha0
        
        
    def computeValues(self, xk, sK):
        def f_alpha(alpha):
            return self.f(xk+alpha*sK)
        def f_prime_alpha(alpha):
            return (f_alpha(alpha+self.dx)-f_alpha(alpha))/self.dx
        self.f_alphaL = f_alpha(self.alphaL)
        self.f_alpha0 = f_alpha(self.alpha0)
        self.f_prime_aL = f_prime_alpha(self.alphaL)
        self.f_prime_a0 = f_prime_alpha(self.alpha0)
    
    #Compute the left and right conditions according to Goldstein
    def LC(self):
        return self.f_alpha0 >= (self.f_alphaL + (1-self.rho)*(self.alpha0 - self.alphaL)*self.f_prime_aL)
    def RC(self):
        return self.f_alpha0 <= (self.f_alphaL + self.rho*(self.alpha0 - self.alphaL)*self.f_prime_aL)
#==============================================================================
# 
#      #Compute the left and right conditions according to Wolfe-Powel
#     def LC(self):
#          return self.f_prime_a0 >= self.sigma*self.f_prime_aL
#     def RC(self):
#          return self.f_alpha0 <= self.f_alphaL + self.rho*(self.alpha0 - self.alphaL)*self.f_prime_aL
# 
#         
#==============================================================================
    
    def block1(self):
        if not (self.f_prime_aL-self.f_prime_a0):
            deltaAlpha = 0
        else:
            deltaAlpha = (self.alpha0-self.alphaL)*(self.f_prime_a0/(self.f_prime_aL-self.f_prime_a0))
        deltaAlpha = np.max([deltaAlpha, self.tau*(self.alpha0-self.alphaL)])
        deltaAlpha = np.min([deltaAlpha, self.chi*(self.alpha0-self.alphaL)])
        self.alphaL = self.alpha0
        self.alpha0 = self.alpha0 + deltaAlpha
    def block2(self):
        self.alphaU = np.min([self.alpha0,self.alphaU])
        if not (self.f_alphaL-self.f_alpha0 + (self.alpha0-self.alphaL)*self.f_prime_aL):
            alphaBar = 0
        else:
            alphaBar = (self.alpha0-self.alphaL)**2*self.f_prime_aL/(2*(self.f_alphaL-self.f_alpha0 + (self.alpha0-self.alphaL)*self.f_prime_aL))
        alphaBar = np.max([alphaBar,self.alphaL + self.tau*(self.alphaU-self.alphaL)])
        alphaBar = np.min([alphaBar,self.alphaU - self.tau*(self.alphaU-self.alphaL)])
        self.alpha0 = alphaBar
     

class goodBroyden(Newton):
    
    def update(self,x):
        delta = self.deltaX
        gamma = self.g(x)-self.g(x-delta)
        u = delta - self.Hinv.dot(gamma)
        uuT = np.outer(u,u.T)
        self.Hinv = self.Hinv + uuT/(u.dot(gamma))
        
class badBroyden(Newton):
    def update(self, x):
        gamma = self.g(x)-self.g(x-self.deltaX)
        delta = self.deltaX
        dtd = delta.dot(delta)
        Hd = self.Hinv.dot(delta)
        self.Hinv = self.Hinv + np.dot((gamma - Hd)/dtd,delta)
class DFP(Newton):
    def update(self,x):
        delta = self.deltaX
        gamma = self.g(x)-self.g(x-delta)
        ddT = np.outer(delta,delta)
        dtg = delta.T.dot(gamma)
        Hg = self.Hinv.dot(gamma)
        gtH = gamma.T.dot(self.Hinv)
        self.Hinv = self.Hinv + ddT/dtg - (np.outer(Hg,Hg.T))/(gtH.dot(gamma))
               
class BFGS(Newton):
    
    def update(self,x):
        delta = self.deltaX
        gamma = (self.g(x)-self.g(x-delta))
        gth = gamma.dot(self.Hinv)
        dtg = delta.T.dot(gamma)
        ddT = np.outer(delta,delta)
        hg = self.Hinv.dot(gamma)
        hgd = np.outer(hg,delta)
        self.Hinv = self.Hinv + (1 + gth.dot(gamma)/dtg)*(ddT/dtg) - (hgd + hgd.T)/dtg
def main():
    def f(x):
        #return x[1]**2 + x[0]**2
        return x[0]**2 + x[1]**2 + x[2]**2
        return 100*(x[1]-x[0]**2)**2 + (1 -x[0])**2
    opt = badBroyden(f,3,10**-9)
    x = opt.step([1,1,1])
    print(f(x))
    print(x)
    
main()