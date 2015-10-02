# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:39:19 2015

@author: Edward
"""

# Project 2 FMNN25

import numpy as np
import scipy as sp

class optimizationProblem(object):
    
    def __init__(self,function, dimensions,tolerance, gradient = None):
        self.tol = tolerance
        self.dx = 0.000001
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
    
    def step(self,x0):
        x=x0
        while True:
            g=self.g(x)
            H=self.hessian(x)
            print(H)
            U = np.linalg.cholesky(H)
            Uinv = np.linalg.inv(U)
            Hinv = np.dot(Uinv,Uinv.T)
            sK= -1*np.dot(Hinv,g)
            alpha= self.lineSearch(x,sK)
            print(alpha)
            x= x+alpha*sK
            print(alpha*sK)
            if abs(np.linalg.norm(alpha*sK)) < self.tol:
                return x

    def lineSearch(self,xk,sK):
        alpha = np.linspace(0.,10**4,10**5)
        
        f_alpha = np.array([self.f(xk+alpha[i]*sK) for i in range(len(alpha))])
        alpha_k = alpha[np.argmin(f_alpha)]
        
        return alpha_k

class inexactNewton(Newton):
    
    def lineSearch(self,xk,sK):
        def f_alpha(alpha):
            return self.f(xk+alpha*sK)
        def f_prime_alpha(alpha):
            return (f_alpha(alpha+self.dx)-f_alpha(alpha))/self.dx
        alpha0 = 0
        alphaL = 0
        alphaU = 10**99
        self.rho = 0.1
        self.sigma = 0.7
        self.tau  =0.1
        self.chi = 9.
        while not (LC and RC):
            if not LC:
                [alpha0,alphaL] = self.notLC(alpha0,alphaL,f_prime_alpha(alpha0),f_prime_alpha(alphaL))
                
            else:
                [alpha0,alphaU] = self.notRC(alpha0,alphaU,alphaL,f_alpha(alphaL),f_alpha(alpha0),f_prime_alpha(alphaL))
            
        
    def notLC(self,alpha0,alphaL,f_prime_a0, f_prime_aL):
        deltaAlpha = (alpha0-alphaL)*(f_prime_a0/(f_prime_aL-f_prime_a0))
        deltaAlpha = np.max([deltaAlpha, self.tau*(alpha0-alphaL)])
        deltaAlpha = np.min([deltaAlpha, self.chi*(alpha0-alphaL)])
        alphaL = alpha0
        alpha0 = alpha0 + deltaAlpha
        return np.array([alpha0, alphaL])
        
    def notRC(self,alpha0,alphaU,alphaL,f_alphaL,f_alpha0,f_prime_aL):
        alphaU = np.min([alpha0,alphaU])
        alphaBar = (alpha0-alphaL)**2*f_prime_aL/(2*(f_alphaL-f_alpha0 + (alpha0-alphaL)*f_prime_aL))
        alphaBar = np.max([alphaBar,alphaL + self.tau*(alphaU-alphaL)])
        alphaBar = np.min([alphaBar,alphaU - self.tau*(alphaU-alphaL)])
        return np.array([alphaBar,alphaU])

def main():
    def f(x):
        return x[1]**2 + x[0]**2
        return 100*(x[1]-x[0]**2)**2 + (1 -x[0])**2
        
    opt = Newton(f,2,0.001)
    x = opt.step([500,5.])
    print(f(x))
    print(x)
    
main()