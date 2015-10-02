# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:39:19 2015

@author: Edward
"""

# Project 2 FMNN25

import numpy as np
import scipy as sp

class optimizationProblem(object):
    
    def __init__(self,function, dimensions, gradient = None):
        self.dx = 0.0000000001
        self.f = function
        self.dimensions
        if gradient:
            self.g = gradient
        else:
            def grad(x):
                return np.array([(self.f(x+self.dx*delta[i]/2.)-self.f(x-self.dx*delta[i])/2.)/self.dx] for i in range(dimensions))
            delta = sp.identity(self.dimensions)
            self.g = grad
        
        self.hessian = self.computeHessian()
        
    def computeHessian(self):
        def hess(x):
            G = np.zeros((self.dimensions,self.dimensions))
            for row in range(self.dimensions):
                for col in range(self.dimensions):
                    G[row,col] = self.secondDerivative(row,col,x)
            return G
        return hess
        
    def secondDerivative(self,i,j,x):
        deltax = np.zeros(self.dimensions)
        deltay = np.zeros(self.dimensions)
        deltax[i] = self.dx
        deltay[j] = self.dx
        return (self.f(x+deltax + deltay)-self.f(x + deltax) -self.f(x + deltay) + self.f(x))/self.dx**2
        

class Newton(object):
    
    def __init__(self,optimizationProblem,x0 = None):
        self.op = optimizationProblem
        if not x0 == None:
            self.x0 = x0
        else:
            x0 = 0 #change to achieve correct dimensions
            
    def step(f, x0, gradient, tolerance):
        x=x0
        while true:
            g=gradient(x)
            H=hessian(x)
            Hinv=np.linalg.inv(H)
            d=-np.multiply(Hinv,g)
            alpha= findAlpha(f,x, d)
            x+=alpha*d
            if abs(np.linalg.norm(alpha*d)) < tolerance:
                return x


    def exactLineSearch(self,xk):
        
        alpha = linspace()
        
        f_alpha = np.array([self.op.f(xk-alpha[i]*sk for i in range(len(alpha))])
        alpha_k = alpha[np.argmin(f_alpha)]
        
        return alpha_k

