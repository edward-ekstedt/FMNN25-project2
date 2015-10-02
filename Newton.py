# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:39:19 2015

@author: Edward
"""

# Project 2 FMNN25

class optimizationProblem(object):
    
    def __init__(self,function, gradient = None):
        self.f = function
        if not gradient == None:
            self.g = gradient

class optimizationMethod(object):
    
    def __init__(self,optimizationProblem):
        