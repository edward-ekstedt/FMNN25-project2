# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:39:19 2015

@author: Edward
"""

# Project 2 FMNN25

import numpy as np


class optimizationProblem(object):
    
    def __init__(self,function, gradient = None):
        self.f = function
        if not gradient == None:
            self.g = gradient
        else:
            self.g = 
class optimizationMethod(object):
    
    def __init__(self,optimizationProblem,x0 = None):
        self.op = optimizationProblem
        if not x0 == None:
            self.x0 = x0
        else:
            x0 = 0 #change to achieve correct dimensions
            

            
