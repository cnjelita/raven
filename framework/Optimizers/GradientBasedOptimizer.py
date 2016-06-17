"""
  This module contains the Gradient Based Optimization sampling strategy

  Created on June 16, 2016
  @author: alfoa
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
from collections import OrderedDict
import sys
import os
import copy
import abc
import numpy as np
import json
from operator import mul,itemgetter
from collections import OrderedDict
from functools import reduce
from scipy import spatial
from scipy.interpolate import InterpolatedUnivariateSpline
import xml.etree.ElementTree as ET
import itertools
from math import ceil
from collections import OrderedDict
from sklearn import neighbors
from sklearn.utils.extmath import cartesian

if sys.version_info.major > 2: import pickle
else: import cPickle as pickle
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Optimizer import Optimizer
# from Assembler import Assembler
# import Distributions
# import DataObjects
# import TreeStructure as ETS
# import SupervisedLearning
# import pyDOE as doe
# import Quadratures
# import OrthoPolynomials
# import IndexSets
# import Models
# import PostProcessors
# import MessageHandler
# import GridEntities
from AMSC_Object import AMSC_Object
#Internal Modules End--------------------------------------------------------------------------------

class GradientBasedOptimizer(Optimizer):    
  def evaluateGradient(self, optVarsValues):
    """
    For explicit gradient case, optVarsValues are the point value to be evaluated;
    For implicit gradient case, optVarsValues are the perturbed parameter values for gradient estimation
    """
    gradient = self.localEvaluateGradient(optVarsValues)
    return gradient
      
  @abc.abstractmethod
  def localEvaluateGradient(self, optVars, gradient):
    return gradient
    
    
    
    






