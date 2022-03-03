# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 20:08:42 2022

@author: angus
"""

class test:
    def __init__(self, x):
        self.x = x

class testing:
    def __init__(self, x):
        self.x = x
     

class Outer:
    """Outer Class"""

    def __init__(self):
        ## Instantiating the 'Inner' class
        self.inner = self.Inner()
        ## Instantiating the '_Inner' class
        self._inner = self._Inner()

    def show_classes(self):
        print("This is Outer class")
        print(inner)
        print(_inner)

    class Inner:
        """First Inner Class"""

        def inner_display(self, msg):
            print("This is Inner class")
            print(msg)

    class _Inner:
        """Second Inner Class"""

        def inner_display(self, msg):
            print("This is _Inner class")
            print(msg)

      
## instantiating the classes

## 'Outer' class
outer = Outer()
## 'Inner' class
inner = outer.Inner() ## inner = outer.inner or inner = Outer().Inner()
## '_Inner' class
_inner = outer._Inner() ## _inner = outer._outer or _inner = Outer()._Inner()

## calling the methods
outer.show_classes()

print()

## 'Inner' class
inner.inner_display("Just Print It!")

print()

## '_Inner' class
_inner.inner_display("Just Show It!")


"""
Results of the above
This is Outer class
<__main__.Outer.Inner object at 0x0000021B37962048>
<__main__.Outer._Inner object at 0x0000021B37962160>

This is Inner class
Just Print It!

This is _Inner class
Just Show It!
"""

import numpy as np
import pandas as pd
import os
import warnings
import re

os.chdir('C:/Users/angus/Desktop/SteinmetzLab/Analysis')
import getSteinmetz2019data as stein


class channels:
    #class to store .npy files laoded from the clusters
    def __init__(self, session_path, brainLocation,
                 probe, rawRow):
        self.session_path = x # the path to the session
        self.brainLocation = "Not loaded, use attribute.get() to load"
        self.probe = "Not loaded, use attribute.get() to load"
        self.rawRow = "Not loaded, use attribute.get() to load"
        

from functools import lru_cache, partial
from glob import glob

from os.path import join, basename

SESSION_DIR = r'C:\Users\angus\Desktop\SteinmetzLab\9598406\spikeAndBehavioralData\allData'

to_key = basename

def to_key(key, prefix):
    """
    Sets the individual objects eg trials, clusters etc,
    to a spcific key for being called in the get_all_data method
    for calling all the session info at once.    
    """
    key = basename(key)
    key = key[len(prefix)+1:]
    key, ext = os.path.splitext(key)
    return key

class Session:
    def __init__(self, name):
        self.name = name
        self.path = join(SESSION_DIR, name)
        if not os.path.exists(self.path):
            raise ValueError('The directory ' + self.path + ' does not exist.')
        
    def get_data(self, key):
        return np.load(join(self.path, key))

    def get_all_data(self, key_prefix):
        #consider having and if function here to check for .tsvs and adding a way to laod those
        return {to_key(f, key_prefix) : self.get_data(f) for f in glob(join(self.path, key_prefix + '.*.npy'))}
    
    @property
    @lru_cache(maxsize=None)
    def trials(self):
        return self.get_all_data('trials')
    
    @property
    @lru_cache(maxsize=None)
    def channels(self):
        return self.get_all_data('channels')
    
    @property
    @lru_cache(maxsize=None)
    def clusters(self):
        return self.get_all_data('clusters')




            