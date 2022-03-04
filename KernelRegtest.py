# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:26:06 2022

@author: angus
"""

import os
import numpy as np
import pandas as pd
from math import ceil
from math import floor
import scipy.ndimage
import timeit #for testing and tracking run times
import scipy.stats 
os.chdir('C:/Users/angus/Desktop/SteinmetzLab/Analysis')
import getSteinmetz2019data as stein
import warnings
import KernelRegDraft as kreg

start = timeit.timeit()
#These trials selected because they contain all types of choices, left 2 rights then a no go
# [4,5,6,7]

#test this fucntion out
#note steinmetz mthods uses P and X interchanably so
# I thought ti would be appropriate here

P = kreg.make_toeplitz_matrix(session = 'Theiler_2017-10-11', 
                     bin_size = 0.005, 
                     kernels = [True, True, True],
                     select_trials=np.array([4,5,6,7])
                     )


end= timeit.timeit()
print(start-end)

import KernelRegDraft as kreg
start = timeit.timeit()
# only use these clusters includes first 10 clusters in clusters_idx that pass quality
Y, clusters_index = kreg.frequency_array(session = 'Theiler_2017-10-11', 
                                    bin_size = 0.005, 
                                    only_use_these_clusters=[ 3,  4, 7],
                                    select_trials = np.array([4,5,6,7])
                                    )
end= timeit.timeit()
print(start-end)


import seaborn as sns

sns.heatmap(P)