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
import sklearn

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


### Making b
# CCA between P and Y to get b
sklearn.cross_decomposition.CCA(n_components=2, *, 
                                scale=True, 
                                max_iter=500, 
                                tol=1e-06, 
                                copy=True)
#Test
from sklearn.cross_decomposition import CCA
Xtest = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
Ytest = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
cca = CCA(n_components=2)
cca.fit(Xtest, Ytest)
CCA(n_components=2)
X_c, Y_c = cca.transform(Xtest, Ytest)


# run the regression
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression

Ytest = np.array(Ytest)
for n in range(0, Ytest.shape[0]):
    print(n)
    y = Ytest[n,:]
    #X_c, y = make_regression(n_features=2, random_state=0)
    regr = ElasticNetCV(cv=5, random_state=0)
    regr.fit(X_c, y)




#https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html


import seaborn as sns

sns.heatmap(P)