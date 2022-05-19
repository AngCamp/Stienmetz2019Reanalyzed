# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:26:06 2022

@author: angus
"""
# %%
from distutils.command.build_scripts import first_line_re
import os
import numpy as np
import pandas as pd
from math import ceil
from math import floor
import scipy.ndimage
import timeit #for testing and tracking run times
import scipy.stats 
#os.chdir('C:/Users/angus/Desktop/SteinmetzLab/Analysis')
import getSteinmetz2019data as stein
import warnings
import sklearn

# %%
import KernelRegDraft as kreg

#path on Angus Laptop
# path_to_data = os.fspath(r'C:\Users\angus\Desktop\SteinmetzLab\9598406\spikeAndBehavioralData\allData\')
#for path on the NINC desktop
path_to_data =  os.fspath(r'/home/ninc-user/Desktop/9598406/spikeAndBehavioralData/allData/')
# %%


#As abjit discussed, these papers are relevant
# It would be nice to show trial history effects on current decisions 
"""
    Urai, A. E., de Gee, J. W., & Donner, T. H. (2018). Choice history biases subsequent evidence accumulation. bioRxiv, 251595.
-shows across various modalities that drift bias is a better fit for data than a shifted starting point

Athena Akrami, Charles D. Kopec, Mathew Diamond, and Carlos D. Brody, Posterior parietal cortex represents sensory history and mediates its effects on behaviour. Nature, 2018

Also relevant highly relevant:
    Edward H. Nieh, Manuel Schottdorf, Nicolas W. Freeman, Ryan J. Low,
    Sam Lewallen, Sue Ann Koay, Lucas Pinto, Jeffrey L. Gauthier, Carlos D. Brody
    & David W. Tank, Geometry of abstract learned knowledge in the hippocampus, 
    Nature 2021.

https://www.nature.com/articles/s41586-021-03652-7?WT.ec_id=NATURE-202106&sap-outbound-id=DBB1ED36DA74022A92B31BD53A1EE0017E8112F2

"""
"""
For use_only_these_clsuters chose from the first 10 clsuters passing annotation
 threshold for quality.
 
 These are the fist 10 from Theiler_2017-10-11
 foudn via....
 pathforus = os.fspath(r'C:\Users\angus\Desktop\SteinmetzLab\9598406\spikeAndBehavioralData\allData')

 spikesderp = stein.calldata('Theiler_2017-10-11', ['spikes.clusters.npy',
                          'spikes.times.npy',
                          'clusters._phy_annotation.npy'],
                steinmetzpath= pathforus)
 
 anot = spikes['clusters_phy_annotation']
 clust = spikes['spikesclusters']
 clust = unique(clust)
 test = clust[anot.reshape(clust.shape)>=2]
 
 returning...
 array([ 3,  4,  7,  9, 12, 14, 16, 17, 18, 19])
 
 The trials chosed were the first to contain a no-go, left and right
 animals responses were not factored in
 
 
 Also this may be worth a read:
     Wikle, C. K. (2015). Modern perspectives on statistics for spatio‐temporal data. Wiley Interdisciplinary Reviews: Computational Statistics, 7(1), 86-98.
     
     
"""
# %%
start = timeit.timeit()
#These trials selected because they contain all types of choices, left 2 rights then a no go
# [4,5,6,7]

#test this fucntion out
#note steinmetz mthods uses P and X interchanably so
# I thought ti would be appropriate here

P = kreg.make_toeplitz_matrix(session = 'Theiler_2017-10-11', 
                     bin_size = 0.005, 
                     kernels = [True, True, True],
                     filepath = path_to_data,
                     select_trials=np.array([4,5,6,7])
                     )


end= timeit.timeit()
print(start-end)
# %%

fetched_obj = stein.calldata('Theiler_2017-10-11',
                                    ['wheelMoves.intervals.npy',
                                    'licks.times.npy'],
                                    steinmetzpath = path_to_data)

movesintervals = fetched_obj['wheelMovesintervals']
lickstimes = fetched_obj['lickstimes']

licksintervals = kreg.generate_event_interval(lickstimes, [-0.025,0.025])
print( licksintervals.shape )
print( type(licksintervals) )

# %%
I =  np.row_stack([movesintervals, licksintervals])


# %%

#combine two intervals

#first asses which of the two intervals should be used  first,
# then write a function that you can call 
#setting input values for function
intervals_x = movesintervals
intervals_y = licksintervals
# pandas does interval operations may be able to do this fast using it 
# https://pandas.pydata.org/docs/reference/api/pandas.arrays.IntervalArray.html
## this package may work https://piso.readthedocs.io/en/latest/
# make the intervals disjoint intervals then just take the union of them
 # interval index from arrays https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.IntervalIndex.from_arrays.html#pandas.IntervalIndex.from_arrays
 # then we 

# bind the two arrays, turn them into an index, then run piso.union
import piso



t = 0 #overall counter to check if there are itnervals to include
i = 0 # counter for X intervals
j = 0 # counter for Y intervals
in_bounds = True # variable that assesess if we have reached end of
# our intervals
while in_bounds:
    in_bounds = (t =< len(interval_x) ) | (t =< len(interval_y) ) 
    t = t + 1

    #another while loop?  until no more overlaps?

    neighbourhood_X = intervals_x[i, :]

    neighbourhood_Y = intervals_y[j, :]

    




if intervals_x[0,0] < intervals_y[0,0]:
    earliest_time = intervals_x[0,0]
else:
    earliest_time = intervals_y[0,0]

for i in intervals_x[:,0]:
    i > earliest_time


np.argmin(np.where(intervals_y == earliest_time))

        #y first
#need a funciton that just checks within the interval until it
# finds an end value that is not within the interval


# %%

start = timeit.timeit()
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

pathforus = os.fspath(r'C:\Users\angus\Desktop\SteinmetzLab\9598406\spikeAndBehavioralData\allData')

trialstest = stein.calldata('Theiler_2017-10-11', ['trials.intervals.npy',
                                      'trials.included.npy'],
                steinmetzpath=pathforus)

#select_these = np.array([4,5,6,7])
select_these = []

if len(select_these)!=0:
    trialsincludedtest = select_these
elif True:   #filter by engagement
    trialsincludedtest = trialstest['trialsincluded']

[ i for i in range(0,len(trialsincluded)) if trialsincluded[i]]

trialsintervalstest = trialstest['trialsintervals']
trialsintervalstest = trialsintervalstest[trialsincludedtest,:]

trialsintervalstest = trialstest['trialsintervals']
trialsintervalstest = trialsintervalstest[trialsincludedtest.reshape(trialsintervalstest.shape),:]


#again with more cl"usters, 
"""
Fixed the last error but now it's printing out the clusters for some weird reason'


"""
start = timeit.timeit()
# only use these clusters includes first 10 clusters in clusters_idx that pass quality
Y, clusters_index = kreg.frequency_array(session = 'Theiler_2017-10-11', 
                                    bin_size = 0.005, 
                                    only_use_these_clusters=[ 3,  4,  7,  9, 12, 14, 16, 17, 18, 19]
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

#could use make regression to simulate data
#X, y = make_regression(n_features=2, random_state=0)

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