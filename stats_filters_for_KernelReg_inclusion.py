# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 23:08:38 2022

@author: angus

YET TO BE RUN TAKES A VERY LONG TIME, MAY NEED TO BE CHANGED TO REPORT WHICH TESS ARE
BEING PASSED

Please Note all functions here assume all times tested will be within trial
        intervals will need some reworking if we want to use non-trial events as well.
        Lines 60,  61,  58, 151, 152, 207 all use trials.included.npy to
        filter various data, so these would need to be changed.  Be aware of other spots as well
        and double check code, usually it will beocme appernte when you run this as you
        will get an index mismatch error or something at some point but be careful.
        
Ignore the unexpected indent in spyder, it just doesnt like stein.calldata

NOTE:  Also currently the code skips neurons that pass one of the tests and does not demarcate
which test it passed simply outputs a boolean saying which test was passed.

Currently stats_filter() is only running the bare-bones first four stats tests
at a threshold of p<0.05, instead of the bonferonni one (0.05/6) with 6 tests which
is used to select neurons for visualizing.  I can easily implement that though.


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


def stats_filter(session, datapath, threshold):    
    """
        Here we test for the first 4 critieria used in the publication, basically
        if a neurons passes these at a threshold of 0.05.  Despite doing 4 tests
        a neurons transientyl firing would be excluded so this threshold was
        chosen instead.
        
        We autopass neurons that passed an earlier test.
        
        According to Steinmetz et al. (2019) neurons were further tested before
        inclusion in the kernel regression...
        '...a set of six statistical tests were used to detect changes in activity 
        during various task epochs and conditions: 
        """
        
        fetched_objects =stein.calldata(recording = session,
                                        list_of_data = ['spikes.clusters.npy',
                                                        'spikes.times.npy',
                                                        'clusters._phy_annotation.npy',
                                                        'trials.visualStim_times.npy',
                                                        'trials.intervals.npy',
                                                        'trials.included.npy',
                                                        'trials.response_times.npy',
                                                        'trials.response_choice.npy'],
                                        steinmetzpath = datapath)
        
        #Filtering by quality first
        spikesclusters = fetched_objects['spikesclusters'] #the idneity in sequence of 
        #each cluster, match it with spikestimes to get timing and identity info
        spikestimes = fetched_objects['spikestimes'] #times corresponding to clusters firing
        clusterquality = fetched_objects['clusters_phy_annotation'] #quality rating of clsuters
        clusters_idx = np.arange(0, len(clusterquality)).reshape(clusterquality.shape)
        
        #Getting spike identiy and times and filtering out low quality
        good_clusters = clusters_idx[clusterquality >= 2] 
        cluster_mask = np.isin(spikesclusters, good_clusters) #boolean mask
        spikestimes = spikestimes[cluster_mask] 
        spikesclusters = spikesclusters[cluster_mask]
        clusters_idx = np.unique(spikesclusters)
        
        #trials to be included 
        trialsintervals = fetched_objects["trialsintervals"]
        #wheter or not a trial was included based on engagement, logical 
        trialsincluded = fetched_objects["trialsincluded"]
        #filter trialsintervals by trialsincluded reshape prevents indexing error
        trialsintervals = trialsintervals[trialsincluded.reshape(trialsintervals.shape[0]),:]
        
        """
            (1) Wilcoxon sign-rank test between trial firing rate (rate of 
        spikes between stimulus onset and 400 ms post-stimulus) and baseline
        rate (defined in period −0.2 to 0 s relative to stimulus onset on each 
        trial); 
        """
        stimOn = fetched_objects['trialsvisualStim_times']
        stimOn = stimOn[trialsincluded]
        
        stats_filter = np.zeros((1,len(clusters_idx)), dtype = bool)
        
        pvals = []
        for cluster in clusters_idx:
            baseline = []
            trialrate = []
            this_clusters_spikes = spikestimes[np.isin(spikesclusters, cluster)]
            for trial in range(0, trialsintervals.shape[0]):
                #first we make the baserate
                begin = stimOn[trial] - 0.2
                end = stimOn[trial]
                rate = sum(np.logical_and(this_clusters_spikes>=begin, this_clusters_spikes<=end))
                rate= rate/(begin-end)
                baseline.append(rate)
                
                #now we do the stimulus onset rate
                begin = stimOn[trial]
                end = stimOn[trial] + 0.4
                rate = sum(np.logical_and(this_clusters_spikes>=begin, this_clusters_spikes<=end))
                rate= rate/(begin-end)
                trialrate.append(rate)
            #end of trial for loop
            if sum(trialrate+baseline)==0:
                pvals.append(1)
            else:
                pvals.append(scipy.stats.wilcoxon(x=baseline,y = trialrate)[1])
        #end of cluster for loop
                
        passed_tests = np.array(pvals)<0.05

        """
            (2) sign-rank test between stimulus-driven rate (firing rate 
        between 0.05 and 0.15 s after stimulus onset) and baseline rate; 
        """
        #this chunk runs fine
        
        i = 0
        pvals = []
        for i in range(0, len(clusters_idx)):
            cluster = clusters_idx[i]
            this_clusters_spikes = spikestimes[np.isin(spikesclusters, cluster)]
            if passed_tests[i]:
                pvals.appen(0) #auto pass for neurons that passed one previous tests
            else:
                baseline = []
                trialrate = []
                this_clusters_spikes = spikestimes[np.isin(spikesclusters, cluster)]
                for trial in range(0, trialsintervals.shape[0]):
                    #first we make the baserate
                    begin = stimOn[trial]-0.2
                    end = stimOn[trial]
                    rate = sum(np.logical_and(this_clusters_spikes>=begin, this_clusters_spikes<=end))
                    rate = rate/(begin-end)
                    baseline.append(rate)
                    
                    #now we do the stimulus onset rate
                    begin = stimOn[trial] + 0.05
                    end = stimOn[trial] + 0.15
                    rate = sum(np.logical_and(this_clusters_spikes>=begin, this_clusters_spikes<=end))
                    rate=rate/(begin-end)
                    trialrate.append(rate)
                #end of trial for loop
                if sum(trialrate+baseline)==0:
                    pvals.append(1)
                else:
                    pvals.append(scipy.stats.wilcoxon(x=baseline,y = trialrate)[1])
        #end of cluster for loop
        passed_tests = np.array(pvals)<0.05
        
        """
            (3) sign-rank test between pre-movement rates (−0.1 to 0.05 s 
       relative to movement onset) and baseline rate (for trials with movements);
       """
       #passed tests this is working
       i = 0
       responsechoice = fetched_objects['trialsresponse_choice']
       responsetimes = fetched_objects['trialsresponse_times']
       responsechoice = responsechoice[trialsincluded]
       responsetimes = responsetimes[trialsincluded]
       moved = np.array(responsechoice, dtype= bool)
       responsetimes = responsetimes[moved]
       
       # we are done with trialsintervals so we can modify it without changing it back
       trialsintervals = trialsintervals[moved,:] 
       
       #this needs to be fixed, we need to remove the wheel moves not occuring in
       #one of the trials

       pvals = []
       for i in range(0, len(clusters_idx)):
            cluster = clusters_idx[i]
            this_clusters_spikes = spikestimes[np.isin(spikesclusters, cluster)]
            if passed_tests[i]:
                pvals.append(0) #auto pass for neurons that passed one previous test
            else:
                baseline = []
                trialrate = []
                this_clusters_spikes = spikestimes[np.isin(spikesclusters, cluster)]
                for trial in range(0, trialsintervals.shape[0]):
                    #first we make the baserate
                    begin = trialsintervals[trial,0]-0.2
                    end = trialsintervals[trial,0]
                    rate = sum(np.logical_and(this_clusters_spikes>=begin, this_clusters_spikes<=end))
                    rate = rate/(begin-end)
                    baseline.append(rate)
                    
                for move in range(0, len(responsetimes)):
                    print(move)
                    #now we do the stimulus onset rate
                    begin = responsetimes[move] - 0.1
                    end = responsetimes[move] + 0.05
                    rate = sum(np.logical_and(this_clusters_spikes>=begin, this_clusters_spikes<=end))
                    rate=rate/(begin-end)
                    trialrate.append(rate)
                #end of for loops to get rates
                if sum(trialrate+baseline)==0:
                    pvals.append(1)
                else:
                    pvals.append(scipy.stats.wilcoxon(x=baseline,y = trialrate)[1])
        #end of cluster for loop
        passed_tests = np.array(pvals)<0.05
       
       
       """
            (4) Wilcoxon rank-sum test between pre-movement rates on left choice 
        trials and those on right choice trials; 
        
        #Note:  here we use the mannwhitney becasue it is equvilent but can handle
                   #different sample sizes, which arrise in this test
        """
        
       i = 0
       responsechoice = fetched_objects['trialsresponse_choice']
       responsechoice = responsechoice[trialsincluded]
       moved = np.array(responsechoice, dtype= bool)
       responsechoice = responsechoice[moved]
       # left choice
       leftchoice = responsechoice == 1
       leftchoice = responsetimes[leftchoice]
       # right choice
       rightchoice = responsechoice == -1
       rightchoice = responsetimes[rightchoice]
       pvals = []
       for i in range(0, len(clusters_idx)):
           cluster = clusters_idx[i]
           this_clusters_spikes = spikestimes[np.isin(spikesclusters, cluster)]
           if passed_tests[i]:
               pvals.append(0) #auto pass for neurons that passed one previous tests
           else:
               baseline = []
               trialrate = []
               this_clusters_spikes = spikestimes[np.isin(spikesclusters, cluster)]
               for move in range(0, len(leftchoice)):
                   #first we make the baserate
                   begin = leftchoice[move] - 0.1
                   end = leftchoice[move] + 0.05
                   rate = sum(np.logical_and(this_clusters_spikes>=begin, this_clusters_spikes<=end))
                   rate = rate/(begin-end)
                   baseline.append(rate)
                    
               for move in range(0, len(rightchoice)):
                   #now we do the stimulus onset rate
                   begin = rightchoice[move] - 0.1
                   end = rightchoice[move] + 0.05
                   rate = sum(np.logical_and(this_clusters_spikes>=begin, this_clusters_spikes<=end))
                   rate = rate/(begin-end)
                   trialrate.append(rate)
                #end of for loops to get rates
               if sum(trialrate + baseline)==0:
                   pvals.append(1)
               else:
                   #here we use the mannwhitney becasue ti is equvilent but can handle
                   #different sample sizes, which arrise in this test
                   pvals.append(scipy.stats.mannwhitneyu(x=baseline,y = trialrate)[1])
        #end of cluster for loop
        passed_tests = np.array(pvals)<0.05

        
        
        """
        (5) sign-rank test between post-movement rates (−0.05 to 0.2 s 
        relative to movement onset) and baseline rate; 
        """
        """
        (6) rank–sum test between post-reward rates (0 to 0.15 s relative 
        to reward delivery for correct NoGos) and baseline rates. 
        
        A neuron was considered active during the task, or to have detectable 
        modulation during some part of the task, if any of the P values on 
        these tests were below a Bonferroni-corrected alpha value (0.05/6 = 0.0083). 
        
        However, because the tests were coarse and would be relatively insensitive
        to neurons with transient activity, a looser threshold was used to 
        determine the neurons included for statistical analyses (Figs. 3–5): 
        if any of the first four tests (that is, those concerning the period 
        between stimulus onset and movement onset) had a P value less than 0.05.'    
    
    """
    return passed_tests