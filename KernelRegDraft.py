#Kernal Regression from Steimetz et al. (2019)
#
#Feb 6th 2022
#Angus Campbell

"""
We need to first reun CCA to generate B then we want to find the matrix a 
(also denoted as a matrix W with vectors w_n for each neruon n).  CCA
is first run from the toeplitz matrix of diagonalized kernel functions this will reduce
the dimensionality of the entire time course, we then optimize the weights of the components of 
this reduced representation.  Minimizations of square error is done by elastic net regularizatuion applied
on a neuron by neuron basis.  

Currently has matlab code sprinkled in comments to guide development.

Eventually the goal is to turn this into a .ipyn file, the all caps comments
are notes which denote sections, multi line commebts are quotes from the paper
which will be included or written up for description of the workflow.

"""
##INTRODUCTION
####START WITH AN IMAGE OF THE WORKFLOW AND A BRIEF EXPLANATION OF THE MODEL

import os
import numpy as np
import pandas as pd
from math import ceil

#for ubuntu....
#cd mnt/c/Users/angus/Desktop/SteinmetzLab/Analysis 

os.chdir('C:/Users/angus/Desktop/SteinmetzLab/Analysis')
import getSteinmetz2019data as stein

############ FILTERING
##Going from neurons across all regions and mice

# Which neurons to include
"""
clusters._phy_annotation.npy [enumerated type] (nClusters) 0 = noise (these are already
excluded and don't appear in this dataset at all); 1 = MUA 
(i.e. presumed to contain spikes from multiple neurons; these are not analyzed 
 in any analyses in the paper); 2 = Good (manually labeled); 3 = Unsorted. In 
this dataset 'Good' was applied in a few but not all datasets to included neurons, 
so in general the neurons with _phy_annotation>=2 are the ones that should be included.
"""

#So we should apply the criteria we want and search the data that way.

#when querrying the clusters data we can apply the quality score criteria

allsessions = list(stein.recording_key())
datapath = os.fspath(r'C:\Users\angus\Desktop\SteinmetzLab\9598406\spikeAndBehavioralData\allData')

for session in stein.recording_key():
    print(session)
    #to be filled with criteria bellow

# first we want trial times since we are initially only going to look at
# data withing the trial times, may as well collect the data we need from them
# for feeding into toeplitz matrix later
session = list(stein.recording_key())[1]


trials = stein.calldata(session, ['trials.visualStim_contrastLeft.npy',
                                       'trials.visualStim_contrastRight.npy',
                                       'trials.response_choice.npy',
                                       'trials.feedbackType.npy',
                                       'trials.intervals.npy'], 
                        steinmetzpath= datapath, propertysearch = False)
     

spikes = stein.calldata(session, ['spikes.clusters.npy',
                                  'spikes.times.npy',
                                  'clusters._phy_annotation.npy'],
                        steinmetzpath=datapath)


def frequency_array(session, filepath, bin_size,
                    return_meta_data = True, filter_by_quality= True, minquality = 2):
    """
    Takes Alyx format .npy files and load them into a numpy array,
    can either give you 
    
    spikeclusterIDs:  from the 'spikes.clusters.npy' file
    spikestimes: from the 'spikes.times.npy'
    start_times: times to start collecting from should have corrresponding equal length 
    vector of end_times
    end_times: time to stop collecting spikes
    
    return_meta_data: returns ABA location and cluster depth
    
    filter_by_quality= false by default if supplied with a vector of quality scores will auto 
    remove the ones below a certain threshold
    
    Returns: A numpy array of spike frequencies for each neuron,
    if return_meta_data also supplies a dataframe of the cluster ID and
    corresponding Allen onotlogy data as well as session label
    
    """
    
    def get_and_filter_spikes():
        """
        calls the spikes datat from the session we are interested in,
        removes the low quality scores, I.e. those listed as 1 
        steinmetz annotated the kilosorts clusters as 1, 2, or 3 recomended
        using nothing below a 2
        -returns 2 numpy arrays one for the clusters 
        """
        spikes = stein.calldata(session, ['spikes.clusters.npy',
                                  'spikes.times.npy',
                                  'clusters._phy_annotation.npy'],
                        steinmetzpath=filepath)
        
        spikesclusters = np.array(list(spikes['spikesclusters']))
        spikestimes = np.array(list(spikes['spikestimes']))
        clusterquality = np.array(list(spikes['clusters_phy_annotation']))
        
        if filter_by_quality:
            clusterquality = clusterquality >=minquality
            spikestimes = spikestimes[clusterquality]
            spikesclusters = spikesclusters[clusterquality]
        
        return(spikesclusters, spikestimes )
    
    clusters, times = get_and_filter_spikes()
    
    def bin_spikes_in_trials():
        """
        Returns the bhin by bin frequencies of each neuron,
        first we pull only the clusters that fired, then we use their cluster 
        to add that to the index
        
        """
        trialintervals = np.array(trials["trialsintervals"])
        #trials starts are trialintervals[, 0]
        #trial ends are trialintervals[, 0]
        for trial in trailsintervals.shape[0]:
            #find out number of step in the trial
            n_steps = ceil((trialintervals[trial,1]-trialintervals[trial,0])/bin_size)
            t_i = trialintervals[trial,0]
            t_plus_dt = t_i + bin_size
            trial_arr = np.empty(len(np.unique(clusters)), dtype=float) # will be concatenated
            
            for i in n_steps:
                
                
                #bin_arr will be the frequency for this trial, will be added to trail_arr each step and the reset
                bin_arr = np.zeros(len(np.unique(clusters)), dtype=float) 
                
                #this bin will filter our timing and clusters so we can
                # just work on the slice of spikeclusters corresponding to
                #each bin step
                this_bin = np.logical_and(times>=t_i,
                                          times<=t_plus_dt)
                
                #we find the index of the clusters and convert spike counts to hertz
                (unique, counts) = np.unique(clusters[this_bin], return_counts=True)
                frequencies = np.asarray((unique, counts/bin_size))

                
                # double check this step, smoothing should correct it
                #counts/bin_size gives the moment by moment firing rate
                #but since it can't take into account every other time step
                #we can correct this later with smoothing
                #should we may be include times just before and after trials
                #so the smoothing calculations won't be clipped?
                j = 0
                for neuron in frequencies[0,]:
                    
                    #make cluster identiy in frequencies into int so it can be found in clusters_idx
                    #for adding firirng rate to bin_arr 
                    neuron = int(neuron) 

                    bin_arr[neuron] = frequencies[1,j] #add the freq in Hz to the vector
                    print(j)
                    print(frequencies[1,j])
                    print(type(bin_arr[neuron]))
                    j = j + 1
                #bin_arr is now ready to be concatenated to trial_arr
                test = np.concatenate(trial_arr, bin_arr) # this throws an error
                    
                    
                    
                
                
                
                
                
            

        
        
        
        
        #end
        
        

    spikesclusters = np.array(list(spikes['spikesclusters'])) #delete after testing
    spikestimes = np.array(list(spikes['spikestimes'])) 
   
    #filter out low quality scores only if quality scores are supplied

    
    #MAKING BINNED TIME SERIES
    #1) Identify trial starts 

    

    


        
df = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=['A', 'B'])   
m = df % 3 == 0    
    
    for t_1 in start_times:
        this_bin
        
        
    
    
    
test = unique(spikes['spikes.'])
    
    

"""
We need a for loop over the data, could consider adding this to stein as
I will likely need to do this again

print(spikes['spikesclusters'][0:5])
[[1261]
 [ 961]
 [ 773]
 [ 778]
 [ 996]]

print(spikes['spikestimes'][0:5])
[[-3.3504999 ]
 [-3.3503999 ]
 [-3.3503999 ]
 [-3.35023323]
 [-3.3501999 ]]
"""




"""
The Methods section Data Analysis describes a battery of criteria that are needed, 
read in detail

- remeber to check the quality scores of the neurons online it says not to include 
any neuron below a 2 I think, please double check this.
-other filtering methods were applied but below is the list of criteria we will need to apply
ourselves

To determine whether a neuron’s firing rate was significantly modulated during the task 
(Supplementary Fig. 1), a set of six statistical tests were used to detect changes in activity 
during various task epochs and conditions: (1) Wilcoxon sign-rank test between trial firing rate 
(rate of spikes between stimulus onset and 400 ms post-stimulus) and baseline rate (defined in 
period −0.2 to 0 s relative to stimulus onset on each trial); (2) sign-rank test between 
stimulus-driven rate (firing rate between 0.05 and 0.15 s after stimulus onset) and baseline rate;
 (3) sign-rank test between pre-movement rates (−0.1 to 0.05 s relative to movement onset) and 
 baseline rate (for trials with movements); (4) Wilcoxon rank-sum test between pre-movement rates
  on left choice trials and those on right choice trials; (5) sign-rank test between post-movement 
  rates (−0.05 to 0.2 s relative to movement onset) and baseline rate; (6) rank–sum test between 
  post-reward rates (0 to 0.15 s relative to reward delivery for correct NoGos) and baseline rates.
   A neuron was considered active during the task, or to have detectable modulation during some part
    of the task, if any of the P values on these tests were below a Bonferroni-corrected alpha value
     (0.05/6 = 0.0083). However, because the tests were coarse and would be relatively insensitive
      to neurons with transient activity, a looser threshold was used to determine the neurons
       included for statistical analyses (Figs. 3–5): if any of the first four tests (that is,
        those concerning the period between stimulus onset and movement onset) had a P value less
         than 0.05.

In determining the neurons statistically significantly responding during different task conditions 
(Figs. 2d–h, right sub-panels, 5b), the mean firing rate in the post-stimulus window (0 to 0.25 s), 
taken across trials of the desired condition, was z-scored relative to trial-by-trial baseline rates
 (from the window −0.1 to 0) and taken as significant when this value was >4 or <−4, equivalent to
  a two-sided t-test at P < 10−4.

For visualizing firing rates (Extended Data Fig. 4), the activity of each neuron was then binned
 at 0.005 s, smoothed with a causal half-Gaussian filter with standard deviation 0.02 s, averaged 
 across trials, smoothed with another causal half-Gaussian filter with standard deviation 0.03 s, 
 baseline subtracted (baseline period −0.02 to 0 s relative to stimulus onset, including all trials
  in the task), and divided by baseline + 0.5 spikes s−1. Neurons were selected for display if they
   had a significant difference between firing rates on trials with both stimuli and movements 
   versus trials with neither, using a sliding window 0.1 s wide and in steps of 0.005 s 
   (rank-sum P < 0.0001 for at least three consecutive bins).
"""





###BINNING, SMOOTHING AND MAKING Y AND X
#preprocess spiking data credit goes to this stack answer: https://stackoverflow.com/questions/71003634/applying-a-half-gaussian-filter-to-binned-time-series-data-in-python/71003897#71003897
#We need to bin the spikes
import scipy.ndimage

def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)




#create the Toeplitz matrix X
# composed of the kernels for visual stimuli and action Refered to as L_c and L_d resprectfully.
#L_c had 6 kinds on for left-low,left-medium and left-low cotnrast and vice verse for right contrast.
#L_d simply composed of if a left or right whelle turn was occuring.
#we need to look up the definition they used for the wheel turns

#CCA to get the matrix B (aka b or )
#from the matlab code for canonCorall.m
#b = CXXMH * c;
"""
allX = vertcat(X{:});
CXX = cov(allX);
eps = 1e-7; 
CXX = CXX+eps*eye(nX); % prevents imaginary results in some cases

CXXMH = CXX ^ -0.5; 
"""

# could use this: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html
import sklearn.cross_decomposition


#use CCA in cross validated regularized regression
#note that CCA is calcualted once and is not cross validated
"""
%% basic regression
tic
[a, b, R2] = CanonCor2all({r.bs}, {r.A});
toc

% cut some dimensions just to start with 
b = b(:,1:200);
"""

#elastic net regression of the weights of the CCA components 
"""
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
"""
# we may need to train one of these for each neuron individually



#calculate cross validated varience explained per neuron and return the median cross validated varience




#there are also steps
