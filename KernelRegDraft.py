#Kernal Regression from Steimetz et al. (2019)
#
#Feb 6th 2022
#Angus Campbell

"""

frequency_array still needs testing.
       
Ignore the unexpected indent in spyder, it just doesnt like stein.calldata

Description of Kernel Regression Implementation:
    
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
from math import floor
import scipy.ndimage
import timeit #for testing and tracking run times
import scipy.stats 
os.chdir('C:/Users/angus/Desktop/SteinmetzLab/Analysis')
import getSteinmetz2019data as stein
import warnings


FILEPATH = os.fspath(r'C:\Users\angus\Desktop\SteinmetzLab\9598406\spikeAndBehavioralData\allData')



"""
start = timeit.timeit()

end = timeit.timeit()
print(end - start)"
"""
#for ubuntu....
#cd mnt/c/Users/angus/Desktop/SteinmetzLab/Analysis 



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



# first we want trial times since we are initially only going to look at
# data withing the trial times, may as well collect the data we need from them
# for feeding into toeplitz matrix later

"""
A NOTE ON THE TIMESTAMP FILES AND LFP DATA

So each session contains a few files named like this:
    'Forssmann_2017-11-01_K1_g0_t0.imec.lf.timestamps.npy'
    
    These are the time base offsets for the probes internal clocks.  In order
to align the time base here for the events occuring in the trials to the LFP
you will need to account for these.  They bear no relevance for the spikes,
stimuli, movement etc etc these are set to the same time base which starts 
prior to the begining of the trials.

"""







#For smoothing we make halfguassian_kernel1d and halfgaussian_filter1d
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



#now we can make the function that will generate our Y matrix, the firing rates to predict
#based on our kernels
def frequency_array(session, bin_size, 
                    only_use_these_clusters=[],
                    quality_annotation_filter = True,
                    select_trials = [],
                    filter_by_engagement = True):
    """
    Input:
        session:  the name of the desired session, we take it and generate....
            Takes Alyx format .npy files and load them into a numpy array,
            can either give you 
            spikeclusterIDs:  from the 'spikes.clusters.npy' file
            spikestimes: from the 'spikes.times.npy'
            start_times: times to start collecting from should have corrresponding equal length 
                vector of end_times
            end_times: time to stop collecting spikes
        
        bin_size:  the length in seconds of the bins we calculate frqncy over
        
        only_use_these_clusters: a list or array of clusters to filter, should be
                            supplied as an actual list of indices a boolean will not works
                            
        quality_annotation_filter: default to true overwritten byonly_use_these_clusters,
                                  removes clusters below quality annotation of 2 (out of 3)
              
        select_trials: may be boolean or an array of ints, limits trials to particular set,
                    should match that of the X you are pulling from
                    
        filter_by_engagement: by default set to true removes trials based on engagement index
        
        

    
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
        
        THIS SECTION MAY BE UNNESCESSARY
        
        """
       
        #We call the relvant objects for clusters (neurons) identity of a firing
        #the time at which a firing occured and the quality of the recording
        spikes = stein.calldata(session, ['spikes.clusters.npy',
                                  'spikes.times.npy',
                                  'clusters._phy_annotation.npy'],
                        steinmetzpath=FILEPATH)
        
        
        
        spikesclusters = spikes['spikesclusters'] #the idneity in sequence of 
        #each cluster, match it with spikestimes to get timing and identity info
        spikestimes = spikes['spikestimes'] #times corresponding to clusters firing
        
        # by default remove clusters wiht a rating of 1 
        if quality_annotation_filter:
            clusterquality = spikes['clusters_phy_annotation'] #quality rating of clsuters
            clusters_idx = np.arange(0, len(clusterquality)).reshape(clusterquality.shape)
            clusters_mask = clusterquality >=2 #boolean mask
            clusters_idx = clusters_idx[clusters_mask] #filter out low qualit clusters
            
            #remove those clusters from the time series
            spikestimes = spikestimes[np.isin(spikesclusters, clusters_idx)] 
            spikesclusters = spikesclusters[np.isin(spikesclusters, clusters_idx)]

        # if provided clusters to use instead....
        if len(only_use_these_clusters)!=0:
            #finds the clusters in the time series with bad quality (q<2) and removes them
            #from the series holding when a pike occured and what it's identity was
            cluster_mask = np.isin(spikesclusters, only_use_these_clusters) #boolean mask
            spikestimes = spikestimes[cluster_mask] 
            spikesclusters = spikesclusters[cluster_mask]
            clusters_idx = np.unique(spikesclusters)
            
        return(spikesclusters, spikestimes, clusters_idx )
    
    # run above function and get the spikes serieses for this session
    clusters, times, filteredclusters_idx = get_and_filter_spikes()
    
    #getting thetrials objects we need
    trials = stein.calldata(session, ['trials.intervals.npy',
                                      'trials.included.npy'],
                steinmetzpath=FILEPATH)
    
    
    # filter by the engagfement index filter provided is set tp ture by default
    # alternately a filter may be supplied
    if filter_by_engagement:
        trialsincluded = trials['trialsincluded']
    
    # Supplying this filter overwrites the engagement-index
    if len(select_trials)!=0:
        trialsincluded = select_trials
    
    # filter trialsintervals by trialsincluded
    trialsintervals = trialsintervals[trialsincluded,:]
    
    #this will be our output
    session_arr = np.zeros([len(np.unique(clusters)),2], dtype=float)
    
    #trials starts are trialintervals[, 0]
    #trial ends are trialintervals[, 0]
    for trial in range(0,trialintervals.shape[0]):
        #find out number of step in the trial
        n_steps = ceil((trialintervals[trial,1]-trialintervals[trial,0])/bin_size)
        t_i = trialintervals[trial,0]
        t_plus_dt = t_i + bin_size
        trial_arr = np.zeros([len(np.unique(clusters)),2], dtype=float) # will be concatenated
        
        for i in range(0,n_steps):
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

            
            j = 0 #initializing and index to move down frequncy 2d frequency values array with
            for neuron in frequencies[0,]:

                ### !!!!
                ####!!!! there is an error in this loop
                ## !!!!!

                #make cluster identiy in frequencies into int so it can be found in clusters_idx
                #for adding firirng rate to bin_arr 
                match_idx = int(neuron)==filteredclusters_idx #this evaluats to True,
                
                
                bin_arr[match_idx] = frequencies[1,j] #add the freq in Hz to the vector
                #bin_arr is now ready to be concatenated to trial_arr

                j = j + 1
                trial_arr = np.column_stack([trial_arr, bin_arr])
            #end of neuron for-loop
        #end of i for-loop

        #trimming array, then smoothing our firing rates
        trial_arr = trial_arr[:,2:]
        trial_arr = halfgaussian_filter1d(input = trial_arr,
                              sigma = 0.25)
        
        #clipping intialization array
        session_arr = np.column_stack([session_arr, trial_arr])
        
    #end of trial for-loop
    session_arr = session_arr[:,2:] # cuts off initialization array from session_arr
    
    return (session_arr, filteredclusters_idx)



def make_toeplitz_matrix(session, bin_size,
                         kernels, filter_by_engagement = True, select_trials = []):
    """
    Makes the matrix X aka P in Steinmetz et al., (2019), the Toeplitz matrix of
    dimension.  THe kernel is either 0 or 1 or -1
    Input:
        session: session name see stein.recording_key()
        bin_size:  needs to matech taht used for frequency array
        kernels:  which kernels to inlcude should be a three entry
                boolean list
                
        Please Note this function assumes all times tested will be within trial
        intervals will need some reworking if we want to use non-trial events as well
        
    
    """

    
    #Run this before trial_section()
    fetched_objects = stein.calldata(session,
                                     ['trials.intervals.npy',
                                      'trials.included.npy',
                                      'trials.response_choice.npy',
                                      'trials.response_times.npy',
                                      'trials.visualStim_contrastLeft.npy',
                                      'trials.visualStim_contrastRight.npy',
                                      'trials.visualStim_times.npy'],
                                     steinmetzpath = FILEPATH)
    
    # filter by the engagfement index filter provided is set tp ture by default
    # alternately a filter may be supplied
    if filter_by_engagement:
        include = fetched_objects['trialsincluded']
        trialsintervals = fetched_objects['trialsintervals']
        trialsintervals = trialsintervals[include.reshape(trialsintervals.shape[0]),:]
    
    # Supplying this filter overwrites the engagement-index
    if len(select_trials)!=0:
        include = select_trials
        trialsintervals = fetched_objects['trialsintervals']
        trialsintervals = trialsintervals[include]
    

    
    responsechoice = fetched_objects['trialsresponse_choice'][include]
    responsetimes = fetched_objects['trialsresponse_times'][include]
    Leftcontrasts = fetched_objects['trialsvisualStim_contrastLeft'][include]
    Rightcontrasts = fetched_objects['trialsvisualStim_contrastRight'][include]
    stim_times = fetched_objects['trialsvisualStim_times'][include]

    
    
    # the vision kenels, L_c, are supported for -0.05 to 0.4 post stimulus onset
    # the L_c kernels are therefore 90 high
    # the L_d kernels, for actions and choice are 55 high while L_c are 90 
    # the action kernels are supported over -025 to 
    def trial_section(trial):
        """
        Requires a fetched_objects = stein.calldata(session,
                                     ['trails.intervals.npy',
                                      'trials.included.npy',
                                      'trials.response_choice.npy',
                                      'trials.visualStim_contrastLeft.npy',
                                      'trials.visualStim_contrastRight.npy'])
        
        to be run before hand.
        
        Input:  
            trial, specifies which trial interval this is running on, be sure to
            filter trialsintervals and the behavioural measures as well with
            trialsincluded to drop the trials with low engagement
            
            kernel: a three item boolean list specifcying which kernels to include
            in this run kernel = [vision, action, choice],
            should be specified beforehand if this is run in make_toeplitz_matrix()
        
        """
        
        def make_kernel(trialkernel, T_start, T_stop,
                        L_start, L_stop, coef = 1):
            """
            Creates an np.diag array and replaces the provided the specified 
            indices of  trialkernel with this array, coef is by default 1 but
            will be changed for right choice kernels to -1
            
            """
            
            #these four lines scale the starting and stopping based on bin_size
            #prevents making non-mathcing trialkernels and kernels
            L_start = (bin_size/0.005)*L_start
            L_start = floor(L_start)
            L_stop = (bin_size/0.005)*L_stop
            L_stop = ceil(L_stop)
            
            kernel_length = L_stop-L_start
            kernel = np.diag(np.ones(kernel_length))*coef
            trialkernel[T_start:T_stop, L_start:L_stop] = kernel

            return trialkernel

        
        #here the timesteps are length and each kernel is hieght
        # T_trial is calculated same as s_steps in frequency_array()
        trial_start = trialsintervals[trial,0]
        trial_end = trialsintervals[trial,1]
        T_trial = ceil((trial_end - trial_start)/bin_size)

        #same thing is assumed in frequency_array and they need to match lengths
        
        
        
        #the 6 vision kernels (left low, left med, left high, right low, etc..)
        """
        The Vision kernels Kc,n(t) are supported over the window −0.05 to 0.4 s 
        relative to stimulus onset,
        """
        if kernels[0] == True:
            
            # instatiating zeros to fill in with diagonal 1's
            visionkernel = np.zeros(( T_trial, 6*90+90), dtype =  int)
            # indices for looping over

            #in bin count from start of trial when the kernel begins
            stim_start = stim_times[trial] - trial_start - 0.05
            stim_start = floor(stim_start/bin_size)
            
            # stim_end at +.45s/binsize because vision kernel k_c covers...
            #  -0.05s >= stimulation start time =< 0.4s therefore...
            stim_end = int( stim_start + (0.45/bin_size) )

            #  Left Low Contrast
            if Leftcontrasts[trial] == 0.25:
                visionkernel = make_kernel(visionkernel, stim_start, stim_end,
                                           L_start =0, L_stop = 90, coef = 1)
            # Left Medium Contrast
            if Leftcontrasts[trial] == 0.5:
                visionkernel = make_kernel(visionkernel, stim_start, stim_end,
                                           L_start =90, L_stop = 180, coef = 1)
            #Left High Contrast
            if Leftcontrasts[trial] == 1.0:
                visionkernel = make_kernel(visionkernel, stim_start, stim_end,
                                           L_start =180, L_stop = 270, coef = 1)
            
            # Right Low Contrat
            if Rightcontrasts[trial] == 0.25:
                visionkernel = make_kernel(visionkernel, stim_start, stim_end,
                                           L_start =270, L_stop = 360, coef = 1)
            # Right Medium Contrast
            if Rightcontrasts[trial] == 0.5:
                visionkernel = make_kernel(visionkernel, stim_start, stim_end,
                                           L_start =450, L_stop = 540, coef = 1)
            # Right High Contrast
            if Rightcontrasts[trial] == 1.0:
                visionkernel = make_kernel(visionkernel, stim_start, stim_end,
                                           L_start =540, L_stop = 630, coef = 1)
        
        ##### Movement Kernel
        """
        the Action and Choice kernels are supported over the window −0.25 
        to 0.025 s relative to movement onset. 
        """
        if kernels[1]==True:
            
            # instantiate matrix
            actionkernel = np.zeros((T_trial, 55), dtype = int)
            
            #when movementstarts
            move_start = responsetimes[trial] - trial_start - 0.25
            move_start = floor(move_start/bin_size)
            
            # move_end at +.45s/binsize because movement kernel k_d covers...
            #  -0.25s >= movement start time =< 0.025s therefore...
            move_end = int( move_start + (0.275/bin_size) )
            
            if responsechoice[trial]!=0:
                #add contrast to our matrix if there is no movement
                actionkernel = make_kernel(actionkernel, move_start, move_end,
                                             L_start = 0, L_stop = 55, coef =1)
       
        #Choice Kernel
        """
        the Action and Choice kernels are supported over the window −0.25 
        to 0.025 s relative to movement onset. 
        """
        if kernels[2]==True:
            
            # instantiate matrix
            choicekernel = np.zeros((T_trial, 55), dtype = int)
            
            #when movementstarts
            move_start = responsetimes[trial] - trial_start - 0.25
            move_start = floor(move_start/bin_size)
            
            # move_end at +.45s/binsize because movement kernel k_d covers...
            #  -0.25s >= movement start time =< 0.025s therefore...
            move_end = ceil( move_start + (0.275/bin_size) )
            
            ##!!!! this is causing an error needs testing
            #add contrast to our matrix
            #Left Choice Kernel contrast = 1 along diagonal aligned to movement start
            if responsechoice[trial]==1:
                #Left choice
                choicekernel = make_kernel(choicekernel, move_start, move_end,
                                           L_start = 0, L_stop = 55, coef = 1)
            if responsechoice[trial]==-1:
                #Right choice Kernel contrast = 1 along diagonal aligned to movement start
                # so here we set coef to -1
                choicekernel = make_kernel(choicekernel, move_start, move_end,
                                           L_start = 0, L_stop = 55, coef = -1)
        
        # Stitiching kernels together and warning about how kernel should be given
        
        def kernel_improperly_spcified():
            warnings.warn(
                "kernel must be input including vision kernel, also you cannot\
                include choice kernel without action kernel."
        )
        
        if kernels[0] & kernels[1] & kernels[2]:
            X_trial_i = np.column_stack([visionkernel , actionkernel, choicekernel])
        elif kernels[0] & kernels[1]:
            X_trial_i = np.column_stack([visionkernel , actionkernel])
        elif kernels[0]:
            X_trial_i = visionkernel
        else:
            kernel_improperly_spcified()
            
        
        return(X_trial_i)
    
    #instantiate the array to stack based on kernels included
    #this will need to be changed if you change the kernels included
    if kernels[0] & kernels[1] & kernels[2]:
        X = np.zeros((2, 740))
    elif kernels[0] & kernels[1]:
        X = np.zeros((2, 685))
    elif kernels[0]:
        X = np.zeros((2, 630))
    else:
        kernel_improperly_spcified()
    # loop to rowstack all these things
    for i in range(0, trialsintervals.shape[0]):
        X_i = trial_section(i)
        X = np.row_stack([X, X_i])
    #end of this for loop
    
    #clip instatiation array
    X = X[2:,:]
    
    return X



############# ACTUALLY RUNNING THE CODE AND KERNEL REG

start = timeit.timeit()
#These trials selected because they contain all types of choices, left 2 rights then a no go
# [4,5,6,7]

#test this fucntion out
#note steinmetz mthods uses P and X interchanably so
# I thought ti would be appropriate here

P = make_toeplitz_matrix(session = 'Theiler_2017-10-11', 
                     bin_size = 0.005, 
                     kernels = [True, True, True],
                     select_trials=np.array([4,5,6,7])
                     )

"""

It's worth noting that even at a full session length P runs no problem

P = make_toeplitz_matrix(session = 'Theiler_2017-10-11', 
                     bin_size = 0.005, 
                     kernels = [True, True, True]
                     )
"""

end= timeit.timeit()
print(start-end)


import seaborn as sns

sns.heatmap(P)

"""
#How test spikes were selected...
list_of = ['spikes.clusters.npy',
                          'spikes.times.npy',
                          'clusters._phy_annotation.npy',
                          'trials.intervals.npy',
                          'trials.included.npy',
                          'trials.response_choice.npy',
                          'trials.visualStim_contrastLeft.npy',
                          'trials.visualStim_contrastRight.npy']

theiler = stein.calldata(session = ['Theiler_2017-10-11'],
                         list_of,
                         steinmetzpath=FILEPATH)




clusterquality = theiler['clusters_phy_annotation']
spikesclusters = theiler['spikesclusters'] #quality rating of clsuters
clusters_idx = np.unique(spikesclusters).reshape(clusterquality.shape)
clusters_mask = clusterquality >=2 #boolean mask
clusters_idx = clusters_idx[clusters_mask] #filter out low quality clusters
clusters_idx[0:10]

only_use_these_clusters=[ 3,  4,  7,  9, 12, 14, 16, 17, 18, 19]
"""
start = time.it()
# only use these clusters includes first 10 clusters in clusters_idx that pass quality
Y, clusters_index = frequency_array(session = 'Theiler_2017-10-11', 
                                    bin_size = 0.005, 
                                    only_use_these_clusters=[ 3,  4])
"""
Traceback (most recent call last):

  File "<ipython-input-3-c46d16d81243>", line 1, in <module>
    Y, clusters_index = frequency_array(session = 'Theiler_2017-10-11',

  File "<ipython-input-1-6ada96e8e5de>", line 204, in frequency_array
    trialsintervals = trialsintervals[trialsincluded,:]

UnboundLocalError: local variable 'trialsintervals' referenced before assignment

"""


end = timeit.timeit()
print(start - end)



# at some point we need to write a function for clusters_index that pulls its ABA

#CCA on P and Y for this session

#Mutiply 

#Eventually we want Y_estimated






####NOTE FROM ABHJIT JUST PREDICT BEHAVIOUR, use that body publication

#the model is calculated neuron by neuron using a reduced representation 
# across 


    #need to be making index of columns as well
    """
    FINDING THE CHANNELS ABA-Ontollogy Location
    
    def fetch_channel_locations(session):
    
    #finds a session and returns the locations of its channels 
    
    locations = stein.calldata(session, ['channels.brainLocation.tsv','channels.probe.npy'], 
                        steinmetzpath= FILEPATH, propertysearch = False) 
    locations = locations['channelsbrainLocation']
    locations = pd.Series(locations['allen_ontology'])
    
    
    #search
    locations = stein.calldata(tate, ['clusters.peakChannel.npy',
                                      'channels.brainLocation.tsv',
                                      'channels.probe.npy'], 
                        steinmetzpath= FILEPATH, propertysearch = False)
    
    #give channel(on probe) to associate to cluster
    clusterpeaks = locations['clusterspeakChannel'] 
    
    #probe cluster was recorded from
    clusterprobe = tatumclusters['clustersprobes'] 
    
    #df with associated allen ontology
    brainLocation = locations['channelsbrainLocation']
    
    The goal is to store it all in a single string for each
    Session_cluster#_in_the_ABAlocation                         
    Example: Forssmann_2017-11-01_Cluster1738_in_the_CA1
    
    """
    
            """
            FROM Methods - KERNEL REGRESSION ANALYSIS
            'For the current study, only visual stimulus onset and wheel 
            movement onset kernels were required,'              
            """      
                    
                


                
                
                
            

        
        
        
        
        #end
        
        

  
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
