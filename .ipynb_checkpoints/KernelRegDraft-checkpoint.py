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
import getSteinmetz2019data as stein
import warnings
import piso


# From the local path on Angus's PC, toeplitz and freq_array,
# use this as the default consider changing
DEFAULT_FILEPATH = os.fspath(r'C:\Users\angus\Desktop\SteinmetzLab\9598406\spikeAndBehavioralData\allData')



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
                    filter_by_engagement = True,
                    FILEPATH = DEFAULT_FILEPATH):
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
        if len(only_use_these_clusters)!=0:
            #finds the clusters in the time series with bad quality (q<2) and removes them
            #from the series holding when a spike occured and what it's identity was
            clusters_mask = np.isin(spikesclusters, only_use_these_clusters) #boolean mask
            spikestimes = spikestimes[clusters_mask] 
            spikesclusters = spikesclusters[clusters_mask]
            clusters_idx = np.unique(spikesclusters)
        elif quality_annotation_filter:
            clusterquality = spikes['clusters_phy_annotation'] #quality rating of clsuters
            clusters_idx = np.arange(0, len(clusterquality)).reshape(clusterquality.shape)
            clusters_mask = clusterquality >=2 #boolean mask
            clusters_idx = clusters_idx[clusters_mask]
             #filter out low quality clusters
            
            #remove those clusters from the time series, here we do it with np.isin
            spikestimes = spikestimes[np.isin(spikesclusters, clusters_idx)] 
            spikesclusters = spikesclusters[np.isin(spikesclusters, clusters_idx)]
            clusters_idx = np.unique(spikesclusters)
        

        # if provided clusters to use instead....

            
        return(spikesclusters, spikestimes, clusters_idx )
    
    # run above function and get the spikes serieses for this session
    clusters, times, filteredclusters_idx = get_and_filter_spikes()
    
    #getting thetrials objects we need
    trials = stein.calldata(session, ['trials.intervals.npy',
                                      'trials.included.npy'],
                steinmetzpath=FILEPATH)
    
    
    # filter by the engagfement index filter provided is set tp ture by default
    # alternately a list of trials to include may be supplied
    # Supplying this filter overwrites the engagement-index
    if len(select_trials)!=0:
        trialsincluded = select_trials
    elif filter_by_engagement:
        trialsincluded = trials['trialsincluded']
        trialsincluded = [ i for i in range(0,len(trialsincluded)) if trialsincluded[i]]
        trialsincluded = np.array(trialsincluded)
    

    
    # filter trialsintervals by trialsincluded
    trialsintervals = trials['trialsintervals']
    trialsintervals = trialsintervals[trialsincluded,:]
    
    #this will be our output
    session_arr = np.zeros([len(np.unique(clusters)),2], dtype=float)
    
    #trials starts are trialsintervals[, 0]
    #trial ends are trialsintervals[, 0]
    for trial in range(0,trialsintervals.shape[0]):

        #find out number of step in the trial
        n_steps = ceil((trialsintervals[trial,1]-trialsintervals[trial,0])/bin_size)
        t_i = trialsintervals[trial,0]
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

            #This runs if there are no spikes, i.e. frequency array has 2nd dim = 0
            if frequencies.shape[1]==0:
                bin_arr = np.zeros([trial_arr.shape[0],1])
                trial_arr = np.column_stack([trial_arr, bin_arr])
                
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



def make_toeplitz_matrix(session,
                         bin_size,
                         kernels, 
                         filter_by_engagement = True, 
                         select_trials = [],
                         FILEPATH = DEFAULT_FILEPATH):
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
        
        def kernel_improperly_specified():
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
            kernel_improperly_specified()
            
        
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
        kernel_improperly_specified()
    # loop to rowstack all these things
    for i in range(0, trialsintervals.shape[0]):
        X_i = trial_section(i)
        X = np.row_stack([X, X_i])
    #end of this for loop
    
    #clip instatiation array
    X = X[2:,:]
    
    return X



def generate_event_interval(events, offset):
    """testetest
    makes a Alyx format .npy intervals array 0 index for interval beginings and 
    1 index for intervals end

    Args:
        events (numpy 1d, or list of int or floats): list of events in seconds from trial start
        offset(a tuple or 2 item list): time from event to make the interval extend from and to,
    """
    # lists to be later converted to numpy arrays and stacked
    starts = []
    ends = []

    #extends lsits with values from offset
    for occurence in range(0, len(events)):
        starts.append(events[occurence] + offset[0])
        ends.append(events[occurence] + offset[1])
    
    # turn them into arrays make sure they are shaped right, as numpy is weird like that
    starts = np.asarray(starts)
    starts = np.reshape(starts, (len(starts), 1) )
    ends = np.asarray(ends)
    ends = ends.reshape(starts.shape)
    out_arr = np.column_stack([starts, ends])

    return( out_arr )

def combine_intervals(intervals_x, intervals_y):
    """combines two alyx intervals objects into a single one by removing overlapping ones
    and sorting intervals STILL IN PROGRESS

    Args:
        intervals_x (_type_): _description_
        intervals_y (_type_): _description_
    """
    # combine the intervals, convert into an Interval array
    combined_intervals =  np.row_stack([intervals_x, intervals_y])
    combined_intervals = pd.arrays.IntervalArray.from_arrays(left = combined_intervals[:,0],
                                                             right = combined_intervals[:,1],
                                                             closed = 'left')
    
    #run a union operation from piso on the array, change them to being all open and make them into an array of tuples
    combined_intervals = piso.union(combined_intervals)
    combined_intervals = combined_intervals.set_closed('neither')
    combined_intervals = combined_intervals.to_tuples()
    
    #convert them to a list of lists, and make this into a numpy array
    combined_intervals = [list(x) for x in combined_intervals]
    combined_intervals = np.array(combined_intervals)
    
    return( combined_intervals )



def frequency_array_v2(session, bin_size, 
                    only_use_these_clusters=[],
                    quality_annotation_filter = True,
                    select_trials = [],
                    filter_by_engagement = True):
    """
    Second version of this to make it more modular, the other assumes we are obly interested
    in data within the start and end of a trial this one will be more general
    allowing the user to provide their own start and end times
    -not functioning just yet



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
        if len(only_use_these_clusters)!=0:
            #finds the clusters in the time series with bad quality (q<2) and removes them
            #from the series holding when a spike occured and what it's identity was
            clusters_mask = np.isin(spikesclusters, only_use_these_clusters) #boolean mask
            spikestimes = spikestimes[clusters_mask] 
            spikesclusters = spikesclusters[clusters_mask]
            clusters_idx = np.unique(spikesclusters)
        elif quality_annotation_filter:
            clusterquality = spikes['clusters_phy_annotation'] #quality rating of clsuters
            clusters_idx = np.arange(0, len(clusterquality)).reshape(clusterquality.shape)
            clusters_mask = clusterquality >=2 #boolean mask
            clusters_idx = clusters_idx[clusters_mask]
             #filter out low quality clusters
            
            #remove those clusters from the time series, here we do it with np.isin
            spikestimes = spikestimes[np.isin(spikesclusters, clusters_idx)] 
            spikesclusters = spikesclusters[np.isin(spikesclusters, clusters_idx)]
            clusters_idx = np.unique(spikesclusters)
        

        # if provided clusters to use instead....

            
        return(spikesclusters, spikestimes, clusters_idx )
    
    # run above function and get the spikes serieses for this session
    clusters, times, filteredclusters_idx = get_and_filter_spikes()
    
    #getting thetrials objects we need
    trials = stein.calldata(session, ['trials.intervals.npy',
                                      'trials.included.npy'],
                steinmetzpath=FILEPATH)
    
    
    # filter by the engagfement index filter provided is set tp ture by default
    # alternately a list of trials to include may be supplied
    # Supplying this filter overwrites the engagement-index
    if len(select_trials)!=0:
        trialsincluded = select_trials
    elif filter_by_engagement:
        trialsincluded = trials['trialsincluded']
        trialsincluded = [ i for i in range(0,len(trialsincluded)) if trialsincluded[i]]
        trialsincluded = np.array(trialsincluded)
    

    
    # filter trialsintervals by trialsincluded
    trialsintervals = trials['trialsintervals']
    trialsintervals = trialsintervals[trialsincluded,:]
    
    #this will be our output
    session_arr = np.zeros([len(np.unique(clusters)),2], dtype=float)
    
    #trials starts are trialsintervals[, 0]
    #trial ends are trialsintervals[, 0]
    for trial in range(0,trialsintervals.shape[0]):

        #find out number of step in the trial
        n_steps = ceil((trialsintervals[trial,1]-trialsintervals[trial,0])/bin_size)
        t_i = trialsintervals[trial,0]
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

            #This runs if there are no spikes, i.e. frequency array has 2nd dim = 0
            if frequencies.shape[1]==0:
                bin_arr = np.zeros([trial_arr.shape[0],1])
                trial_arr = np.column_stack([trial_arr, bin_arr])
                
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

