# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 23:45:10 2021

@author: angus
Steinmetz, N. A., Zatka-Haas, P., Carandini, M., & Harris, K. D. (2019). 
Distributed coding of choice, action and engagement across the mouse brain. 
Nature, 576(7786), 266-273.

Invovles 10 male and female mice (not specified which ones are which)
from 29 134 neurons across 39 regions.  Time in the trials data
is from the first trial.  Each trial was initiated by the animal holding
the wheel still, inter trial intervals were uniformly distributed between
0.2 and 0.5 seconds.

google docs in my drive:
    https://docs.google.com/document/d/1EDb9PKYOx3MXALORrP4zsTuwOb_SziqC2NF6IiMFdkA/edit
 and Howe 2020 thesis:
     https://escholarship.org/uc/item/9d35603d

Steinmetz et al., 2019 as .nyp 

data is described here: https://github.com/nsteinme/steinmetz-et-al-2019/wiki/data-files

LFP data is here: https://figshare.com/articles/dataset/LFP_data_from_Steinmetz_et_al_2019/9727895
Allen Inswtitute tutorial on how to analyze neuropixels data
https://allensdk.readthedocs.io/en/v1.4.0/_static/examples/nb/ecephys_lfp_analysis.html

How to analyze it here: https://github.com/cortex-lab/neuropixels/wiki/Other_analysis_methods#basic-lfp-characterization
This may also help: https://mark-kramer.github.io/Case-Studies-Python/intro.html

From Nick Steinmetz: https://github.com/cortex-lab/spikes/blob/master/analysis/lfpBandPower.m

"That function does everything: open the file, pull in some data, and compute power
 spectra. You could stop it at line 34 to take "thisDat" (matrix of channels x time) 
 for some other analysis. Note that "lfpFs" (sampling frequency) is 2500 Hz, and 
 "nChansInFile" is 385. That function loads data using "memmapfile" which makes 
 it easy to pull out data segments from discontinuous parts of the file - but you 
 can also just do "fread" to read a 385 x Nsamples data segment of int16 datatype. "

Analysis Idea: Autoencoder analysis https://www.youtube.com/watch?v=KmQkDgu-Qp0&ab_channel=SteveBrunton
Hierarchical Deep learning for multiscale timeseries https://www.youtube.com/watch?v=Jfl3dIlSTrU&ab_channel=SteveBrunton

Highly relevant paper:
Van Der Meer, M. A., & Redish, A. D. (2011). Theta phase precession in rat ventral striatum links place and reward information. Journal of neuroscience, 31(8), 2843-2854.
https://www.jneurosci.org/content/31/8/2843#:~:text=Theta%2Dmodulated%20cells%20in%20ventral,up%20to%20the%20reward%20sites.

Peters, A. J., Fabre, J. M., Steinmetz, N. A., Harris, K. D., & Carandini, M. (2021). Striatal activity topographically reflects cortical activity. Nature, 591(7850), 420-425.
https://www.nature.com/articles/s41586-020-03166-8#Sec1
-this article is also directly relevant and has data freely available
-data is available on request

BASIC IDEA:
-Howe et al., 2020 describes in his thesis how CA3 theta correaltes to
subiculum and lateral (medial?) septum before some regions in striatum
becomes active (clear this up)

Try to relate this to the concept of urgecny described here:
    Thura, D., & Cisek, P. (2017). The basal ganglia do not select reach 
    targets but control the urgency of commitment. Neuron, 95(5), 1160-1170.
    https://www.sciencedirect.com/science/article/pii/S0896627317306876
-how does theta relate to urgency models of  BG activity?
"To test these predictions, we recorded the activity of 107 task-related
 pallidal neurons (51 GPe; 56 GPi;"
                   
Although the task here does not allow for us to dissociate changes in speed 
accuracy based on sensory input alone other factors can influence S-A trade off.
Including the trial history.

Akrami, A., Kopec, C. D., Diamond, M. E., & Brody, C. D. (2018).
 Posterior parietal cortex represents sensory history and mediates
 its effects on behaviour. Nature, 554(7692), 368-372.
 -see figure 1c for a description of how previous trial history can shape
 the effects of future responses
 

Miller, Kevin J., Matthew M. Botvinick, and Carlos D. Brody. 
"From predictive models to cognitive models: Separable behavioral
 processes underlying reward learning in the rat." bioRxiv (2021): 461129.
https://www.biorxiv.org/content/10.1101/461129v3.full
-We can potentially look for 

Brain state:
    The foster et al., 2017 paper describes how hippocampal theta may be related to
    brain state.  It would be interesting to take the frame work described in
    Tim Murphies paper as it would.
    https://pubmed.ncbi.nlm.nih.gov/28674167/
    
    Xiao, D., Vanni, M. P., Mitelut, C. C., Chan, A. W., LeDue, J. M.,
    Xie, Y., ... & Murphy, T. H. (2017). Mapping cortical mesoscopic 
    networks of single spiking cortical or sub-cortical neurons.
    Elife, 6, e19976.
    

Object oriented data science python example if you want to try it out
https://opendatascience.com/an-introduction-to-object-oriented-data-science-in-python/

"""

#import onelight as one
import os
import numpy as np
import pandas as pd

#for ubuntu....
#cd mnt/c/Users/angus/Desktop/SteinmetzLab/Analysis 

os.chdir('C:/Users/angus/Desktop/SteinmetzLab/Analysis')
import getSteinmetz2019data as stein

datapath = os.fspath(r'C:\Users\angus\Desktop\SteinmetzLab\9598406\spikeAndBehavioralData\allData')

#finding all the unique locations in a recording
tate = 'Tatum_2017-12-09'


def fetch_channel_locations(session):
    """
    finds a session and returns the locations of its channels 
    """
    locations = stein.calldata(session, ['channels.brainLocation.tsv','channels.probe.npy'], 
                        steinmetzpath= datapath, propertysearch = False) 
    locations = locations['channelsbrainLocation']
    locations = pd.Series(locations['allen_ontology'])
    
    return(locations.value_counts())

tatumlocations = fetch_channel_locations(tate)


########
"""
Getting the clusters from a recording, getting their anatomical location

"""

tatumclusters = stein.calldata(tate, ['clusters.'], 
                        steinmetzpath= datapath, propertysearch = True)
tatumclusters = tatumclusters['clusters']

tatumclusters = pd.DataFrame({'site':list(tatumclusters['clusterspeakChannel']),
                              'probe':list(tatumclusters['clustersprobes']),
                               'quality':list(tatumclusters['clusters_phy_annotation'])})


tatumpeak = stein.calldata(tate, ['clusters.peakChannel.npy'], 
                        steinmetzpath= datapath, propertysearch = False)

tatumchannel = stein.calldata(tate, ['channels.'], 
                        steinmetzpath= datapath, propertysearch = True)

tatumchannel = tatumchannel['channels']
tatumchannel_df = pd.DataFrame({'allen_ontology':list(tatumchannel['channelsbrainLocation'].allen_ontology), 
              'site':list(tatumchannel['channelssite']), 
              'sitepostion':list(tatumchannel['channelssitePositions']), 
              'rawRow':list(tatumchannel['channelsrawRow']), 
              'probe':list(tatumchannel['channelsprobe'])})


"""

We are trying to extract the allen ontology row from the tatumchannel_df 
-the site value in channel data corresponds to the channel value
for tatumclusters, ENSURE THE PROBES VALUES MATCH AS WELL

"""


#Adding in the allen ontollogy annotation
ABA_ontollogy = []
for i in tatumclusters.site:
    ABA_ontollogy.append(str(tatumchannel_df.allen_ontology.loc[i].item()))

tatumclusters['allen_ontollogy'] = ABA_ontollogy
tatumclusters = tatumclusters.sort_values(by=['allen_ontollogy'])
#from this we can see that CA3 is on probe 1


sum(tatumclusters.quality[tatumclusters.allen_ontollogy=='CA3']>1)


##############################################################################
#Characterizing all trial types 

#all times are recorded in seconds

#for sesh in stein.recording_key():
# this loop is not ready yet  
    sesh = tate
    trials = stein.calldata(tate, ['trials.visualStim_contrastLeft.npy',
                                       'trials.visualStim_contrastRight.npy',
                                       'trials.response_choice.npy',
                                       'trials.feedbackType.npy',
                                       'trials.intervals.npy'], 
                        steinmetzpath= datapath, propertysearch = False)
     
    trials = pd.DataFrame({'Choice':list(trials['trialsresponse_choice']),
                       'LeftContrast':list(trials['trialsvisualStim_contrastLeft']),
                       'RightContrast':list(trials['trialsvisualStim_contrastRight']),
                       'feedback':list(trials['trialsfeedbackType']),
                       'trialstart':list(trials['trialsintervals'][:,0]),
                       'trialend':list(trials['trialsintervals'][:,1])})
    
    
    session_df = pd.DataFrame({'Trialtype':list(range(0,len(trials)))})
    session_df[:] = np.NAN
    #Note this should be changed so trials just gets a new column
    
    #Note add pandas series to dataframe with assing
    # df1 = df1.assign(e=pd.Series(np.random.randn(sLength)).values)
    #will produce a session_df containing trail type and difficulty
    # and session kind
    
    Righttrial = trials.RightContrast > trials.LeftContrast
    Lefttrial = trials.RightContrast < trials.LeftContrast
    nogo =(trials.LeftContrast==0)&(trials.RightContrast==0)
    random = (trials.RightContrast == trials.LeftContrast)&(~nogo) 
    #in case you want to check the values of the random trials
    #trials[['Choice','LeftContrast','RightContrast','feedback']][random]
    
    
    """
    A Hit denotes a propoerly chosen gabor grating.  The grating contrast on 
    that side matches the choice.  Note that some times this occurs on trials where
    gratings were equally strong, if the mouse was rewarded for its action also 
    note that.
    
    Incorrect trials mean trial where the wrong action was taken.  Note this also 
    includes the randomly rewarded equal contrast trials where the aniaml recieved 
    negative reinforcement.  The L or R denotes which actionw as chosen
    
    Correct rejection means a no-go trial was correctly responded to.
    
    FA is false alarm meaning the animal responded with the corresponding action 
    to a no-go trial.
    
    Miss means that no action was taken when a choice should have been made.
    R or L here corresponds to the action that would have been rewarded.
    Very rarely there was a trial with equal trial grating but since the
    animal didn't respond there's no way to know what would have been
    the rewarded action so this is simply labelled Miss without an R or L
    """
    
    #this needs to be rewritten so trials just gets a new column and we modify it 
    #using these statements here
    session_df.Trialtype[Righttrial&(trials.Choice==-1)] = 'HitR'
    session_df.Trialtype[random&(trials.Choice==-1)&(trials.feedback == 1)] = 'HitR'
    session_df.Trialtype[Lefttrial&(trials.Choice==-1)&(~nogo)] = 'IncorrectR'
    session_df.Trialtype[random&(trials.Choice==-1)&(trials.feedback == -1)] = 'IncorrectR'
    session_df.Trialtype[Lefttrial&(trials.Choice==1)&(~nogo)] = 'HitL'
    session_df.Trialtype[random&(trials.Choice==1)&(trials.feedback == 1)] = 'HitL'
    session_df.Trialtype[(Righttrial)&(trials.Choice==1)] = 'IncorrectL'
    session_df.Trialtype[random&(trials.Choice==1)&(trials.feedback == -1)] = 'IncorrectL'
    session_df.Trialtype[nogo&(trials.Choice == 0)] = 'CorrectRejection'
    session_df.Trialtype[(~nogo)&(trials.Choice == 0)&Righttrial] = 'MissR'
    session_df.Trialtype[(~nogo)&(trials.Choice == 0)&Lefttrial] = 'MissL'
    session_df.Trialtype[(~nogo)&random&(trials.Choice == 0)] = 'Miss'
    session_df.Trialtype[nogo&(trials.Choice == 1)] = 'FA_L'
    session_df.Trialtype[nogo&(trials.Choice == -1)] = 'FA_R'
    
    trials['Trialtype'] = session_df.Trialtype
    
    """
    Most of the time mice responded very quickly and made the correct choice but
    on ~4% of trials there were mistaken moves.  This may be an intersting place to begin 
    looking for clashes of prediction vs perception.
    
    We call the variables wheelMoves and the interevals they take place during
    to see what occured during each trial.
    """
    
    is_reversal = pd.DataFrame({'Trialtype':list(range(0,len(trials)))})
    is_reversal = np.empty((len(trials),1))
    is_reversal[:] = np.NAN

        
    wheelmoves = stein.calldata(sesh, ['wheelMoves.type.npy',
                                           'wheelMoves.intervals.npy'], 
                        steinmetzpath= datapath, propertysearch = False)
    
    wheelmoves = pd.DataFrame({'movetype':list(wheelmoves['wheelMovestype']),
                  'movestart':list(wheelmoves['wheelMovesintervals'][:,0]),
                  'moveend':list(wheelmoves['wheelMovesintervals'][:,1])})
    
    # right turns are coded as 2 but in trials.Choices its -1 so this 
    #line chacnes them to make it comparable
    wheelmoves.movetype[wheelmoves.movetype == 2] = -1.0
    
    wheelposition = stein.calldata(sesh, ['wheel.position.npy'], 
                        steinmetzpath= datapath, propertysearch = False)
    wheelpostion = pd.DataFrame({'wheelposition':list(wheelposition['wheelposition'])})
    
    
#################
    t = 0
    moves_in_trials = []
    for t in list(trials.index):
        moves_made = []
        #these two condition check that the wheelmove began during the trial interval
        idx = (wheelmoves.movestart>trials.trialstart[t])&(wheelmoves.movestart<trials.trialend[t])
        
        #converts the array output into a list of integers for later analysis
        moves_made = list(wheelmoves.movetype[idx])
        moves_made = [int(x) for x in moves_made]
        moves_in_trials.append(moves_made)
        
     
        """
        Some notes at this point.  Prehaps this is not the best definiton of 
        movement during a trial, for one the goCue could be important as well.
        I am noting several trails in Tatum_2017-12-09 that show trials where 
        within the trial the movements do not appear to correspond to 
        the animals response.  It could be that the change jsut wasn't strong 
        enough but these trials may warrant further investigation just to get
        a good working definiton of reversal.  Specficially trail 27 (indexed as 26),
        the response is right (-1) and was correct but the 
        list of movements withing the trial are as follows: [-1, 0, 0, 0, 1, 0, 0, 0].
        So there was a small reversal and several weak movements.
        
        Does global signal increas in line with flinches?  Indicating some degree of
        uncertainty?
                                                                     
        
        Prehaps it would be interesting to look at reversals generally,
        so at points when there were flinches and changes from -1 to 1.
        What is happening to hippocampal theta and subiculum just preceding these
        incidences?
        
        In Cortical State Fluctiations during perceptrual decision making, the
        Harris et al., 2020 publication with concurrent neuropixels and GCaMp
        recording in visual cortex.  They note that the most significant difference
        in cortical state was whether or not the animal chose to move or not,
        not the choice of left or right.
        
        -see if theta tracks pupillary diallation as well
        
        Another interesting phenomenon could be the bias towards perviously
        rewarded trials.  Just from looking at the number of Incorrect L and
        FA_L there seems to be a bias (as would be expected from the literature)
        towards L.  Given that recent Replay expeirment from Loren showing that
        previously rewarded spaces tend to be rewarded it would be interesting
        to see if replay could be driving the bias towards previously rewarded
        responses in uncertain situations like the no-go trials or the difficult
        trials (trials with near identical contrasts).
        
             
        Kaefer, K., Nardin, M., Blahna, K., & Csicsvari, J. (2020). Replay of 
        behavioral sequences in the medial prefrontal cortex during rule
        switching. Neuron, 106(1), 154-165.
        
        Jai, Y. Y., Liu, D. F., Loback, A., Grossrubatscher, I., & 
        Frank, L. M. (2018). Specific hippocampal representations are linked 
        to generalized cortical representations in memory. Nature 
        communications, 9(1), 1-11.
        
        Tang, W., Shin, J. D., Frank, L. M., & Jadhav, S. P. (2017). 
        Hippocampal-prefrontal reactivation during learning is stronger in 
        awake compared with sleep states. Journal of Neuroscience, 37(49), 
        11789-11805.
        
        Jadhav, S. P., Rothschild, G., Roumis, D. K., & Frank, L. M. (2016).
        Coordinated excitation and inhibition of prefrontal ensembles during 
        awake hippocampal sharp-wave ripple events. Neuron, 90(1), 113-127.
        
                 Time Cell analysis:
        -note that there are not really great methods for doing this
        -the methods below require some decently complicated matlab scripts,
        we could try emailing the authors and seeing if they will help or
        supply code because there is no github
        
        Tiganj, Z., Cromer, J. A., Roy, J. E., Miller, E. K., & Howard, M. W. 
        (2018). Compressed timeline of recent experience in monkey lateral 
        prefrontal cortex. Journal of cognitive neuroscience, 30(7), 935-950.
        
        
        Tiganj, Z., Jung, M. W., Kim, J., & Howard, M. W. (2017). Sequential 
        firing codes for time in rodent medial prefrontal cortex. Cerebral 
        Cortex, 27(12), 5663-5671.
        -the time it took rodents to navigate the maze seems comperable to the trials
        times in the task here the maze took between 3.0-4.7 seconds
        -could be interesting ot see to what extent time cells are present in 
        different regions of the cortex see if we cna make a story about that
        -it is tricky to define these as "time cells" may be worth avoiding
        but the idea of a cell having a place in an order, like for instance with 
        a markov chain may be of value
        -the model could be used for a collapsed timeline, see if any cells
        fit within the "relative time" to a choice or absolute time within a
        trial as well
        
        
        """
            
        
    
    
    
    
    CorrectRight = Righttrial&(trials.Choice==1)
    IncorrectRight = not(Righttrial)&(trials.Choice==-1) #1 is a right choice, -1 left, 0 nogo
    CorrectLeft = not(Righttrial)&(trials.Choice==-1)
    IncorrectLeft = not(Righttrial)&(trials.Choice==-1)
    CorrectRejection = nogo&(trials.Choice == 0)
    FalseAlarmR = nogo&(trials.Choice == 1)
    FalseAlarmL = nogo&(trials.Choice == -1)
    
    #end

df_c = pd.concat([df_a.reset_index(drop=True), df_b], axis=1)


test = pd.DataFrame.from_records(trials, index = idx)

"""

Characterizing trials for searching for LFP signatures of interest

"""
Righttrial = pd.Index(trials.RightContrast > trials.LeftContrast)
nogo =~((trials.LeftContrast>0)|(trials.RightContrast>0))
random = pd.Index((trials.RightContrast == trials.LeftContrast)&(~nogo) )

#Trial Types
CorrectRight = Righttrial&(trials.Choice==1)
IncorrectRight = not(Righttrial)&(trials.Choice==-1) #1 is a right choice, -1 left, 0 nogo
CorrectLeft = not(Righttrial)&(trials.Choice==-1)
IncorrectLeft = not(Righttrial)&(trials.Choice==-1)
CorrectRejection = nogo&(trials.Choice == 0)
FalseAlarmR = nogo&(trials.Choice == 1)
FalseAlarmL = nogo&(trials.Choice == -1)

"""
A note on the recordings from the methods:
    "Recordings were made in external reference mode with local field potential
    gain = 250 and action potential gain = 500. Recordings were repeated at 
    different locations on each of multiple subsequent days (Supplementary 
    Table 2), performing new craniotomy procedures as necessary. All recordings
    were made in the left hemisphere. The ability of a single probe to record 
    from multiple areas, and the use of multiple probes simultaneously, led to
    a number of areas being recorded simultaneously in each session (Supplementary
    Table 3)"

"""


"""

channels.site.npy is the index that peakChannels is reffering to,
brain location in the allen ontology is in channels.brainLocation which is 
a df containing allen_ontollogy and the stereotaxic coordinates,
make sure it is on the same probe

clusters.peakChannel.npy [integer] (nClusters) The channel number of the location of 
the peak of the cluster's waveform.
-this value is how we relate clusters to channels which gives us locations
-note you also need clusters.probes.npy as well to know the probe the channel
number corresponds to
-note clusters.cluster_phy_annotation is a quallity control measure,
2 or 3 may be used but those with 1 should be excluded


clusters.npy (nSpikes) The cluster number of the spike, 0-indexed, matching 
rows of the clusters object.
-this value is how we relate things in the spikes object to clusters


"""

test = stein.calldata(sesh, ['trials.response_choice.npy',
                               'trials.visualStim_contrastLeft.npy',
                               'trials.visualStim_contrastRight.npy',
                               'trials.goCue_times.npy',
                               'trials.feedbackType.npy'], 
                        steinmetzpath= datapath, propertysearch = False)

tatum = stein.calldata(sesh, ['channels.', 'trials.',
                               'clusters.', 'spikes.'], 
                        steinmetzpath= datapath, propertysearch = True)

tatumchannels = tatum['channels']
tatumspikes = tatum['spikes']
tatumspikesdf = pd.DataFrame.from_dict(tatumspikes, orient='columns')


tatumclusters = tatum['clusters']
tatumtrials = tatum['trials']

for i in test.keys():
    print(i)
    print(test[i].shape)


trials = pd.DataFrame({'Choice':list(test['trialsresponse_choice']),
                       'LeftContrast':list(test['trialsvisualStim_contrastLeft']),
                       'RightContrast':list(test['trialsvisualStim_contrastRight']),
                       'feedback':list(test['trialsfeedbackType'])})

#allr recordings done on left hemispehere corresponding to right visual field
# this marks if the contrast matches the side that shoudl be activated by it
for i in range(0, trials.shape[0]):
    
#making logical vectors of choices



def 


#make a df with the following columns Session, OGIndex, Trialtype, is_random
#Trial type will contain HitsR, HitsL, CorrectRejection, MissL, MissR,
#IncorrectL, IncorrectR
#is_random will be binary indicating these were trials with equal chance of 
#either L or R being correct and stimuli at 50/50 contrast



Ipsilateralchoice = trials.RightContrast[i] > trials.LeftContrast[i]


if Ipsilateralchoice&

#the kinds of responses a mouse can have in the task
trialtypes = ['CorrectLeftChoice', 'IncorrectRightChoice', 'MissLeftChoice',
'CorrectRightChoice', 'IncorrectLeftChoice', 'MissRightChoice',
'CorrectRejection','RightChoiceFalseP','LeftChoiceFalseP']

getdata.datatype_key()

test2 = getdata.recording_key()
test3 = getdata.datatype_key()
test4 = getdata.datatype_key(False,True)
#For searching steinmetz data names







test2 = getdata.calldata('Theiler_2017-10-11', ['times.'], 
                        steinmetzpath= datapath, propertysearch = True)



