# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:07:24 2021

@author: angus

Getting LFP signals: https://github.com/cortex-lab/neuropixels/wiki/Other_analysis_methods#basic-lfp-characterization

Gillespie, A. K., Maya, D. A. A., Denovellis, E. L., Liu, D. F., Kastner, 
D. B., Coulter, M. E., ... & Frank, L. M. (2021). Hippocampal replay reflects 
specific past experiences rather than a plan for subsequent choice. bioRxiv.
https://www.biorxiv.org/content/10.1101/2021.03.09.434621v2.full



Denovellis, Eric L., Anna K. Gillespie, Michael E. Coulter, Marielena Sosa, 
Jason E. Chung, Uri T. Eden, and Loren M. Frank. "Hippocampal replay of 
experience at real-world speeds." bioRxiv (2021): 2020-10.
https://www.biorxiv.org/content/10.1101/2020.10.20.347708v5.full

https://github.com/Eden-Kramer-Lab/replay_trajectory_classification

-it is possible to set your own maze up with the model, circles are possible 
try using wheel position or turning speed as a "maze" for the decoders position 

this tutorial even has code to set up a circular track basically a wheel
https://github.com/Eden-Kramer-Lab/replay_trajectory_classification/blob/master/notebooks/tutorial/02-Decoding_with_Sorted_Spikes.ipynb
"""

import os
import numpy as np
import pandas as pd


os.chdir("C:/Users/angus/Desktop/SteinmetzLab/Analysis")

# we need to find out how many cells are required for replay


timestamp_tateprobe1 = np.load(
    r"C:\Users\angus\Desktop\SteinmetzLab\9598406\spikeAndBehavioralData\allData\Tatum_2017-12-09_K3_g0_t0.imec.lf.timestamps.npy"
)

"""
The LFP analysis:
    
https://github.com/cortex-lab/spikes/blob/master/analysis/lfpBandPower.m

That function does everything: open the file, pull in some data, and compute 
power spectra. You could stop it at line 34 to take "thisDat" (matrix of
channels x time) for some other analysis. Note that "lfpFs" (sampling 
frequency) is 2500 Hz, and "nChansInFile" is 385. That function loads data 
using "memmapfile" which makes it easy to pull out data segments from 
discontinuous parts of the file - but you can also just do "fread" to read a 
385 x Nsamples data segment of int16 datatype. 

LFP data:
https://figshare.com/articles/dataset/LFP_data_from_Steinmetz_et_al_2019/9727895
-currrently at 03/09/2021 I only have Tatum_2017


EXAMPLE OF PROBE TIMESTAMP

timestamp_tateprobe1
Out[16]: 
array([[0.00000000e+00, 1.38937876e+00],
       [7.45897100e+06, 2.98499185e+03]])

So the first sample in row 1 occured at ~1.39 seconds
and the last sample, the 745 897th sample occured at ~2985.99 seconds.
Using this information you can allign LFP data with the alex format behavioural
and spike/waveform data.

"""


"""
Reference to understand statespace modelling: 
Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.
-see chapter 18
"""
