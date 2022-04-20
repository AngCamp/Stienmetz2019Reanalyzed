""""
The purpose of this script is to analyze the behaviour of the animals specifically looking for effects described
in Lak et al., (2020) as well as Akrami et al., (2018).  Both publications show how previous reinforcement of correct trials results in the
choice of a particular stimuli on hard trials.  We can easily train and estimate such effects for mice.
In difficult trials mice tend to rely on previously reinforced decisions, this effect appears across a number of 
tasks in a variety of species (Lak et a., 2020).  Recent evidence is favored (Akrami et al., 2018) and 
behavioural states will fluctuate throughout a tasks run (Ashwood et al., 2020).  We can start with simpler analyses
and move on to more complex models if we want to explore these effects in more detail.


Lak, A., Hueske, E., Hirokawa, J., Masset, P., Ott, T., Urai, A. E., ... & Kepecs, A. (2020).
 Reinforcement biases subsequent perceptual decisions when confidence is low, a widespread 
 behavioral phenomenon. ELife, 9, e49834.

Akrami, A., Kopec, C. D., Diamond, M. E., & Brody, C. D. (2018).
 Posterior parietal cortex represents sensory history and mediates
 its effects on behaviour. Nature, 554(7692), 368-372.

 Espescially relevant is extended data 6 where they describe other models to the one they found was the best fit.
 https://www.nature.com/articles/nature25510/figures/10

Hidden Markov model incorperation engagement and reinforcement bias on a trial by trial level:

Ashwood, Z. C., Roy, N. A., Stone, I. R., Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022). 
Mice alternate between discrete strategies during perceptual decision-making. Nature Neuroscience, 25(2), 201-212.
-this was done by the IBL so may be of most use to us

"""

#CODE GOES HERE
#libraries
import pandas as pd
import numpy as np
import os
import scikit.learn
import KernelRegDraft as kreg
import getSteinmetz2019data as stein

#setting a filepath up
# NINC_filepath = find this out
# command to start python REPL in visual code is 'ctrl + r' then 'ctrl + p' 
anguslocal_filepath = os.fspath(r'C:\Users\angus\Desktop\SteinmetzLab\9598406\spikeAndBehavioralData\allData')

FILEPATH = anguslocal_filepath

# 1) LOOK FOR BIAS WITH A SIMPLE ANALYSIS
# Look for effect with linear model of engagement index and trial difficulty on bias







# 2) TRIAL HISTORY EFFECTS
# Akrami model







# 3) TRIAL BY TRIAL PREDICTION OF THESE STRATEGIES
# A) for removal and focusing on engaged trials
# B) for 



"""
# IMPLEMENTING ASHWOOD
# while I cannot use the same optimizations I should be able to build the same testing procedures
# python library to impliment HMM-GLM

https://github.com/janclemenslab/pyGLMHMM

-the writers of that repo rcomend using this repo....
https://github.com/lindermanlab/ssm
...and provide code below to implement it

as well as this line of code that shows how to use it....
true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, 
                      observations="input_driven_obs", 
                      observation_kwargs=dict(C=num_categories),
                      transitions="inputdriven")

And a notebook detailing GLM-HMM, written by Zoe Ashwood, author of the other study
https://github.com/lindermanlab/ssm/blob/master/notebooks/2b%20Input%20Driven%20Observations%20(GLM-HMM).ipynb

GLM-HMM is described in this paper: 
Calhoun, A. J., Pillow, J. W., & Murthy, M. (2019). 
Unsupervised identification of the internal states that shape natural behavior. 
Nature neuroscience, 22(12), 2040-2049.
-this is also cited in Ashwood et al., (2022)




"""









"""
Furhter Readings:
Other Effects to look into:
PPC as a history buffer, is this accomplished via hippocampus or more locally driven?

 Bitzidou, M., Bale, M. R., & Maravall, M. (2018). Cortical Lifelogging: the posterior parietal cortex as sensory history buffer. 
Neuron, 98(2), 249-252.

Jarzebowski, P. (2022). Encoding and recall of memory for reward location in the mouse hippocampus
 (Doctoral dissertation, University of Cambridge).
https://www.repository.cam.ac.uk/handle/1810/334631
-pdf downloaded in Seteinmetzlab on your pc as PhDThesis-Jarzebowski-2022.pdf

Worth reading as well:
Zabeh, E., Foley, N. C., Jacobs, J., & Gottlieb, J. P. (2022). 
Traveling waves in the monkey frontoparietal network predict recent reward memory. bioRxiv.
https://www.biorxiv.org/content/10.1101/2022.02.03.478583v1.abstract



"""