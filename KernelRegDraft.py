#Kernal Regression from Steimetz et al. (2019)
#
#Feb 6th 2022
#Angus Campbell

"""
We need to first reun CCA to generate B then we want to find the matrix a 
(also denoted as a matrix W with vectors w_n for each neruon n).  CCA
is first run from the toeplitz matrix of diagonalized kernel functions this will reduce
the dimensionality of the entire time course, we then optimize the weights of the copneents of 
this reduced representation.  Minimizations of square error is done by elastic net regularizatuion applied
on a neuron by neuron basis.  

Currently has matlab code sprinkled in comments to guide development.

"""

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

# Which neurons to include
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
