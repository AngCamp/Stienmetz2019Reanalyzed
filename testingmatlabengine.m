%%% test script to see if we can rewrite between matlab and jupyter using
%%% the matlab engine in jupyter

%adding the functions we need to read npy files
addpath('matlab_npy/','-end')
%testing will remove later
channels_of_interest = readNPY('these_channels.npy');

channels_of_interest = channels_of_interest*2;

