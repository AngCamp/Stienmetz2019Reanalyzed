# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 14:17:07 2021

@author: angus

Written as an attmept to be able to search the sitenment list code names without retunring data
ended up being more time than it was worth

"""


def data_options( searchlist = [], ):
    """
    Shows options for various data types stored in spiking an behavioural data.
    """
    
    data_set = set('channels.brainLocation.tsv',
                   'channels.probe.npy',
                   'channels.rawRow.npy',
                   'channels.site.npy',
                   'channels.sitePositions.npy',
                   'clusters.depths.npy',
                   'clusters.originalIDs.npy',
                   'clusters.peakChannel.npy',
                   'clusters.probes.npy',
                   'clusters.templateWaveformChans.npy',
                   'clusters.templateWaveforms.npy',
                   'clusters.waveformDuration.npy',
                   'clusters._phy_annotation.npy',
                   'eye.area.npy',
                   'eye.blink.npy',
                   'eye.timestamps.npy',
                   'eye.xyPos.npy',
                   'face.motionEnergy.npy',
                   'face.timestamps.npy',
                   'lickPiezo.raw.npy',
                   'lickPiezo.timestamps.npy',
                   'licks.times.npy',
                   'passiveBeeps.times.npy',
                   'passiveValveClick.times.npy',
                   'passiveVisual.contrastLeft.npy',
                   'passiveVisual.contrastRight.npy',
                   'passiveVisual.times.npy',
                   'passiveWhiteNoise.times.npy',
                   'probes.description.tsv',
                   'probes.insertion.tsv',
                   'probes.rawFilename.tsv',
                   'probes.sitePositions.npy',
                   'sparseNoise.positions.npy',
                   'sparseNoise.times.npy',
                   'spikes.amps.npy',
                   'spikes.clusters.npy',
                   'spikes.depths.npy',
                   'spikes.times.npy',
                   'spontaneous.intervals.npy',
                   'imec.lf.timestamps.npy',
                   'trials.feedbackType.npy',
                   'trials.feedback_times.npy',
                   'trials.goCue_times.npy',
                   'trials.included.npy',
                   'trials.intervals.npy',
                   'trials.repNum.npy',
                   'trials.response_choice.npy',
                   'trials.response_times.npy',
                   'trials.visualStim_contrastLeft.npy',
                   'trials.visualStim_contrastRight.npy',
                   'trials.visualStim_times.npy',
                   'wheel.position.npy',
                   'wheel.timestamps.npy',
                   'wheelMoves.intervals.npy',
                   'wheelMoves.type.npy')
    
    def property_search(data_with_this_property):
        """if property is true use this to make a lsit to run through
        regular_search for each variable in list_of_data"""
        find_this_data=[]
        for entry in list(data_set):
            if data_with_this_property in entry:
                find_this_data.append(entry)
        return(find_this_data) 
    
    
    search = (len(searchlist) == 0)
    if search:
        matches = list(searchlist in data_list)
        if (len(matches) == 0):
            matches = {}
            for entry in matches:
                matches[entry] = property_search(entry)
    
    
    
    
    
    
    
    
    def search_list(list):
    if not(search):
        
        
def recording_options( searchlist = [], ):
    """
    Shows options for various data types stored in spiking an behavioural data.
    """
    isEmpty = (len(searchlist) == 0)
    if isEmpty:
        search = False
    else:
        search = True
    
    recordings_set = set('Cori_2016-12-14', 'Cori_2016-12-17', 'Cori_2016-12-18',
    'Forssmann_2017-11-01', 'Forssmann_2017-11-02', 'Forssmann_2017-11-04', 'Forssmann_2017-11-05',
    'Hench_2017-06-15', 'Hench_2017-06-16', 'Hench_2017-06-17', 'Hench_2017-06-18',
    'Lederberg_2017-12-06', 'Lederberg_2017-12-07', 'Lederberg_2017-12-08', 'Lederberg_2017-12-09',
    'Lederberg_2017-12-10', 'Lederberg_2017-12-11',
    'Moniz_2017-05-15', 'Moniz_2017-05-16', 'Muller_2017-01-07', 'Muller_2017-01-08',
    'Muller_2017-01-09',
    'Radnitz_2017-01-08', 'Radnitz_2017-01-09', 'Radnitz_2017-01-10', 'Radnitz_2017-01-11',
    'Radnitz_2017-01-12',
    'Richards_2017-10-29', 'Richards_2017-10-30', 'Richards_2017-10-31', 'Richards_2017-11-01',
    'Richards_2017-11-02',
    'Tatum_2017-12-06', 'Tatum_2017-12-07', 'Tatum_2017-12-08', 'Tatum_2017-12-09',
    'Theiler_2017-10-11')