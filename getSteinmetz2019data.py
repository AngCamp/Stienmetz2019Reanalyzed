# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:09:45 2021

@author: angus
 A module to load all Steinmetz et al., 2019 data
 
FUNCTIONS
calldata() -creates dictionaries containind pandas dataframes of Steimetz data
descriptions() -prints link to Steinmetz data descriptions
recording_key() -prints names of all recording sessions
datatype_key() -prints names of all datatypes in each session folder

"""
import numpy as np
import pandas as pd
import os
import warnings
import re

# A note on working with paths in python....
# https://www.btelligent.com/en/blog/best-practice-working-with-paths-in-python-part-1/


##############################################################################
def calldata(recording, list_of_data, steinmetzpath=os.getcwd(), propertysearch=False):
    """
    Returns dictionary of data from the Steinmetz et al., 2019 paper.  Data types
    described here: https://github.com/nsteinme/steinmetz-et-al-2019/wiki/data-files
    Based on the ALF naming conventions described here: https://int-brain-lab.github.io/iblenv/one_docs/one_reference.html#alf
    Data from here must be organized and extracted as https://figshare.com/articles/dataset/LFP_data_from_Steinmetz_et_al_2019/9727895

    Parameters:
        recording: A string argument to set the path to the recording session
        list_of_data: A list of strings of the data data to be called, sets will be made into lists
        stienmetzpath: the path to the steinmetz data, defaults to the current directory but can be manually set
        propertysearch: default to false, if true will search all data types
        for a specific propoerty like 'times.' or 'channels.', string must
        end in a period or an error will be thrown

    Returns:
        recording_dict: a dictionary with the keys set to be the .npy filenames without periods
        or the .npy extension.  So 'clusters.peakChannel.npy' becomes  stored
        as a numpy array in the dict with the key: clusterspeakChannel
        Defaults to searching spcifically for those strings provided.
        If propoerty search is true it will return all datatypes whose names
        contain that string.

    """
    # for testing and debugging, these three should be deleted at the end of this

    # checks to
    if propertysearch:
        for entry in list_of_data:
            if not (entry[-1:] == "."):
                raise ValueError(
                    "propertysearch is true, all list_of_data entries must be properties which end in periods '.'\
                                 Examples 'times.', please see https://github.com/nsteinme/steinmetz-et-al-2019/wiki/data-files\
                                 for description of acceptable datatypes and properties"
                )

    # all the recording sessions that should be present to
    allsteinmetzrecordings = {
        "Cori_2016-12-14",
        "Cori_2016-12-17",
        "Cori_2016-12-18",
        "Forssmann_2017-11-01",
        "Forssmann_2017-11-02",
        "Forssmann_2017-11-04",
        "Forssmann_2017-11-05",
        "Hench_2017-06-15",
        "Hench_2017-06-16",
        "Hench_2017-06-17",
        "Hench_2017-06-18",
        "Lederberg_2017-12-06",
        "Lederberg_2017-12-07",
        "Lederberg_2017-12-08",
        "Lederberg_2017-12-09",
        "Lederberg_2017-12-10",
        "Lederberg_2017-12-11",
        "Moniz_2017-05-15",
        "Moniz_2017-05-16",
        "Muller_2017-01-07",
        "Muller_2017-01-08",
        "Muller_2017-01-09",
        "Radnitz_2017-01-08",
        "Radnitz_2017-01-09",
        "Radnitz_2017-01-10",
        "Radnitz_2017-01-11",
        "Radnitz_2017-01-12",
        "Richards_2017-10-29",
        "Richards_2017-10-30",
        "Richards_2017-10-31",
        "Richards_2017-11-01",
        "Richards_2017-11-02",
        "Tatum_2017-12-06",
        "Tatum_2017-12-07",
        "Tatum_2017-12-08",
        "Tatum_2017-12-09",
        "Theiler_2017-10-11",
    }

    # Checking to make sure steinmetzpath contains the recordings data in the proper format
    def check_recordings_in_cwd():
        # checks steinmetzpath for missing recordings
        list_recs = os.listdir(steinmetzpath)
        
        
        mismatches = set(allsteinmetzrecordings).difference(list_recs)

        return mismatches

    def missing_recordings_warning(missing):
        # warning when recordings in directory are missing, i.e. mismatches is non-empty
        missing = list(missing)
        missing = " ".join(missing)
        warnings.warn(
            "Not all recordings from Steimetz et al., 2019 are in the chosen directory.\
                      Please re-enter the recording value or check steinmetz path.\
                      The following recordings are missing: "
            + missing
        )

    if not (propertysearch):
        missing_recordings = check_recordings_in_cwd()
        isEmpty = len(missing_recordings) == 0
        if not (isEmpty):
            missing_recordings_warning(missing_recordings)

    # warnings for incorrect recording variable entry
    def recording_must_be_string():
        warnings.warn(
            "recording variable must be a string.\
                      Please re-enter the recording value."
        )

    def incorrect_recording_entry(recording):
        warnings.warn(
            recording
            + " is not a Steinmetz et al., 2019 session name.\
                      Please re-enter the recording value."
        )

    if not (isinstance(recording, str)):
        recording_must_be_string()
    elif not (recording in allsteinmetzrecordings):
        incorrect_recording_entry(recording)

    # creating paths for for pulling data and set of all datatypes to be searched
    recording_path = os.path.join(steinmetzpath, recording)
    datatypes_set = set(os.listdir(recording_path))

    # check list_of_data variable is entered correctly
    def check_list_of_data(list_of_data):
        # checks steinmetzpath for missing recordings
        input_set = set(list_of_data)

        mismatches = set()
        if not (input_set in datatypes_set):
            mismatches = list(input_set.difference(datatypes_set))
        return mismatches

    def incorrect_entry_warning(missing):
        # warning when recordings in directory are missing, i.e. mismatches is non-empty
        missing = list(missing)
        missing = ", ".join(missing)
        warnings.warn(
            missing
            + " datatypes do not exist.\
                      Please re-enter the datatype values in list_of_data.\
                      Check https://github.com/nsteinme/steinmetz-et-al-2019/wiki/data-files to see acceptable datatype entries."
        )

    # if this is a datatype search we flag a warning for incorrect entries
    if not (propertysearch):
        incorrect_datatypes = check_list_of_data(list_of_data)
        isEmpty = len(incorrect_datatypes) == 0
        if not (isEmpty):
            incorrect_entry_warning(incorrect_datatypes)

    # actually fetching the values
    if not (isinstance(list_of_data, list)):
        list_of_data = list(list_of_data)

    def regular_search(datalist):
        # if property search is false use this
        out_dict = {}
        for entry in datalist:
            key = entry
            key = re.sub("[!@#$.-]", "", key)
            if entry[-3:] == "npy":
                out_dict[key[:-3]] = np.load(os.path.join(recording_path, entry))
            elif entry[-3:] == "tsv":
                out_dict[key[:-3]] = pd.read_csv(os.path.join(recording_path, entry), sep="\t", header=0)
        return out_dict

    def property_search(data_with_this_property):
        """if property is true use this to make a lsit to run through
        regular_search for each variable in list_of_data"""
        find_this_data = []
        for entry in list(datatypes_set):
            if data_with_this_property in entry:
                find_this_data.append(entry)
        return find_this_data

    if propertysearch:
        recording_dict = {}
        for property in list_of_data:
            sharedproperty = property_search(property)
            recording_dict[property[:-1]] = regular_search(sharedproperty)
    else:
        recording_dict = regular_search(list_of_data)

    # End of the calldata() function
    return recording_dict


########################################################################
# these variables are loaded so that they can be used for searching
def descriptions():
    """
    Prints link the github with data descriptions
    """
    print("Details of data type values found here: https://github.com/nsteinme/steinmetz-et-al-2019/wiki/data-files")


###########################################################################
def recording_key():
    """
    Returns the set of all recordigns
    """

    steinmetz_recordings_set = {
        "Cori_2016-12-14",
        "Cori_2016-12-17",
        "Cori_2016-12-18",
        "Forssmann_2017-11-01",
        "Forssmann_2017-11-02",
        "Forssmann_2017-11-04",
        "Forssmann_2017-11-05",
        "Hench_2017-06-15",
        "Hench_2017-06-16",
        "Hench_2017-06-17",
        "Hench_2017-06-18",
        "Lederberg_2017-12-06",
        "Lederberg_2017-12-07",
        "Lederberg_2017-12-08",
        "Lederberg_2017-12-09",
        "Lederberg_2017-12-10",
        "Lederberg_2017-12-11",
        "Moniz_2017-05-15",
        "Moniz_2017-05-16",
        "Muller_2017-01-07",
        "Muller_2017-01-08",
        "Muller_2017-01-09",
        "Radnitz_2017-01-08",
        "Radnitz_2017-01-09",
        "Radnitz_2017-01-10",
        "Radnitz_2017-01-11",
        "Radnitz_2017-01-12",
        "Richards_2017-10-29",
        "Richards_2017-10-30",
        "Richards_2017-10-31",
        "Richards_2017-11-01",
        "Richards_2017-11-02",
        "Tatum_2017-12-06",
        "Tatum_2017-12-07",
        "Tatum_2017-12-08",
        "Tatum_2017-12-09",
        "Theiler_2017-10-11",
    }

    return steinmetz_recordings_set


###############################################################################
def datatype_key(supress_git_link=True, structured=False):
    """
    Returns the set of all recordigns and link to more complete  description

    Parameters:
        suppress_git_link: default is true, prints link to github describing data in detail
        structured: default is false, if true returns dataset as a dictioanry with keys of grouping
        propoerties

    Returns:
        text for git link
        data_set: structured or unstructured datatypes name object
    """
    if not (structured):
        steinmetz_data_set = {
            "channels.brainLocation.tsv",
            "channels.probe.npy",
            "channels.rawRow.npy",
            "channels.site.npy",
            "channels.sitePositions.npy",
            "clusters.depths.npy",
            "clusters.originalIDs.npy",
            "clusters.peakChannel.npy",
            "clusters.probes.npy",
            "clusters.templateWaveformChans.npy",
            "clusters.templateWaveforms.npy",
            "clusters.waveformDuration.npy",
            "clusters._phy_annotation.npy",
            "eye.area.npy",
            "eye.blink.npy",
            "eye.timestamps.npy",
            "eye.xyPos.npy",
            "face.motionEnergy.npy",
            "face.timestamps.npy",
            "lickPiezo.raw.npy",
            "lickPiezo.timestamps.npy",
            "licks.times.npy",
            "passiveBeeps.times.npy",
            "passiveValveClick.times.npy",
            "passiveVisual.contrastLeft.npy",
            "passiveVisual.contrastRight.npy",
            "passiveVisual.times.npy",
            "passiveWhiteNoise.times.npy",
            "probes.description.tsv",
            "probes.insertion.tsv",
            "probes.rawFilename.tsv",
            "probes.sitePositions.npy",
            "sparseNoise.positions.npy",
            "sparseNoise.times.npy",
            "spikes.amps.npy",
            "spikes.clusters.npy",
            "spikes.depths.npy",
            "spikes.times.npy",
            "spontaneous.intervals.npy",
            "imec.lf.timestamps.npy",
            "trials.feedbackType.npy",
            "trials.feedback_times.npy",
            "trials.goCue_times.npy",
            "trials.included.npy",
            "trials.intervals.npy",
            "trials.repNum.npy",
            "trials.response_choice.npy",
            "trials.response_times.npy",
            "trials.visualStim_contrastLeft.npy",
            "trials.visualStim_contrastRight.npy",
            "trials.visualStim_times.npy",
            "wheel.position.npy",
            "wheel.timestamps.npy",
            "wheelMoves.intervals.npy",
            "wheelMoves.type.npy",
        }

    elif structured:
        steinmetz_data_set = {
            "channels.": [
                "channels.brainLocation.tsv",
                "channels.probe.npy",
                "channels.rawRow.npy",
                "channels.site.npy",
                "channels.sitePositions.npy",
            ],
            "clusters.": [
                "clusters.depths.npy",
                "clusters.originalIDs.npy",
                "clusters.peakChannel.npy",
                "clusters.probes.npy",
                "clusters.templateWaveformChans.npy",
                "clusters.templateWaveforms.npy",
                "clusters.waveformDuration.npy",
                "clusters._phy_annotation.npy",
            ],
            "eye.": ["eye.area.npy", "eye.blink.npy", "eye.timestamps.npy", "eye.xyPos.npy"],
            "face.": ["face.motionEnergy.npy", "face.timestamps.npy"],
            "lickPiezo.": ["lickPiezo.raw.npy", "lickPiezo.timestamps.npy"],
            "licks.": "licks.times.npy",
            "passive": [
                "passiveBeeps.times.npy",
                "passiveValveClick.times.npy",
                "passiveVisual.contrastLeft.npy",
                "passiveVisual.contrastRight.npy",
                "passiveVisual.times.npy",
                "passiveWhiteNoise.times.npy",
            ],
            "probes.": [
                "probes.description.tsv",
                "probes.insertion.tsv",
                "probes.rawFilename.tsv",
                "probes.sitePositions.npy",
            ],
            "sparseNoise.": [
                "sparseNoise.positions.npy",
                "sparseNoise.times.npy",
            ],
            "spikes.": ["spikes.amps.npy", "spikes.clusters.npy", "spikes.depths.npy", "spikes.times.npy"],
            "spontaneous.": "spontaneous.intervals.npy",
            ".imec.lf.timestamps": [
                "time stamps of probe recordings, first and last sample time, names and number of probes vary between recordings"
            ],
            "trials.": [
                "trials.feedbackType.npy",
                "trials.feedback_times.npy",
                "trials.goCue_times.npy",
                "trials.included.npy",
                "trials.intervals.npy",
                "trials.repNum.npy",
                "trials.response_choice.npy",
                "trials.response_times.npy",
                "trials.visualStim_contrastLeft.npy",
                "trials.visualStim_contrastRight.npy",
                "trials.visualStim_times.npy",
            ],
            "wheel": ["wheel.position.npy", "wheel.timestamps.npy", "wheelMoves.intervals.npy", "wheelMoves.type.npy"],
        }

    if not (supress_git_link):
        print(
            "Details of data type values found here: https://github.com/nsteinme/steinmetz-et-al-2019/wiki/data-files"
        )

    return steinmetz_data_set
