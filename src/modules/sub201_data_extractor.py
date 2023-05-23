import os
import numpy as np
import mne
import scipy
from modules.properties import sub201_properties as props
from modules.utils import eegFilters

def get_all_days(data_dir, eyes_state):
    dirPath = data_dir + '/RA' + eyes_state
    onlyfiles = [f for f in os.listdir(dirPath) if f.endswith('.mat')]
    days = []
    for item in onlyfiles:
        days.append(int(item.split('-')[1][3:]))
    
    return days

def createMontage(chanLabels):
    """
    Creates standard 10-20 location montage for given channel set
    """
    montageGeneral = mne.channels.make_standard_montage('standard_1020')
    locationDict = montageGeneral.get_positions()
    locationDict = locationDict['ch_pos']
    montageDict = {}
    
    for elec_i in chanLabels:
        montageDict[elec_i] = locationDict[elec_i]

    montage = mne.channels.make_dig_montage(montageDict)
    return montage 


def stackBlocks(eegDictList, block_N):
    """
    Stack blocks from same day into one EEG + labels dictionary
    """
    stackedList = []
    count = 0
    for i, eegDict in enumerate(eegDictList):
        if i % block_N == 0:
            tempArray = eegDict['segmentedEEG']
            tempLabels = eegDict['labels']
        else:
            tempArray = np.concatenate((tempArray, eegDict['segmentedEEG']))
            tempLabels = np.concatenate((tempLabels, eegDict['labels']))
            count += 1
        if count == block_N - 1:
            stackedDict = {'segmentedEEG': tempArray, 'labels': tempLabels, 'fs': eegDict['fs'],
           'chanLabels': eegDict['chanLabels'], 'trigLabels': eegDict['trigLabels'], 'trials_N': len(tempLabels)}
            stackedList.append(stackedDict)
            count = 0
    
    return stackedList


def segment_EEG(eegArrangedDict, trialLen, printFlag = 1):
    """
    Segment the data into epochs of MI and idle.
    """
    EEG = []
    labels = []
    removedCount = 0
    idleCount = 0
    imagineCount = 0
    
    # Timestamps of "move" command
    imgIdx = np.where(eegArrangedDict['triggers'] == 3)[0]
    # Timestamps of 1st pause
    idleIdx = np.where(eegArrangedDict['triggers'] == 2)[0]
    for idx in imgIdx:
            # Check if theres artifacts in trial (more then half the trial is labeled with artificats)          
        if np.sum(eegArrangedDict['artifacts'][idx + 1 : idx + 1 + int(trialLen * eegArrangedDict['fs'])]) > \
        trialLen * eegArrangedDict['fs'] * 0.9:
            removedCount += 1
            # Check that the trial is atleast as the given trial length (not ended before)
        elif np.sum(eegArrangedDict['triggers'][idx + 1 : idx + 1 + int(trialLen * eegArrangedDict['fs'])]) == 0:
            EEG.append(eegArrangedDict['EEG'][:, idx : idx + int(trialLen * eegArrangedDict['fs'])])
            labels.append(1)
            imagineCount += 1
        else:
            removedCount += 1
            
    for idx in idleIdx:
        if np.sum(eegArrangedDict['artifacts'][idx + 1 : idx + 1 + int(trialLen * eegArrangedDict['fs'])]) > 0:
            removedCount += 1
        else:
            EEG.append(eegArrangedDict['EEG'][:, idx : idx + int(trialLen * eegArrangedDict['fs'])])
            labels.append(0)
            idleCount += 1
    
    # Add to the dictionary the segmented data
    eegArrangedDict['segmentedEEG'] = np.asarray(EEG)
    eegArrangedDict['labels'] = np.asarray(labels)
    
    if printFlag:
        # Print number of trials of each class and number of removed trials
        print(f'Imagine Trials-{imagineCount} \nIdle Trials- {idleCount} \nRemoved Trials- {removedCount}\n')
    
    # Return the dictionary
    return eegArrangedDict

def arange_data(eegDict):
    """
    Arrange the given dictionary to more comfort dictionary
    """
    # EEG will be channels_N X timestamps_N
    EEG = eegDict['dat']['X'][0][0].T
    # Triggers
    triggers = np.squeeze(eegDict['dat']['Y'][0][0])
    # Artifacts marker
    artifacts = np.squeeze(eegDict['dat']['E'][0][0])
    # Sampling rate 
    fs = eegDict['header']['sampleFreq'][0][0][0][0]
    # Electrodes labels
    chanLabels = [ch[0] for ch in eegDict['header']['Xlabels'][0][0][0]]
    # Triggers labels
    trigLabels = [trig[0] for trig in eegDict['header']['Ymarkers'][0][0][0]]    
    # Trials time (in secs)
    imagineLength = eegDict['paramRA']['c_robot'][0][0][0][0]
    idleLength = eegDict['paramRA']['b_pause'][0][0][0][0]

    Data = {'EEG': EEG, 'triggers': triggers, 'artifacts': artifacts, 'fs': fs,
           'chanLabels': chanLabels, 'trigLabels': trigLabels, 'imagineLength': imagineLength,
           'idleLength': idleLength}
    return Data

def extract_data(data_dir, sub, eyes_state, day, block=[1]):
    """
    Iterate over days given, of specific subject and get a list of all the files of the relevant days
    """
    
    data = []
    dirPath = data_dir  + '/RA' + eyes_state
    for day_i in day:
        dayStr = str(day_i)
        if len(dayStr) == 1:
            dayStr = '0' + dayStr
        for block_i in block:
            fileFormat = 'sub' + sub + '-day' + dayStr + '-block' + str(block_i) + '-condRA' + eyes_state + '.mat'
            data.append(scipy.io.loadmat(dirPath + '/' +fileFormat))
    
    return data

def get_all_subs_EEG_dict(sub = '201', eyes_state = 'CC', block = [1]):
    data_dir = props['data_dir']
    n_days = get_all_days(data_dir, eyes_state)
    dataList = extract_data(data_dir, sub, eyes_state, n_days, block)

    # Extract and segment all the data
    dictList = []
    for dayData in dataList:
        # Extract each day data
        day_data = arange_data(dayData)
        
        # This condition is to remove some corrupted files in subject 201
        if day_data['EEG'].dtype != np.dtype('float64'):
            continue
            
        # Filter the data
        day_data['EEG'] = eegFilters(day_data['EEG'], day_data['fs'], props['filter_lim'])
        day_data['EEG'] = day_data['EEG'][props['elec_idxs'], :]

        # Segment the data
        dictList.append(segment_EEG(day_data, props['trial_len'], printFlag=0))

    # Stack block of same day
    EEG_dict = {sub : stackBlocks(dictList, len(block))}
    return EEG_dict