import numpy as np
import mne
import os

from .properties import IEEE_properties as props
from .utils import remove_noisy_trials

class LazyProperty:
    def __init__(self, method):
        self.method = method
        self.method_name = method.__name__
        # print('function overriden: {}'.format(self.fget))
        # print("function's name: {}".format(self.func_name))

    def __get__(self, obj, cls):
        if not obj:
            return None
        value = self.method(obj)
        # print('value {}'.format(value))
        setattr(obj, self.method_name, value)
        return value


class data_4class(object):
    def __init__(self, path, tmin, tmax):
        ch_names = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2',
            'CP4', 'CP6', 'hEOG', 'vEOG', 'F5', 'AF3', 'AF4', 'P5', 'P3', 'P1',
            'Pz', 'P2', 'P4', 'P6', 'PO3', 'POz', 'PO4', 'Oz', 'F6']  # 41ch EEG + 2ch EOG
        ch_types = ['eeg'] * 26 + ['eog'] * 2 + ['eeg'] * 15
    
        self.path = path
        npz_data = np.load(path)
        self.npz_data_dict = dict(npz_data)
        self.fs = self.npz_data_dict['SampleRate'][0]
        self.index = self.npz_data_dict['MarkOnSignal'].copy()
        self.index[:, 0] = self.index[:, 0]
        data = self.npz_data_dict['signal']  # (samples, channels)
        self.data = data[:, 0:28] if data.shape[1] == 29 \
            else np.delete(data, [28, 29, 45], 1)  # 去M1 M2 事件通道
#         self.data = data[:, 0:28]
#         self.ch_names, self.ch_types = ch_names[:28], ch_types[:28]
        self.ch_names, self.ch_types = (ch_names[:28], ch_types[:28]) \
            if self.data.shape[1] == 28 else (ch_names, ch_types)
        
        self.events = None
        self.epoch_data = None
        self._info = None
        self._raw_mne = None
        self._epochs_mne = None
        self.mne_scallings = dict(eeg=20, eog=500)
        self.tmin = tmin
        self.tmax = tmax
        
    @LazyProperty
    def info(self):
        self._info = self.create_info()
        return self._info

    @LazyProperty
    def raw_mne(self):
        self._get_epoch_event()
#         self.info['events'] = self.events
        self._raw_mne = mne.io.RawArray(self.data.T, self.info)  # RawArray input (n_channels, n_times)
        # self._raw_mne.pick_types(eog=False, eeg=True)
        return self._raw_mne

    @LazyProperty
    def epochs_mne(self):
        # epochs: (n_epochs, n_chans, n_times)
        if self.events is None:
            self._get_epoch_event()
        # reject_criteria = dict(eeg=150e-6, eog=250e-6)  # eeg150 µV  eog250 µV Exclude the signal with large amplitude
        self._epochs_mne = mne.Epochs(self.raw_mne, self.events, self.event_id, tmin=-4, tmax=5, baseline=None,
                                      preload=True)
        return self._epochs_mne

    def spectral_conv(self, fmin=0, fmax=np.inf, tmin=-2, tmax=0, method='pli'):
        # Functional connection
        epochs = self.epochs_mne.pick_types(meg=False, eeg=True)
        # epochs = epochs.drop_channels(['M1', 'M2']) if len(self.ch_names) == 45 else epochs
        epochs = epochs.crop(tmin=tmin, tmax=tmax, include_tmax=True)
        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            epochs, method=method, mode='multitaper', sfreq=self.fs, fmin=fmin, fmax=fmax,
            faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)
        return con[:, :, 0], epochs.ch_names

    def plot_connectivity_2Dcircle(self, con, labels):
        node_angles = circular_layout(labels, labels, start_pos=90, group_boundaries=[0, len(labels) / 2])
        # Plot the graph using node colors from the FreeSurfer parcellation. We only
        # show the 100 strongest connections.
        plot_connectivity_circle(con, labels, n_lines=100, node_angles=node_angles,
                                 title='All-to-All Connectivity left-Auditory Condition (PLI)')

    def psd_multitaper(self, type=None, fmin=0, fmax=np.inf, tmin=None, tmax=None):
        # psds:
        # if raw input: (n_channels, n_freqs)
        # if epoch input: (n_epochs, n_channels, n_freqs)
        type_dict = dict(raw=self.raw_mne, epoch=self.epochs_mne)
        inst = type_dict[type] if type else self.raw_mne
        psds, freqs = mne.time_frequency.psd_multitaper(inst, low_bias=True, tmin=tmin, tmax=tmax,
                                                        fmin=fmin, fmax=fmax, proj=True, picks='eeg', n_jobs=1)
        return psds, freqs

    def psd_welch(self, type=None, fmin=0, fmax=np.inf, tmin=None, tmax=None, event_label=None):
        # psds:
        # if raw input: (n_channels, n_freqs)
        # if epoch input: (n_epochs, n_channels, n_freqs)
        type_dict = dict(raw=self.raw_mne, epoch=self.epochs_mne)
        inst = type_dict[type] if type else self.raw_mne
        if event_label:
            inst = inst[event_label]
        psds, freqs = mne.time_frequency.psd_welch(inst, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, proj=True, picks='eeg', n_jobs=1)
        return psds, freqs

    def _get_epoch_event(self, tmin=0, tmax=5):
        # epoch_data: n_samples, n_channal, n_trial
        j = 0
        tmin = self.tmin
        tmax = self.tmax
        delete_epoch_idx = []
        channal_num = self.data.shape[1]
        trial_num = len(np.where(self.index[:, 1] == 800)[0])
        self.epoch_data = np.zeros([(tmax - tmin) * self.fs, channal_num, trial_num])
        self.events = np.zeros([trial_num, 3], dtype=np.int)
        for i in range(self.index.shape[0]):
            if self.index[i, 1] in [769, 770, 771, 780]:
                try:
                    start = self.index[i, 0] + tmin * self.fs
                    end = self.index[i, 0] + tmax * self.fs
                    self.epoch_data[:, :, j] = self.data[start:end, :]
                    self.events[j, 0] = self.index[i, 0]
                    self.events[j, 2] = self.index[i, 1] - 768
                    j += 1
                except ValueError:
                    delete_epoch_idx.append(j)  # Data Loss
                except IndexError:
                    print(self.path)
                    print('Index ', j, ' out of bounds, incomplete data file!')  # The experimental interrupt
                    break
        if delete_epoch_idx:
            print(self.path)
            print('data lost at trial', delete_epoch_idx)
            self.epoch_data = np.delete(self.epoch_data, delete_epoch_idx, 2)
            self.events = np.delete(self.events, delete_epoch_idx, 0)
        self.events[np.where(self.events[:, 2] == 12), 2] = 4
        self.event_id = dict(left=1, right=2, foot=3, rest=4)
        # label = events[:, -1]

    def get_raw_data(self):
        # return (samples, channels)
        if self._raw_mne:
            return self.raw_mne.get_data(picks='eeg').T  # get_data (n_channels, n_times)
        else:
            return self.data[:, :-2]

    def set_raw_data(self, raw_data):
        # (samples, channels)
        self.eog = self.data[:, -2:]
        self.data = np.concatenate((raw_data, self.eog), axis=1)

    def get_epoch_data(self, tmin=-1, tmax=5, select_label=None):
        # return: (sample, channel, trial) label:1-left 2-right 3-foot 4-rest
        if self._epochs_mne:
            label = self.epochs_mne.events[:, -1]
            epochs = self.epochs_mne.crop(tmin=tmin, tmax=tmax)
            data = epochs.get_data(picks='eeg')  # (n_epochs, n_channels, n_times)
            data = data.swapaxes(0, 2)
#             data = data[:, :26, :]  # 取前26EEG通道
            data = np.delete(data, [26, 27], 1)  # 45通道移除M1 M2
        else:
            if self.epoch_data is None:
                self._get_epoch_event(tmin, tmax)
            data = self.epoch_data
            label = self.events[:, -1]
        # data = np.delete(data, [4, 20], 1)  # 移除f4 cp3
        if select_label:
            idx = [i for i in range(len(label)) if label[i] in select_label]
            label = label[idx]
            data = data[:, :, idx]

        return data, label

    def create_info(self):
        montage = 'standard_1005'
        info = mne.create_info(self.ch_names, self.fs, self.ch_types)
        info.set_montage(montage)
        return info

    def plot_raw_psd(self, fmin=1, fmax=40):
        self.raw_mne.pick_types(eeg=True, meg=False, stim=False, eog=True)
        self.raw_mne.plot(duration=5, n_channels=28, show=True, scalings=self.mne_scallings)
        self.raw_mne.plot_psd(fmin, fmax, n_fft=2 ** 10, spatial_colors=True)  # 功率谱

    def plot_events(self):
        if self.events is None:
            self._get_epoch_event()
        mne.viz.plot_events(self.events, event_id=self.event_id, sfreq=self.fs, first_samp=self.raw_mne.first_samp)

    def plot_epoch(self, fmin=1, fmax=40):
        self.epochs_mne.pick_types(eeg=True, meg=False, stim=False, eog=True)
        self.epochs_mne.plot(scalings=self.mne_scallings, n_epochs=5, n_channels=28)
        self.epochs_mne.plot_psd(fmin, fmax, average=True, spatial_colors=True)
        self.epochs_mne.plot_psd_topomap()

    def plot_oneclass_epoch(self, event_name, ch_name):
        oneclass_epoch = self.epochs_mne[event_name]
        oneclass_epoch.plot_image(picks=ch_name)
        self.epochs_mne.plot_psd_topomap()

    def plot_tf_analysis(self, event_name, ch_name):
        # time-frequency analysis via Morlet wavelets
        event_epochs = self.epochs_mne[event_name]
        frequencies = np.arange(7, 30, 3)
        power = mne.time_frequency.tfr_morlet(event_epochs, n_cycles=2, return_itc=False, freqs=frequencies, decim=3)
        power.plot(picks=ch_name, baseline=(-2, 0), mode='logratio', title=event_name + ch_name)

    def bandpass_filter(self, l_freq=1, h_freq=40):
        self.raw_mne.filter(l_freq, h_freq, verbose='warning')
        # filter_params = mne.filter.create_filter(self.raw_mne.get_data(), self.fs, l_freq=l_freq, h_freq=h_freq)
        # mne.viz.plot_filter(filter_params, self.fs, flim=(0.01, 5))  # plot 滤波器

    def notch_filter(self):
        self.raw_mne.notch_filter(freqs=np.arange(50, 250, 50), notch_widths=1)  # ?

    def set_reference(self, ref='average'):
        # ref: CAR
        self.raw_mne.set_eeg_reference(ref_channels=ref, projection=True).apply_proj()
        # self.raw_mne.set_eeg_reference(ref_channels=['M1','M2'], projection=True).apply_proj()

    def plot_ICA_manually(self):
        # 手动选择伪迹IC 并去除
        ica = mne.preprocessing.ICA(n_components=12, random_state=97, max_iter=800)
        ica.fit(self.raw_mne)
        ica.plot_sources(inst=self.raw_mne)
        ica.plot_components(inst=self.raw_mne)
        orig_raw = self.raw_mne.copy()
        ica.apply(self.raw_mne)
        orig_raw.plot(start=0, duration=5, n_channels=28, block=False, scalings=self.mne_scallings)
        self.raw_mne.plot(start=0, duration=5, n_channels=28, block=True, scalings=self.mne_scallings)

    def removeEOGbyICA(self):
        # 根据EOG寻找相似IC 去眼电
        ica = mne.preprocessing.ICA(n_components=12, random_state=97, max_iter=800)
        ica.fit(self.raw_mne)
        ica.exclude = []
        eog_indices, eog_scores = ica.find_bads_eog(self.raw_mne, threshold=2.3)
        ica.exclude = eog_indices
        # orig_raw = self.raw_mne.copy()
        ica.apply(self.raw_mne)
        # ica.plot_scores(eog_scores)
        # ica.plot_properties(orig_raw, picks=eog_indices)  # plot diagnostics
        # ica.plot_sources(orig_raw)  # plot ICs applied to raw data, with EOG matches highlighted
        # eog_epoch = mne.preprocessing.create_eog_epochs(orig_raw, tmin=-1.5, tmax=1.5, thresh=200).average()
        # eog_epoch.apply_baseline(baseline=(-1, -0.5))
        # eog_epoch.plot_joint()
        # ica.plot_sources(eog_epoch)
        # orig_raw.plot(start=0, duration=5, n_channels=28, block=False, scalings=self.mne_scallings)
        # self.raw_mne.plot(start=0, duration=5, n_channels=28, block=True, scalings=self.mne_scallings)


# In[ ]:


def extract_ieee_data(sub, filterLim, tmin=-1, tmax=5, select_label=None, data_dir = 'data/ieee_dataset/'):

    # ch_names = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    #             'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2',
    #             'CP4', 'CP6', 'hEOG', 'vEOG', 'F5', 'AF3', 'AF4', 'P5', 'P3', 'P1',
    #             'Pz', 'P2', 'P4', 'P6', 'PO3', 'POz', 'PO4', 'Oz', 'F6']  # 41ch EEG + 2ch EOG
    # ch_types = ['eeg'] * 26 + ['eog'] * 2 + ['eeg'] * 15

    data_dir = os.path.abspath(data_dir + sub)
    all_days_dirs = os.listdir(data_dir)
    all_days_data = []

    for days in all_days_dirs:
        all_day_trials = []
        all_day_labels = []

        files_ind_dir = os.listdir(data_dir + '/' + days)

        for file in files_ind_dir:
            file_path = data_dir + '/' + days+ '/' + file
            d = data_4class(file_path, tmin, tmax)
            d.data = np.delete(d.data, [26,27] , 1)  #remove weird channels
            del d.ch_names[26 : 28]
            del d.ch_types[26 : 28]
            d.bandpass_filter(l_freq=filterLim[0], h_freq=filterLim[1])
    #         d.data = mne.filter.filter_data(d.data.T , d.fs, filterLim[0], filterLim[1], verbose=0)
            d.data = d.data.T

            trail_data, label = d.get_epoch_data(tmin, tmax, select_label)
            all_day_trials.append(trail_data)
            all_day_labels.append(label)
        segmentedEEG = np.concatenate(all_day_trials, axis = 2)
        segmentedEEG = np.swapaxes(segmentedEEG, 0, -1)
        labels = np.concatenate(all_day_labels)
        fs = d.fs
        stackedDict = {'segmentedEEG': segmentedEEG, 'labels': labels, 'sub': sub, 'fs': fs,
               'chanLabels': d.ch_names[:26], 'trigLabels': ['left', 'right', 'foot', 'rest'], 'trials_N': len(labels)}

        all_days_data.append(stackedDict)
    
    return all_days_data


# In[ ]:


def extract_ieee_data_by_sub(sub_list, filterLim, tmin=-1, tmax=5, select_label=None, data_dir = 'data/ieee_dataset/'):

    all_subs_data = []
    for sub in sub_list:
        data_dir = os.path.abspath('data/ieee_dataset/' + sub)
        all_days_dirs = os.listdir(data_dir)
        all_days_data = []
        all_day_trials = []
        all_day_labels = []
        for days in all_days_dirs:

            files_ind_dir = os.listdir(data_dir + '/' + days)

            for file in files_ind_dir:
                file_path = data_dir + '/' + days+ '/' + file
                d = data_4class(file_path, tmin, tmax)
                d.data = np.delete(d.data, [26,27] , 1)  #remove weird channels
                d.ch_names = d.ch_names[:26]
                d.ch_types = d.ch_types[:26]
                d.bandpass_filter(l_freq=filterLim[0], h_freq=filterLim[1])
        #       d.data = mne.filter.filter_data(d.data.T , d.fs, filterLim[0], filterLim[1], verbose=0)
                d.data = d.data.T

                trail_data, label = d.get_epoch_data(tmin, tmax, select_label)
                all_day_trials.append(trail_data)
                all_day_labels.append(label)
                
        segmentedEEG = np.concatenate(all_day_trials, axis = 2)
        segmentedEEG = np.swapaxes(segmentedEEG, 0, -1)
        labels = np.concatenate(all_day_labels)
        fs = d.fs
            
        stackedDict = {'segmentedEEG': segmentedEEG, 'labels': labels, 'sub': sub, 'fs': fs,
               'chanLabels': d.ch_names[:26], 'trigLabels': ['left', 'right', 'foot', 'rest'], 'trials_N': len(labels)}

        all_subs_data.append(stackedDict)
    
    return all_subs_data


def get_all_subs_EEG_dict():
    all_sub_EEG_dict = {}
    
    try:
        for sub in props['sub_list']:  
            dictListStacked = extract_ieee_data(
                sub, props['filterLim'],props['tmin'], props['tmax'], props['select_label'], props['data_dir'])
        
            # Remove noisy trials using amplitude threshold
            clean_EEG = remove_noisy_trials(dictListStacked, props['amplitude_th'], props['min_trials'])
            all_sub_EEG_dict[sub] = clean_EEG
    
        return all_sub_EEG_dict
        
    except Exception as e:
        print(e)
        print(f'Could\'nt load data files for sub')
        




