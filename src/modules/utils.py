#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import mne
import scipy
import torch
import sklearn
import lightgbm as lgb

from scipy.io import savemat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from torch.utils.data import random_split, DataLoader, Dataset





# In[ ]:


def csp_score(signal, labels, cv_N = 5, classifier = False):
    
    # Set verbose to 0
    mne.set_log_level(verbose='WARNING', return_old_level=False, add_frames=None)

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    
    if classifier:
        y_pred = classifier.predict(signal)
        acc = sklearn.metrics.accuracy_score(labels, y_pred)
        return acc
    
    else:
        # Assemble a classifier
        svm = sklearn.svm.SVC()
        lda = LinearDiscriminantAnalysis()
#         lda = sklearn.ensemble.RandomForestClassifier()
        csp = mne.decoding.CSP(n_components=99, reg=None, log=False, norm_trace=True)
        # Use scikit-learn Pipeline with cross_val_score function
        clf = Pipeline([('CSP', csp), ('LDA', lda)])
#         clf = Pipeline([('CSP', csp), ('SVM', svm)])
        scores = cross_val_score(clf, signal, labels, cv=cv_N, n_jobs=1)
        _ = clf.fit(signal, labels)
        return np.mean(scores), clf


# In[ ]:



def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    ly += yerr[num1]
    ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh+0.05)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


def origin_day_clf(EEGdict, AE_model):
    # Use day zero classifier for classifying the reconstructed eeg per day
    
    # get relevant data
    signal_test_data = EEGDataSet_signal_by_day(EEGdict, [0, len(EEGdict)])
    orig_signal, _, labels = signal_test_data.getAllItems()
    rec_signal = AE_model(orig_signal).detach().numpy()
    res_signal = orig_signal - rec_signal
    
    # change labels from 1hot to int
    labels = np.argmax(labels, axis=1)
    
    score_orig, _ = csp_score(np.float64(orig_signal.detach().numpy()), labels, cv_N = 5, classifier = False)
    score_rec, _ = csp_score(np.float64(rec_signal), labels, cv_N = 5, classifier = False)
    score_res, _ = csp_score(np.float64(res_signal), labels, cv_N = 5, classifier = False)
    return score_orig, score_rec, score_res



class EEGDataSet_signal_by_day(Dataset):
    def __init__(self, EEGDict, days_range=[0,1]):
        
        # Concat dict      
        X, y, days_y = self.concat(EEGDict, days_range)
        

        
        # Convert from numpy to tensor
        self.X = torch.tensor(X)
        self.n_samples = self.X.shape[0]
        self.n_channels = self.X.shape[1]
        self.y = y
        self.days_y = days_y
        self.days_labels_N = days_range[1] - days_range[0]
        self.task_labels_N = y.shape[1]

        
    def __getitem__(self, index):
        return self.X[index].float(), self.y[index], self.days_y[index]
    
    def __len__(self):
        return self.n_samples
    
    def getAllItems(self):
        return self.X.float() , self.y, self.days_y
    
        
    def concat(self, EEGDict, days_range):
        X = []
        y = []
        days_y = []
        for day, d in enumerate(EEGDict[days_range[0]:days_range[1]]):
            X.append(d['segmentedEEG'])
            y.append(d['labels'])
            days_y.append(np.ones_like(d['labels']) * day)

        X = np.asarray(X)
        y = np.asarray(y)
        X = np.concatenate(X)
        y = np.concatenate(y)
        days_y = np.concatenate(days_y)
        #  one hot encode days labels
        y_temp = np.zeros((days_y.size, days_y.max() + 1))
        y_temp[np.arange(days_y.size), days_y] = 1
        days_y = y_temp
        # One hot encode task labels
        y_temp = np.zeros((y.size, y.max() + 1))
        y_temp[np.arange(y.size), y] = 1
        y = y_temp
        return X, y, days_y


class EEGDataSet_signal(Dataset):
    def __init__(self, EEGDict, days_range=[0,1]):
        
        # Concat dict      
        X, y = self.concat(EEGDict, days_range)
        

        
        # Convert from numpy to tensor
        self.X = torch.tensor(X)
        self.n_samples = self.X.shape[0]
        self.n_channels = self.X.shape[1]
        self.y = y

        
    def __getitem__(self, index):
        return self.X[index].float(), self.y[index]
    
    def __len__(self):
        return self.n_samples
    
    def getAllItems(self):
        return self.X.float() , self.y
    
    def concat(self, EEGDict, days_range):
        X = []
        y = []
        for d in EEGDict[days_range[0]:days_range[1]]:
            X.append(d['segmentedEEG'])
            y.append(d['labels'])

        X = np.asarray(X)
        y = np.asarray(y)
        X = np.concatenate(X)
        y = np.concatenate(y)
        return X, y


def remove_noisy_trials(dictListStacked, amp_thresh, min_trials):
    # Remove noisy trials using amplitude threshold
    new_dict_list = []
    for i, D in enumerate(dictListStacked):
        max_amp = np.amax(np.amax(D['segmentedEEG'], 2), 1)
        min_amp = np.amin(np.amin(D['segmentedEEG'], 2), 1)
        max_tr = max_amp > amp_thresh
        min_tr = min_amp < -amp_thresh
        noisy_trials = [a or b for a, b in zip(max_tr, min_tr)]
        D['segmentedEEG'] = np.delete(D['segmentedEEG'], noisy_trials,axis=0)
        D['labels'] = np.delete(D['labels'], noisy_trials,axis=0)
    #    # One hot the labels
    #     D['labels'][D['labels']==4] = 3
    #     D['labels'] = torch.as_tensor(D['labels']).to(torch.int64) - 1
    #     D['labels'] = F.one_hot(D['labels'], 3)
        if D['segmentedEEG'].shape[0] > min_trials:
                new_dict_list.append(D)

    return new_dict_list


def eegFilters(eegMat, fs, filterLim):
    eegMatFiltered = mne.filter.filter_data(eegMat, fs, filterLim[0], filterLim[1], verbose=0)
    return eegMatFiltered




def training_loop(train_days, dictListStacked, ae_learning_rt, convolution_filters, batch_sz, epoch_n, proccessor):
    
    # check if enough train days exists
    if train_days[1] >= len(dictListStacked):
        raise Exception("Not enough training days")


    # device settings
    device = torch.device(proccessor)
    accelerator = proccessor if proccessor=='cpu' else 'gpu' 
    devices = 1 if proccessor=='cpu' else -1 
    
    # Logger
    logger = TensorBoardLogger('../tb_logs', name='EEG_Logger')
    # Shuffle the days
    random.shuffle(dictListStacked)
    # Train Dataset
    signal_data = EEGDataSet_signal_by_day(dictListStacked, train_days)
    signal_data_loader = DataLoader(dataset=signal_data, batch_size=batch_sz, shuffle=True, num_workers=0)
    x, y, days_y = signal_data.getAllItems()
    y = np.argmax(y, -1)
    days_labels_N = signal_data.days_labels_N
    task_labels_N = signal_data.task_labels_N

    # Train model on training day
    metrics = ['classification_loss', 'reconstruction_loss']
    day_zero_AE = convolution_AE(signal_data.n_channels, days_labels_N, task_labels_N, ae_learning_rt, filters_n=convolution_filters, mode='supervised')
    day_zero_AE.to(device)

    trainer_2 = pl.Trainer(max_epochs=epoch_n, logger=logger, accelerator=accelerator , devices=devices)
    trainer_2.fit(day_zero_AE, train_dataloaders=signal_data_loader)
    
    # CV On the training set (with and without ae)
    ws_ae_train, day_zero_AE_clf = csp_score(np.float64(day_zero_AE(x).detach().numpy()), y, cv_N=5, classifier=False)
    ws_train, day_zero_bench_clf = csp_score(np.float64(x.detach().numpy()), y, cv_N=5, classifier=False)
    

    test_days = [train_days[1], len(dictListStacked)]

    # Create test Datasets
    signal_test_data = EEGDataSet_signal(dictListStacked, test_days)

    # get data
    signal_test, y_test = signal_test_data.getAllItems()
    # reconstruct EEG using day 0 AE
    rec_signal_zero = day_zero_AE(signal_test).detach().numpy()


    # Use models
    # within session cv on the test set (mean on test set)
    ws_test, _ = csp_score(np.float64(signal_test.detach().numpy()), y_test, cv_N=5, classifier = False)
    # Using day 0 classifier for test set inference (mean on test set)
    bs_test = csp_score(np.float64(signal_test.detach().numpy()), y_test, cv_N=5, classifier=day_zero_bench_clf)
    # Using day 0 classifier + AE for test set inference (mean on test set)
    bs_ae_test = csp_score(rec_signal_zero, y_test, cv_N=5, classifier=day_zero_AE_clf)
    
    return ws_train, ws_ae_train, ws_test, bs_test, bs_ae_test, day_zero_AE


# #### Load the files - IEEE

def run_all_subs_multi_iterations(props, subs_EEG_dict, train_days_range = [1,7], iterations_per_day = 250):
    
# This function runs multi iterations experiment over all subjects.
# The experiment runs all ranges of traning days from 0-{train_days_range[0]} to 0-{train_days_range[1]}.
# Every iteration models are trained for all ranges of training days and all subjects.
# the function saves 2 dictionaries to 2 files:
    # task clasification results dictionary
    # origin day clasification results dictionary
# The function returns the 2 file pathes
    
    ts = time.strftime("%Y%m%d-%H%M%S")
    
    task_iter_dict = {} # keys: iterations, vals: dict of dicts of dicts of scores for each sub
    origin_iter_dict = {} 
    
    for itr in range(iterations_per_day):
        task_days_range_dict = {} # keys: train days range, vals: dict of dicts of scores for each sub
        origin_days_range_dict = {}
        
        for last_train_day in range(train_days_range[0],train_days_range[1]):
            task_sub_dict = {} # keys: sub, vals: dict of list of the scores dicts for each sub
            origin_sub_dict = {}
            
            curr_days_rng=[0, last_train_day] # determine the current range for training days 
            rng_str = '-'.join(str(e) for e in curr_days_rng) # turn days range list to str to use as key name
                  
            for sub in list(subs_EEG_dict.keys()):
                print(f'\niter: {itr}, last training day: {last_train_day}, sub: {sub}...')
                
                task_per_sub_scores_dict = {} # keys: method(ws,bs,AE), vals: scores
                origin_per_sub_scores_dict = {} # keys: signal(orig,rec,res), vals: scores
                  
                print('training model...')
                try:
                    ws_train, ws_ae_train, ws_test, bs_test, ae_test, day_zero_AE = \
                    training_loop(curr_days_rng, subs_EEG_dict[sub], props['ae_lrn_rt'], \
                                props['cnvl_filters'], props['btch_sz'], props['n_epochs'], props['device'])
                except Exception as e:
                    print(f'Can\'t train a model for sub: {sub} with last training day: {last_train_day} because:')
                    print(e)
                    continue
                
                # Add task classification results
                task_per_sub_scores_dict['ws_train'] = ws_train
                task_per_sub_scores_dict['ae_train'] = ws_ae_train
                task_per_sub_scores_dict['ws_test'] = ws_test
                task_per_sub_scores_dict['bs_test'] = bs_test
                task_per_sub_scores_dict['ae_test'] = ae_test
                
                # Day classfication using residuals original and recontrusted EEG
                print('classifying origin day...')
                orig_score, rec_score, res_score = origin_day_clf(subs_EEG_dict[sub], day_zero_AE)
                origin_per_sub_scores_dict['orig'] = orig_score
                origin_per_sub_scores_dict['rec'] = rec_score
                origin_per_sub_scores_dict['res'] = res_score
            
                task_sub_dict[sub] = task_per_sub_scores_dict
                origin_sub_dict[sub] = origin_per_sub_scores_dict

            
            task_days_range_dict[rng_str] = task_sub_dict
            origin_days_range_dict[rng_str] = origin_sub_dict
                   
        task_iter_dict[itr] = task_days_range_dict
        origin_iter_dict[itr] = origin_days_range_dict
        
        # save to file
        print('save to file...')
        f_task_path = f'./results/task_iters_timestr_{ts}.pickle'
        f_origin_path = f'./results/origin_iters_timestr_{ts}.pickle'
        
        try:
            f_task = open(f_task_path, 'wb')
            f_origin = open(f_origin_path, 'wb')
            pickle.dump(task_iter_dict, f_task)
            pickle.dump(origin_iter_dict, f_origin)
        except Exception as e:
            print(e)
            print("Couldn't save to file")
        finally:
            f_task.close()
            f_origin.close()
        
        print(f'stopped after {itr+1} iterations')
    return f_task_path, f_origin_path


def get_mean_result_from_file(f_name):
    # extract the data from {f_name} and calculates the mean for exch method over iterations and subjects
    # possible methods,
        # for task: ws_test, bs_test, ae_test, ws_train, ae_train
        # for origin day: orig, rec, res
    #return
        # {mtd_by_rng_result_dict} mean result for each range by mthod
        # {all_train_ranges} list of all ranges
        # {methods} list of all methods
        
    with open(f_name, 'rb') as f:
        results_dict = pickle.load(f)
    
    all_iters = list(results_dict.keys())
    all_train_ranges = list(results_dict[all_iters[0]].keys())
    sub_list = list(results_dict[all_iters[0]][all_train_ranges[0]].keys())
    methods = list(results_dict[all_iters[0]][all_train_ranges[0]][sub_list[0]].keys())

    all_rng_result_dict = {}
    
    for rng in all_train_ranges:
        mtd_result_dict = {}
        range_subs = list(results_dict[0][rng].keys()) # range might not contains all subs

        for mtd in methods:
            # collect all results for rng and mtd (from all iters and subs)
            result_per_rng_and_mtd = [results_dict[itr][rng][sub][mtd] for itr in all_iters for sub in range_subs]
            mtd_result_dict[mtd] = np.mean(result_per_rng_and_mtd)
        
        all_rng_result_dict[rng] = mtd_result_dict
    return all_rng_result_dict, all_train_ranges, methods, len(all_iters)



