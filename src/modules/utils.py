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


class CSP:
    def __init__(self,m_filters):
        self.m_filters = m_filters

    def fit(self,x_train,y_train):
        x_data = np.copy(x_train)
        y_labels = np.copy(y_train)
        n_trials, n_channels, n_samples = x_data.shape
        cov_x = np.zeros((2, n_channels, n_channels), dtype=np.float64)
        for i in range(n_trials):
            x_trial = x_data[i, :, :]
            y_trial = y_labels[i]
            cov_x_trial = np.matmul(x_trial, np.transpose(x_trial))
            cov_x_trial /= np.trace(cov_x_trial)
            cov_x[y_trial, :, :] += cov_x_trial

        cov_x = np.asarray([cov_x[cls]/np.sum(y_labels==cls) for cls in range(2)])
        cov_combined = cov_x[0]+cov_x[1]
        eig_values, u_mat = scipy.linalg.eig(cov_combined,cov_x[0])
        sort_indices = np.argsort(abs(eig_values))[::-1]
        eig_values = eig_values[sort_indices]
        u_mat = u_mat[:,sort_indices]
        u_mat = np.transpose(u_mat)

        return eig_values, u_mat

    def transform(self,x_trial,eig_vectors):
        z_trial = np.matmul(eig_vectors, x_trial)
        z_trial_selected = z_trial[:self.m_filters,:]
        z_trial_selected = np.append(z_trial_selected,z_trial[-self.m_filters:,:],axis=0)
        sum_z2 = np.sum(z_trial_selected**2, axis=1)
        sum_z = np.sum(z_trial_selected, axis=1)
        var_z = (sum_z2 - (sum_z ** 2)/z_trial_selected.shape[1]) / (z_trial_selected.shape[1] - 1)
        sum_var_z = sum(var_z)
        return np.log(var_z/sum_var_z)


# In[ ]:


class FBCSP:
    def __init__(self,m_filters):
        self.m_filters = m_filters
        self.fbcsp_filters_multi=[]

    def fit(self,x_train_fb,y_train):
        y_classes_unique = np.unique(y_train)
        n_classes = len(y_classes_unique)
        self.csp = CSP(self.m_filters)

        def get_csp(x_train_fb, y_train_cls):
            fbcsp_filters = {}
            for j in range(x_train_fb.shape[0]):
                x_train = x_train_fb[j, :, :, :]
                eig_values, u_mat = self.csp.fit(x_train, y_train_cls)
                fbcsp_filters.update({j: {'eig_val': eig_values, 'u_mat': u_mat}})
            return fbcsp_filters

        for i in range(n_classes):
            cls_of_interest = y_classes_unique[i]
            select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            fbcsp_filters=get_csp(x_train_fb,y_train_cls)
            self.fbcsp_filters_multi.append(fbcsp_filters)

    def transform(self,x_data,class_idx=0):
        n_fbanks, n_trials, n_channels, n_samples = x_data.shape
        x_features = np.zeros((n_trials,self.m_filters*2*len(x_data)),dtype=np.float64)
        for i in range(n_fbanks):
            eig_vectors = self.fbcsp_filters_multi[class_idx].get(i).get('u_mat')
            eig_values = self.fbcsp_filters_multi[class_idx].get(i).get('eig_val')
            for k in range(n_trials):
                x_trial = np.copy(x_data[i,k,:,:])
                csp_feat = self.csp.transform(x_trial,eig_vectors)
                for j in range(self.m_filters):
                    x_features[k, i * self.m_filters * 2 + (j+1) * 2 - 2]  = csp_feat[j]
                    x_features[k, i * self.m_filters * 2 + (j+1) * 2 - 1]= csp_feat[-j-1]

        return x_features


# In[ ]:


class FeatureSelect:
    def __init__(self, n_features_select=4, n_csp_pairs=2):
        self.n_features_select = n_features_select
        self.n_csp_pairs = n_csp_pairs
        self.features_selected_indices=[]

    def fit(self,x_train_features,y_train):
        MI_features = self.MIBIF(x_train_features, y_train)
        MI_sorted_idx = np.argsort(MI_features)[::-1]
        features_selected = MI_sorted_idx[:self.n_features_select]

        paired_features_idx = self.select_CSP_pairs(features_selected, self.n_csp_pairs)
        x_train_features_selected = x_train_features[:, paired_features_idx]
        self.features_selected_indices = paired_features_idx

        return x_train_features_selected

    def transform(self,x_test_features):
        return x_test_features[:,self.features_selected_indices]

    def MIBIF(self, x_features, y_labels):
        def get_prob_pw(x,d,i,h):
            n_data = d.shape[0]
            t=d[:,i]
            kernel = lambda u: np.exp(-0.5*(u**2))/np.sqrt(2*np.pi)
            prob_x = 1 / (n_data * h) * sum(kernel((np.ones((len(t)))*x- t)/h))
            return prob_x

        def get_pd_pw(d, i, x_trials):
            n_data, n_dimensions = d.shape
            if n_dimensions==1:
                i=1
            t = d[:,i]
            min_x = np.min(t)
            max_x = np.max(t)
            n_trials = x_trials.shape[0]
            std_t = np.std(t)
            if std_t==0:
                h=0.005
            else:
                h=(4./(3*n_data))**(0.2)*std_t
            prob_x = np.zeros((n_trials))
            for j in range(n_trials):
                prob_x[j] = get_prob_pw(x_trials[j],d,i,h)
            return prob_x, x_trials, h

        y_classes = np.unique(y_labels)
        n_classes = len(y_classes)
        n_trials = len(y_labels)
        prob_w = []
        x_cls = {}
        for i in range(n_classes):
            cls = y_classes[i]
            cls_indx = np.where(y_labels == cls)[0]
            prob_w.append(len(cls_indx) / n_trials)
            x_cls.update({i: x_features[cls_indx, :]})

        prob_x_w = np.zeros((n_classes, n_trials, x_features.shape[1]))
        prob_w_x = np.zeros((n_classes, n_trials, x_features.shape[1]))
        h_w_x = np.zeros((x_features.shape[1]))
        mutual_info = np.zeros((x_features.shape[1]))
        parz_win_width = 1.0 / np.log2(n_trials)
        h_w = -np.sum(prob_w * np.log2(prob_w))

        for i in range(x_features.shape[1]):
            h_w_x[i] = 0
            for j in range(n_classes):
                prob_x_w[j, :, i] = get_pd_pw(x_cls.get(j), i, x_features[:, i])[0]

        t_s = prob_x_w.shape
        n_prob_w_x = np.zeros((n_classes, t_s[1], t_s[2]))
        for i in range(n_classes):
            n_prob_w_x[i, :, :] = prob_x_w[i] * prob_w[i]
        prob_x = np.sum(n_prob_w_x, axis=0)
        # prob_w_x = np.zeros((n_classes, prob_x.shape[0], prob_w.shape[1]))
        for i in range(n_classes):
            prob_w_x[i, :, :] = n_prob_w_x[i, :, :]/prob_x

        for i in range(x_features.shape[1]):
            for j in range(n_trials):
                t_sum = 0.0
                for k in range(n_classes):
                    if prob_w_x[k, j, i] > 0:
                        t_sum += (prob_w_x[k, j, i] * np.log2(prob_w_x[k, j, i]))

                h_w_x[i] -= (t_sum / n_trials)

            mutual_info[i] = h_w - h_w_x[i]

        mifsg = np.asarray(mutual_info)
        return mifsg


    def select_CSP_pairs(self,features_selected,n_pairs):
        features_selected+=1
        sel_groups = np.unique(np.ceil(features_selected/n_pairs))
        paired_features = []
        for i in range(len(sel_groups)):
            for j in range(n_pairs-1,-1,-1):
                paired_features.append(sel_groups[i]*n_pairs-j)

        paired_features = np.asarray(paired_features,dtype=np.int)-1

        return paired_features


# In[ ]:


def fit_selection(x_features, y, n_features_select):
        selector = FeatureSelect(n_features_select)
        selector.fit(x_features, y)
        return selector


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


def fbcsp_score(signal, labels, m_filters, fs, cv_N = 5, classifier = [], n_select = 50):
    # Create 4D matrix
    filter_range = [4,40]
    step = 4
    filters_start = range(filter_range[0], filter_range[1], step)
    filtered_data_4d = []
    

    for start_freq in filters_start:
        filtered = mne.filter.filter_data(np.float64(signal), fs, start_freq, start_freq+step, method='fir', verbose=False)
        filtered_data_4d.append(filtered)
    filtered_data_4d = np.stack(filtered_data_4d, axis=0)
    # Set verbose to 0
    mne.set_log_level(verbose='WARNING', return_old_level=False, add_frames=None)
    if classifier:
        features = classifier[0].transform(filtered_data_4d) # fbcsp
        features = classifier[1].transform(features) # feature selection
        y_pred = classifier[2].predict(features) # classifier
        acc = sklearn.metrics.accuracy_score(labels, y_pred)
        return acc
    else:
        # implement CV
        n_trials = labels.shape[0]
        idx_perm = np.random.permutation(n_trials)
        scores = []
        # CV loop
        for i in range(cv_N):
            fractions = np.array_split(idx_perm, cv_N)

            val_idx = list(fractions.pop(i))
            train_idx = list(np.concatenate(fractions))
            
            X_train = filtered_data_4d[:,train_idx,:,:]
            y_train = labels[train_idx]
            X_val = filtered_data_4d[:,val_idx,:,:]
            y_val = labels[val_idx]
            
            # Assemble classifier
            fbcsp = FBCSP(m_filters=m_filters)
#             clf = LinearDiscriminantAnalysis()
#             clf = lgb.LGBMClassifier()
            clf = sklearn.svm.SVC(verbose=0)
            
            # Train classifier
            _ = fbcsp.fit(X_train, y_train)
            features_train = fbcsp.transform(X_train)
            selector = fit_selection(features_train, y_train, n_select)
            features_train = selector.transform(features_train)
            clf.fit(features_train, y_train)             
            
            # Evaluate classifier
            features_val = fbcsp.transform(X_val)
            features_val = selector.transform(features_val)
            y_hat = clf.predict(features_val)
            scores.append(sklearn.metrics.accuracy_score(y_val, y_hat)) 
           
        # Train on full data
        fbcsp = FBCSP(m_filters=m_filters)
#         clf = LinearDiscriminantAnalysis()
#         clf = lgb.LGBMClassifier()
        clf = sklearn.svm.SVC()
        
        fbcsp.fit(filtered_data_4d, labels)
        features = fbcsp.transform(filtered_data_4d)
        selector = fit_selection(features, labels, n_select)
        features = selector.transform(features)
        clf.fit(features, labels)            
        
    return np.mean(scores), [fbcsp, selector, clf]


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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





