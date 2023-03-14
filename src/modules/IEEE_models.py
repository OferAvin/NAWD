import torch
import scipy.io
import mne
import sklearn
import os 
import time
import random
import time
import scipy.linalg
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb

from itertools import chain, product
import pickle # to write results to file

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from mne_features.feature_extraction import FeatureExtractor
from torch.utils.data import random_split, DataLoader, Dataset
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import norm, wasserstein_distance
from torchmetrics.classification import BinaryAccuracy


# from CHIST_ERA_data import *
from .utils import *

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
print('here')
# get_ipython().run_line_magic('load_ext', 'tensorboard')



# Assess whether GPU is availble
if torch.cuda.is_available():
    print("PyTorch is using the GPU.")
    print("Device name - ", torch.cuda.get_device_name(torch.cuda.current_device()))
else: 
    print("PyTorch is not using the GPU.")
    

# ### Datset and Model classes

class convolution_AE(LightningModule):
    def __init__(self, input_channels, days_labels_N, task_labels_N, learning_rate=1e-3, filters_n = [32, 16, 4], mode = 'supervised'):
        super().__init__()
        self.input_channels = input_channels
        self.filters_n = filters_n
        self.learning_rate = learning_rate
        self.float()
        self.l1_filters, self.l2_filters, self.l3_filters = self.filters_n
        self.mode = mode
        self.switcher = True
        ### The model architecture ###
        

        # Encoder
        self.encoder = nn.Sequential(
        nn.Conv1d(self.input_channels, self.l1_filters, kernel_size=25, stride=5, padding=1),
#         nn.Dropout1d(p=0.2),
#         nn.MaxPool1d(kernel_size=15, stride=3),
        nn.LeakyReLU(),
#         nn.AvgPool1d(kernel_size=2, stride=2),
        nn.Conv1d(self.l1_filters, self.l2_filters, kernel_size=10, stride=2, padding=1),
#         nn.Dropout1d(p=0.2),
        nn.LeakyReLU(),
#         nn.AvgPool1d(kernel_size=2, stride=2),
        nn.Conv1d(self.l2_filters, self.l3_filters, kernel_size=5, stride=2, padding=1),
#         nn.Dropout1d(p=0.2),
        nn.LeakyReLU()
        )
                
        # Decoder
        self.decoder = nn.Sequential(
        # IMPORTENT - on the IEEE dataset - the output padding needs to be 1 in the row below -on CHIST-ERA its 1
        nn.ConvTranspose1d(self.l3_filters, self.l2_filters, kernel_size=5, stride=2, padding=1, output_padding=0),
#         nn.Dropout1d(p=0.33),
        nn.LeakyReLU(),
#         nn.Upsample(scale_factor=2, mode='linear'),
        nn.ConvTranspose1d(self.l2_filters, self.l1_filters, kernel_size=10, stride=2, padding=1, output_padding=0),
#         nn.Dropout1d(p=0.33),
        nn.LeakyReLU(),
#         nn.Upsample(scale_factor=2, mode='linear'),
        nn.ConvTranspose1d(self.l1_filters, self.input_channels, kernel_size=25, stride=5, padding=1, output_padding=2),
        )
        
        # Residuals Encoder
        self.res_encoder = nn.Sequential(
        nn.Conv1d(self.input_channels, self.l1_filters, kernel_size=25, stride=5, padding=1),
        nn.LeakyReLU(),
        nn.Conv1d(self.l1_filters, self.l2_filters, kernel_size=10, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Conv1d(self.l2_filters, self.l3_filters, kernel_size=5, stride=2, padding=1),
        nn.LeakyReLU()
        )
                
        # Classifier Days
        self.classiffier_days = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4704, days_labels_N),
        nn.Dropout(0.5),
        )
        
        # Classifier Task
        self.classiffier_task = nn.Sequential(
        nn.Flatten(),
        nn.Linear(4704, task_labels_N),
        nn.Dropout(0.5),

        )
        
      
    def forward(self, x):
        # Forward through the layeres
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        # Forward through the layeres
        # Encoder
        x = self.encoder(x)
        return x
    
    def on_train_epoch_end(self):
        if self.current_epoch > 200:
            self.unfreeze_decoder()
            self.unfreeze_encoder()
            self.mode = 'all'
    
        if self.current_epoch % 20 == 0:
            self.switcher = not self.switcher
            if self.switcher == True:
                self.freeze_decoder()
                self.unfreeze_encoder()
                self.mode = 'task'
            elif self.switcher == False:
                self.freeze_encoder()
                self.unfreeze_decoder()
                self.mode = 'reconstruction'
        
    def training_step(self, batch, batch_idx):
        # Extract batch
        x, y, days_y = batch
        # Define loss functions
        loss_fn_days = nn.CrossEntropyLoss()
        loss_fn_rec = nn.MSELoss()
        loss_fn_task = nn.CrossEntropyLoss()
            
        # Encode
        encoded = self.encode(x)
        
        # Get predictions for task
        preds_task = self.classiffier_task(encoded)
        task_loss = loss_fn_task(preds_task, y)

        # Compute task classification accuracy
        task_acc = sklearn.metrics.accuracy_score(np.argmax(F.softmax(preds_task, dim=-1).detach().cpu().numpy(), axis=1),
                                             np.argmax(y.detach().cpu().numpy(), axis=1))

        # Log scalars
        self.log('task_loss', task_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('task_accuracy', task_acc, prog_bar=True, on_step=False, on_epoch=True)

        # Decode
        reconstructed = self.decoder(encoded)

        # Compute residuals
        residuals = torch.sub(x, reconstructed)

        # Encode residuals
        residuals_compact = self.res_encoder(residuals)

        # Get predictions per day
        preds_days = self.classiffier_days(residuals_compact)

        # Compute all losses
        days_loss = loss_fn_days(preds_days, days_y)
        reconstruction_loss = loss_fn_rec(reconstructed, x)

        # Compute days classification accuracy
        days_acc = sklearn.metrics.accuracy_score(np.argmax(F.softmax(preds_days, dim=-1).detach().cpu().numpy(), axis=1),
                                             np.argmax(days_y.detach().cpu().numpy(), axis=1))

        # Log results
        self.log('days_loss', days_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('reconstruction_loss', reconstruction_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('days_accuracy', days_acc, prog_bar=True, on_step=False, on_epoch=True)

        if self.mode == 'task':
            return days_loss + task_loss
        elif self.mode == 'reconstruction':
            return reconstruction_loss
        elif self.mode == 'all':
            return reconstruction_loss + days_loss + task_loss
   
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    
    def freeze_encoder(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        for name, param in self.encoder.named_parameters():
            param.requires_grad = True
            
    def freeze_decoder(self):
        for name, param in self.decoder.named_parameters():
            param.requires_grad = False
            
    def unfreeze_decoder(self):
        for name, param in self.decoder.named_parameters():
            param.requires_grad = True
            
            
    def change_mode(self, mode):
        self.mode = mode
        
        
    def configure_optimizers(self):
        # Optimizer
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    


# ## Training loop function

# In[ ]:


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


# In[ ]:


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


