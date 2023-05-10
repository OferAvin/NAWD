from .utils import *
import pickle 
import pytorch_lightning as pl

import time
import torch
import random
from pytorch_lightning.loggers import TensorBoardLogger

from .models import convolution_AE         
from .properties import hyper_params as params  



def training_loop(props, train_days, dictListStacked):
    
    # check if enough train days exists
    if train_days[1] >= len(dictListStacked):
        raise Exception("Not enough training days")

    # device settings
    proccessor = params['device']
    device = torch.device(proccessor)
    accelerator = proccessor if proccessor=='cpu' else 'gpu' 
    devices = 1 if proccessor =='cpu' else -1 
    
    # Logger
    logger = TensorBoardLogger('../tb_logs', name='EEG_Logger')
    # Shuffle the days
    random.shuffle(dictListStacked)
    # Train Dataset
    signal_data = EEGDataSet_signal_by_day(dictListStacked, train_days)
    signal_data_loader = DataLoader(dataset=signal_data, batch_size=params['btch_sz'], shuffle=True, num_workers=0)
    x, y, days_y = signal_data.getAllItems()
    y = np.argmax(y, -1)
    n_days_labels = signal_data.n_days_labels
    n_task_labels = signal_data.n_task_labels

    # Train model on training day
    metrics = ['classification_loss', 'reconstruction_loss']
    day_zero_AE = convolution_AE(signal_data.n_channels, n_days_labels, n_task_labels, props, \
                                 params['ae_lrn_rt'], filters_n=params['cnvl_filters'], mode='supervised')
    day_zero_AE.to(device)

    trainer_2 = pl.Trainer(max_epochs=params['n_epochs'], logger=logger, accelerator=accelerator , devices=devices)
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



def run_all_subs_multi_iterations(props, subs_EEG_dict, train_days_range = [1,7], iterations_per_day = 250):
    
# This function runs multi iterations experiment over all subjects.
# The experiment runs all ranges of traning days from 0-{train_days_range[0]} to 0-{train_days_range[1]}.
# Every iteration models are trained for all ranges of training days and all subjects.
# the function saves 2 dictionaries to 2 files:
    # task clasification results dictionary
    # origin day clasification results dictionary
# The function returns the 2 file pathes
    
    ts = time.strftime("%Y%m%d-%H%M%S")
    print(f'START EXPERIMENT!!! {ts}\n')

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
                print(f'\niter: {itr}, last training day: {last_train_day}, sub: {sub}...\n')
                
                task_per_sub_scores_dict = {} # keys: method(ws,bs,AE), vals: scores
                origin_per_sub_scores_dict = {} # keys: signal(orig,rec,res), vals: scores
                  
                print('training model...\n')
                try:
                    ws_train, ws_ae_train, ws_test, bs_test, ae_test, day_zero_AE = \
                    training_loop(props, curr_days_rng, subs_EEG_dict[sub])
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
