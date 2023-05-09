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

def remove_subs_by_accuracy_and_method(results_dict, min_acc = 0.6, method = 'ws_train', range = '0-1'):
    # Remove subs whith accuracy smaller than min_acc for method and range
    all_iters = list(results_dict.keys())
    all_train_ranges = list(results_dict[all_iters[0]].keys())
    subs = list(results_dict[all_iters[0]][range].keys())
    
    # Find bed subs
    subs_to_remove = [] #only for mehod and range
    for sub in subs:
        accuracies = [results_dict[itr][range][sub][method] for itr in all_iters]
        sub_mean_acc = np.mean(accuracies)
        if sub_mean_acc < min_acc:
            subs_to_remove.append(sub)
            # Remove subs from result dict
            for itr in all_iters:
                for rng in all_train_ranges:
                    results_dict[itr][rng].pop(sub, None)       
            
    n_sub_left = len(subs) - len(subs_to_remove)
    subs = [s for s in subs if s not in subs_to_remove]
    return subs, subs_to_remove 

def get_mean_result_from_file(f_name):
    # extract the data from {f_name} and calculates the mean for exch method over iterations and subjects
    # possible methods,
        # for task: ws_test, bs_test, ae_test, ws_train, ae_train
        # for origin day: orig, rec, res
    #return:
        # {mtd_by_rng_result_dict} mean result for each range by mthod
        # {all_train_ranges} list of all ranges
        # {methods} list of all methods
        
    with open(f_name, 'rb') as f:
        results_dict = pickle.load(f)
    
    subs, bed_subs = remove_subs_by_accuracy_and_method(results_dict, min_acc=0.6)

    all_iters = list(results_dict.keys())
    all_train_ranges = list(results_dict[all_iters[0]].keys())
    methods = list(results_dict[all_iters[0]][all_train_ranges[0]][subs[0]].keys())

    all_rng_result_dict = {}
    
    for rng in all_train_ranges:
        mtd_result_dict = {}
        range_subs = list(results_dict[0][rng].keys()) # range might not contains all subs

        for mtd in methods:
            result_per_iter = np.empty([len(all_iters), len(range_subs)])
            for i, itr in enumerate(all_iters):
                # collect results from all subs 
                result_per_iter[i, :] = [results_dict[itr][rng][sub][mtd] for sub in range_subs]

            mean_over_subs = np.mean(result_per_iter, axis=1) # we want the std over the iterations
            mtd_result_dict[mtd] = [np.mean(mean_over_subs),np.std(mean_over_subs)]
        
        all_rng_result_dict[rng] = mtd_result_dict
    return all_rng_result_dict, all_train_ranges, methods, len(all_iters)


def get_results_for_plots(res_dict, ranges, methods):
    # This function translate the result dict to np 2d array 
    # range X method for easy plotting

    mean_results_mat = np.empty((len(ranges),len(methods)))
    std_results_mat = np.empty((len(ranges),len(methods)))

    for i, mtd in enumerate(methods):
        mean_res_per_mtd = [res_dict[rng][mtd][0] for rng in ranges] 
        std_res_per_mtd = [res_dict[rng][mtd][1] for rng in ranges] 
        mean_results_mat[:, i] = mean_res_per_mtd
        std_results_mat[:, i] = std_res_per_mtd
    
    return mean_results_mat, std_results_mat

def plot_scores_mean_and_std(x, Y_mean, Y_std, colors, legend):
    if 'ae_train' in legend:
        Y_mean = np.delete(Y_mean, [1,2], axis=1)
        Y_std = np.delete(Y_std, [1,2], axis=1)
        legend = np.delete(np.asanyarray(legend), [1,2])

    fig, ax = plt.subplots()
    for i in range(Y_mean.shape[1]):
        ax.plot(x, Y_mean[:,i],  
                        label = legend[i],       
                        marker = ',',           
                        linestyle = '-',   
                        color = colors[i],      
                        linewidth = '3.5'      
                            ) 
        ax.fill_between(x, Y_mean[:,i] - Y_std[:,i], Y_mean[:,i] + Y_std[:,i], color=colors[i], alpha=0.2)
 
    ax.legend()
    # plt.legend(handles=[l1, l2, l3])
# Add labels and title to the plot
    plt.xlabel('Range of training data')
    plt.ylabel('Accuracy')
    plt.title('Session clasification')
    plt.show()