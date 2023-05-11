import matplotlib.pyplot as plt
import numpy as np
import pickle
from copy import deepcopy
from modules.properties import result_params
from os import path
from sys import exit

# for task: ws_test, bs_test, ae_test, ws_train, ae_train
            # for origin day: orig, rec, res

class ResultsNotProcessedException(Exception):
    pass

class Results_Processor:
    
    def __init__(self, f_name):
        self.f_name = f_name
        self.results_dir = result_params['result_dir']
        self.f_path = self.results_dir + '/' + self.f_name
        self.min_acc = 0
        self.remove_sub_by_method = None
        self.remove_sub_by_range = None

        try:
            with open(self.f_path, 'rb') as f:
                self.results =  pickle.load(f)
        except:
            print("Couldn't load result file!!!")
            exit()        
        self.all_iters = list(self.results.keys())
        self.train_ranges = list(self.results[self.all_iters[0]].keys())
        self.all_subs = list(self.results[self.all_iters[0]][self.train_ranges[0]].keys())
        self.methods = list(self.results[self.all_iters[0]][self.train_ranges[0]][self.all_subs[0]].keys())
        self.n_iters = len(self.all_iters)
        self.n_ranges = len(self.train_ranges)
        self.n_total_subs = len(self.all_subs)
        self.n_methods = len(self.methods)

        # Followin fields will be set when applying filter_sub()
        self.filtered_result = None  
        self.filtered_subs = self.all_subs
        self.removed_subs = None
        self.n_filtered_subs = self.n_total_subs

        # Following fields will be set when applying process_result()
        # mean = mean result per training range
        self.mean_resulrs = None
        self.mean_matrix = None
        self.std_matrix = None 
        self.is_processed = False

        self.colors = result_params['colors']
        self.alpha = result_params['alpha']

    def get_methods(self):
        return self.methods
    
    def get_all_subs(self):
        return self.all_subs
    
    def get_filtered_subs(self):
        return self.filtered_subs

    def print_filter_settings(self):
        if not self.remove_sub_by_method:
            print('No filteration was made')
        else:
            print(f'Filteration was made according to:\n\
                  Minimum accuracy: {self.min_acc:4d}\n\
                  Method: {self.remove_sub_by_method:4d}\n\
                  Range: {self.remove_sub_by_range:4d}' )
    
    def filter_out_sub_from_list(self, subs_to_remove = []):

        self.filtered_subs = [s for s in self.all_subs if s not in subs_to_remove]
        self.removed_subs = subs_to_remove
        self.n_filtered_subs = len(self.filtered_subs)  


    def filter_sub_by_acc(self, min_acc = 0.6, method = 'ws_train', range = '0-1'):
        '''
         This method filters out subs whith accuracy smaller than `min_acc`

         the accuracy calculated as the mean over `method` and `range` for each sub 

         in `subs`. in case `subs` is empty than the filter wil go over `self.all_subs`

        '''

        # Find bad subs
        self.filtered_result = deepcopy(self.results)
        subs_to_remove = [] #only for mehod and range
        for sub in self.all_subs:
            accuracies = [self.results[itr][range][sub][method] for itr in self.all_iters]
            sub_mean_acc = np.mean(accuracies)
            if sub_mean_acc < min_acc:
                subs_to_remove.append(sub)
                
                # Remove bad subs from result dict
                for itr in self.all_iters:
                    for rng in self.train_ranges:
                        self.filtered_result[itr][rng].pop(sub, None) 

        n_removed = len(subs_to_remove)
        self.filter_out_sub_from_list(subs_to_remove)
        self.min_acc = min_acc
        self.remove_sub_by_method = method
        self.remove_sub_by_range = range

        print(f'\
            Total number of subjects:{self.n_total_subs:>4}\n\
            Number of subjects with accuracy higer than {min_acc}:{self.n_filtered_subs:>4}\n\
            Number of subjects with accuracy less than {min_acc} (removed): {n_removed:>4}'
        )

    

    def _calculate_mean_result_over_iters_and_subs(self):
        '''
        #This function calculates the mean over the subjects and then

        calculates mean and standard error over iterations for each training range and method

        the mean results then assigned to `self.mean_results` dict
        '''

        results = self.filtered_result if self.filtered_result else self.results

        mean_per_range_result = {}
        
        for rng in self.train_ranges:
            mean_per_method_result = {}
            range_subs = list(results[0][rng].keys()) # range might not contains all subs

            for mtd in self.methods:
                result_per_iter = np.empty([self.n_iters, len(range_subs)])
                for i, itr in enumerate(self.all_iters):
                    # collect results from all subs 
                    result_per_iter[i, :] = [results[itr][rng][sub][mtd] for sub in range_subs]

                mean_over_subs = np.mean(result_per_iter, axis=1) # we want the standard error over the iterations
                mean_per_method_result[mtd] = [np.mean(mean_over_subs),np.std(mean_over_subs)/np.sqrt(self.n_iters)]
            
            mean_per_range_result[rng] = mean_per_method_result
        self.mean_resulrs = mean_per_range_result


    def _prepare_results_for_plots(self):
        '''
        This function translate the self.mean_result dict to np 2d array 

        of shape `n_range` X `n_method` for easy plotting
        '''

        mean_results_mat = np.empty((self.n_ranges,self.n_methods))
        std_results_mat = np.empty((self.n_ranges,self.n_methods))

        for i, mtd in enumerate(self.methods):
            mean_res_per_mtd = [self.mean_resulrs[rng][mtd][0] for rng in self.train_ranges] 
            std_res_per_mtd = [self.mean_resulrs[rng][mtd][1] for rng in self.train_ranges] 
            mean_results_mat[:, i] = mean_res_per_mtd
            std_results_mat[:, i] = std_res_per_mtd
        
        self.mean_matrix = mean_results_mat
        self.std_matrix = std_results_mat


    def process_result(self):
        self._calculate_mean_result_over_iters_and_subs()
        self._prepare_results_for_plots()
        self.is_processed = True


    def _plot_mean_and_sd(self, x, Y_mean, Y_std, legend, title):
        fig, ax = plt.subplots()
        for i in range(Y_mean.shape[1]):
            ax.plot(x, Y_mean[:,i],  
                            label = legend[i],       
                            marker = ',',           
                            linestyle = '-',   
                            color = self.colors[i],      
                            linewidth = '2.5'      
                                ) 
            ax.fill_between(x, Y_mean[:,i] - Y_std[:,i], Y_mean[:,i] + Y_std[:,i], color=self.colors[i], alpha=self.alpha)
    
        ax.legend()
        # Add labels and title to the plot
        plt.xlabel('Range of training data')
        plt.ylabel('Accuracy')
        plt.suptitle(title)
        plt.text(0.5, 0.92, f'({self.n_filtered_subs} subjects)', ha='center', va='center', transform=plt.gcf().transFigure)
        plt.show()
        
    
    def plot_result(self, dont_plot = ['ae_train','ws_test'], title = ''):

        if not self.is_processed:
            raise ResultsNotProcessedException(\
                'In order to plot results you must first apply process_result() ')


        x = self.train_ranges
        Y_mean = self.mean_matrix
        Y_std = self.std_matrix
        legend = np.asarray(self.methods)
        
        if dont_plot:
            dont_plot_idx = [i for i, mtd in enumerate(self.methods) if mtd in dont_plot]
            Y_mean = np.delete(Y_mean, dont_plot_idx, axis=1)
            Y_std = np.delete(Y_std, dont_plot_idx, axis=1)
            legend =  np.delete(legend, dont_plot_idx)

        self._plot_mean_and_sd(x, Y_mean, Y_std, legend, title)
