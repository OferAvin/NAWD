import matplotlib.pyplot as plt
import numpy as np
import pickle
from copy import deepcopy


class Results_Processor:
    
    def __init__(self, f_name, results_dir='C:/Users/ofera/studies/NAWD/results'):
        self.f_name = f_name
        self.results_dir = results_dir
        self.f_path = results_dir + '/' + f_name
        self.min_acc = 0
        self.remove_sub_by_method = 'ws_train'
        self.remove_sub_by_range = '0-1'

        with open(self.f_path, 'rb') as f:
            self.results =  pickle.load(f)
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
        self.mean_dict = None
        self.mean_matrix = None
        self.std_matrix = None 




    def filter_sub(self, min_acc = 0.6, method = 'ws_train', range = '0-1'):
        # This method filters out subs whith accuracy smaller than min_acc
        # the accuracy calculated as the mean over method and range for each sub
        
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

        self.removed_subs = subs_to_remove
        n_removed = len(subs_to_remove)
        self.filtered_subs = [s for s in self.all_subs if s not in subs_to_remove]

        self.n_filtered_subs = len(self.filtered_subs)

        print(f'\
            Total number of subjects:{self.n_total_subs:>4}\n\
            Number of subjects with accuracy higer than {min_acc}:{self.n_filtered_subs:>4}\n\
            Number of subjects with accuracy less than {min_acc} (removed): {n_removed:>4}'
        )

    

    def _calculate_mean_result_over_iters_and_subs(self):
        # extract the data from {f_name} and calculates the mean for exch method over iterations and subjects
        # possible methods,
            # for task: ws_test, bs_test, ae_test, ws_train, ae_train
            # for origin day: orig, rec, res
        #return:
            # {mtd_by_rng_result_dict} mean result for each range by mthod
            # {train_ranges} list of all ranges
            # {methods} list of all methods
            
        mean_per_range_result = {}
        
        for rng in self.train_ranges:
            mean_per_method_result = {}
            range_subs = list(self.filtered_result[0][rng].keys()) # range might not contains all subs

            for mtd in self.methods:
                result_per_iter = np.empty([self.n_iters, len(range_subs)])
                for i, itr in enumerate(self.all_iters):
                    # collect results from all subs 
                    result_per_iter[i, :] = [self.results[itr][rng][sub][mtd] for sub in range_subs]

                mean_over_subs = np.mean(result_per_iter, axis=1) # we want the std over the iterations
                mean_per_method_result[mtd] = [np.mean(mean_over_subs),np.std(mean_over_subs)]
            
            mean_per_range_result[rng] = mean_per_method_result
        self.mean_dict = mean_per_range_result


    def _prepare_results_for_plots(self, res_dict, ranges, methods):
        # This function translate the result dict to np 2d array 
        # of shape n_range X n_method for easy plotting

        mean_results_mat = np.empty(self.n_ranges,self.n_methods)
        std_results_mat = np.empty(self.n_ranges,self.n_methods)

        for i, mtd in enumerate(methods):
            mean_res_per_mtd = [res_dict[rng][mtd][0] for rng in ranges] 
            std_res_per_mtd = [res_dict[rng][mtd][1] for rng in ranges] 
            mean_results_mat[:, i] = mean_res_per_mtd
            std_results_mat[:, i] = std_res_per_mtd
        
        self.mean_matrix = mean_results_mat
        self.std_matrix = std_results_mat

    def process_result(self):
        self._calculate_mean_result_over_iters_and_subs()
        self._prepare_results_for_plots()

    
    def plot_scores_mean_and_std(self, dont_plot = ['ae_train','ws_test'], colors = ['#008080', '#800080', '#800000']):
        if dont_plot:
            dont_plot_idx = [i for i, mtd in enumerate(self.methods) if e in dont_plot]

            x = self.train_ranges
            Y_mean = np.delete(self.mean_matrix, dont_plot_idx, axis=1)
            Y_std = np.delete(self.std_matrix, dont_plot_idx, axis=1)
            legend = [mtd for mtd in self.methods if mtd not in dont_plot]

        fig, ax = plt.subplots()
        for i in range(Y_mean.shape[1]):
            ax.plot(x, Y_mean[:,i],  
                            label = legend[i],       
                            marker = ',',           
                            linestyle = '-',   
                            color = colors[i],      
                            linewidth = '2.5'      
                                ) 
            ax.fill_between(x, Y_mean[:,i] - Y_std[:,i], Y_mean[:,i] + Y_std[:,i], color=colors[i], alpha=0.15)
    
        ax.legend()
        # Add labels and title to the plot
        plt.xlabel('Range of training data')
        plt.ylabel('Accuracy')
        plt.title('Session clasification')
        plt.show()