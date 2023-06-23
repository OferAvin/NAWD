from matplotlib import pyplot as plt
import matplotlib.cm as cm
from modules.results import ResultsProcessor as rp
import numpy as np
from modules.properties import result_params
from typing import Any, Tuple, List
import os
from scipy.stats import linregress, pearsonr

class Figuers():

    def __init__(self, exp_names: Tuple[str, str], main_exp: bool, subplots_dim: Tuple[int,int], fig_size: Tuple[float,float], title: str,
                 title_fontsize = 9):
        """
        Initialize an instance of Figuers.

        Args:
            exp_names : Tuple of len 2 contains the exps names.
            main_exp (bool): define the main experiment, this is the expirament that from which all results,
                            corresponds to exp_name:
                            0 represents the first exp 1 represents the second.
        """
        self.exp_names = exp_names
        self.res_dir = result_params['result_dir']
        self.exps_task_results: List[rp, rp] = [None,None]
        self.exps_origin_results: List[rp, rp] = [None,None]
        self.main_exp = main_exp
        self.secondary_exp = not main_exp
        self._create_results_for_exps()

        self.fig, self.axes = plt.subplots(*subplots_dim)
        self.fig.set_size_inches(fig_size)
        self.title = self.fig.suptitle(title, x=0.05, y=0.98, ha='left', va='top')
        self.title.set_fontsize(title_fontsize)

        self.x = None
        self.Y_mean = None
        self.Y_std = None
        self.legend = None


    def _create_results_for_exps(self):
        for i, exp_name in enumerate(self.exp_names):

            exp1_task_file, exp1_origin_file = self._get_files_for_exp(exp_name)

            self.exps_task_results[i] = rp(exp1_task_file)
            self.exps_origin_results[i] = rp(exp1_origin_file)

        self._process_all_results()


    def _process_all_results(self):
        if self.exps_task_results[0] is None:
            raise TypeError

        for i in range(len(self.exp_names)):
            self.exps_task_results[i].process_result()
            self.exps_origin_results[i].process_result()    


    def _get_files_for_exp(self, exp_name) -> list[str,str]:
        for file_name in os.listdir(self.res_dir):
            if exp_name in file_name and 'task' in file_name:
                task_file = file_name
            elif exp_name in file_name and 'origin' in file_name:
                origin_file = file_name

        return task_file, origin_file


    def filter_results_by_acc(self, min_acc: float):
        main_exp_results = self.exps_task_results[self.main_exp]
        main_exp_results.filter_sub_by_acc(min_acc)
        removed_subs = main_exp_results.removed_subs

        for i in range(len(self.exp_names)):

            self.exps_task_results[i].filter_out_subs_from_results(removed_subs)
            self.exps_origin_results[i].filter_out_subs_from_results(removed_subs)

        self._process_all_results()

        print(f'Results are now filtered by min  of: {min_acc}!')

    def _get_next_empty_axis(self):

        for ax in self.axes.flatten():
            if len(ax.lines) == 0:
                return ax
            
    def plot_all_basic_results(self, do_subplots = True):
       
        fig, axes = plt.subplots(2,2) if do_subplots else (None,None)

        for i in range(len(self.exp_names)):
            exp_name = self.exp_names[i].replace('_', ' ')
            if axes:
                self.exps_task_results[i].plot_result(title=f'{exp_name} task results', ax=axes[i, 0])
                self.exps_origin_results[i].plot_result(title=f'{exp_name} origin results', ax=axes[i, 1]) 
            else:
                self.exps_task_results[i].plot_result(title=f'{exp_name} task results')
                self.exps_origin_results[i].plot_result(title=f'{exp_name} origin results')

        return fig, axes

    def _combine_results(self, result_mode, unique_methods):

        """
            Combines the results of self.main_exp with the results of unique_methods from self.secondary_exp
            based on the specified result_mode.

            Args:
                result_mode (str): Specifies the mode for selecting the results. Valid options are 'task' and 'origin'.
                unique_methods (list): A list of unique methods whose results from self.secondary_exp will be appended to self.main_exp.
        """
        
        if result_mode == 'task':
            main_exp = self.exps_task_results[self.main_exp]
            secondary_exp = self.exps_task_results[self.secondary_exp]
        elif result_mode == 'origin':
            main_exp = self.exps_origin_results[self.main_exp]
            secondary_exp = self.exps_origin_results[self.secondary_exp]

        unique_idxes = [secondary_exp.methods.index(unique_method) for unique_method in unique_methods]

        unique_result = secondary_exp.mean_matrix[:, unique_idxes]
        unique_std = secondary_exp.std_matrix[:, unique_idxes]

        self.x = range(1,len(main_exp.train_ranges)+1)
        self.Y_mean = np.append(main_exp.mean_matrix, unique_result, axis=1)
        self.Y_std = np.append(main_exp.std_matrix, unique_std, axis=1)
        self.legend = main_exp.methods + unique_methods

    def add_combined_results_subplot(self, result_mode = 'task', unique_methods = ['ae_test'],
                              ax = None , title = '', legend = None, xlable='', ylable='', legend_fontsize = "6"):

        if ax is None:
            ax = self._get_next_empty_axis()
        elif isinstance(ax, int):
            ax = self.axes.flatten()[ax] 

        self._combine_results(result_mode, unique_methods)

        legend = self.legend if legend is None else legend
        self.exps_task_results[self.main_exp]._plot_mean_and_sd(
            self.x,
            self.Y_mean,
            self.Y_std,
            ax=ax,
            title=title,
            legend=legend,
            xlable=xlable,
            ylabel=ylable,
            legend_fontsize=legend_fontsize 
            )
        
        return ax
    
    def corr_res(self, method, mean_over = None, ax = None):

        task_mat = self.exps_task_results[self.main_exp].get_subs_X_ranges_mat('ae_test')
        origin_mat = self.exps_origin_results[self.main_exp].get_subs_X_ranges_mat(method)

        if mean_over == 'range':
            task_mat = np.expand_dims(np.nanmean(task_mat, axis=1), axis=1)
            origin_mat = np.expand_dims(np.nanmean(origin_mat, axis=1), axis=1)
        elif mean_over == 'subs':
            task_mat = np.expand_dims(np.nanmean(task_mat, axis=0), axis=1).T
            origin_mat = np.expand_dims(np.nanmean(origin_mat, axis=0), axis=1).T

        task_vec = task_mat.flatten()
        origin_vec = origin_mat.flatten()

        corr, p_value = pearsonr(task_vec, origin_vec)
        corr_text = f"Corr: {corr:.2f}\np-value: {p_value:.4f}"

        slope, intercept, _, _, _ = linregress(task_vec, origin_vec)
        regression_line = slope * task_vec + intercept

        #### Plot results ####
        if method is 'rec':
            mtd_str = 'Reconstructed'  
        elif method is 'res':
            mtd_str = 'Residuals'
        else:
            mtd_str = 'Original'

        num_lines, num_columns = task_mat.shape
        cmap = cm.get_cmap('tab20')
        colors = cmap(range(num_lines))
        if not ax:
            fig, ax = plt.subplots()
        for line in range(num_lines):
            for column in range(num_columns):
                x = task_mat[line, column]
                y = origin_mat[line, column]
                label = f"Line {line} Column {column}"
                color = colors[line] # Normalize row values between 0 and 1
                label = str(column + 1) if num_columns > 1 else 'O'
                ax.text(x, y, label, ha='center', va='center', color=color)
        
        ax.text(0.95, 0.05, corr_text, ha='right', va='bottom', transform=ax.transAxes)
        ax.plot(task_vec, regression_line, color='r', label='Regression Line')

        ax.set_title('Correlation')
        if mean_over is not None:
            ax.text(0.5, 0.995, f'(mean over {mean_over})', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_xlabel('Task classification')
        ax.set_ylabel(f'{mtd_str} classification')
        ax.set_xlim(np.min(task_vec)-0.1, np.max(task_vec)+0.1)
        ax.set_ylim(np.min(origin_vec)-0.1, np.max(origin_vec)+0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)   
