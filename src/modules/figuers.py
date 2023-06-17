from matplotlib import pyplot as plt
from modules.results import ResultsProcessor as rp
import numpy as np
from modules.properties import result_params
from typing import Any, Tuple, List
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Figuers():

    def __init__(self, exp_names: Tuple[str, str], main_exp: bool, subplots_dim: Tuple[int,int], fig_size: Tuple[float,float], title: str,
                 title_fontsize = 9, xlable = 'Number of sessions in training set', ylable= 'Accuracy' ):
        """
        Initialize an instance of Figuers.

        Args:
            exp_names : Tuple of len 2 contains the exps names.
            main_exp (bool): define the main experiment corresponds to exp_name:
                             0 represents the first exp 1 represents the second.
        """
        self.exp_names = exp_names
        self.res_dir = result_params['result_dir']
        self.exps_task_results: List[rp, rp] = [None,None]
        self.exps_origin_results: List[rp, rp] = [None,None]
        self.main_exp = main_exp
        self._create_results_for_exps()

        self.fig, self.axes = plt.subplots(*subplots_dim)
        self.fig.set_size_inches(fig_size)
        self.title = self.fig.suptitle(title, x=0.05, y=0.98, ha='left', va='top')
        self.title.set_fontsize(title_fontsize)
        self.fig.text(0.5, 0.04, xlable, ha='center')
        self.fig.text(0.02, 0.5, ylable, va='center', rotation='vertical')

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
        
        if result_mode == 'task':
            main_exp = self.exps_task_results[self.main_exp]
            secondery_exp = self.exps_task_results[not self.main_exp]
        elif result_mode == 'origin':
            main_exp = self.exps_origin_results[self.main_exp]
            secondery_exp = self.exps_origin_results[not self.main_exp]

        unique_idxes = [secondery_exp.methods.index(unique_method) for unique_method in unique_methods]

        unique_result = secondery_exp.mean_matrix[:, unique_idxes]
        unique_std = secondery_exp.std_matrix[:, unique_idxes]

        self.x = range(1,len(main_exp.train_ranges)+1)
        self.Y_mean = np.append(main_exp.mean_matrix, unique_result, axis=1)
        self.Y_std = np.append(main_exp.std_matrix, unique_std, axis=1)
        self.legend = main_exp.methods + unique_methods


    def add_combined_results_subplot(self, result_mode = 'task', unique_methods = ['ae_test'],
                              ax = None , title = '', legend = None, xlable='', ylable= '', legend_fontsize = "6"):

        if ax is None:
            ax = self._get_next_empty_axis()
        else:
            ax = self.axes.flatten[ax] 

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
        