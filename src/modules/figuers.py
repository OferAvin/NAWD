from matplotlib import pyplot as plt
from modules.results import ResultsProcessor as rp
import numpy as np
from modules.properties import result_params
from typing import Any, Tuple, List
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Figuers():

    def __init__(self, exp_names: Tuple[str, str], main_exp: bool):
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

    def plot_all_basic_results(self):

        for i in range(len(self.exp_names)):
            exp_name = self.exp_names[i].replace('_', ' ')
            self.exps_task_results[i].plot_result(title=f'{exp_name} task results')
            self.exps_origin_results[i].plot_result(title=f'{exp_name} origin results') 

    def _combine_results(self, result_mode, unique_methods):
        
        if result_mode is 'task':
            main_exp = self.exps_task_results[self.main_exp]
            secondery_exp = self.exps_task_results[not self.main_exp]
        elif result_mode is 'origin':
            main_exp = self.exps_origin_results[self.main_exp]
            secondery_exp = self.exps_origin_results[not self.main_exp]

        unique_idxes = [secondery_exp.methods.index(unique_method) for unique_method in unique_methods]

        unique_result = secondery_exp.mean_matrix[:, unique_idxes]
        unique_std = secondery_exp.std_matrix[:, unique_idxes]

        self.x = main_exp.train_ranges
        self.Y_mean = np.append(main_exp.mean_matrix, unique_result, axis=1)
        self.Y_std = np.append(main_exp.std_matrix, unique_std, axis=1)
        self.legend = main_exp.methods + unique_methods


    def plot_combined_results(self, result_mode = 'task', unique_methods = ['ae_test'],
                              ax = None , title = '', legend = None, xlable='Range of training data', ylable= 'Accuracy'):

        
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
            ylabel=ylable 
            )
        