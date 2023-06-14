from modules.figuers import Figuers as figs
import matplotlib.pyplot as plt


ieee_results_files = ['9_IEEE_unsupervised_4c','9_IEEE_supervised_4c']


IEEE_figure, (ax1,ax2) = plt.subplots(2,1)
ieee_results = figs(ieee_results_files, main_exp=1)
ieee_results.filter_results_by_acc(0.25)
# ieee_results.plot_all_basic_results()
ieee_results.plot_combined_results(result_mode = 'task', unique_methods = ['ae_test'], ax = ax1)
ieee_results.plot_combined_results(result_mode = 'origin', unique_methods = ['rec', 'res'], ax=ax2)


IEEE_figure1, (ax11,ax12) = plt.subplots(2,1)
ieee_results.filter_results_by_acc(0)

ieee_results.plot_combined_results(result_mode = 'task', unique_methods = ['ae_test'], ax = ax11)
ieee_results.plot_combined_results(result_mode = 'origin', unique_methods = ['rec', 'res'], ax=ax12)
plt.show()

