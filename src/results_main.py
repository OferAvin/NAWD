from modules.figuers import Figuers as figs
import matplotlib.pyplot as plt


ieee_results_files = ['9_IEEE_unsupervised_4c','9_IEEE_supervised_4c']
sub201_results_files = ['9_sub201_unsupervised','9_sub201_supervised']
shu_results_files = ['shu_unsupervised','shu_supervised']
xlable = 'Number of sessions in training set'
ylable= 'Accuracy'

# Sub201 panel
sub_201_figure, (ax1,ax2) = plt.subplots(2,1)
sub_201_results = figs(sub201_results_files, main_exp=1)
sub_201_results.plot_combined_results(result_mode='task', unique_methods=['ae_test'], ax=ax1, title="Sub201 task")
sub_201_results.plot_combined_results(result_mode='origin', unique_methods=['rec', 'res'], ax=ax2, title="Sub201 origin")

sub_201_figure.suptitle('Figure 2')
sub_201_figure.text(0.5, 0.04, xlable, ha='center')
sub_201_figure.text(0.02, 0.5, ylable, va='center', rotation='vertical')


# IEEE panel
ieee_figure, axs = plt.subplots(1,3)
ieee_results = figs(ieee_results_files, main_exp=1)
# ieee_results.plot_all_basic_results()
ieee_results.plot_combined_results(result_mode='task', unique_methods=['ae_test'], ax=axs[0], title="IEEE task")
ieee_results.plot_combined_results(result_mode='origin', unique_methods=['rec', 'res'], ax=axs[2], title="IEEE origin")

ieee_results.filter_results_by_acc(0.3)
ieee_results.plot_combined_results(result_mode='task', unique_methods=['ae_test'], ax=axs[1], title="IEEE task > 30%")

ieee_figure.suptitle('Figure 3')
ieee_figure.text(0.5, 0.04, xlable, ha='center')
ieee_figure.text(0.02, 0.5, ylable, va='center', rotation='vertical')



# Shu panel
shu_figure, axs1 = plt.subplots(1,3)
shu_results= figs(shu_results_files, main_exp=1)
shu_results.plot_combined_results(result_mode='task', unique_methods=['ae_test'], ax=axs1[0], title="Shu task")
shu_results.plot_combined_results(result_mode='origin', unique_methods=['rec', 'res'], ax=axs1[2], title="Shu origin")

shu_results.filter_results_by_acc(0.55)
shu_results.plot_combined_results(result_mode='task', unique_methods=['ae_test'], ax=axs1[1], title="Shu task > 55%")

shu_figure.suptitle('Figure 4')
shu_figure.text(0.5, 0.04, xlable, ha='center')
shu_figure.text(0.02, 0.5, ylable, va='center', rotation='vertical')


# Show all panels
plt.show()

