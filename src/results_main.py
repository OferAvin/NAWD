from modules.figuers import Figuers as figs
import matplotlib.pyplot as plt


ieee_results_files = ['9_IEEE_unsupervised_4c','9_IEEE_supervised_4c']
sub201_results_files = ['9_sub201_unsupervised','9_sub201_supervised']
shu_results_files = ['shu_unsupervised','shu_supervised']

xlable = 'Number of sessions in training set'
ylable= 'Accuracy'

sub201_subplots_dim = (2,1)
ieee_subplots_dim = shu_subplots_dim = (1,3)

sub201_fig_size = (3.35, 5.51) # inches
ieee_figs_size = shu_figs_size = (7.09, 2.36)

task_legend = ['WS', 'BS', 'SAE', 'AE']
origin_legend = ['Origin', 'Reconstructed SAE', 'Residuals SAE', 'Reconstructed AE', 'Residuals AE']


##### Sub201 panel #####
sub201_results = figs(
    sub201_results_files,
    main_exp=1,
    subplots_dim=sub201_subplots_dim,
    fig_size=sub201_fig_size,
    title="Figure2")

ax = sub201_results.add_combined_results_subplot(result_mode='task', unique_methods=['ae_test'],
                                            title="Sub201 task", legend=task_legend, 
                                            ylable='Accuracy', plot_n_subs = False)
ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

ax = sub201_results.add_combined_results_subplot(result_mode='origin', unique_methods=['rec', 'res'], 
                                            title="Sub201 origin", legend=origin_legend, 
                                            xlable = 'Number of sessions in training set', ylable='Accuracy',
                                            plot_n_subs = False)
ax.text(0.02, 0.98, 'B', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

plt.subplots_adjust(hspace=0.4, left=0.15)


#### IEEE panel #####
ieee_results = figs(
    ieee_results_files, 
    main_exp=1,
    subplots_dim=ieee_subplots_dim,
    fig_size=ieee_figs_size,
    title="Figure3")

# ieee_results.plot_all_basic_results(do_subplots=True)
ax1 = ieee_results.add_combined_results_subplot(result_mode='task', unique_methods=['ae_test'], 
                                            title="IEEE task", legend=task_legend, 
                                            ylable='Accuracy', plot_n_subs = False)
ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

ax2 = ieee_results.add_combined_results_subplot(result_mode='origin', unique_methods=['rec', 'res'], 
                                            ax=2, title="IEEE origin", legend=origin_legend,
                                            plot_n_subs = False)
ax2.text(0.02, 0.98, 'C', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

ieee_results.filter_results_by_acc(0.3)
ax3 = ieee_results.add_combined_results_subplot(result_mode='task', unique_methods=['ae_test'], 
                                            title="IEEE task > 30%", legend=task_legend,
                                            xlable = 'Number of sessions in training set',
                                            plot_n_subs = False) 
ax3.text(0.02, 0.98, 'B', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

# set the same y ticks for ax1 and ax3
yticks = ax3.get_yticks()
ax1.set_yticks(yticks)
ylim = ax3.get_ylim()
ax1.set_ylim(ylim)
plt.subplots_adjust(bottom=0.15)


##### Shu panel #####
shu_results= figs(
    shu_results_files,
    main_exp=1,
    subplots_dim=shu_subplots_dim,
    fig_size=shu_figs_size,
    title="Figure4")


ax1 = shu_results.add_combined_results_subplot(result_mode='task', unique_methods=['ae_test'], title="Shu task", legend=task_legend,
                                            ylable='Accuracy', plot_n_subs = False)
ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

ax2 = shu_results.add_combined_results_subplot(result_mode='origin', unique_methods=['rec', 'res'], 
                                            ax=2, title="Shu origin", legend=origin_legend, plot_n_subs = False)
ax2.text(0.02, 0.98, 'C', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

shu_results.filter_results_by_acc(0.55)
ax3 = shu_results.add_combined_results_subplot(result_mode='task', unique_methods=['ae_test'], 
                                            title="Shu task > 55%", legend=task_legend,
                                            xlable = 'Number of sessions in training set', 
                                            plot_n_subs = False)
ax3.text(0.02, 0.98, 'B', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

# set the same y ticks for ax1 and ax3
yticks = ax3.get_yticks()
ax1.set_yticks(yticks)
ylim = ax3.get_ylim()
ax1.set_ylim(ylim)

plt.subplots_adjust(bottom=0.15)

# Show all panels
plt.show()

# fig, axs = plt.subplots(2,2)

# shu_results.corr_res(method='res', ax=axs[0,0])
# shu_results.corr_res(method='rec', ax=axs[0,1])
# shu_results.corr_res(method='res', mean_over='range', ax=axs[1,0])
# shu_results.corr_res(method='rec', mean_over='range', ax=axs[1,1])