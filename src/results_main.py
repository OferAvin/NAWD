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


# Sub201 panel
# sub201_results = figs(
#     sub201_results_files,
#     main_exp=1,
#     subplots_dim=sub201_subplots_dim,
#     fig_size=sub201_fig_size,
#     title="Figure2")

# sub201_results.add_combined_results_subplot(result_mode='task', unique_methods=['ae_test'], title="Sub201 task")
# sub201_results.add_combined_results_subplot(result_mode='origin', unique_methods=['rec', 'res'], title="Sub201 origin")


# IEEE panel
ieee_results = figs(
    ieee_results_files, 
    main_exp=1,
    subplots_dim=ieee_subplots_dim,
    fig_size=ieee_figs_size,
    title="Figure3")
ieee_results.plot_all_basic_results(do_subplots=False)
plt.show()
ieee_results.add_combined_results_subplot(result_mode='task', unique_methods=['ae_test'], title="IEEE task")
ieee_results.add_combined_results_subplot(result_mode='origin', unique_methods=['rec', 'res'], title="IEEE origin")

ieee_results.filter_results_by_acc(0.3)
ieee_results.add_combined_results_subplot(result_mode='task', unique_methods=['ae_test'], title="IEEE task > 30%")




# Shu panel
shu_results= figs(
    shu_results_files,
    main_exp=1,
    subplots_dim=shu_subplots_dim,
    fig_size=shu_figs_size,
    title="Figure4")

shu_results.add_combined_results_subplot(result_mode='task', unique_methods=['ae_test'], title="Shu task")
shu_results.add_combined_results_subplot(result_mode='origin', unique_methods=['rec', 'res'], title="Shu origin")

shu_results.filter_results_by_acc(0.55)
shu_results.add_combined_results_subplot(result_mode='task', unique_methods=['ae_test'], title="Shu task > 55%")



# Show all panels
plt.show()

