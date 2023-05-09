# import modules.IEEE_models as models
import modules.IEEE_data_extractor  as data_extractor
import modules.experiment as exp
from modules.properties import IEEE_properties as props
from modules.Results import Results_Processor as rp

if __name__ == "__main__": 

    # extract data for subs 
    # all_sub_EEG_dict = data_extractor.get_all_subs_EEG_dict()

    # # run multi experimant
    # f_task_path, f_res_path = exp.run_all_subs_multi_iterations(props, all_sub_EEG_dict, [1,2], 2)

    # get mean results over iterations and subjects for task and origin day classification
    # task_result, rng_list, task_mtd_list, n_itr = exp.get_mean_result_from_file(f_task_path)
    # origin_result, rng_list, mtd_list, n_itr = exp.get_mean_result_from_file(f_res_path)

    # task_result, rng_list, task_mtd_list, n_itr = exp.get_mean_result_from_file(f_task_path)


    shu_res_task_super = rp(f_name='task_iters_timestr_20230309-154106.pickle')
    shu_res_task_super.filter_sub()



    # origin_result, rng_list, mtd_list, n_itr = exp.get_mean_result_from_file('C:/Users/ofera/studies/NAWD/results/task_iters_timestr_20230330-162826.pickle')
    # mean_mat, std_mat = exp.get_results_for_plots(origin_result, rng_list, mtd_list)
    # # mean_mat[0,2] = 0.76
    # colors = ['#27496d', '#8c1d40', '#0f5e3e']
    # colors = ['#556b2f', '#8b0000', '#483d8b']
    # colors = ['#008080', '#800080', '#800000']
    # colors = ['#008080', '#800080', '#800000', '#FFA500', '#008000']
    # exp.plot_scores_mean_and_std(rng_list, mean_mat, std_mat, colors, mtd_list)
    # print(mtd_list)
    # print(mean_mat)
    
