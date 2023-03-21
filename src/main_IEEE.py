import modules.IEEE_models as models
import modules.IEEE_data_extractor  as data_extractor
import modules.expirament as exp
if __name__ == "__main__": 

    # extract data for subs 
    all_sub_EEG_dict = data_extractor.get_all_subs_EEG_dict()

    # run multi experimant
    f_task_path, f_res_path = exp.run_all_subs_multi_iterations(all_sub_EEG_dict, [1,2], 2)

    # get mean results over iterations and subjects for task and origin day classification
    task_result, rng_list, task_mtd_list, n_itr = exp.get_mean_result_from_file(f_task_path)
    origin_result, rng_list, mtd_list = exp.get_mean_result_from_file(f_res_path)