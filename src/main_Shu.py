import modules.IEEE_models as models
import modules.Shu_data_extractor as data_extractor
import modules.utils as utils

if __name__ == "__main__": 

    # extract data for subs 
    all_sub_EEG_dict = data_extractor.get_all_subs_EEG_dict()
    
    # run multi experimant
    f_task_path, f_res_path = utils.run_all_subs_multi_iterations(all_sub_EEG_dict, [1,2], 2)

    # get mean results over iterations and subjects for task and origin day classification
    task_result, rng_list, task_mtd_list, n_itr = utils.get_mean_result_from_file(f_task_path)
    origin_result, rng_list, mtd_list = utils.get_mean_result_from_file(f_res_path)