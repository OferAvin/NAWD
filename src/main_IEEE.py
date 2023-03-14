import modules.IEEE_models as models
import modules.IEEE_data  as data
from modules.properties import props_dict

if __name__ == "__main__": 

    # extract data for subs 
    all_sub_EEG_dict = data.get_all_subs_EEG_dict(props_dict)

    # run multi experimant
    f_task_path, f_res_path = models.run_all_subs_multi_iterations(props_dict, all_sub_EEG_dict, [1,2], 2)

    # get mean results over iterations and subjects for task and origin day classification
    task_result, rng_list, task_mtd_list, n_itr = models.get_mean_result_from_file(f_task_path)
    origin_result, rng_list, mtd_list = models.get_mean_result_from_file(f_res_path)