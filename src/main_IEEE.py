# import modules.IEEE_models as models
import modules.IEEE_data_extractor as ieee_data_extractor
import modules.Shu_data_extractor as shu_data_extractor
from modules.Experiment import Experiment as exp
from modules.properties import IEEE_properties as props
from modules.Results import Results_Processor as rp

if __name__ == "__main__": 

    # extract data for subs 
    # all_sub_EEG_dict = ieee_data_extractor.get_all_subs_EEG_dict()

    # # run multi experimant
    # f_task_path, f_res_path = exp.run_all_subs_multi_iterations(props, all_sub_EEG_dict, [1,2], 2)

    # get mean results over iterations and subjects for task and origin day classification
    # task_result, rng_list, task_mtd_list, n_itr = exp.get_mean_result_from_file(f_task_path)
    # origin_result, rng_list, mtd_list, n_itr = exp.get_mean_result_from_file(f_res_path)

    # task_result, rng_list, task_mtd_list, n_itr = exp.get_mean_result_from_file(f_task_path)

    # super_shu = exp("super_ieee", shu_data_extractor, props, [1,2], 2)
    super_ieee = exp("super_ieee", ieee_data_extractor, props, [1,2], 2)
    super_ieee.extract_data()
    print(super_ieee.get_subs)
    super_ieee.run_experiment()

    # shu_res_task_super = rp(f_name='task_iters_timestr_20230309-154106.pickle')
    # shu_res_task_super.filter_sub(min_acc=0.55)
    # shu_res_task_super.process_result()
    # shu_res_task_super.plot_result(title="Nice Shot")




    
