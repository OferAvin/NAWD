import modules.IEEE_data_extractor as ieee_data_extractor
import modules.Shu_data_extractor as shu_data_extractor
from modules.Experiment import Experiment as exp
from modules.properties import IEEE_properties as props
from modules.Results import Results_Processor as rp

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

IEEE_super = rp(f_name='task_iters_timestr_20230309-154106.pickle')
IEEE_super.filter_sub_by_acc(min_acc=0.6)
removed_subs = IEEE_super.removed_subs
print(IEEE_super.n_iters)
IEEE_super.process_result()
# IEEE_super.plot_result(title="IEEE dataset - supervised")

IEEE_super = rp(f_name='origin_iters_timestr_20230309-154106.pickle')
IEEE_super.filter_out_sub_from_list(removed_subs)
print(IEEE_super.n_iters)
IEEE_super.process_result()
IEEE_super.plot_result(title="IEEE dataset - supervised (session clasification)")