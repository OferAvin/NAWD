import modules.IEEE_data_extractor as ieee_data_extractor
import modules.Shu_data_extractor as shu_data_extractor
from modules.Experiment import Experiment as exp
from modules.properties import IEEE_properties as props
from modules.Results import Results_Processor as rp

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

IEEE_unsuper = rp(f_name='task_iters_timestr_20230314-164800.pickle')
IEEE_unsuper.filter_sub_by_acc(min_acc=0.6)
removed_subs = IEEE_unsuper.removed_subs
print(IEEE_unsuper.n_iters)
IEEE_unsuper.process_result()
# # IEEE_unsuper.plot_result(title="IEEE dataset - unsupervised")

IEEE_unsuper = rp(f_name='origin_iters_timestr_20230314-164800.pickle')
IEEE_unsuper.filter_out_subs_from_results(removed_subs)
print(IEEE_unsuper.n_iters)
IEEE_unsuper.process_result()
IEEE_unsuper.plot_result(title="IEEE dataset - unsupervised (session clasification)")