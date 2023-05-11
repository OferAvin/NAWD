import modules.IEEE_data_extractor as ieee_data_extractor
import modules.Shu_data_extractor as shu_data_extractor
from modules.Experiment import Experiment as exp
from modules.properties import IEEE_properties as ieee_props

if __name__ == "__main__":

    IEEE_unsupervised_2c = exp('IEEE_unsupervised_2c', ieee_data_extractor, ieee_props, [1,6], 10, mode = 'unsupervised')
    IEEE_unsupervised_2c.run_experiment()
    
    ieee_props['select_label'] = [1, 2, 3, 4]

    IEEE_unsupervised_4c = exp('IEEE_unsupervised_4c', ieee_data_extractor, ieee_props, [1,6], 10, mode = 'unsupervised')
    IEEE_unsupervised_4c.run_experiment()

    IEEE_supervised_4c = exp('IEEE_supervised_4c', ieee_data_extractor, ieee_props, [1,6], 10, mode = 'supervised')
    IEEE_supervised_4c.run_experiment()