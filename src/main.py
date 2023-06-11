import modules.IEEE_data_extractor as ieee_data
import modules.Shu_data_extractor as shu_data
from modules.chist_era_data_extractor import Chist_Era_data_extractor as sub201_data_extractor
from modules.experiment import Experiment as exp
from modules.properties import IEEE_properties as ieee_props
from modules.properties import sub201_properties  as sub201_props

if __name__ == "__main__":
    sub201_data = sub201_data_extractor() 
    sub201_supervised = exp('sub201_supervised', sub201_data, sub201_props, [1,99], 1, mode = 'supervised')
    sub201_supervised.run_experiment()

    sub201_unsupervised = exp('sub201_supervised', sub201_data, sub201_props, [1,2], 1, mode = 'unsupervised')
    sub201_unsupervised.run_experiment()

    # IEEE_unsupervised_2c = exp('IEEE_unsupervised_2c', ieee_data, ieee_props, [1,6], 10, mode = 'unsupervised')
    # IEEE_unsupervised_2c.run_experiment()
    
    # ieee_props['select_label'] = [1, 2, 3, 4]

    # IEEE_unsupervised_4c = exp('IEEE_unsupervised_4c', ieee_data, ieee_props, [1,6], 10, mode = 'unsupervised')
    # IEEE_unsupervised_4c.run_experiment()

    # IEEE_supervised_4c = exp('IEEE_supervised_4c', ieee_data, ieee_props, [1,6], 10, mode = 'supervised')
    # IEEE_supervised_4c.run_experiment()