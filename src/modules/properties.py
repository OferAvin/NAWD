props_dict = {
    'device' : 'cpu', # 'cuda'
    
    'tmin' : 0,
    'tmax' : 6,
    'select_label' : [1, 4],

    'filterLim' : [1,40], # In Hz

    'fs' : 100,
    'ae_lrn_rt' : 3e-4,
    'n_epochs' : 250,
    'btch_sz' : 8,
    'cnvl_filters' : [8, 16, 32],

    'n_feature_select' : '250',

    'amp_thresh' : 250,
    'min_trials' : 10,

    'sub_list' : ['A2', 'A3'],# 'A4', 'A5', 'A6', 'A7', 'A8','S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8','S9','S10','S11', 'S12'],
    
    'data_dir' : './data/ieee_dataset/'
}
