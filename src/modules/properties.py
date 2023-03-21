IEEE_properties = {
    'device' : 'cuda', # 'cpu
    
    # Data
    'tmin' : 0,
    'tmax' : 6,
    'select_label' : [1, 4],
    'filterLim' : [1,40], # In Hz
    'fs' : 100,
    'amplitude_th' : 250,
    'min_trials' : 10,

    'sub_list' : ['A2', 'A3'],# 'A4', 'A5', 'A6', 'A7', 'A8','S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8','S9','S10','S11', 'S12'],
    'data_dir' : './data/ieee_dataset/',

    # Model
    'ae_lrn_rt' : 3e-4,
    'n_epochs' : 250,
    'btch_sz' : 8,
    'cnvl_filters' : [8, 16, 32],
    'enoder_pad' : [1, 1, 1],
    'decoder_pad' : [1, 0, 1, 0, 1, 2],
    'latent_sz' : 4704

}

Shu_properties = {
    'device' : 'cpu', # 'cuda'

    # Data
    'filterLim' : [1,40], # In Hz
    'fs' : 250,
    'sub_list' : ['{:03d}'.format(n) for n in range(1,26)],
    'data_dir' : './data/shu_dataset/',

    # Model
    'enoder_pad' : [1, 1, 1],
    'decoder_pad' : [1, 0, 1, 0, 1, 2],
    'latent_sz' : 1504

}

hyper_params = {
    'ae_lrn_rt' : 3e-4,
    'n_epochs' : 250,
    'btch_sz' : 8,
    'cnvl_filters' : [8, 16, 32],
}