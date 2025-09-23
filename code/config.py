def get_configs():
    paras = {
        'data_name': 'EFB-EMVC-Youtube',
        'choice':1,
        'fusion_ways': ['add', 'mul', 'cat', 'max', 'avg'],
        'feature_statistics': ['a', 'a', 'a', 'a', 'a'],
        'fused_nb_feats': 128,
        'nb_view': 5,
        'pop_size': 28,
        'nb_iters': 20,
        'idx_split': 1,
        'result_save_dir': 'fused_feats=128, t=20, test_1',
        'gpu_list': [0,1,2,3,4,5,6,7],
        'epochs': 200,
        'batch_size': 512,
        'patience': 10,
        'is_remove': True,
        'crossover_rate': 0.9,
        'mutation_rate': 0.2,
        'noisy': True,
        'max_len': 15,
        # data set information
        'image_size': {
            'w': 230, 'h': 230, 'c': 1},
        'classes': 31,
        'split_data': [2, 4, 6, 8, 10],
        'fusion_L': 4,
        'fusion_C': 512,
    }
    return paras










