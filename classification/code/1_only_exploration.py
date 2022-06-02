# start of the analysis
# get information on distribution of y, missing values, outliers, concentrations and skewness
# selection for each file: module get_data.py
# complete file (data)
# first -> (80%,20%) tr_tu - test split (tr_tu, test)
# folds -> fold 0 of 10 of tr_tu (train_x, train_y)

import os as os
from Project_classification.code.get_data import read_data, get_fold
from Project_classification.code.exploration import info_fold, draw_report

nm_files = sorted(os.listdir('../data'))
for nr_file in range(5):
    this_fold = 0
    data_name = nm_files[nr_file].split('.')[0]
    data = read_data('../data/' + nm_files[nr_file])
    # select training fold 0
    train_x, train_y = get_fold(dataset=data, kind='training', fold_nr=this_fold)
    # get relevant info for exploration or preprocessing
    msg_info, graph_info, bounds, idx_info, cl_trx, sc_obj = \
        info_fold(train_x, train_y, pct_mi=0.5, pct_cc=0.05, dist_out=10, n_pc=5)
    # graphical report
    draw_report(data_name, graph_info, msg_info, bounds, nr_fold=this_fold)
