import os as os
from Project_classification.code.get_data import read_data, get_fold
from Project_classification.code.exploration import info_fold, draw_report

# impute values with regression: args ok : random_state needed!
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# main code
nm_files = sorted(os.listdir('../data'))
for nr_file in range(5):
    this_fold = 0
    data_name = nm_files[nr_file].split('.')[0]
    data = read_data('../data/' + nm_files[nr_file])
    train_x, train_y = get_fold(dataset=data, kind='training', fold_nr=this_fold)
    msg_info, graph_info, bounds, idx_info, cl_trx, sc_obj = \
        info_fold(train_x, train_y, pct_mi=0.5, pct_cc=0.05, dist_out=10, n_pc=5)
    draw_report(data_name, graph_info, msg_info, bounds, nr_fold=this_fold)
