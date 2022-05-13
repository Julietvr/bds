import os as os
#os.chdir(os.getcwd()+"/Project_classification/code/")
import pandas as pd
from Project_classification.code.get_data import read_data
from Project_classification.code.preprocessing import prepare_the_data
from Project_classification.code.dimensionreduction import best_representatives, pca, kernel_pca
from Project_classification.code.models import get_hyper_parameters, eval_one_dataset, get_classifier, get_dim_reduced
from sklearn.metrics import balanced_accuracy_score

# main code
seed = 42
nm_files = sorted(os.listdir('../data'))
hyper_parameters = get_hyper_parameters(run='first')
this_metric = balanced_accuracy_score
file_range = range(2,5)
fold_range = range(1,10)
for nr_file in file_range:
    data_name = nm_files[nr_file].split('.')[0]
    data = read_data('../data/' + nm_files[nr_file])
    for this_fold in fold_range:
        clean_tr_x, clean_tr_y, clean_val_x, clean_val_y = prepare_the_data(data, this_fold, seed)
        all_scores = []
        for idx, hp_i in hyper_parameters.iterrows():
            if idx % 5 == 0:
                br_train_x, br_val_x = get_dim_reduced(clean_tr_x, clean_tr_y, clean_val_x, hp_i, "br")
                pc_train_x, pc_val_x = get_dim_reduced(clean_tr_x, clean_tr_y, clean_val_x, hp_i, "pc")
                kpc_train_x, kpc_val_x = get_dim_reduced(clean_tr_x, clean_tr_y, clean_val_x, hp_i, "kpc")
            scores_idx = []
            for dr_train_x, dr_val_x in zip([br_train_x, pc_train_x, kpc_train_x], [br_val_x, pc_val_x, kpc_val_x]):
                scores_idx.extend(eval_one_dataset(dr_train_x, clean_tr_y, dr_val_x, clean_val_y,
                                                   hp_i, this_metric, seed))
            all_scores.append(scores_idx)
        # output first run
        nm_models = [dim + '_' + mdl for dim in ['br', 'pc', 'kpc'] for mdl in ['kd', 'rf', 'lr']]
        part_scores = pd.DataFrame(all_scores,  columns=nm_models, index=hyper_parameters.index)
        balanced_accuracies = hyper_parameters.join(part_scores)
        nm_csv = 'balanced_accuracies_file_' + str(nr_file) + '_fold_' + str(this_fold) + '.csv'
        balanced_accuracies.to_csv('../models/' + nm_csv)
