# first run: calculate balanced accuracy for
# 5 files x 10 folds x 3 dimension reduction techniques x 3 classifiers x 25 sets of hyper parameters
# data cleaning fitted on training set, transformations applied on tuning set
# dimension reduction and classification fitted on training set, predicted for tuning set
# balanced accuracy measured on the tuning set

import os as os
import pandas as pd
from Project_classification.code.get_data import read_data
from Project_classification.code.preprocessing import prepare_the_data
from Project_classification.code.models import get_hyper_parameters, eval_one_dataset, get_dim_reduced
from sklearn.metrics import balanced_accuracy_score

seed = 42
nm_files = sorted(os.listdir('../data'))
hyper_parameters = get_hyper_parameters(run='first')
this_metric = balanced_accuracy_score
file_range = range(2,5)
fold_range = range(0,1)
for nr_file in file_range:
    # read the complete file
    data_name = nm_files[nr_file].split('.')[0]
    data = read_data('../data/' + nm_files[nr_file])
    for this_fold in fold_range:
        # get information on missing, outliers, concentrations, skewness from training fold
        # apply transformations and deletions on both training and tuning set
        clean_tr_x, clean_tr_y, clean_val_x, clean_val_y = prepare_the_data(data, this_fold, seed)

        # dimension reduction, classifier, balanced accuracy
        # 25 rows per file per fold
        all_scores = []
        for idx, hp_i in hyper_parameters.iterrows():
            # hyper_parameters for dimension reduction change every 5 rows - avoid double calculation
            if idx % 5 == 0:
                # dimension reduction: best representatives, pca and kernel pca
                # fit on training fold, apply on both training and tuning fold
                br_train_x, br_val_x = get_dim_reduced(clean_tr_x, clean_tr_y, clean_val_x, hp_i, "br")
                pc_train_x, pc_val_x = get_dim_reduced(clean_tr_x, clean_tr_y, clean_val_x, hp_i, "pc")
                kpc_train_x, kpc_val_x = get_dim_reduced(clean_tr_x, clean_tr_y, clean_val_x, hp_i, "kpc")
            # 3 scores per row
            scores_idx = []
            for dr_train_x, dr_val_x in zip([br_train_x, pc_train_x, kpc_train_x], [br_val_x, pc_val_x, kpc_val_x]):
                # classification: kernel discriminant analysis, random forest, logistic regression
                # fit on training fold, predict on tuning fold returns balanced accuracy
                scores_idx.extend(eval_one_dataset(dr_train_x, clean_tr_y, dr_val_x, clean_val_y,
                                                   hp_i, this_metric, seed))
            all_scores.append(scores_idx)

        # output first run: values of hyper_parameters and corresponding balanced accuracy per file per fold
        nm_models = [dim + '_' + mdl for dim in ['br', 'pc', 'kpc'] for mdl in ['kd', 'rf', 'lr']]
        part_scores = pd.DataFrame(all_scores,  columns=nm_models, index=hyper_parameters.index)
        balanced_accuracies = hyper_parameters.join(part_scores)
        nm_csv = 'balanced_accuracies_file_' + str(nr_file) + '_fold_' + str(this_fold) + '.csv'
        balanced_accuracies.to_csv('../models/' + nm_csv)
