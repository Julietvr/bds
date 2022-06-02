# calculate more performance measures the the models of interest
# candidates were selected in 3_evaluate_first_only_bac
# apply the different models for every fold of the associated file
# i/e. (train,val) split, preprocessing, dimension reduction and classification
# calculate performance measures per fold (recall, precision, f1, auroc, logloss)
# assemble performance measures (columns) per model per fold (rows)

import os as os
import pandas as pd
from Project_classification.code.get_data import read_data
from Project_classification.code.preprocessing import prepare_the_data
from Project_classification.code.evaluation import get_predictions, table_metrics

# settings
seed = 42
nm_files = sorted(os.listdir('../data'))
nms_eval_out = ['top5', 'per_type']
nms_eval_in = ['top5_models_per_file','compare_types_per_file']
fold_range = range(0,10)
for nm_in, nm_out in zip(nms_eval_in, nms_eval_out):
    # data frame per file with candidates: with hyperparameters, scores per fold
    hyper_parameters = pd.read_csv('../evaluation/' + nm_in + '.csv')
    complete_list=[]
    for _,hp_i in hyper_parameters.iterrows():
        # one row is one model,
        nr_file = hp_i.file
        data = read_data('../data/' + nm_files[nr_file])
        dr_type, clf_type = hp_i.model.split('_')
        all_scores = []
        for this_fold in fold_range:
            clean_tr_x, clean_tr_y, clean_val_x, clean_val_y = prepare_the_data(data, this_fold, seed)
            y_hat_class, y_hat_prob = get_predictions(clean_tr_x,clean_tr_y,clean_val_x,hp_i, seed)
            all_scores.append(table_metrics(clean_val_y,y_hat_class, y_hat_prob))
        # all scores for one model to data frame: rows = fold, columns is performance metrics
        metrics_mod_i = pd.DataFrame(all_scores)
        measures = metrics_mod_i.columns
        metrics_mod_i['file'] = hp_i.file
        metrics_mod_i['fold'] = list(fold_range)
        metrics_mod_i['full_model'] = hp_i.full_model
        # complete list with dataframes (performance measures per fold)
        complete_list.append(metrics_mod_i)
    # concatenate over the different models hp_i - write to csv
    complete_eval = pd.concat(complete_list)
    complete_eval.to_csv('../evaluation/' + nm_out + '_all_metrics.csv')
    # make a summary with mean and standarddeviation of the measures - write to cv
    fun_agg = ['mean','std']
    summarize_eval = complete_eval.\
        groupby(['file','full_model']).\
        agg(dict(zip(measures, [fun_agg]*len(measures)))).round(4)
    summarize_eval.to_csv('../evaluation/' + nm_out + '_summ_metrics.csv')