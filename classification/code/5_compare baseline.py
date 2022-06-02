# what happens if no dimension reduction is applied?

import os as os
import pandas as pd
from Project_classification.code.get_data import read_data
from Project_classification.code.preprocessing import prepare_the_data
from Project_classification.code.models import get_hp_baseline
from Project_classification.code.evaluation import get_predictions, table_metrics, all_pars_in_model

# same data, same procedure but with a new set of hyper-parameters
# settings
seed = 42
nm_files = sorted(os.listdir('../data'))
fold_range = range(0,10)
# hyperparameters
hyper_parameters = get_hp_baseline()
hyper_parameters['full_model'] = all_pars_in_model(hyper_parameters)
complete_list=[]
for nr_file in range(5):
    for _,hp_i in hyper_parameters.iterrows():
        # read complete dataset
        data = read_data('../data/' + nm_files[nr_file])
        dr_type, clf_type = hp_i.model.split('_')
        all_scores = []
        for this_fold in fold_range:
            # split and preprocessing
            clean_tr_x, clean_tr_y, clean_val_x, clean_val_y = prepare_the_data(data, this_fold, seed)
            # dimension reduction (does nothing), classification and predictions for validation set
            y_hat_class, y_hat_prob = get_predictions(clean_tr_x,clean_tr_y,clean_val_x,hp_i, seed)
            # list with measures per fold
            all_scores.append(table_metrics(clean_val_y,y_hat_class, y_hat_prob))
        # data frame for one model, rows= folds, columns= performance measures
        metrics_mod_i = pd.DataFrame(all_scores)
        measures = metrics_mod_i.columns
        # add file fold full_model
        metrics_mod_i['file'] = nr_file
        metrics_mod_i['fold'] = list(fold_range)
        metrics_mod_i['full_model'] = hp_i.full_model
        # complete list with measures for all file up tu now
        # rows = model x fold, columns = performance measures
        complete_list.append(metrics_mod_i)
    # make dataframe up to now where rows = file x model x fold, columns = performance measures
    # partial info is written out
    complete_eval = pd.concat(complete_list)
    complete_eval.to_csv('../evaluation/' + 'baseline_all_metrics.csv')
    fun_agg = ['mean','std']
    summarize_eval = complete_eval.\
        groupby(['file','full_model']).\
        agg(dict(zip(measures, [fun_agg]*len(measures)))).round(4)
    summarize_eval.to_csv('../evaluation/' + 'baseline_summ_metrics.csv')