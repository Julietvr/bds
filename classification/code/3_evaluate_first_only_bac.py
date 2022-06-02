# select for every file the best model based on the balanced accuracy for the tuning set
# compromise high mean vs stability: highest value for mean minus 1 stddev
# csv1: parameters for the 5 best models overall (top5)
# csv2: best parameters for every combination of dimension reduction x classifier (per_type)

import matplotlib.pyplot as plt
import pandas as pd
from Project_classification.code.evaluation import read_results
from Project_classification.code.models import get_hyper_parameters

hyper_parameters = get_hyper_parameters(run='first')
retained_info = ['file','model','full_model'] + [hp for hp in hyper_parameters]

fig = plt.figure(0, figsize=(12.0, 9.0))
fig.suptitle('Stability: balanced accuracy score')
top_n_models, list_per_type = [], []
loc_mdl = '../models/balanced_accuracies_file_'
for nr_file in range(5):
    # read results (one score bac per row) after run 2_calculate_models, glue all the folds together
    folds = pd.concat([read_results(nr_file, nr_fold) for nr_fold in range(10)])
    # pivot to a wider format: one model with the bac for all the folds on one row
    scores = folds.pivot(index=retained_info, columns='fold', values='bac')
    # create new columns: mean, stddev and mean_1s_bac (mean - 1 stddev)
    scores_eval = scores.copy()
    scores_eval['mean_bac'] = scores.mean(axis=1)
    scores_eval['std_bac'] = scores.std(axis=1)
    scores_eval['mean_1s_bac'] = scores_eval.mean_bac - scores_eval.std_bac
    scores_eval = scores_eval.reset_index()

    # part 1 best 5 candidates
    # scores of the best 5 candidates
    n_candidates = 5
    scores_top_n = scores_eval.nlargest(columns='mean_1s_bac', n=n_candidates)
    top_n_models.append(scores_top_n)
    # we also need the full name of the best candidate
    best_mean_1s = scores_eval.nlargest(columns='mean_1s_bac', n=1)
    # graphical report 1: best 5 candidates
    ax_n0 = plt.subplot2grid((5, 1), (nr_file, 0))
    ax_n0.plot(scores_top_n.filter(range(10)).T, color='lightgrey', linestyle='dotted', linewidth=1)
    ax_n0.scatter(x=range(10),y=best_mean_1s.filter(range(10)), color='red', s=12)
    rm_ticks = [tick.set_visible(False) for tick in ax_n0.get_xticklabels()]
    dr_nm, clf_nm = best_mean_1s.model.iloc[0].split('_')
    ax_n0.set_title(f"best mean 1s: {best_mean_1s.full_model.iloc[0]}, "
                    f" mean = {best_mean_1s.mean_bac.iloc[0]:.3f}, mean 1s = {best_mean_1s.mean_1s_bac.iloc[0]:.3f}",
                    fontsize=8)
    # output list of best 5 models and their scores
    scores.to_csv('../evaluation/stability_folds_file' + str(nr_file) + '.csv')

    # part 2: best hyper parameters per type of model
    # get the best score per file per model type
    model_per_type = scores_eval.sort_values('mean_1s_bac', ascending=False).groupby(['file', 'model']).head(1)
    list_per_type.append(model_per_type)
    model_per_type.to_csv('../evaluation/stability_per_type_file' + str(nr_file) + '.csv')

#output
plt.savefig('../evaluation/stability_bac.pdf')
top_n_models = pd.concat(top_n_models).set_index("file")
top_per_type = pd.concat(list_per_type).set_index("file")
top_n_models.to_csv('../evaluation/top5_models_per_file.csv')
top_per_type.to_csv('../evaluation/compare_types_per_file.csv')