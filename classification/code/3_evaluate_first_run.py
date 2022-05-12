import os as os
import matplotlib.pyplot as plt
import pandas as pd
from Project_classification.code.evaluation import read_results
fig = plt.figure(0, figsize=(12.0, 9.0))
fig.suptitle('Stability: balanced accuracy score')
best_models,best_hypers,top5 = [],[], []
loc_mdl = '../models/balanced_accuracies_file_'
for nr_file in range(5):
    # read the results
    folds = pd.concat([read_results(nr_file, nr_fold) for nr_fold in range(10)])
    scores = folds.pivot(index=['file','model','full_model'], columns='fold', values='bac')

    # choose the model (mean - 1 sd is maximal over the folds)
    scores_eval = scores.copy()
    scores_eval['mean_bac'] = scores.mean(axis=1)
    scores_eval['std_bac'] = scores.std(axis=1)
    scores_eval['mean_1s_bac'] = scores_eval.mean_bac - scores_eval.std_bac
    scores_eval = scores_eval.reset_index()
    best_mean_1s = scores_eval.nlargest(columns='mean_1s_bac', n=1)
    name_best_model = best_mean_1s.full_model.iloc[0]
    best_hyper = folds.where((folds.full_model == name_best_model) & (folds.fold == 0) ).dropna()
    best_models.append(best_mean_1s)
    best_hypers.append(best_hyper)

    # graphical report
    n_candidates = 5
    scores_graph = scores_eval.nlargest(columns='mean_1s_bac', n=n_candidates)
    top5.append(scores_graph)
    ax_n0 = plt.subplot2grid((5, 1), (nr_file, 0))
#    ax_n0.plot(x=list(range(10)) * n_candidates, y=scores_graph.filter(range(10)).T)
    ax_n0.plot(scores_graph.filter(range(10)).T, color='lightgrey', linestyle='dotted', linewidth=1)
    ax_n0.scatter(x=range(10),y=best_mean_1s.filter(range(10)), color='red', s=12)
    rm_ticks = [tick.set_visible(False) for tick in ax_n0.get_xticklabels()]
    dr_nm, clf_nm = best_mean_1s.model.iloc[0].split('_')
    ax_n0.set_title(f"best mean 1s: {best_mean_1s.full_model.iloc[0]}, "
                    f" mean = {best_mean_1s.mean_bac.iloc[0]:.3f}, mean 1s = {best_mean_1s.mean_1s_bac.iloc[0]:.3f}",
                    fontsize=8)

    scores.to_csv('../models/stability_folds_file' + str(nr_file) + '.csv')

#output
plt.savefig('../models/stability_bac.pdf')
best_models = pd.concat(best_models).set_index("file")
best_models.to_csv('../models/retained_model_per_file.csv')
best_hypers = pd.concat(best_hypers).set_index("file")
best_hypers.to_csv('../models/retained_hyper_pars_per_file.csv')
top5_models = pd.concat(top5).set_index("file")
top5_models.to_csv('../models/top5_models_per_file.csv')