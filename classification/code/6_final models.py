import os as os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import MDS
from Project_classification.code.get_data import read_data
from Project_classification.code.exploration import rm_ticks
from Project_classification.code.preprocessing import prepare_the_data
from Project_classification.code.evaluation import get_predictions, table_metrics,  hp_complete

# look for the best parameters
# get the scores of the best candidates
top5_summ = pd.read_csv('../evaluation/top5_summ_metrics.csv')
part1 = top5_summ.filter(['Unnamed: 0', 'Unnamed: 1','balanced_accuracy', 'balanced_accuracy.1']).iloc[2:]
baseline_summ = pd.read_csv('../evaluation/baseline_summ_metrics.csv')
part2 = baseline_summ.filter(['Unnamed: 0','Unnamed: 1','balanced_accuracy', 'balanced_accuracy.1']).iloc[2:]
models = pd.concat([part1,part2])
# select the best model per file based on bac: mean - 1 std
models.columns = ['file', 'full_model', 'bac_mean', 'bac_std']
models['bac_1s'] = models['bac_mean'].astype(float) - models['bac_std'].astype(float)
winners = models.sort_values(['bac_1s'], ascending=False).groupby(['file']).head(1)
key_winners= winners.set_index(['full_model'])
# add the relevant sets of hyper parameters to the records
hp_parameters = hp_complete()
final_parameters = key_winners.join(hp_parameters, how='left').reset_index().sort_values('file')


# back to modelling, this time on tr_tu/test
seed = 42
nm_files = sorted(os.listdir('../data'))
all_scores = []
fig0 = plt.figure(0, figsize=(12.0, 9.0))
fig1 = plt.figure(1, figsize=(12.0, 9.0))
ax_legend = plt.subplot2grid((3, 2), (2, 1), fig=fig1)
for idx, hp_i in final_parameters.iterrows():
    nr_file = int(hp_i.file)
    print(nr_file)
    # complete data for file idx
    data = read_data('../data/' + nm_files[nr_file])
    dr_type, clf_type = hp_i.model.split('_')
    # split (tr_tu, test), and preprocessing
    clean_tr_x, clean_tr_y, clean_val_x, clean_val_y = prepare_the_data(data, 0, seed, final=True)
    # fitted values, probabilities to assess performance measures
    y_hat_class, y_hat_prob = get_predictions(clean_tr_x, clean_tr_y, clean_val_x, hp_i, seed)
    all_scores.append(table_metrics(clean_val_y, y_hat_class, y_hat_prob))

    # plot: confusion matrix
    ax_i = plt.subplot2grid((3, 2), (int(nr_file/2), nr_file % 2), fig=fig0)

    ax_i.set_title(nm_files[nr_file].split('.')[0], fontsize=8)
    cf_mat = ConfusionMatrixDisplay.from_predictions(clean_val_y, y_hat_class, ax=ax_i)
    if nr_file < 3:
        rm_ticks(ax_i,rm_x=True, rm_y=False)

    # plot: visualization
    embedding = MDS(n_components=2)
    prob_transformed = embedding.fit_transform(y_hat_prob)
    ax_mds = plt.subplot2grid((3, 2), (int(nr_file/2), nr_file % 2), fig=fig1)
    colors = ["purple", "navy", "cornflowerblue", "magenta", "teal"]
    dc_col = dict(zip(clean_val_y.value_counts().index, colors))
    col_y = [dc_col[y] if y == y_h else 'white' for y, y_h in zip(clean_val_y, y_hat_class)]
    ecol_y = [dc_col[y] for y in clean_val_y]
    ax_mds.scatter(x=prob_transformed[:, 0], y=prob_transformed[:, 1], c=col_y, edgecolors=ecol_y)
    ax_mds.set_title('File ' + str(nr_file) + ': ' + nm_files[nr_file].split('.')[0], fontsize=8)

    ord_lab, ord_col = zip(*sorted(dc_col.items()))
    ax_legend.scatter(x=range(5), y= [nr_file]*5, c= ord_col)


# output final metrics (all performance measures), visualisation af fit and confusion matrix
metrics_final = pd.DataFrame(all_scores).round(4)
measures = metrics_final.columns
metrics_final['file'] = [hp_i.file for _, hp_i in final_parameters.iterrows()]
metrics_final['full_model'] = [hp_i.full_model for _, hp_i in final_parameters.iterrows()]
metrics_final.to_csv('../final/final_metrics.csv')

ax_legend.set_xticks(ticks=range(5), labels=ord_lab)
ax_legend.set_title('Legend (empty points are misclassified)', fontsize=8)
ax_legend.set_xlabel('Class')
ax_legend.set_ylabel('File')

fig1.suptitle('Visualisation of the final model: test set')
plt.savefig('../final/visualisation test set.pdf')
plt.close(fig1)

fig0.suptitle('Confusion matrix for final models')
plt.savefig('../final/confusion_matrix.pdf')
plt.close(fig0)

