import numpy as np
import pandas as pd


from Project_classification.code.models import get_classifier, get_dim_reduced, get_hyper_parameters, get_hp_baseline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import log_loss
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackError


def read_results(nr_file, nr_fold):
    """
    reads the file with the balanced accuracies for one file and 1 fold
    the input file has 9 scores per row
    returns the scores in longer format: adds the columns file, fold, model, full_model and bac
    """
    # read file
    loc_mdl = '../models/balanced_accuracies_file_'
    results = pd.read_csv(loc_mdl + str(nr_file) + '_fold_' + str(nr_fold) + '.csv')
    # add columns: file, fold
    results['file'] = nr_file
    results['fold'] = nr_fold
    # columns with scores in nm_cols, all other information in nm_idx
    nm_cols = [dim_red + '_' + nm_mdl for dim_red in ['br', 'pc', 'kpc'] for nm_mdl in ['kd', 'rf', 'lr']]
    nm_idx = list(results.columns.drop(nm_cols))
    # pivot to longer format
    stacked = pd.DataFrame(results.set_index(nm_idx).rename_axis('model',axis=1).stack().rename('bac')).reset_index()
    # add column full_model
    stacked['full_model'] = all_pars_in_model(stacked)
    return stacked


def get_predictions(tr_x, tr_y, val_x, hp_mod, seed):
    """
    predictions and confusion matrix
    :param tr_x: training set (cleaned, before dimension reduction)
    :param tr_y: classes for y from training set
    :param val_x: validations set, explanatory variables (cleaned)
    :param val_y: classes for y from validation set
    :param hp_mod: parameters for the model, defines the classifier
    :param seed: seed for the fitting process
    :return: predictions for y (prob and proba) and the confusion matrix plot
    """
    #  define classifier clf and get prediction
    dim_red_type, clf_type = hp_mod.model.split('_')
    tf_tr_x, tf_val_x = get_dim_reduced(tr_x, tr_y, val_x, hp_mod, dim_red_type)
    clf = get_classifier(hp_mod, clf_type, seed)
    try:
        # fit data get predicted classes and predicted probabilities (not supported in Kfda)
        clf.fit(tf_tr_x, tr_y)
        y_hat = clf.predict(tf_val_x)
        y_hat_prob = None if clf_type == "kd" else clf.predict_proba(tf_val_x)
        return y_hat, y_hat_prob
    # kernel discriminant analysis is not always defined
    except ArpackError as err:
        print(err, hp_mod.full_model)
        return None, None


def table_metrics(y_test, y_hat, y_hat_proba):
    """
    defines several metrics and puts them in a table
    :param y_test: classes for y
    :param y_hat: predicted classes
    :param y_hat_proba: probabilities per class
    :return: table with bac, acc, prec, f1, log-loss and auc
    """
    names = ['balanced_accuracy', 'accuracy', 'recall', 'precision', 'f1', 'auroc', 'logloss']
    if y_hat is None:
        return pd.Series([None]*7, index =names)
    measures_wt = [recall_score, precision_score, f1_score]
    scores_acc = [balanced_accuracy_score(y_test, y_hat), accuracy_score(y_test,y_hat)]
    scores_wt = [score_s(y_test, y_hat, average='weighted') for score_s in measures_wt]
    scores_prob = [None, None] if y_hat_proba is None\
        else [roc_auc_score(y_test, y_hat_proba,multi_class='ovr'), log_loss(y_test, y_hat_proba)]
    return pd.Series(scores_acc + scores_wt + scores_prob, index =names)


def all_pars_in_model(hyp_par):
    """
    returns full name of the model for every set of hyper parameters
    model has to be specified in input
    full_name can function as key when parameters have to be retrieved later on
    """
    # turn column model in 2 list: dim_red and nm_clf
    dim_red, nm_clf = zip(*hyp_par.model.str.split('_'))
    # logistic regression: parameters l1_ratio and regulariation term C
    lr_pars = ['l1_' + str(x.lr_l1) + '_C_' + str(x.lr_reg)
               if x.model.endswith('lr') else '' for _, x in hyp_par.iterrows()]
    # random forest: max_leaf_nodes and nr of estimated trees
    rf_pars = ['max_leaf_' + str(x.rf_max_leaf) + '_max_n_estimates_' + str(x.rf_n_est)
               if x.model.endswith('rf') else '' for _, x in hyp_par.iterrows()]
    # kernel discriminant analysis: gaussian kernels and polynomials of degree 1 to 4
    kd_pars = ['kernel_' + str(x.kda_kernel) + ('_degree_') + str(x.kda_dg)
               if x.model.endswith('kd') else '' for _, x in hyp_par.iterrows()]
    # best representatives removes x with similarity (x,y) lower than rm_y,
    # chooses from features which are more correlated with x then keep_x
    br_pars = ['keep_x_' + str(x.br_keep_x) + ('_rm_y_') + str(x.br_rm_y)
               if x.model.startswith('br') else '' for _, x in hyp_par.iterrows()]
    # principal component analysis - kernel pca with gaussian kernel
    pc_pars = ['pc_' + str(x.n_pc)
               if x.model.startswith(('pc','kpc')) else '' for _, x in hyp_par.iterrows()]
    # join information about the specified model to one string per row
    clf_pars = [''.join(pars) for pars in zip(lr_pars,rf_pars,kd_pars)]
    dim_red_pars = [''.join(pars) for pars in zip(br_pars,pc_pars)]
    return [' '.join(names) for names in zip(nm_clf, clf_pars, dim_red, dim_red_pars)]


def hp_complete():
    """
    retrieves all hyperparameters from the first run and the baseline models
    indexed with the name of the full model
    useful for evaluation
    """
    part1 = get_hyper_parameters('first')
    nrow = part1.shape[0]
    all_models = [pd.DataFrame({'model': [dr + '_' + cl] * nrow})
                  for dr in ['br','pc','kpc'] for cl in ['lr', 'rf', 'kd']]
    parts_first = [part1.join(mod) for mod in all_models]
    part_baseline = get_hp_baseline()
    full = pd.concat(parts_first + [part_baseline])
    full['full_model'] = all_pars_in_model(full)
    return full.set_index(['full_model'])