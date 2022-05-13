import pandas as pd
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from kfda import Kfda
from Project_classification.code.dimensionreduction import best_representatives, pca, kernel_pca


def get_hyper_parameters(run="first", mids=None):
    if run == 'first':
        n_mods = 5
        n_dims = 5
        best_rep_rm_y = [ 0.1, 0.15, 0.2, .25, 0.3]
        best_rep_keep_x = [0.75]*n_dims
        n_components_pc = [150, 200, 250, 300, 350]
        kda_kernel = ['rbf','poly','poly','poly','poly']
        kda_degree = [None, 1, 2, 3, 4]
        rf_max_leaves = [5, 10, 15, 20, 25]
        rf_n_estimates = [20, 30, 40, 50, 75]
        elastic_net_fraction = [0.5]*n_mods
        regularization_fraction = [0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        n_mods = 1
        n_dims = 1
        diffs = range(-2,3)
        best_rep_rm_y = [mids.br_rm_y] * len(diffs)
        best_rep_keep_x = [mids.br_keep_x] * len(diffs)
        n_components_pc = [ mids.n_pc]* len(diffs)
        kda_kernel = [mids.kda_kernel] * len(diffs)
        kda_degree = [mids.kda_dg] * len(diffs)
        rf_max_leaves = [df + mids.rf_max_leaf for df in diffs]
        rf_n_estimates = [df*2 + mids.rf_n_est for df in diffs]
        elastic_net_fraction = [mids.lr_l1] * len(diffs)
        regularization_fraction = [df*0.02 + mids.lr_reg for df in diffs]

    return pd.DataFrame({
        'br_rm_y': [x for x in best_rep_rm_y for _ in range(n_mods)],
        'br_keep_x': [x for x in best_rep_keep_x for _ in range(n_mods)],
        'n_pc': [x for x in n_components_pc for _ in range(n_mods)],
        'kda_kernel':  kda_kernel * n_dims,
        'kda_dg': kda_degree * n_dims,
        'rf_max_leaf': rf_max_leaves * n_dims,
        'rf_n_est': rf_n_estimates * n_dims,
        'lr_l1': elastic_net_fraction * n_dims,
        'lr_reg': regularization_fraction * n_dims})


def eval_one_model(tr_x, tr_y, new_x, new_y, clf, info_pars, score_type):
    """
    balanced accuracy score on 1 classifier clf
    """
    try:
        clf.fit(tr_x, tr_y)
        y_hat = clf.predict(new_x)
        return round(score_type(new_y, y_hat), 4)
    except ArpackError as err:
        print(err, info_pars)
        return 0


def eval_one_dataset(tr_x, tr_y, val_x, val_y, hp_i, score_type, seed):
    """
    balanced accuracy score on 3 classifiers (kernel discriminant analysis/random forest/logistic regression)
    hp_i = hyperparameters of the model
    tr_x, val_x = training set, validation set (after dimension reduction)
    """
    kdc, rfc, lrc = [get_classifier(hp_i, mdl, seed) for mdl in ['kd', 'rf', 'lr']]
    return [eval_one_model(tr_x, tr_y, val_x, val_y, clf, hp_i, score_type) for clf in [kdc, rfc, lrc]]


def get_classifier(hp_i, mdl_type, seed):
    if mdl_type == "kd" and hp_i.kda_kernel == "poly":
        clf = Kfda(kernel=hp_i.kda_kernel, degree=int(hp_i.kda_dg))
    elif mdl_type == "kd":
        clf = Kfda(kernel=hp_i.kda_kernel)
    elif mdl_type == "rf":
        clf = RandomForestClassifier(n_estimators=int(hp_i.rf_n_est), random_state=seed,
                                 max_leaf_nodes=int(hp_i.rf_max_leaf), n_jobs=-1)
    elif mdl_type == "lr":
        clf = LogisticRegression(penalty='elasticnet', random_state=seed, solver='saga',
                             l1_ratio=hp_i.lr_l1, C=hp_i.lr_reg, max_iter=200, n_jobs=-1)
    else:
        clf = None
    return clf


def get_dim_reduced(tr0_x, tr0_y, val0_x, hp_i, dr_type):
    if dr_type == "br":
        tf1 = best_representatives(tr0_x, tr0_y, hp_i.br_keep_x, 'pearson', hp_i.br_rm_y)
        dr_train_x = tr0_x.filter(tf1.index)
        dr_val_x = val0_x.filter(tf1.index)
    elif dr_type == "pc":
        dr_train_x, dr_val_x, dr_object = pca(tr0_x, val0_x, n_pc=int(hp_i.n_pc))
    elif dr_type == "kpc":
        dr_train_x, dr_val_x, dr_object = kernel_pca(tr0_x, val0_x, n_pc=int(hp_i.n_pc))
    else:
        dr_train_x, dr_val_x = tr0_x, val0_x
    return dr_train_x, dr_val_x
