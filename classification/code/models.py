import pandas as pd
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from kfda import Kfda
from Project_classification.code.dimensionreduction import best_representatives, pca, kernel_pca


def get_hyper_parameters(run="first", mids=None):
    """
    retrieves the hyperparameters in the first run (calculate models)
    each row contains one set of hyperparameters for 9 models (dimension reduction x classifier)
    the first run represents 25 such sets of hyperparameters
    this format is only use in the first run
    """
    if run == 'first':
        n_mods = 5
        n_dims = 5
        # dimension reduction
        # best representatives
        best_rep_rm_y = [ 0.1, 0.15, 0.2, .25, 0.3]
        best_rep_keep_x = [0.75]*n_dims
        # PCA and Kernel PCA (with kernel rbf)
        n_components_pc = [150, 200, 250, 300, 350]
        # classifier
        # Kernel discriminant analysis
        kda_kernel = ['rbf','poly','poly','poly','poly']
        kda_degree = [None, 1, 2, 3, 4]
        # Random forest
        rf_max_leaves = [5, 10, 15, 20, 25]
        rf_n_estimates = [20, 30, 40, 50, 75]
        # Logistic regression: elastic net
        elastic_net_fraction = [0.5]*n_mods
        regularization_fraction = [0.2, 0.3, 0.4, 0.5, 0.6]

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


def get_hp_baseline():
    """
    retrieves the hyperparameters for the baseline models (compare_baseline)
    same hyperparameters as in the first run, but this time no dimension reduction is calculated
    longer format than get_hyper_parameters: one model per row, 15 rows in total,
    extra column model
    """
    n_par = 5
    n_mod = 3
    kda_kernel = ['rbf', 'poly', 'poly', 'poly', 'poly']
    kda_degree = [None, 1, 2, 3, 4]
    rf_max_leaves = [5, 10, 15, 20, 25]
    rf_n_estimates = [20, 30, 40, 50, 75]
    elastic_net_fraction = [0.5] * n_par
    regularization_fraction = [0.2, 0.3, 0.4, 0.5, 0.6]
    return pd.DataFrame({
            'model': ['bl_' + mod for mod in ['kd','rf','lr'] for _ in range(n_par)],
            'kda_kernel': kda_kernel * n_mod,
            'kda_dg': kda_degree * n_mod,
            'rf_max_leaf': rf_max_leaves * n_mod,
            'rf_n_est': rf_n_estimates * n_mod,
            'lr_l1': elastic_net_fraction * n_mod,
            'lr_reg': regularization_fraction * n_mod})


def eval_one_model(tr_x, tr_y, new_x, new_y, clf, info_pars, score_type):
    """
    fits a classifier on a training set (after dimension reduction),
    predicts the classes of the validation set and
    returns one validation metric (eg balanced accuracy) for the model
    """
    try:
        clf.fit(tr_x, tr_y)
        y_hat = clf.predict(new_x)
        return round(score_type(new_y, y_hat), 4)
    # kernel discriminant analysis sometimes goes wrong -> error handling
    except ArpackError as err:
        print(err, info_pars)
        return 0


def eval_one_dataset(tr_x, tr_y, val_x, val_y, hp_i, score_type, seed):
    """
    fits 3 classifiers on a training set (after dimension reduction),
    predicts the classes of the validation set per classifier and
    returns one validation metric (eg balanced accuracy) for each model

    hp_i = hyperparameters of the model
    tr_x, val_x = training set, validation set after dimension reduction
    score_type = performance measure (balanced accuracy score)
    classifiers are kernel discriminant analysis/random forest/logistic regression
    returns one list with 3 elements
    """
    kdc, rfc, lrc = [get_classifier(hp_i, mdl, seed) for mdl in ['kd', 'rf', 'lr']]
    return [eval_one_model(tr_x, tr_y, val_x, val_y, clf, hp_i, score_type) for clf in [kdc, rfc, lrc]]


def get_classifier(hp_i, mdl_type, seed):
    """
    takes a set of hyper parameters and a type of classifier
    returns an instance of a classifier of that type with the relevant hyper parameters
    the model is not fitted on any data
    """
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
    """
    takes a set of hyper parameters and a type of dimension reduction and data (train and val)
    fits on train data, transforms both train and validation data
    returns the train and validation data for the features after dimension reduction
    """
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


