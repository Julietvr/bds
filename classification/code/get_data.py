"""
module for reading data and for returning the right folds
"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split


def read_data(filename):
    """
    reads filename into panda dataframe
    gives names to columns, last column is response variable y
    """
    new = pd.read_csv(filename, header=None)
    nr_y_col = new.shape[1] - 1
    new.columns = ['x' + str(nr) if nr != nr_y_col else 'y' for nr in new.columns]
    return new


def get_fold(dataset, kind, fold_nr=0, seed=42):
    """
    returns part of the data you need for the analysis
    kind =  training tuning test or toy
    Seed is fixed, stratified sampling,
    test = 20%, tr_tu = 80% stratified train test split
    tr_tu is split up in 10 folds Stratified K-fold
    returns tuple (df_x, df_y)
    """
    # fixed choices
    p_test = 0.2
    folds = 10
    var_y = 'y'
    # split test
    tr_tu, test = train_test_split(dataset,test_size=p_test, random_state=seed, stratify=dataset[var_y])
    if kind == 'test':
        df_y = test[var_y]
        df_x = test.drop(var_y, axis=1)
        return df_x, df_y
    # split tr_tu in folds
    kfs = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    split = list(kfs.split(tr_tu, tr_tu[var_y]))
    if kind == 'training':
        idx = split[fold_nr][0]
    elif kind == 'tuning':
        idx = split[fold_nr][1]
    else: #toydata
        idx = split[0][0][:500]
    df_y = tr_tu[var_y].iloc[idx]
    df_x = tr_tu.drop([var_y], axis=1).iloc[idx]
    return df_x, df_y