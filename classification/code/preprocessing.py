import pandas as pd
from Project_classification.code.exploration import remove_outliers, adjust_outliers, info_fold
from Project_classification.code.get_data import get_fold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox


# impute values with regression: args ok : random_state needed!
#
def make_flags(cc_data, cc_values, cc_missing, rm_duplicates=True, cols_out=None):
    cols_cc = cols_out if cols_out is not None else cc_data.columns
    cc_dct = {}
    for nm, cc in cc_values:
        cc_name = nm + '_' + str(cc)
        if nm in cols_cc or cc_name in cols_cc:
            cc_dct[cc_name] = (cc_data[nm] == cc)
    for nm in cc_missing:
        cc_name = nm + '_NA'
        if nm in cols_cc or cc_name in cols_cc:
            cc_dct[cc_name] = cc_data[nm].isna()
    cc_per_col = pd.DataFrame(cc_dct, index=cc_data.index)
    if rm_duplicates:
        cc_per_col = cc_per_col.T.drop_duplicates().T
    return cc_per_col


def scale_partial(tr, val, nms):
    small_tr = tr.filter(nms)
    small_val = val.filter(nms)
    not_scaled_tr = tr.drop(nms, axis=1)
    not_scaled_val = val.drop(nms, axis=1)
    scale_obj = StandardScaler()
    scale_obj.fit(small_tr)
    sc_tr = pd.DataFrame(scale_obj.transform(small_tr), columns=nms, index=small_tr.index)
    sc_val = pd.DataFrame(scale_obj.transform(small_val), columns=nms, index = small_val.index)
    scaled_tr_x = sc_tr.join(not_scaled_tr)
    scaled_val_x = sc_val.join(not_scaled_val)
    return scaled_tr_x, scaled_val_x, scale_obj


def transform_data(tr_data_x, tr_data_y, val_data_x, info_idx, info_explore, params):
    # remove columns and rows: missing, constant, outliers (rows only for train_x)
    idx_rm_col = list(info_idx['idx_cst_c']) + list(info_idx['idx_na_c'])
    idx_rm_row = list(info_idx['idx_na_r']) + list(info_idx['idx_out_r'])
    tf_train_x, _, _ = remove_outliers(tr_data_x, info_explore['df_lim'], idx_rm_col)
    tf_train_y = tr_data_y.filter(tf_train_x.index)
    tf_val_x, _, idx_replaced_val = adjust_outliers(val_data_x, info_explore['df_lim'], idx_rm_col)
    tf_train_x = tf_train_x.drop(idx_rm_row)
    tf_train_y = tf_train_y.drop(idx_rm_row)
    col_names = tf_train_x.columns


    # concentrations for continuous variables, calculate flags
    # selected on nr of distinct values (instead of data type)
    # zero values can be hidden missing: let them be for now
    nm_floats = [nm for nm in col_names if len(tf_train_x[nm].value_counts()) > params['max_distinct_n']]
    cc_miss = info_explore['mi_c'].where(info_explore['mi_c'] > params['max_cc_n']).dropna().index
    cc_miss = cc_miss.difference(info_idx['idx_na_c'])
    cc_vls = info_explore['dct_cc']

    flags_cc_tr = make_flags(tf_train_x, cc_vls, cc_miss, rm_duplicates=True, cols_out=nm_floats)
    flags_cc_val = make_flags(tf_val_x, cc_vls, cc_miss, rm_duplicates=False, cols_out=flags_cc_tr.columns)


    # imputation for na-values - simple imputation
    # iterative imputes take too long: let's keep it basic for now
    tf_train_cnt = tf_train_x.filter(nm_floats)
    tf_train_ctg = tf_train_x.drop(nm_floats, axis=1)
    tf_val_cnt = tf_val_x.filter(nm_floats)
    tf_val_ctg = tf_val_x.drop(nm_floats, axis=1)
    imp_mean = SimpleImputer(strategy='mean')
    imp_mode = SimpleImputer(strategy='most_frequent')
    imp_mean.fit(tf_train_cnt)
    imp_mode.fit(tf_train_ctg)

    tf_train_cnt = pd.DataFrame(imp_mean.transform(tf_train_cnt), columns=nm_floats, index=tf_train_cnt.index)
    tf_train_ctg = pd.DataFrame(imp_mode.transform(tf_train_ctg), columns=tf_train_ctg.columns, index=tf_train_ctg.index)
    tf_val_cnt = pd.DataFrame(imp_mean.transform(tf_val_cnt), columns=nm_floats, index=tf_val_cnt.index)
    tf_val_ctg = pd.DataFrame(imp_mode.transform(tf_val_ctg), columns=tf_val_ctg.columns, index=tf_val_ctg.index)
    tf_train_x = tf_train_cnt.join(tf_train_ctg)
    tf_val_x = tf_val_cnt.join(tf_val_ctg)
    # transformations: scaling and centering for continuous variables
    tf_train_x, tf_val_x, scale_cnt = scale_partial(tf_train_x, tf_val_x, nm_floats)

    # make data more normal if abs(skewness) > 2 -  needs truncation of validation set (df_lim)
    # apply on x**2 + epsilon ok for first two dataset, overflow warnings in third, error in 3,2
    skewness = tf_train_x.skew(axis=0, skipna=True)
    skewed = skewness.where(abs(skewness) > params['trigger_skewness']).dropna()
    for nm in skewed.index.intersection(nm_floats):
        xtf, l_opt = boxcox(tf_train_x[nm] ** 2 + params['epsilon'])
        tf_train_x[nm] = xtf
        tf_val_x[nm] = boxcox(tf_val_x[nm] ** 2 + params['epsilon'], lmbda=l_opt)
    tf_train_x, tf_val_x, scale_obj_bc = scale_partial(tf_train_x, tf_val_x, skewed.index)

    # add the flags from na and concentration
    tf_train_x = tf_train_x.join(flags_cc_tr)
    tf_val_x = tf_val_x.join(flags_cc_val)
    return tf_train_x, tf_val_x, tf_train_y, idx_replaced_val


def prepare_the_data(complete_dataset, nr_fold, seed):
    # get data
    train_x, train_y = get_fold(dataset=complete_dataset, kind='training', fold_nr=nr_fold)
    val_x, val_y = get_fold(dataset=complete_dataset, kind='tuning', fold_nr=nr_fold)
    # fixed choices
    rows_n = train_x.shape[0]
    max_cc_pct = 0.05
    nm_choices = ['max_rm_missing_pct', 'max_cc_pct', 'max_out_sd', 'nr_pc_out', 'rows_n',
                  'max_distinct_n', 'max_cc_n', 'trigger_skewness', 'epsilon']
    val_choices = [0.5, 0.05, 10, 5, rows_n, rows_n / 5, rows_n * max_cc_pct, 2, 10 ** -8]
    choices = dict(zip(nm_choices, val_choices))
    # info exploration
    msg_info, graph_info, bounds, idx_info, cl_trx, sc_obj = \
        info_fold(train_x, train_y,
                  pct_mi=choices['max_rm_missing_pct'], pct_cc=choices['max_cc_pct'],
                  dist_out=choices['max_out_sd'], n_pc=choices['nr_pc_out'])
    # preprocessing
    clean_train_x, clean_val_x, clean_train_y, idx_out_val = \
        transform_data(train_x, train_y, val_x, idx_info, graph_info, choices)
    # first remove the painfully concentrated variables
    modes = clean_train_x.mode(axis=0)
    concentration = clean_train_x.eq(modes.loc[0,]).sum() / clean_train_x.shape[0]
    idx_painfully_concentrated = concentration.where(concentration > 0.99).dropna().index
    clean_train_x.drop(idx_painfully_concentrated, axis=1)
    return clean_train_x, clean_train_y, clean_val_x, val_y
