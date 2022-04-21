import os as os
os.chdir(os.getcwd() + '/Project_classification/code')

#import os as os
import pandas as pd
from Project_classification.code.get_data import read_data, get_fold
from Project_classification.code.exploration import info_fold, remove_outliers
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox




def make_flags(cc_data, cc_values, cc_missing):
    cc_dct = {}
    for nm, cc in cc_values:
        cc_name = nm + '_' + str(cc)
        cc_dct[cc_name] = (cc_data[nm] == cc)
    for nm in cc_missing:
        cc_name = nm + '_NA'
        cc_dct[cc_name] = cc_data[nm].isna()

    cc_per_col = pd.DataFrame(cc_dct, index=cc_data.index)
    cc_per_col = cc_per_col.T.drop_duplicates().T
    return cc_per_col


# boxcox transformation, requires positive data


# main code
seed = 42
nm_files = sorted(os.listdir('../data'))
for nr_file in range(1):
    data_name = nm_files[nr_file].split('.')[0]
    data = read_data('../data/' + nm_files[nr_file])
    for this_fold in range(1):
        train_x, train_y = get_fold(dataset=data, kind='training', fold_nr=this_fold)
        val_x, val_y = get_fold(dataset=data, kind='validation', fold_nr=this_fold)
        msg_info, graph_info, bounds, idx_info, cl_trx, sc_obj = \
            info_fold(train_x, train_y, pct_mi=0.5, pct_cc=0.05, dist_out=10, n_pc=5)
        max_skewness = 2
        # remove columns and rows: missing, constant, outliers (rows only for train_x)
        idx_rm_col = list(idx_info['idx_cst_c']) + list(idx_info['idx_na_c'])
        idx_rm_row = list(idx_info['idx_na_y']) + list(idx_info['idx_na_r'])
        tf_train_x, _, _ = remove_outliers(train_x, graph_info['df_lim'], idx_rm_col)
        tf_train_x = tf_train_x.drop(idx_info['idx_na_r'].drop(idx_info['idx_out_r']))
        tf_val_x = val_x.drop(idx_rm_col, axis=1)
        keep_col_names = tf_train_x.columns
        # first selected on dtype, afterwards on nr of distinct values
        nm_floats = [nm for nm in tf_train_x.columns if len(tf_train_x[nm].value_counts()) > tf_train_x.shape[0] / 5]
        cc_miss = graph_info['mi_c'].where(graph_info['mi_c'] > 0.05).dropna().index
        cc_vls = [(ky, val) for ky, vls in graph_info['dct_cc'].items()
                  for val in vls.index if len(vls) > 0 and ky in nm_floats]
        flags_cc_tr = make_flags(tf_train_x, cc_vls, cc_miss)
        flags_cc_val = make_flags(tf_train_x, cc_vls, cc_miss)
        tf_train_x = tf_train_x.merge(flags_cc_tr)
        tf_val_x = tf_val_x.merge(flags_cc_val)
        # transformations: imputation - iterative regression ascending (both sets)
        imp_mean = SimpleImputer()
        imp_mean.fit(tf_train_x)
        tf_train_x = pd.DataFrame(imp_mean.transform(tf_train_x), columns=keep_col_names)
        tf_val_x = pd.DataFrame(imp_mean.transform(tf_val_x),columns=keep_col_names)
        # transformations: extra variable for concentrations -  before scaling/normalizing
        # first trainingset indicates concentrations are often (always) 0.0 - suggesting hidden na-values



        # transformations: scaling and centering
        scale_obj = StandardScaler()
        scale_obj.fit(tf_train_x)
        tf_train_x = pd.DataFrame(scale_obj.transform(tf_train_x),columns=keep_col_names)
        tf_val_x = pd.DataFrame(scale_obj.transform(tf_val_x),columns=keep_col_names)

        # make data more normal if abs(skewness) > 0.5
        skewed = train_x.where(abs(train_x.skew(axis=0, skipna=True) > max_skewness)).dropna().index
        tf_list = []
        for nm in skewed:
            xtf, l_opt = boxcox(tf_train_x)
            tf_train_x[nm] = xtf
            tf_val_x[nm] = boxcox(tf_val_x[nm], lmbda=l_opt)
        # dimreductiom is in next step
        tf_train_x.to_csv('../preprocessing/test_train00.csv')
        tf_val_x.to_csv('../preprocessing/test_val00.csv')
        tf_train_x.to_csv('../preprocessing/test_train00.csv')
        tf_val_x.to_csv('../preprocessing/test_val00.csv')
        print(tf_train_x.shape)
