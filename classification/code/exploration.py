import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import io
from scipy.stats import norm


def rm_ticks(loc, rm_x=True, rm_y=True):
    """" remove ticks from graph """
    remove_x = loc.get_xticklabels() if rm_x else []
    remove_y = loc.get_yticklabels() if rm_y else []
    for tl in remove_x + remove_y:
        tl.set_visible(False)


def get_info_general(data_x):
    """ shape, missing values, data types"""
    buffer = io.StringIO()
    data_x.info(verbose=False, buf=buffer)
    short_info = buffer.getvalue()
    return short_info


def draw_info_general(text, nr_fold, ax_info):
    """ shape, missing values, data types"""
    ax_info.annotate(text, (0.1, 0.5), xycoords='axes fraction', va='center', fontsize='small')
    ax_info.set_title('General info for training fold ' + str(nr_fold), fontsize=8)
    rm_ticks(ax_info)


def get_info_variable(var, lim_c):
    # info
    all_counts = var.value_counts(dropna=False)
    row_nas = var.where(var.isna()).dropna().index
    concentrations_var = dict(all_counts.dropna().where(all_counts > lim_c).dropna())
    return all_counts, row_nas, concentrations_var


def make_msg(nrs, suffix):
    return ', '.join('' if nr == 0 else str(nr) + ' ' + txt for nr, txt in zip(nrs, suffix))


def draw_counts_y(y_counts, ax_counts, msg_counts):
    ax_counts.bar(x=y_counts.index, height=y_counts)
    ax_counts.set_title('Histogram class', fontsize=8)
    if len(msg_counts) > 0:
        ax_counts.annotate(msg_counts, (0.5, 0.9), xycoords='axes fraction', va='center', fontsize=8)


def get_info_missing(data_x, lim_m):
    """ missing values per column """
    count_na_per_col = data_x.isna().sum()
    count_na_per_row = data_x.isna().sum(axis=1)
    na_col_rm = count_na_per_col.where(count_na_per_col > lim_m[0]).dropna().index
    na_row_rm = count_na_per_row.where(count_na_per_row > lim_m[1]).dropna().index
    cst_col_rm = data_x.std().where(data_x.std() == 0.0).dropna().index
    return count_na_per_row, count_na_per_col, na_col_rm, cst_col_rm, na_row_rm


def draw_info_missing(counts_na, n_all, lim_m, type_rc, ax_m, msg_mi):
    info = pd.DataFrame([[0, 'per row', 'Remove rows: '],
                         [1, 'per column', 'Remove columns: ']],
                        columns=['idx', 'per', 'rm'], index=['r', 'c'])
    ax_m.set_title("missing: histogram " + info.per[type_rc], fontsize=8)
    rm_ticks(ax_m)
    if sum(counts_na) == 0:
        ax_m.annotate('No values missing', (0.2, 0.5), xycoords='axes fraction', va='center', fontsize=8)
    else:
        lim_line = lim_m
        ax_m.hist(counts_na.where(counts_na > 0))
        ax_m.set_xlim(0, n_all)
        ax_m.vlines(x=lim_line, ymin=0, ymax=ax_m.get_ylim()[1], colors='red')
        if len(msg_mi) > 0:
            ax_m.annotate(info.rm[type_rc] + msg_mi, (0.25, 0.9), xycoords='axes fraction', va='center', fontsize=8)


def count_concentrations(var, threshold):
    tab = var.value_counts()
    return tab.where(tab > threshold).dropna()


def get_concentrations(data_x, lim_c, excl_int=False):
    nr_concentrated = pd.Series([sum(count_concentrations(data_x[nm], lim_c)) for nm in data_x.columns])
    nr_concentrated.index = data_x.columns
    data_types = data_x.dtypes
    cols = data_types.where(data_types != 'int64').dropna().index if excl_int else data_types.index
    all_cc = dict([(nm, count_concentrations(data_x[nm], lim_c)) for nm in cols])
    return nr_concentrated, all_cc


def draw_quality(n_all, nr_concentrated, nr_missing, ax_c, msg_quality):
    other = n_all - nr_missing - nr_concentrated
    rm_ticks(ax_c)
    ax_c.stackplot(range(len(nr_missing)), nr_concentrated, nr_missing, other, colors='brg', alpha=0.7)
    ax_c.set_title('per column: concentrations (blue), missing values (red) and other (green)', fontsize=8)
    if len(msg_quality) > 0:
        ax_c.annotate(msg_quality, (0.2, 0.9), xycoords='axes fraction', va='center', fontsize=8)


def get_outlier_lim(data_x, threshold=10):
    data_std = data_x.std()
    upper = data_x.mean() + threshold * data_std
    lower = data_x.mean() - threshold * data_std
    df_out = pd.DataFrame({'lower': lower, 'upper': upper})
    return df_out


def remove_outliers(data_x, df_ll_ul, idx_cols_rm):
    # dataset without cst columns, with NA values present
    # count number of outliers per row, leave na values na
    data_var = data_x.drop(idx_cols_rm, axis=1)
    data_var, df_lim = data_var.align(df_ll_ul.T, join='inner', axis=1, copy=False)
    df_too_low = data_var.lt(df_lim.loc['lower'])
    df_too_high = data_var.gt(df_lim.loc['upper'])
    ct_out_per_row = (df_too_low | df_too_high).sum(axis=1)
    ct_out_per_col = (df_too_low | df_too_high).sum(axis=0)
    idx_ok = ct_out_per_row.where(ct_out_per_row == 0).dropna().index
    truncated = data_var.loc[idx_ok]
    ct_rows_rm = data_var.shape[0] - truncated.shape[0]
    df_out = df_ll_ul.assign(count_outlier=ct_out_per_col)
    return truncated, df_out, ct_rows_rm


def adjust_outliers(data_x, df_ll_ul, idx_cols_rm):
    # dataset without cst columns, with NA values present
    # count number of outliers per row, leave na values na
    data_var = data_x.drop(idx_cols_rm, axis=1)
    data_var, df_lim = data_var.align(df_ll_ul.T, join='inner', axis=1, copy=False)
    df_too_low = data_var.lt(df_lim.loc['lower'])
    df_too_high = data_var.gt(df_lim.loc['upper'])
    ct_out_per_col = (df_too_low | df_too_high).sum(axis=0)
    idx_replace = ct_out_per_col.where(ct_out_per_col > 0).dropna().index
    for col in idx_replace:
        lower = df_lim.loc['lower',col]
        upper = df_lim.loc['upper',col]
        data_var[col] = [min(max(val,lower),upper) for val in data_var[col]]
    df_out = df_ll_ul.assign(count_outlier=ct_out_per_col)
    return data_var, df_out, idx_replace


def draw_outliers_var(df_out, ax_ov, msg_out):
    ax_ov.plot(df_out.count_outlier)
    ax_ov.set_ylim(0, round(max(df_out.count_outlier)*1.2))
    if len(msg_out) > 0:
        ax_ov.annotate(msg_out, (0.4, 0.9), xycoords='axes fraction', va='center', fontsize=8)
    rm_ticks(ax_ov, rm_y=False)


def get_outliers_pca(data_x, n_pc=5, threshold=10):
    # no cst cols, no univariate outliers
    # try first with simple impute to track outliers
    if data_x.shape[0] == 0:
        return [], []
    s_imp = SimpleImputer(strategy='mean')
    fill_x = pd.DataFrame(s_imp.fit_transform(data_x))
    scale_obj = StandardScaler()
    pca_obj = PCA(n_components=n_pc)
    scale_obj.fit(fill_x)
    scaled_x = scale_obj.transform(fill_x)
    scores_x = pca_obj.fit_transform(X=scaled_x)
    distances = np.sqrt((pd.DataFrame(scores_x/scores_x.std())**2).sum(axis=1))
    locs_out = distances.where(distances > threshold).dropna().index
    idx_out2 = data_x.iloc[locs_out].index
    return distances, idx_out2


def draw_outliers_pca(dist_pca, ax_opc, msg_pca):
    ax_opc.plot(dist_pca)
    ax_opc.set_title('Outliers PCA: distances', fontsize=8)
    if len(msg_pca) > 0:
        ax_opc.annotate(msg_pca, (0.4, 0.9), xycoords='axes fraction', va='center', fontsize=8)
    rm_ticks(ax_opc, rm_y=False)
    return msg_pca + '\n'


def get_scaled_qq(data_x):
    range_sigma = range(-2, 3)
    proportions = norm.cdf(range_sigma)
    sc_obj = StandardScaler()
    sc_data_f = sc_obj.fit_transform(data_x)
    sc_qq = pd.DataFrame(sc_data_f).quantile(proportions)
    return sc_obj, sc_qq, range_sigma


def draw_scaled_qq(quantiles, range_sigma, ax_qq, msg_qq):
    qq_t = quantiles.T
    qq_t.columns = ['sigma_' + str(x) for x in range_sigma]
    ax_qq.plot(qq_t)
    for sigma in range_sigma:
        ax_qq.hlines(xmin=0, xmax=qq_t.shape[0], y=sigma, colors='grey')
    ax_qq.set_title('Quantiles of the standardized columns', fontsize=8)
    if len(msg_qq) > 0:
        ax_qq.annotate(msg_qq, (0.4, 0.9), xycoords='axes fraction', va='center', fontsize=8)


def draw_skewness(skewnesses, n_all, limits_sk, ax_sk, msg_sk):
    ax_sk.plot(skewnesses)
    for lim in limits_sk:
        ax_sk.hlines(xmin=0, xmax=n_all, y=lim, colors='grey')
    ax_sk.set_title('Skewness of the standardized columns', fontsize=8)
    if len(msg_sk) > 0:
        ax_sk.annotate(msg_sk, (0.4, 0.9), xycoords='axes fraction', va='center', fontsize=8)


def info_messages(na_y, na_rm_c, cst_rm_c, na_rm_r, ct_dup, ct_flags_cc, ct_r_rm_out, idx_out_pca):
    cts = [len(na_y), len(na_rm_c), len(cst_rm_c), len(na_rm_r), ct_dup, ct_flags_cc, ct_r_rm_out, len(idx_out_pca)]
    sfx = ['NA', 'NA', 'constant', 'NA', 'duplicated', 'concentrations', 'outlier rows removed', 'outliers PCA']
    nm_g = ['01_mi_y', '10_mi_c', '10_mi_c', '11_mi_r', '2_cc', '2_cc', '30_out_var', '31_out_pc']
    msg_0 = ['' if ct == 0 else str(ct) + ' ' + sf for ct, sf in zip(cts, sfx)]
    return pd.Series({nm0: ', '.join([mg for mg, nm in zip(msg_0, nm_g) if nm == nm0 and mg != ''])
                      for nm0 in set(nm_g)})


def info_fold(data_x, data_y,pct_mi=0.5, pct_cc=0.05, dist_out=10, n_pc=5):
    n_row, n_col = data_x.shape
    max_missing = (pct_mi * n_row, pct_mi * n_col)
    max_concentration = pct_cc * n_row
    info_general = get_info_general(data_x)
    ct_y, na_y, cst_y = get_info_variable(data_y, n_row)
    ct_na_row, ct_na_col, na_rm_c, cst_rm_c, na_rm_r = get_info_missing(data_x, max_missing)
    ct_dup = data_x.duplicated().sum()
    ct_cc_col, dct_cc = get_concentrations(data_x, max_concentration)
    ct_flags_cc = sum([len(val) for key, val in dct_cc.items()])
    df_lim = get_outlier_lim(data_x, threshold=dist_out)
    rm_c = list(na_rm_c) + list(cst_rm_c)
    cl_data_x, df_outliers, ct_r_rm_out = remove_outliers(data_x, df_lim, rm_c)
    distances_pca, idx_out_pca = get_outliers_pca(cl_data_x, n_pc=n_pc, threshold=dist_out)
    cl_data_x = cl_data_x.drop(idx_out_pca)
    scale_x_obj, df_qq, range_qq = get_scaled_qq(cl_data_x)
    cl_data_x = pd.DataFrame(scale_x_obj.transform(cl_data_x))
    skewness = cl_data_x.skew(axis=0, skipna=True)
    info_msg = info_messages(na_y, na_rm_c, cst_rm_c, na_rm_r, ct_dup, ct_flags_cc, ct_r_rm_out,  idx_out_pca)
    info_idx_rm = {'idx_na_y': na_y,
                   'idx_na_c': na_rm_c,
                   'idx_na_r': na_rm_r,
                   'idx_cst_c': cst_rm_c,
                   'idx_out_r': idx_out_pca}
    info_graph ={'gen': info_general,
                 'y': ct_y,
                 'mi_c': ct_na_col, 'mi_r': ct_na_row,
                 'cc_c': ct_cc_col, 'dct_cc': dct_cc,
                 'df_lim': df_outliers, 'dist_pc': distances_pca,
                 'df_qq': df_qq,
                 'skew': skewness}
    info_bound = {'r': n_row, 'c': n_col, 'mi_r': max_missing[1], 'mi_c': max_missing[0],
                  'cc': max_concentration, 'qq': range_qq, 'lim_sk': (-0.5,0.5)}
    return info_msg, info_graph, info_bound, info_idx_rm, cl_data_x, scale_x_obj


def draw_report(loc, i_graph, i_msg, i_bnd, nr_fold):
    fig = plt.figure(0, figsize=(12.0, 9.0))
    fig.suptitle('Exploration: ' + loc)
    ax_00 = plt.subplot2grid((5, 2), (0, 0))
    draw_info_general(i_graph['gen'], nr_fold, ax_00)
    ax_01 = plt.subplot2grid((5, 2), (0, 1))
    draw_counts_y(i_graph['y'], ax_01, i_msg['01_mi_y'])
    ax_10 = plt.subplot2grid((5, 2), (1, 0))
    draw_info_missing(i_graph['mi_c'], i_bnd['r'], i_bnd['mi_c'], 'c', ax_10, i_msg['10_mi_c'])
    ax_11 = plt.subplot2grid((5, 2), (1, 1))
    draw_info_missing(i_graph['mi_r'], i_bnd['c'], i_bnd['mi_r'], 'r', ax_11, i_msg['11_mi_r'])
    ax_2 = plt.subplot2grid((5, 2), (2, 0), colspan=2)
    draw_quality(i_bnd['r'], i_graph['cc_c'], i_graph['mi_c'], ax_2, i_msg['2_cc'])
    ax_30 = plt.subplot2grid((5, 2), (3, 0))
    draw_outliers_var(i_graph['df_lim'], ax_30, i_msg['30_out_var'])
    ax_31 = plt.subplot2grid((5, 2), (3, 1))
    draw_outliers_pca(i_graph['dist_pc'], ax_31, i_msg['31_out_pc'])
    ax_40 = plt.subplot2grid((5, 2), (4, 0), colspan=1)
    draw_scaled_qq(i_graph['df_qq'], i_bnd['qq'], ax_40, '')
    ax_41 = plt.subplot2grid((5, 2), (4, 1), colspan=1)
    draw_skewness(i_graph['skew'], i_bnd['c'], i_bnd['lim_sk'], ax_41, '')
    plt.savefig('../exploration/graph_' + loc + '.pdf')
    f_msg = open('../exploration/messages_' + loc + '.txt', "w")
    f_msg.writelines([msg0 + '\n' for msg0 in i_msg])
    f_msg.close()




