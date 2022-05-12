import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA


def ctg_cors_with_y(data_x, y):
    """
    approximation for degree of correlation: cov(x,y)/ sd(x) instead of cov(x,y)/[sd(x)), sd(y)]
    fit x ~ y and return fit_ssq(x)/total_ssq(x)
    """
    overall_means = data_x.mean()
    total_ssq = ((data_x - overall_means)**2).sum()
    means_per_y_value = data_x.groupby(y).mean()
    counts_per_y_value = y.value_counts()
    fitted_ssq = ((means_per_y_value - overall_means)**2).apply(lambda col: col * counts_per_y_value).sum()
    return fitted_ssq/total_ssq




def best_representatives(train_data_x, train_data_y, min_cor, method='pearson', rm_small=0):
    """
    dimension reduction method 1
    for each variable xi consider similar variables in train_data_x (corr > min_cor)
    the variable in this pool with the highest correlation with y is the chosen representative for xi
    min_cor also counts as a hyper parameter
    """
    # remove variables x with low correlations with y
    all_cor_y = ctg_cors_with_y(train_data_x, train_data_y).abs()
    mid_cor_y = all_cor_y.where(all_cor_y > rm_small).dropna().sort_values(ascending=False)
    # correlation matrix x, row k contains all x variables with better (x,y)-correlation than x_k
    cor_mat = train_data_x.filter(mid_cor_y.index).corr(method).abs()
    triangle = pd.DataFrame(np.tril(cor_mat.to_numpy(),-1))
    # if an element on row k has correlation > min_cor, that element is a better representative than elt k
    check_cor_x = triangle.max(axis=1)
    check_cor_x.index = mid_cor_y.index
    return mid_cor_y.where(check_cor_x < min_cor).dropna()


def pca(train_data_x, val_data_x, n_pc):
    """
    unsupervised dimension reduction: linear
    n_pc as hyperparameter
    fit on train_data_x, apply on train and validation set
    """
    pca_obj = PCA(n_components=n_pc)
    pca_obj.fit(train_data_x)
    tf_train = pca_obj.transform(train_data_x)
    tf_val = pca_obj.transform(val_data_x)
    return tf_train, tf_val, pca_obj


def kernel_pca(train_data_x, val_data_x, n_pc, kernel='rbf'):
    """
    unsupervised non-linear reduction: linear
    n_pc as hyperparameter
    fit on train_data_x, apply on train and validation set
    """
    pca_obj = KernelPCA(n_components=n_pc, kernel=kernel)
    pca_obj.fit(train_data_x)
    tf_train = pca_obj.transform(train_data_x)
    tf_val = pca_obj.transform(val_data_x)
    return tf_train, tf_val, pca_obj