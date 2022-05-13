import pandas as pd


def read_results(nr_file, nr_fold):
    loc_mdl = '../models/balanced_accuracies_file_'
    results = pd.read_csv(loc_mdl + str(nr_file) + '_fold_' + str(nr_fold) + '.csv')
    results['file'] = nr_file
    results['fold'] = nr_fold
    nm_cols = [dim_red + '_' + nm_mdl for dim_red in ['br', 'pc', 'kpc'] for nm_mdl in ['kd', 'rf', 'lr']]
    nm_idx = list(results.columns.drop(nm_cols))
    stacked = pd.DataFrame(results.set_index(nm_idx).rename_axis('model',axis=1).stack().rename('bac')).reset_index()
    stacked['full_model'] = all_pars_in_model(stacked)
    return stacked

def all_pars_in_model(hyp_par):
    dim_red, model = zip(*hyp_par.model.str.split('_'))
    lr_pars = ['l1_' + str(l1) + ('_C_') + str(reg) if mdl.endswith('lr') else ''
               for l1, reg, mdl in zip(hyp_par.lr_l1, hyp_par.lr_reg, model)]
    rf_pars = ['max_leaf_' + str(leaf) + ('_max_n_estimates_') + str(est) if mdl.endswith('rf') else ''
               for leaf, est, mdl in zip(hyp_par.rf_max_leaf, hyp_par.rf_n_est, model)]
    kd_pars = ['kernel_' + str(krn) + ('_degree_') + str(dg) if mdl.endswith('kd') else ''
                    for krn, dg, mdl in zip(hyp_par.kda_kernel, hyp_par.kda_dg, model)]

    br_pars = ['keep_x_' + str(keep_x) + ('_rm_y_') + str(rm_y) if dr.startswith('br') else ''
                    for keep_x, rm_y, dr in zip(hyp_par.br_keep_x, hyp_par.br_rm_y, dim_red)]
    pc_pars = ['pc_' + str(npc) if dr.startswith(('pc','kpc')) else ''
               for npc, dr in zip(hyp_par.n_pc, dim_red)]
    model_pars = [''.join(pars) for pars in zip(lr_pars,rf_pars,kd_pars)]
    dim_red_pars = [''.join(pars) for pars in zip(br_pars,pc_pars)]
    return [' '.join(names) for names in zip(model, model_pars, dim_red, dim_red_pars)]
