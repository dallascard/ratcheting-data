import matplotlib
matplotlib.use('Agg')

import os
import json
from optparse import OptionParser
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from common import create_poly


def main():
    usage = "%prog path/to/config.json "
    parser = OptionParser(usage=usage)
    parser.add_option('--outdir', type=str, default=None,
                      help='Output dir [basedir of config if None]: default=%default')
    parser.add_option('--save', action="store_true", default=False,
                      help='Save data matrix: default=%default')

    (options, args) = parser.parse_args()
    config_file = args[0]
    print(config_file)
    with open(config_file) as f:
        config = json.load(f)
    for key, value in config.items():
        print(key, value)

    outdir = options.outdir
    if outdir is None:
        outdir = os.path.split(config_file)[0]
    if not os.path.exists(outdir):
        raise RuntimeError("Output directory does not exist")

    train_file = config['train_file']
    max_iter = config['max_iter']
    family = config.get('family', 'NegativeBinomial')
    subset_column = config.get('subset_column', None)
    subset_target = config.get('subset_target', None)

    df_train = pd.read_csv(train_file, header=0, index_col=0)

    if subset_column is not None and subset_target is not None:
        print("Taking subset of df")
        print(df_train.shape)
        df_train = df_train[df_train[subset_column] == subset_target]
        print(df_train.shape)

    target = config['target']

    factors = config['factors']

    interactions = config['interactions']

    intercept = config.get('intercept', True)

    l1_alpha = config.get('l1_alpha', None)

    lists = {}
    columns = {}
    types = {}
    poly_matrices = {}
    val_indices = {}

    y = df_train[target].values
    X = pd.DataFrame()

    X_pred = pd.DataFrame()

    zscore_stds = {}
    zscore_means = {}

    for factor in factors:
        name = factor['name']
        factor_type = factor['type']
        types[name] = factor_type
        transform = factor.get('transform', None)
        if factor_type == 'vector':
            if transform is not None and transform == 'log':
                X['log(' + name + ')'] = np.log(df_train[name].values)
                columns[name] = ['log(' + name + ')']
                X_pred['log(' + name + ')'] = [0]
            elif transform is not None and transform == 'zscore':
                values = df_train[name].values
                zscored_values, zmean, zstd = zscore_set(values)
                X['zscore(' + name + ')'] = zscored_values
                zscore_means[name] = zmean
                zscore_stds[name] = zstd
                X_pred['zscore(' + name + ')'] = [0]
            else:
                X[name] = df_train[name].values
                X_pred[name] = [0]

        elif factor_type == 'int':
            linear = factor.get('linear', False)
            quadratic = factor.get('quadratic', False)
            cubic = factor.get('cubic', False)
            include = factor.get('include', [])
            first = factor.get('first', None)
            last = factor.get('last', None)
            pred_val = factor.get('pred_val', 0)
            print(name, 'pred_val', pred_val)
            components_excl_linear = factor.get('components_excl_linear', None)
            factor_df, poly_matrix, val_index = convert_int_list_to_matrix(name, df_train[name].values, linear=linear, components_excl_linear=components_excl_linear)
            poly_matrices[name] = poly_matrix
            val_indices[name] = val_index
            columns[name] = list(factor_df.columns)
            for col in factor_df.columns:
                X[col] = factor_df[col].values

            factor_df_pred, _, _ = convert_int_list_to_matrix(name, [pred_val], linear=linear, poly_matrix=poly_matrices[name], val_index=val_index)
            for col in factor_df_pred.columns:
                X_pred[col] = factor_df_pred[col].values

        elif factor_type == 'str':
            exclude = factor.get('exclude', None)
            exclude_most_common = factor.get('exclude_most_common', False)
            min_count = factor.get('min_count', 0)
            factor_df = convert_string_list_to_matrix(name, df_train[name].values, exclude_most_common=exclude_most_common, exclude=exclude, min_count=min_count)
            lists[name] = list(factor_df.columns)
            columns[name] = list(factor_df.columns)
            for col in factor_df.columns:
                X[col] = factor_df[col].values
                X_pred[col] = [0]
        else:
            print(factor)
            raise RuntimeError("Factor type not recognized")

    if intercept:
        print("Adding intercept")
        X['const'] = 1.
        X_pred['const'] = 1.

    if options.save:
        X_copy = X.copy()
        X_copy[target] = y
        X_copy.to_csv(os.path.join(outdir, 'Xy.csv'))

    X, interaction_cols = add_interactions(X, interactions, columns)
    X_pred, _ = add_interactions(X_pred, interactions, columns)

    if family == 'Logistic':
        print("Using Logistic model")
        model = sm.Logit(y, X)
    elif family.lower() == 'linear':
        print("Using Linear model")
        model = sm.OLS(y, X)
    elif family == 'NegativeBinomial':
        print("Using negative Binomial model")
        model = sm.NegativeBinomial(y, X)
    else:
        raise ValueError("Model family not recognized", family)

    if l1_alpha is None:
        fit = model.fit(maxiter=max_iter)
    else:
        fit = model.fit_regularized(alpha=l1_alpha)

    params = fit.params
    intervals = fit.conf_int()
    stder = fit.bse

    pvalues = fit.pvalues
    aic = fit.aic
    bic = fit.bic
    llf = fit.llf
    print("AIC:", aic)
    print("BIC:", bic)

    params.to_csv(os.path.join(outdir, 'params.csv'))
    intervals.to_csv(os.path.join(outdir, 'intervals.csv'))
    stder.to_csv(os.path.join(outdir, 'stder.csv'))
    pvalues.to_csv(os.path.join(outdir, 'pvalues.csv'))
    for name, poly_matrix in poly_matrices.items():
        np.savez(os.path.join(outdir, name + '.npz'), matrix=poly_matrix)

    if options.save:
        fit.save(os.path.join(outdir, 'model.pkl'))

    report = {'aic': aic,
              'bic': bic,
              'llf': llf,
              'nans': int(np.isnan(pvalues.values).any())
              }
    with open(os.path.join(outdir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(outdir, 'columns.json'), 'w') as f:
        json.dump(columns, f, indent=2)

    with open(os.path.join(outdir, 'interactions.json'), 'w') as f:
       json.dump(interaction_cols, f, indent=2)

    resids = np.array(y) - np.array(fit.fittedvalues)
    order = np.arange(len(y))
    np.random.shuffle(order)
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(resids)), resids, alpha=0.2)
    plt.savefig(os.path.join(outdir, 'resids.pdf'), bbox_inches='tight')

    with open(os.path.join(outdir, 'zscore_means.json'), 'w') as f:
        json.dump(zscore_means, f, indent=2)

    with open(os.path.join(outdir, 'zscore_stds.json'), 'w') as f:
        json.dump(zscore_stds, f, indent=2)

    prediction = fit.predict(X_pred)
    print(prediction)
    X_pred['pred'] = prediction
    X_pred.to_csv(os.path.join(outdir, '2009_0_pred.json'))

    fit.save(os.path.join(outdir, 'model.pickle'))


def convert_data(df, factors, interactions, lists, columns, poly_matrices, val_indices):
    X = pd.DataFrame()
    for factor in factors:
        name = factor['name']
        factor_type = factor['type']
        transform = factor.get('transform', None)
        if factor_type == 'vector':
            if transform is not None and transform == 'log':
                X['log(' + name + ')'] = np.log(df[name].values)
            else:
                X[name] = df[name].values
        if factor_type == 'int':
            linear = factor.get('linear', False)
            quadratic = factor.get('quadratic', False)
            cubic = factor.get('cubic', False)
            include = factor.get('include', [])
            factor_df, _, _ = convert_int_list_to_matrix(name, df[name].values, linear=linear, poly_matrix=poly_matrices[name], val_index=val_indices[name])
            for col in factor_df.columns:
                X[col] = factor_df[col].values
        elif factor_type == 'str':
            factor_df = convert_string_list_to_matrix(name, df[name].values, existing_list=lists[name])
            for col in factor_df.columns:
                X[col] = factor_df[col].values

    X, _ = add_interactions(X, interactions, columns)

    return X


def add_interactions(X, interactions, columns):
    interaction_cols = defaultdict(list)
    for interaction in interactions:
        assert type(interaction) == list
        if len(interaction) == 2:
            f1, f2 = interaction
            name = f1 + '_X_' + f2
            if f1 in columns and f2 in columns:
                for col1 in columns[f1]:
                    for col2 in columns[f2]:
                        X[col1 + '_X_' + col2] = np.array(X[col1].values) * np.array(X[col2].values)
                        interaction_cols[name].append(col1 + '_X_' + col2)
            elif f1 in columns:
                for col in columns[f1]:
                    X[col + '_X_' + f2] = np.array(X[col].values) * np.array(X[f2].values)
                    interaction_cols[name].append(col + '_X_' + f2)
            elif f2 in columns:
                for col in columns[f2]:
                    X[f1 + '_X_' + col] = np.array(X[f1].values) * np.array(X[col].values)
                    interaction_cols[name].append(f1 + '_X_' + col)
            else:
                X[f1 + '_X_' + f2] = np.array(X[f1].values) * np.array(X[f2].values)
                interaction_cols[name].append(f1 + '_X_' + f2)
        else:
            factors = interaction
            print(factors)
            name = '_X_'.join([f for f in factors])
            product = np.array(X[factors[0]])
            for factor in factors[1:]:
                product = product * np.array(X[factor])
            X[name] = product
            interaction_cols[name].append(name)

    return X, interaction_cols


def convert_string_list_to_matrix(name, input_vals, exclude_most_common=False, exclude=None, min_count=0, existing_list=None):
    if existing_list is None:
        input_counter = Counter(input_vals)
        val_list = sorted([k for k, v in input_counter.items() if v >= min_count and type(k) == str])
        if exclude_most_common:
            print("Excluding", input_counter.most_common(n=1)[0][0])
            val_list.remove(input_counter.most_common(n=1)[0][0])
        elif exclude is not None:
            print("Excluding", exclude)
            val_list.remove(exclude)
    else:
        val_list = existing_list
    val_dict = dict(zip(val_list, range(len(val_list))))
    val_matrix = np.zeros([len(input_vals), len(val_list)])
    for i, v in enumerate(input_vals):
        if v in val_dict:
            val_matrix[i, val_dict[v]] = 1
    df = pd.DataFrame()
    for v_i, val in enumerate(val_list):
        df[name + '_' + str(val)] = val_matrix[:, v_i]
    return df


def convert_int_list_to_matrix(name, input_vals, linear=True, components_excl_linear=None, poly_matrix=None, val_index=None):

    if val_index is None or poly_matrix is None:
        val_list = sorted(set(input_vals))
        n_levels = len(val_list)
        val_index = dict(zip(val_list, range(n_levels)))

        # get a matrix mapping from n_levels levels to polynomial factors of length (n_levels-1)
        poly_matrix = create_poly(n_levels)
        # drop additional components if desired
        if components_excl_linear is not None:
            poly_matrix = poly_matrix[:, :components_excl_linear+1]
        # drop the linear term if desired
        if not linear:
            poly_matrix = poly_matrix[:, 1:]
    n_rows, n_cols = poly_matrix.shape

    if linear:
        column_names = [name + '_' + str(i) for i in range(1, n_cols+1)]
    else:
        column_names = [name + '_' + str(i) for i in range(2, n_cols+2)]

    rows = []
    for v in input_vals:
        index = val_index[v]
        rows.append(poly_matrix[index, :])

    df = pd.DataFrame(rows, columns=column_names)

    return df, poly_matrix, val_index


def zscore_set(vals, set_vals=None):
    if set_vals is None:
        mean = np.mean(vals)
        std = np.std(vals)
    else:
        mean = np.mean(set_vals)
        std = np.std(set_vals)
    print(mean, std)
    normalized = [(v - mean) / std for v in vals]
    return normalized, mean, std


if __name__ == '__main__':
    main()
